from __future__ import annotations

import argparse
import logging
import queue
import signal
import threading
from pathlib import Path

from . import pulse
from .audio_capture import AudioCapture
from .config import (
    VALID_COMPUTE_TYPES,
    VALID_WHISPER_DEVICES,
    AppConfig,
    config_path,
    load_config,
    merge_cli_overrides,
    save_config,
)
from .devices import list_input_devices, print_devices, resolve_device
from .levels import log_level_report, measure_input_level
from .lock import SingleInstanceError, acquire_singleton_lock
from .logging_setup import configure_logging
from .transcriber import TranscriptionWorker, ensure_cuda_libs_on_path
from .types import Utterance
from .vad import UtteranceAssembler
from .writer import TranscriptWriter

logger = logging.getLogger("mictranscribe.cli")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mictranscribe",
        description="Local, offline microphone transcription to daily text logs.",
    )
    parser.add_argument("--device", default=None, help="Microphone name (substring match)")
    parser.add_argument(
        "--model-size", default=None, help="Whisper model size, e.g. large-v3-turbo (default), small, medium"
    )
    parser.add_argument("--whisper-device", default=None, choices=VALID_WHISPER_DEVICES, help="Inference backend")
    parser.add_argument("--compute-type", default=None, choices=VALID_COMPUTE_TYPES, help="Whisper compute precision")
    parser.add_argument("--output-dir", default=None, help="Directory for transcripts and logs")
    parser.add_argument("--silence-ms", type=int, default=None, help="Trailing silence (ms) that ends an utterance")
    parser.add_argument("--vad-aggressiveness", type=int, default=None, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--gain-db",
        type=float,
        default=None,
        help="Digital mic gain in dB applied after capture (e.g. 6 to boost a quiet mic, -6 to attenuate a hot one)",
    )
    parser.add_argument(
        "--pulse-source",
        dest="pulse_source",
        default=None,
        help="PulseAudio source name to set the volume of at startup (Linux only, e.g. "
        "alsa_input.usb-...-mono-fallback; see --list-pulse-sources)",
    )
    parser.add_argument(
        "--pulse-volume",
        dest="pulse_volume",
        type=int,
        default=None,
        help="PulseAudio source volume percent to set at startup, e.g. 100 or 300 for a soft boost "
        "(requires --pulse-source or the equivalent config value)",
    )
    parser.add_argument(
        "--pulse-server",
        dest="pulse_server",
        default=None,
        help="PulseAudio server address override, e.g. unix:/path/to/pulse.sock "
        "(for a non-default PulseAudio instance)",
    )
    parser.add_argument(
        "--allowed-languages",
        dest="allowed_languages",
        default=None,
        help="Comma-separated Whisper language codes to accept, e.g. 'en,ja,ko,fr,es'. "
        "A single code skips auto-detection entirely; an utterance detected as any other "
        "language is discarded (helps reject noise misheard as an exotic language). "
        "Unset = accept any detected language.",
    )
    parser.add_argument(
        "--no-speech-threshold",
        dest="no_speech_threshold",
        type=float,
        default=None,
        help="Discard a segment if Whisper's no-speech probability exceeds this (0-1)",
    )
    parser.add_argument(
        "--log-prob-threshold",
        dest="log_prob_threshold",
        type=float,
        default=None,
        help="Discard a segment if its average token log-probability is below this",
    )
    parser.add_argument(
        "--compression-ratio-threshold",
        dest="compression_ratio_threshold",
        type=float,
        default=None,
        help="Discard a segment if its gzip compression ratio exceeds this (catches repetitive/looping hallucinations)",
    )
    parser.add_argument("--list-devices", action="store_true", help="Print input devices and exit")
    parser.add_argument(
        "--list-pulse-sources", action="store_true", help="Print PulseAudio sources (Linux only) and exit"
    )
    parser.add_argument(
        "--check-levels",
        action="store_true",
        help="Sample the selected microphone for a moment, report its input level, and exit",
    )
    parser.add_argument("--config", default=None, help="Path to a specific config file")
    parser.add_argument("--verbose", action="store_true", help="Debug-level console logging")
    return parser.parse_args(argv)


def _apply_pulse_volume(cfg: AppConfig) -> None:
    if cfg.pulse_source_name is None or cfg.pulse_source_volume_percent is None:
        return
    pulse.set_source_volume(cfg.pulse_source_name, cfg.pulse_source_volume_percent, cfg.pulse_server)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_devices:
        print_devices(list_input_devices())
        return

    if args.list_pulse_sources:
        try:
            print(pulse.list_sources(args.pulse_server), end="")
        except RuntimeError as exc:
            print(f"Error: {exc}")
        return

    # Must happen before any interactive/side-effecting startup work (mic
    # picker, PulseAudio volume, level check): this may re-exec the whole
    # process once to fix up LD_LIBRARY_PATH for GPU use, and anything run
    # before that point would otherwise appear to happen twice.
    ensure_cuda_libs_on_path()

    config_file = Path(args.config) if args.config else None

    # A one-shot diagnostic (no persistent capture/transcription) doesn't
    # conflict with a running instance, so it's exempt from the lock and can
    # be run freely alongside one. Anything that reaches this point and
    # isn't --check-levels is about to run the persistent pipeline, so grab
    # the lock immediately — before the mic picker/PulseAudio/level-check —
    # so a duplicate launch fails fast with a clear message instead of
    # walking through the whole interactive startup sequence first.
    lock = None
    if not args.check_levels:
        try:
            lock = acquire_singleton_lock(config_file or config_path())
        except SingleInstanceError as exc:
            print(f"Error: {exc}")
            return

    cfg = merge_cli_overrides(load_config(config_file), args)

    configure_logging(cfg.output_dir, verbose=args.verbose)

    devices = list_input_devices()
    chosen = resolve_device(devices, args.device, cfg.device_name)
    if chosen is None:
        print(f"Error: no microphone matching '{args.device}' was found. Available devices:")
        print_devices(devices)
        if lock is not None:
            lock.release()
        return
    if chosen.name != cfg.device_name:
        cfg.device_name = chosen.name
        save_config(cfg, config_file)

    _apply_pulse_volume(cfg)

    level_report = measure_input_level(chosen.index)
    log_level_report(level_report)
    if args.check_levels:
        return

    frame_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=500)
    utterance_queue: "queue.Queue[Utterance]" = queue.Queue(maxsize=8)

    writer = TranscriptWriter(cfg.output_dir)
    capture = AudioCapture(chosen.name, frame_queue=frame_queue, gain_db=cfg.mic_gain_db, device_index=chosen.index)
    assembler = UtteranceAssembler(
        frame_queue=frame_queue,
        utterance_queue=utterance_queue,
        vad_aggressiveness=cfg.vad_aggressiveness,
        end_trigger_ms=cfg.silence_duration_ms,
    )
    transcriber = TranscriptionWorker(
        utterance_queue=utterance_queue,
        writer=writer,
        model_size=cfg.model_size,
        device=cfg.whisper_device,
        compute_type=cfg.compute_type,
        allowed_languages=cfg.allowed_languages,
        no_speech_threshold=cfg.no_speech_threshold,
        log_prob_threshold=cfg.log_prob_threshold,
        compression_ratio_threshold=cfg.compression_ratio_threshold,
    )

    assembler_thread = threading.Thread(target=assembler.run, daemon=True)
    transcriber_thread = threading.Thread(target=transcriber.run, daemon=True)
    assembler_thread.start()
    transcriber_thread.start()
    capture.start()

    logger.info("Listening on '%s'. Press Ctrl-C to stop.", chosen.name)

    stop_event = threading.Event()
    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is not None:
            signal.signal(sig, lambda *_: stop_event.set())
    stop_event.wait()

    logger.info("Shutting down...")
    capture.stop()
    assembler.stop()
    transcriber.stop()
    assembler_thread.join(timeout=2)
    transcriber_thread.join(timeout=2)
    writer.close()
    lock.release()


if __name__ == "__main__":
    main()
