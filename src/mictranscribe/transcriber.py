from __future__ import annotations

import importlib
import logging
import os
import queue
import sys
import threading

import numpy as np
from faster_whisper import WhisperModel

from .types import Utterance
from .writer import TranscriptWriter

logger = logging.getLogger("mictranscribe.transcriber")

_NVIDIA_LIB_PACKAGES = ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc")


def ensure_cuda_libs_on_path() -> None:
    """CTranslate2's CUDA backend dlopen()s cuBLAS/cuDNN lazily, on first GPU
    use. When installed via the `gpu` extra, those .so files come from pip
    packages (nvidia-cublas-cu12, nvidia-cudnn-cu12) that don't sit on the
    linker's default search path. Without this, `--whisper-device auto`
    happily selects CUDA but then fails on the first real transcription with
    "Library libcublas.so.12 is not found". glibc's dynamic linker only
    reads LD_LIBRARY_PATH once, at process start — mutating os.environ mid-
    process doesn't affect dlopen() calls already in flight in this process,
    so we re-exec ourselves once with the corrected environment instead. A
    CPU-only install simply won't have these packages, so this no-ops there.

    Called once from cli.main(), near the very start — before the device
    picker, PulseAudio volume setup, or level check — not at module import
    time and not from TranscriptionWorker.__init__. Two failure modes drove
    that: re-exec()ing during import means anything that merely imports
    this module (e.g. a test file exercising the pure functions below, with
    no GPU involved) inherits whatever the importer had done with stdio —
    under pytest this silently swallowed all output, since its fd-level
    capture redirect was in effect during collection and the re-exec'd
    process inherited that now-orphaned redirect instead of the real
    terminal. And calling it from TranscriptionWorker.__init__ meant the
    re-exec (a full process restart) happened *after* the interactive
    prompt/PulseAudio/level-check startup sequence had already run once,
    so the user saw all of that duplicated — the restarted process runs
    main() from scratch, right down to the mic picker prompt.
    """
    lib_dirs = []
    for pkg in _NVIDIA_LIB_PACKAGES:
        try:
            module = importlib.import_module(pkg)
        except ImportError:
            continue
        # These nvidia-*-cu12 packages are PEP 420 namespace packages (no
        # __init__.py), so __file__ is None; the package directory lives in
        # __path__ instead.
        bases = list(getattr(module, "__path__", None) or [])
        if not bases and getattr(module, "__file__", None):
            bases = [os.path.dirname(module.__file__)]
        for base in bases:
            lib_dir = os.path.join(base, "lib")
            if os.path.isdir(lib_dir):
                lib_dirs.append(lib_dir)
                break

    if not lib_dirs:
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_entries = existing.split(os.pathsep) if existing else []
    if all(d in existing_entries for d in lib_dirs):
        return  # already applied, e.g. a prior re-exec or a manual export

    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs + existing_entries)
    os.execv(sys.executable, [sys.executable] + sys.argv)


def forced_language(allowed_languages: tuple[str, ...]) -> str | None:
    """When exactly one language is allowed, pass it directly to Whisper to
    skip language auto-detection entirely (faster, and immune to the
    auto-detector guessing an exotic language on pure noise). With zero or
    multiple allowed languages, auto-detect and rely on the post-hoc
    language-allowlist filter below instead."""
    return allowed_languages[0] if len(allowed_languages) == 1 else None


def segment_passes_quality_filter(
    segment,
    no_speech_threshold: float,
    log_prob_threshold: float,
    compression_ratio_threshold: float,
) -> bool:
    """Whisper's classic hallucination heuristics: a segment decoded from
    silence/background noise tends to have high no_speech_prob, low average
    token log-probability, and/or a high compression ratio (repetitive
    looping text). faster-whisper already applies its own, more lenient
    versions of these internally; this is a second, independently-tunable
    pass specifically for a service that expects mostly silence, where a
    false positive is pure annoyance and a missed word is comparatively
    cheap.
    """
    if segment.no_speech_prob > no_speech_threshold:
        return False
    if segment.avg_logprob < log_prob_threshold:
        return False
    if segment.compression_ratio > compression_ratio_threshold:
        return False
    return True


def assemble_transcript(
    segments,
    info,
    allowed_languages: tuple[str, ...],
    no_speech_threshold: float,
    log_prob_threshold: float,
    compression_ratio_threshold: float,
) -> str:
    """Turn a raw (segments, info) result from WhisperModel.transcribe() into
    final transcript text, applying the language allowlist and per-segment
    quality filters. Kept separate from TranscriptionWorker so it's testable
    without constructing a real model."""
    if allowed_languages and info.language not in allowed_languages:
        logger.debug(
            "Dropping utterance: detected language '%s' not in allowed set %s", info.language, allowed_languages
        )
        return ""

    kept = [
        segment
        for segment in segments
        if segment_passes_quality_filter(segment, no_speech_threshold, log_prob_threshold, compression_ratio_threshold)
    ]
    return " ".join(segment.text.strip() for segment in kept).strip()


class TranscriptionWorker:
    """Pulls assembled utterances off a queue and transcribes them one at a
    time with a local Whisper model. A failure transcribing a single
    utterance is logged and skipped rather than crashing the service.
    """

    def __init__(
        self,
        utterance_queue: "queue.Queue[Utterance]",
        writer: TranscriptWriter,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "default",
        allowed_languages: tuple[str, ...] = (),
        no_speech_threshold: float = 0.6,
        log_prob_threshold: float = -1.0,
        compression_ratio_threshold: float = 2.4,
    ) -> None:
        self.utterance_queue = utterance_queue
        self.writer = writer
        self.allowed_languages = allowed_languages
        self.no_speech_threshold = no_speech_threshold
        self.log_prob_threshold = log_prob_threshold
        self.compression_ratio_threshold = compression_ratio_threshold
        self._stopped = threading.Event()
        logger.info("Loading Whisper model '%s' (device=%s, compute_type=%s)", model_size, device, compute_type)
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def run(self) -> None:
        while not self._stopped.is_set():
            try:
                utterance = self.utterance_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                text = self._transcribe(utterance)
                if text:
                    self.writer.write_line(utterance.start, utterance.end, text)
            except Exception:
                logger.exception("Transcription failed for an utterance; skipping it")

    def stop(self) -> None:
        self._stopped.set()

    def _transcribe(self, utterance: Utterance) -> str:
        audio = utterance.audio.astype(np.float32) / 32768.0
        segments, info = self._model.transcribe(
            audio,
            language=forced_language(self.allowed_languages),
            vad_filter=True,
        )
        return assemble_transcript(
            list(segments),
            info,
            self.allowed_languages,
            self.no_speech_threshold,
            self.log_prob_threshold,
            self.compression_ratio_threshold,
        )
