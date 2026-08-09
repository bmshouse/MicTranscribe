from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd

from .devices import find_device_by_name, list_input_devices

logger = logging.getLogger("mictranscribe.audio_capture")

_INT16_MIN = -32768
_INT16_MAX = 32767


def gain_db_to_factor(gain_db: float) -> float:
    """Convert a gain in decibels to a linear amplitude multiplier."""
    return 10 ** (gain_db / 20.0)


def apply_gain(samples: np.ndarray, gain_factor: float) -> bytes:
    """Scale int16 PCM samples by a linear factor, clipping to avoid wraparound.

    A factor of 1.0 is a no-op passthrough (default: mic hardware level
    unchanged). Values above 1.0 boost a mic that's too quiet to reliably
    trigger VAD/transcription; values below 1.0 attenuate a hot input that's
    clipping.
    """
    if gain_factor == 1.0:
        return bytes(samples)
    boosted = samples.astype(np.float32) * gain_factor
    clipped = np.clip(boosted, _INT16_MIN, _INT16_MAX).astype(np.int16)
    return clipped.tobytes()


class AudioCapture:
    """Streams raw PCM frames from a named input device into a queue.

    Runs a watchdog that detects a stalled/disconnected stream (e.g. a
    USB mic unplugged) and reopens it by device name with exponential
    backoff, so the service keeps running through a hot-unplug/replug.
    """

    def __init__(
        self,
        device_name: str,
        frame_queue: "queue.Queue[bytes]",
        samplerate: int = 16000,
        frame_ms: int = 20,
        stall_timeout: float = 3.0,
        gain_db: float = 0.0,
        device_index: int | None = None,
    ) -> None:
        self.device_name = device_name
        self.frame_queue = frame_queue
        self.samplerate = samplerate
        self.frame_ms = frame_ms
        self.frame_samples = int(samplerate * frame_ms / 1000)
        self.stall_timeout = stall_timeout
        self.gain_db = gain_db
        self._gain_factor = gain_db_to_factor(gain_db)
        if gain_db != 0.0:
            logger.info("Mic gain set to %.1f dB (%.3fx)", gain_db, self._gain_factor)

        # Use the exact device the caller already resolved (e.g. from the
        # interactive picker) for the first open, rather than re-deriving it
        # from device_name — on Windows the same physical mic commonly shows
        # up multiple times under different host APIs (MME/DirectSound/
        # WASAPI/WDM-KS) sharing an identical name, and name-based lookup
        # would silently pick whichever comes first rather than what was
        # actually chosen. Cleared on reconnect, where a replugged device
        # may genuinely have a new index and name-based re-lookup is the
        # only option.
        self._known_index = device_index

        self._stream: sd.InputStream | None = None
        self._stopped = threading.Event()
        self._last_callback_time = time.time()
        self._watchdog_thread: threading.Thread | None = None

    def start(self) -> None:
        self._stopped.clear()
        self._open_stream()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("Error closing audio stream")

    def _resolve_device_index(self) -> int:
        if self._known_index is not None:
            return self._known_index
        devices = list_input_devices()
        device = find_device_by_name(devices, self.device_name)
        if device is None:
            raise RuntimeError(f"Audio input device '{self.device_name}' not found")
        return device.index

    def _open_stream(self) -> None:
        index = self._resolve_device_index()
        self._stream = sd.InputStream(
            device=index,
            channels=1,
            samplerate=self.samplerate,
            dtype="int16",
            blocksize=self.frame_samples,
            callback=self._callback,
        )
        self._stream.start()
        self._last_callback_time = time.time()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.warning("Audio stream status flag: %s", status)
        self._last_callback_time = time.time()
        try:
            self.frame_queue.put_nowait(apply_gain(indata, self._gain_factor))
        except queue.Full:
            logger.warning("Frame queue full, dropping an audio frame")

    def _watchdog_loop(self) -> None:
        while not self._stopped.wait(1.0):
            if time.time() - self._last_callback_time > self.stall_timeout:
                logger.warning("Audio stream stalled (no callback for >%.1fs), reconnecting", self.stall_timeout)
                self._reopen_with_backoff()

    def _reopen_with_backoff(self) -> None:
        self._known_index = None
        delay = 1.0
        while not self._stopped.is_set():
            try:
                if self._stream is not None:
                    self._stream.close()
            except Exception:
                pass
            try:
                self._open_stream()
                logger.info("Audio device reconnected")
                return
            except Exception as exc:
                logger.error("Reconnect attempt failed: %s; retrying in %.1fs", exc, delay)
                if self._stopped.wait(delay):
                    return
                delay = min(delay * 2, 30.0)
