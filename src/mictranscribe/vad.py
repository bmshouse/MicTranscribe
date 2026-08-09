from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from datetime import datetime
from enum import Enum, auto
from typing import Callable

import numpy as np
import webrtcvad

from .types import Utterance

logger = logging.getLogger("mictranscribe.vad")


class _State(Enum):
    SILENCE = auto()
    SPEECH = auto()


class UtteranceAssembler:
    """Turns a stream of fixed-size PCM frames into discrete Utterances.

    Uses hysteresis (N consecutive frames) on both the speech-start and
    speech-end transitions to avoid false triggers on brief noises and to
    require a real pause before finalizing an utterance. A rolling
    pre-roll buffer is prepended to each utterance so the first syllable
    isn't clipped.
    """

    def __init__(
        self,
        frame_queue: "queue.Queue[bytes]",
        utterance_queue: "queue.Queue[Utterance]",
        samplerate: int = 16000,
        frame_ms: int = 20,
        vad_aggressiveness: int = 2,
        start_trigger_frames: int = 3,
        end_trigger_ms: int = 1200,
        preroll_ms: int = 300,
        max_utterance_ms: int = 30000,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.frame_queue = frame_queue
        self.utterance_queue = utterance_queue
        self.samplerate = samplerate
        self.frame_ms = frame_ms
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.start_trigger_frames = start_trigger_frames
        self.end_trigger_frames = max(1, end_trigger_ms // frame_ms)
        self.max_utterance_frames = max(1, max_utterance_ms // frame_ms)
        self._clock = clock
        self._stopped = threading.Event()

        preroll_frames = max(1, preroll_ms // frame_ms)
        self._preroll: deque[tuple[bytes, datetime]] = deque(maxlen=preroll_frames)

        self._state = _State.SILENCE
        self._speech_run = 0
        self._silence_run = 0
        self._buffer: list[bytes] = []
        self._utterance_start: datetime | None = None
        self._last_speech_time: datetime | None = None

    def run(self) -> None:
        while not self._stopped.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            utterance = self.process_frame(frame, self._is_speech(frame), self._clock())
            if utterance is not None:
                self.utterance_queue.put(utterance)

    def stop(self) -> None:
        self._stopped.set()

    def _is_speech(self, frame: bytes) -> bool:
        try:
            return self.vad.is_speech(frame, self.samplerate)
        except Exception:
            logger.exception("VAD classification failed on a frame; treating it as silence")
            return False

    def process_frame(self, frame: bytes, is_speech: bool, now: datetime) -> Utterance | None:
        """Advance the state machine by one frame. Returns a completed
        Utterance when a pause (or the max-duration safety valve) ends one.
        Exposed separately from run() so the state machine can be unit
        tested with scripted is_speech/now values, without real audio or
        webrtcvad.
        """
        self._preroll.append((frame, now))

        if self._state == _State.SILENCE:
            if is_speech:
                self._speech_run += 1
                if self._speech_run >= self.start_trigger_frames:
                    self._state = _State.SPEECH
                    self._buffer = [f for f, _ in self._preroll]
                    self._utterance_start = self._preroll[0][1]
                    self._last_speech_time = now
                    self._silence_run = 0
            else:
                self._speech_run = 0
            return None

        # _State.SPEECH
        self._buffer.append(frame)
        if is_speech:
            self._silence_run = 0
            self._last_speech_time = now
        else:
            self._silence_run += 1

        if self._silence_run >= self.end_trigger_frames or len(self._buffer) >= self.max_utterance_frames:
            return self._emit()
        return None

    def _emit(self) -> Utterance:
        audio = np.frombuffer(b"".join(self._buffer), dtype=np.int16)
        utterance = Utterance(
            audio=audio,
            samplerate=self.samplerate,
            start=self._utterance_start,
            end=self._last_speech_time,
        )
        self._reset()
        return utterance

    def _reset(self) -> None:
        self._state = _State.SILENCE
        self._speech_run = 0
        self._silence_run = 0
        self._buffer = []
        self._utterance_start = None
        self._last_speech_time = None
