from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class Utterance:
    """A single assembled speech segment, ready for transcription."""

    audio: np.ndarray  # int16 mono PCM samples
    samplerate: int
    start: datetime
    end: datetime


@dataclass
class InputDevice:
    index: int
    name: str
    max_input_channels: int
