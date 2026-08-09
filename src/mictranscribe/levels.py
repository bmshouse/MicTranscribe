from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import sounddevice as sd

logger = logging.getLogger("mictranscribe.levels")

# Peak amplitude out of a possible 32767 (int16 full scale).
QUIET_PEAK_THRESHOLD = 300
CLIPPING_PEAK_THRESHOLD = 32000


class LevelVerdict(Enum):
    OK = auto()
    QUIET = auto()
    CLIPPING = auto()


@dataclass
class LevelReport:
    peak: int
    rms: float

    @property
    def verdict(self) -> LevelVerdict:
        if self.peak <= QUIET_PEAK_THRESHOLD:
            return LevelVerdict.QUIET
        if self.peak >= CLIPPING_PEAK_THRESHOLD:
            return LevelVerdict.CLIPPING
        return LevelVerdict.OK


def measure_input_level(device_index: int, samplerate: int = 16000, duration: float = 1.0) -> LevelReport:
    """Sample ambient input for a short window and report its level.

    Doesn't require the user to speak — even room tone from a working mic
    reads as a small nonzero signal, whereas a muted/misrouted input reads
    as near-total silence. That distinction is what QUIET_PEAK_THRESHOLD
    catches (e.g. a PulseAudio source volume left at 26%, as opposed to
    someone simply being quiet in the room).

    Takes a resolved device index, not a name: on Windows the same physical
    mic commonly appears multiple times under different host APIs (MME,
    DirectSound, WASAPI, WDM-KS) sharing the exact same name string, and
    sounddevice raises rather than guessing which one you meant if you pass
    an ambiguous name directly.
    """
    data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16", device=device_index)
    sd.wait()
    peak = int(np.abs(data).max())
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    return LevelReport(peak=peak, rms=rms)


def log_level_report(report: LevelReport) -> None:
    logger.info("Input level check: peak=%d/32767 rms=%.1f", report.peak, report.rms)
    if report.verdict is LevelVerdict.QUIET:
        logger.warning(
            "Input level is very low (peak=%d/32767). VAD may fail to detect speech. "
            "Check your OS/PulseAudio mixer volume for this device, verify the correct "
            "microphone is selected, or raise --gain-db as a software fallback.",
            report.peak,
        )
    elif report.verdict is LevelVerdict.CLIPPING:
        logger.warning(
            "Input level is at or near clipping (peak=%d/32767). Lower your OS/PulseAudio "
            "mixer volume for this device or reduce --gain-db.",
            report.peak,
        )
