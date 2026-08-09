from __future__ import annotations

from mictranscribe.levels import LevelReport, LevelVerdict


def test_quiet_verdict_at_or_below_threshold():
    assert LevelReport(peak=0, rms=0.0).verdict is LevelVerdict.QUIET
    assert LevelReport(peak=300, rms=50.0).verdict is LevelVerdict.QUIET


def test_ok_verdict_in_healthy_range():
    assert LevelReport(peak=301, rms=100.0).verdict is LevelVerdict.OK
    assert LevelReport(peak=11410, rms=2771.8).verdict is LevelVerdict.OK
    assert LevelReport(peak=31999, rms=8000.0).verdict is LevelVerdict.OK


def test_clipping_verdict_at_or_above_threshold():
    assert LevelReport(peak=32000, rms=9000.0).verdict is LevelVerdict.CLIPPING
    assert LevelReport(peak=32767, rms=10000.0).verdict is LevelVerdict.CLIPPING
