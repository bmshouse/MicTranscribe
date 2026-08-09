from __future__ import annotations

import numpy as np

from mictranscribe.audio_capture import apply_gain, gain_db_to_factor


def test_zero_db_is_unity_factor():
    assert gain_db_to_factor(0.0) == 1.0


def test_positive_db_boosts_amplitude():
    factor = gain_db_to_factor(6.0)
    assert factor > 1.9 and factor < 2.1  # +6dB is ~2x amplitude


def test_negative_db_attenuates_amplitude():
    factor = gain_db_to_factor(-6.0)
    assert factor > 0.4 and factor < 0.6  # -6dB is ~0.5x amplitude


def test_unity_gain_is_passthrough():
    samples = np.array([[100], [-100], [30000]], dtype=np.int16)
    assert apply_gain(samples, 1.0) == bytes(samples)


def test_gain_boosts_quiet_signal():
    samples = np.array([[100], [-100]], dtype=np.int16)
    result = np.frombuffer(apply_gain(samples, 2.0), dtype=np.int16)
    assert list(result) == [200, -200]


def test_gain_clips_instead_of_wrapping_around():
    samples = np.array([[20000], [-20000]], dtype=np.int16)
    result = np.frombuffer(apply_gain(samples, 3.0), dtype=np.int16)
    # 20000 * 3 = 60000, far above int16 range: must clip, never wrap negative
    assert list(result) == [32767, -32768]
