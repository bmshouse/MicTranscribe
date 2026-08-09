from __future__ import annotations

import queue
from datetime import datetime, timedelta

from mictranscribe.vad import UtteranceAssembler

FRAME_BYTES = 640  # 320 int16 samples, matches default 16kHz/20ms frame


def make_frame(marker: int) -> bytes:
    return bytes([marker % 256]) * FRAME_BYTES


def make_assembler(**overrides) -> UtteranceAssembler:
    kwargs = dict(
        frame_queue=queue.Queue(),
        utterance_queue=queue.Queue(),
        start_trigger_frames=3,
        end_trigger_ms=40,  # 2 frames at 20ms
        preroll_ms=100,  # 5 frames at 20ms
    )
    kwargs.update(overrides)
    return UtteranceAssembler(**kwargs)


def times(n: int, start: datetime | None = None, step_ms: int = 20) -> list[datetime]:
    base = start or datetime(2026, 1, 1, 0, 0, 0)
    return [base + timedelta(milliseconds=step_ms * i) for i in range(n)]


def test_brief_speech_below_start_threshold_produces_no_utterance():
    assembler = make_assembler()
    ts = times(4)

    results = [
        assembler.process_frame(make_frame(0), True, ts[0]),
        assembler.process_frame(make_frame(1), True, ts[1]),  # only 2 consecutive: below threshold of 3
        assembler.process_frame(make_frame(2), False, ts[2]),
        assembler.process_frame(make_frame(3), False, ts[3]),
    ]

    assert all(r is None for r in results)


def test_sustained_speech_then_pause_emits_one_utterance_with_correct_timestamps():
    assembler = make_assembler()
    ts = times(6)

    assert assembler.process_frame(make_frame(0), True, ts[0]) is None
    assert assembler.process_frame(make_frame(1), True, ts[1]) is None
    assert assembler.process_frame(make_frame(2), True, ts[2]) is None  # 3rd consecutive: start confirmed
    assert assembler.process_frame(make_frame(3), True, ts[3]) is None
    assert assembler.process_frame(make_frame(4), False, ts[4]) is None  # 1st silence frame
    utterance = assembler.process_frame(make_frame(5), False, ts[5])  # 2nd silence frame: end confirmed

    assert utterance is not None
    assert utterance.start == ts[0]
    assert utterance.end == ts[3]  # timestamp of the last frame classified as speech

    # back in SILENCE state, ready for the next utterance
    assert assembler.process_frame(make_frame(6), False, times(1, start=ts[5] + timedelta(milliseconds=20))[0]) is None


def test_preroll_frames_are_included_in_emitted_utterance():
    assembler = make_assembler()
    ts = times(16)

    # 10 silence frames roll through the 5-frame preroll buffer (markers 5-9 remain)
    for i in range(10):
        assembler.process_frame(make_frame(i), False, ts[i])

    # 3 consecutive speech frames confirm start (markers 10, 11, 12)
    for i in range(10, 13):
        assembler.process_frame(make_frame(i), True, ts[i])

    # one more speech frame, then 2 silence frames to end the utterance
    assembler.process_frame(make_frame(13), True, ts[13])
    assembler.process_frame(make_frame(14), False, ts[14])
    utterance = assembler.process_frame(make_frame(15), False, ts[15])

    assert utterance is not None
    expected_markers = [8, 9, 10, 11, 12, 13, 14, 15]  # preroll (8,9) + speech (10-13) + trailing silence (14,15)
    expected_bytes = b"".join(make_frame(m) for m in expected_markers)
    assert utterance.audio.tobytes() == expected_bytes
    assert utterance.start == ts[8]
    assert utterance.end == ts[13]
