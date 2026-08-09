from __future__ import annotations

from types import SimpleNamespace

from mictranscribe.transcriber import assemble_transcript, forced_language, segment_passes_quality_filter


def _segment(text="hello", no_speech_prob=0.1, avg_logprob=-0.3, compression_ratio=1.5):
    return SimpleNamespace(
        text=text, no_speech_prob=no_speech_prob, avg_logprob=avg_logprob, compression_ratio=compression_ratio
    )


def _info(language="en"):
    return SimpleNamespace(language=language)


def test_forced_language_none_when_unrestricted():
    assert forced_language(()) is None


def test_forced_language_none_when_multiple_allowed():
    assert forced_language(("en", "ja")) is None


def test_forced_language_single_value_when_exactly_one_allowed():
    assert forced_language(("en",)) == "en"


def test_quality_filter_accepts_confident_speech_segment():
    assert segment_passes_quality_filter(_segment(), 0.6, -1.0, 2.4) is True


def test_quality_filter_rejects_high_no_speech_prob():
    seg = _segment(no_speech_prob=0.9)
    assert segment_passes_quality_filter(seg, 0.6, -1.0, 2.4) is False


def test_quality_filter_rejects_low_avg_logprob():
    seg = _segment(avg_logprob=-2.0)
    assert segment_passes_quality_filter(seg, 0.6, -1.0, 2.4) is False


def test_quality_filter_rejects_high_compression_ratio():
    # repetitive/looping hallucinated text compresses unusually well
    seg = _segment(compression_ratio=3.0)
    assert segment_passes_quality_filter(seg, 0.6, -1.0, 2.4) is False


def test_quality_filter_boundary_values_pass():
    seg = _segment(no_speech_prob=0.6, avg_logprob=-1.0, compression_ratio=2.4)
    assert segment_passes_quality_filter(seg, 0.6, -1.0, 2.4) is True


def test_assemble_transcript_joins_good_segments():
    segments = [_segment("hello"), _segment("world")]
    text = assemble_transcript(segments, _info("en"), (), 0.6, -1.0, 2.4)
    assert text == "hello world"


def test_assemble_transcript_drops_utterance_when_language_not_allowed():
    segments = [_segment("garbled nonsense")]
    text = assemble_transcript(segments, _info("km"), ("en", "ja", "ko", "fr", "es"), 0.6, -1.0, 2.4)
    assert text == ""


def test_assemble_transcript_keeps_utterance_when_language_allowed():
    segments = [_segment("bonjour")]
    text = assemble_transcript(segments, _info("fr"), ("en", "ja", "ko", "fr", "es"), 0.6, -1.0, 2.4)
    assert text == "bonjour"


def test_assemble_transcript_drops_only_bad_segments_keeps_good_ones():
    segments = [
        _segment("real speech", no_speech_prob=0.1),
        _segment("ʕ ʕ ʔ hallucination", no_speech_prob=0.95),
    ]
    text = assemble_transcript(segments, _info("en"), (), 0.6, -1.0, 2.4)
    assert text == "real speech"


def test_assemble_transcript_empty_when_no_segments_survive():
    segments = [_segment(no_speech_prob=0.99)]
    text = assemble_transcript(segments, _info("en"), (), 0.6, -1.0, 2.4)
    assert text == ""
