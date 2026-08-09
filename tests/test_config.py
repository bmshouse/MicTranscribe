from __future__ import annotations

import argparse
from pathlib import Path

from mictranscribe.config import AppConfig, load_config, merge_cli_overrides, save_config


def test_load_config_defaults_when_file_missing(tmp_path):
    cfg = load_config(tmp_path / "does-not-exist.toml")
    assert cfg == AppConfig()


def test_save_then_load_round_trips_values(tmp_path):
    path = tmp_path / "config.toml"
    original = AppConfig(
        device_name="USB Mic",
        model_size="medium",
        whisper_device="cuda",
        compute_type="float16",
        output_dir=Path("/tmp/transcripts"),
        silence_duration_ms=900,
        vad_aggressiveness=3,
        mic_gain_db=6.5,
        pulse_source_name="alsa_input.usb-example.mono-fallback",
        pulse_source_volume_percent=300,
        pulse_server="unix:/tmp/pulse.sock",
        allowed_languages=("en", "ja", "ko", "fr", "es"),
        no_speech_threshold=0.5,
        log_prob_threshold=-0.8,
        compression_ratio_threshold=2.2,
    )
    save_config(original, path)

    loaded = load_config(path)
    assert loaded == original


def test_save_then_load_round_trips_none_pulse_fields(tmp_path):
    path = tmp_path / "config.toml"
    original = AppConfig()  # pulse_* fields default to None
    save_config(original, path)

    loaded = load_config(path)
    assert loaded.pulse_source_name is None
    assert loaded.pulse_source_volume_percent is None
    assert loaded.pulse_server is None


def _namespace(**overrides) -> argparse.Namespace:
    base = dict(
        device=None,
        model_size=None,
        whisper_device=None,
        compute_type=None,
        output_dir=None,
        silence_ms=None,
        vad_aggressiveness=None,
        gain_db=None,
        pulse_source=None,
        pulse_volume=None,
        pulse_server=None,
        allowed_languages=None,
        no_speech_threshold=None,
        log_prob_threshold=None,
        compression_ratio_threshold=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cli_overrides_only_apply_when_explicitly_set():
    cfg = AppConfig(model_size="small", silence_duration_ms=1200)
    args = _namespace(model_size="large-v3")

    merged = merge_cli_overrides(cfg, args)

    assert merged.model_size == "large-v3"  # overridden
    assert merged.silence_duration_ms == 1200  # untouched, args value was None
    assert merged.device_name == cfg.device_name


def test_cli_output_dir_override():
    cfg = AppConfig(output_dir=Path("."))
    args = _namespace(output_dir="/data/transcripts")

    merged = merge_cli_overrides(cfg, args)

    assert merged.output_dir == Path("/data/transcripts")


def test_cli_gain_db_override():
    cfg = AppConfig(mic_gain_db=0.0)
    args = _namespace(gain_db=6.0)

    merged = merge_cli_overrides(cfg, args)

    assert merged.mic_gain_db == 6.0


def test_cli_pulse_overrides():
    cfg = AppConfig()
    args = _namespace(pulse_source="alsa_input.usb-x", pulse_volume=200, pulse_server="unix:/tmp/pulse.sock")

    merged = merge_cli_overrides(cfg, args)

    assert merged.pulse_source_name == "alsa_input.usb-x"
    assert merged.pulse_source_volume_percent == 200
    assert merged.pulse_server == "unix:/tmp/pulse.sock"


def test_cli_allowed_languages_override_parses_comma_separated_list():
    cfg = AppConfig()
    args = _namespace(allowed_languages="en, ja,ko ,fr,es")

    merged = merge_cli_overrides(cfg, args)

    assert merged.allowed_languages == ("en", "ja", "ko", "fr", "es")


def test_cli_hallucination_threshold_overrides():
    cfg = AppConfig()
    args = _namespace(no_speech_threshold=0.4, log_prob_threshold=-0.7, compression_ratio_threshold=2.0)

    merged = merge_cli_overrides(cfg, args)

    assert merged.no_speech_threshold == 0.4
    assert merged.log_prob_threshold == -0.7
    assert merged.compression_ratio_threshold == 2.0
