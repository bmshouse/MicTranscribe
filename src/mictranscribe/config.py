from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import tomli_w
from platformdirs import user_config_dir

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

APP_NAME = "mictranscribe"

VALID_WHISPER_DEVICES = ("auto", "cpu", "cuda")
VALID_COMPUTE_TYPES = ("default", "int8", "int8_float16", "float16")


@dataclass
class AppConfig:
    device_name: str | None = None
    model_size: str = "large-v3-turbo"
    whisper_device: str = "auto"
    compute_type: str = "default"
    output_dir: Path = field(default_factory=Path.cwd)
    silence_duration_ms: int = 1200
    vad_aggressiveness: int = 3
    mic_gain_db: float = 0.0
    pulse_source_name: str | None = None
    pulse_source_volume_percent: int | None = None
    pulse_server: str | None = None
    allowed_languages: tuple[str, ...] = ()
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4


def config_path() -> Path:
    return Path(user_config_dir(APP_NAME)) / "config.toml"


def load_config(path: Path | None = None) -> AppConfig:
    path = path or config_path()
    if not path.exists():
        return AppConfig()

    with open(path, "rb") as f:
        data = tomllib.load(f)

    device = data.get("device", {})
    whisper = data.get("whisper", {})
    output = data.get("output", {})
    vad = data.get("vad", {})
    audio = data.get("audio", {})
    pulse = data.get("pulse", {})

    defaults = AppConfig()
    return AppConfig(
        device_name=device.get("last_selected_name", defaults.device_name),
        model_size=whisper.get("model_size", defaults.model_size),
        whisper_device=whisper.get("device", defaults.whisper_device),
        compute_type=whisper.get("compute_type", defaults.compute_type),
        allowed_languages=tuple(whisper.get("allowed_languages", defaults.allowed_languages)),
        no_speech_threshold=whisper.get("no_speech_threshold", defaults.no_speech_threshold),
        log_prob_threshold=whisper.get("log_prob_threshold", defaults.log_prob_threshold),
        compression_ratio_threshold=whisper.get("compression_ratio_threshold", defaults.compression_ratio_threshold),
        output_dir=Path(output.get("directory", str(defaults.output_dir))),
        silence_duration_ms=vad.get("silence_duration_ms", defaults.silence_duration_ms),
        vad_aggressiveness=vad.get("aggressiveness", defaults.vad_aggressiveness),
        mic_gain_db=audio.get("gain_db", defaults.mic_gain_db),
        pulse_source_name=pulse.get("source_name", defaults.pulse_source_name),
        pulse_source_volume_percent=pulse.get("source_volume_percent", defaults.pulse_source_volume_percent),
        pulse_server=pulse.get("server", defaults.pulse_server),
    )


def save_config(cfg: AppConfig, path: Path | None = None) -> None:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "device": {"last_selected_name": cfg.device_name or ""},
        "whisper": {
            "model_size": cfg.model_size,
            "device": cfg.whisper_device,
            "compute_type": cfg.compute_type,
            "allowed_languages": list(cfg.allowed_languages),
            "no_speech_threshold": cfg.no_speech_threshold,
            "log_prob_threshold": cfg.log_prob_threshold,
            "compression_ratio_threshold": cfg.compression_ratio_threshold,
        },
        "output": {"directory": str(cfg.output_dir)},
        "vad": {
            "silence_duration_ms": cfg.silence_duration_ms,
            "aggressiveness": cfg.vad_aggressiveness,
        },
        "audio": {"gain_db": cfg.mic_gain_db},
        "pulse": {
            k: v
            for k, v in (
                ("source_name", cfg.pulse_source_name),
                ("source_volume_percent", cfg.pulse_source_volume_percent),
                ("server", cfg.pulse_server),
            )
            if v is not None
        },
    }
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def merge_cli_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    """CLI flags that were explicitly passed (non-None) win over the config file."""
    field_to_arg = {
        "device_name": "device",
        "model_size": "model_size",
        "whisper_device": "whisper_device",
        "compute_type": "compute_type",
        "silence_duration_ms": "silence_ms",
        "vad_aggressiveness": "vad_aggressiveness",
        "mic_gain_db": "gain_db",
        "pulse_source_name": "pulse_source",
        "pulse_source_volume_percent": "pulse_volume",
        "pulse_server": "pulse_server",
        "no_speech_threshold": "no_speech_threshold",
        "log_prob_threshold": "log_prob_threshold",
        "compression_ratio_threshold": "compression_ratio_threshold",
    }
    updates: dict = {
        field_name: getattr(args, arg_name)
        for field_name, arg_name in field_to_arg.items()
        if getattr(args, arg_name, None) is not None
    }
    if getattr(args, "output_dir", None) is not None:
        updates["output_dir"] = Path(args.output_dir)
    if getattr(args, "allowed_languages", None) is not None:
        updates["allowed_languages"] = tuple(
            lang.strip() for lang in args.allowed_languages.split(",") if lang.strip()
        )

    return replace(cfg, **updates)
