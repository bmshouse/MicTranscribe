from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

logger = logging.getLogger("mictranscribe.pulse")


def is_available() -> bool:
    """PulseAudio source-volume control is Linux+pactl only; everywhere else
    this module quietly no-ops rather than failing the whole service."""
    return sys.platform == "linux" and shutil.which("pactl") is not None


def build_pulse_env(base_env: dict, pulse_server: str | None) -> dict:
    """Copy base_env, overlaying PULSE_SERVER if one was configured (needed
    for non-default sockets, e.g. a Home-Assistant-managed PulseAudio
    instance rather than a normal per-user session bus)."""
    env = dict(base_env)
    if pulse_server:
        env["PULSE_SERVER"] = pulse_server
    return env


def set_source_volume(source_name: str, volume_percent: int, pulse_server: str | None = None) -> bool:
    """Best-effort: set a PulseAudio source's volume. Never raises — a
    misconfigured or unreachable PulseAudio server shouldn't block the
    service from starting with whatever volume is already set."""
    if not is_available():
        logger.debug("pactl not available on this platform; skipping source volume setup")
        return False

    env = build_pulse_env(os.environ, pulse_server)
    try:
        subprocess.run(
            ["pactl", "set-source-volume", source_name, f"{volume_percent}%"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Failed to set PulseAudio source volume for '%s': %s", source_name, exc)
        return False

    logger.info("Set PulseAudio source '%s' volume to %d%%", source_name, volume_percent)
    return True


def list_sources(pulse_server: str | None = None) -> str:
    """Return the raw `pactl list sources short` output, to help a user find
    the exact source name to put in config. Raises on failure — this is
    used from an explicit CLI diagnostic flag, not startup, so surfacing the
    error directly is more useful than silently swallowing it."""
    if not is_available():
        raise RuntimeError("pactl is not available (Linux + PulseAudio only)")

    env = build_pulse_env(os.environ, pulse_server)
    result = subprocess.run(
        ["pactl", "list", "sources", "short"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout
