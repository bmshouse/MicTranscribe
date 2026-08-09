from __future__ import annotations

import subprocess

from mictranscribe import pulse


def test_build_pulse_env_overlays_server_when_given():
    base = {"PATH": "/usr/bin"}
    env = pulse.build_pulse_env(base, "unix:/tmp/pulse.sock")
    assert env["PULSE_SERVER"] == "unix:/tmp/pulse.sock"
    assert env["PATH"] == "/usr/bin"
    assert base == {"PATH": "/usr/bin"}  # original untouched


def test_build_pulse_env_leaves_env_alone_when_no_server_given():
    base = {"PATH": "/usr/bin"}
    env = pulse.build_pulse_env(base, None)
    assert "PULSE_SERVER" not in env
    assert env == base


def test_is_available_false_on_non_linux(monkeypatch):
    monkeypatch.setattr(pulse.sys, "platform", "win32")
    assert pulse.is_available() is False


def test_is_available_false_when_pactl_missing(monkeypatch):
    monkeypatch.setattr(pulse.sys, "platform", "linux")
    monkeypatch.setattr(pulse.shutil, "which", lambda name: None)
    assert pulse.is_available() is False


def test_is_available_true_when_linux_and_pactl_present(monkeypatch):
    monkeypatch.setattr(pulse.sys, "platform", "linux")
    monkeypatch.setattr(pulse.shutil, "which", lambda name: "/usr/bin/pactl")
    assert pulse.is_available() is True


def test_set_source_volume_returns_false_when_unavailable(monkeypatch):
    monkeypatch.setattr(pulse, "is_available", lambda: False)
    assert pulse.set_source_volume("some_source", 100) is False


def test_set_source_volume_runs_pactl_with_expected_args(monkeypatch):
    monkeypatch.setattr(pulse, "is_available", lambda: True)
    captured = {}

    def fake_run(cmd, env, check, capture_output, text, timeout):
        captured["cmd"] = cmd
        captured["env"] = env

    monkeypatch.setattr(pulse.subprocess, "run", fake_run)

    ok = pulse.set_source_volume("my_source", 150, pulse_server="unix:/tmp/pulse.sock")

    assert ok is True
    assert captured["cmd"] == ["pactl", "set-source-volume", "my_source", "150%"]
    assert captured["env"]["PULSE_SERVER"] == "unix:/tmp/pulse.sock"


def test_set_source_volume_returns_false_on_subprocess_failure(monkeypatch):
    monkeypatch.setattr(pulse, "is_available", lambda: True)

    def fake_run(*a, **k):
        raise subprocess.CalledProcessError(1, "pactl")

    monkeypatch.setattr(pulse.subprocess, "run", fake_run)

    assert pulse.set_source_volume("my_source", 100) is False


def test_list_sources_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(pulse, "is_available", lambda: False)
    try:
        pulse.list_sources()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
