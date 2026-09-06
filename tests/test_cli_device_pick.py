from __future__ import annotations

import builtins

from mictranscribe.devices import prompt_for_device, resolve_device
from mictranscribe.types import InputDevice

DEVICES = [
    InputDevice(index=0, name="Mic A", max_input_channels=1),
    InputDevice(index=3, name="Mic B", max_input_channels=2),
]


def _script_inputs(monkeypatch, responses: list[str]) -> None:
    responses_iter = iter(responses)
    monkeypatch.setattr(builtins, "input", lambda *_args: next(responses_iter))


def test_blank_input_selects_the_default(monkeypatch):
    _script_inputs(monkeypatch, [""])
    chosen = prompt_for_device(DEVICES, default_name="Mic B")
    assert chosen == DEVICES[1]


def test_valid_numeric_input_selects_that_device(monkeypatch):
    _script_inputs(monkeypatch, ["0"])
    chosen = prompt_for_device(DEVICES, default_name=None)
    assert chosen == DEVICES[0]


def test_invalid_input_reprompts_until_valid(monkeypatch):
    _script_inputs(monkeypatch, ["5", "not-a-number", "1"])
    chosen = prompt_for_device(DEVICES, default_name=None)
    assert chosen == DEVICES[1]


def test_unmatched_default_name_prints_a_clear_explanation(monkeypatch, capsys):
    _script_inputs(monkeypatch, ["0"])
    prompt_for_device(DEVICES, default_name="some-pulseaudio-source-name")
    out = capsys.readouterr().out
    assert "No microphone matching 'some-pulseaudio-source-name' was found" in out


def test_matched_default_name_prints_no_warning(monkeypatch, capsys):
    _script_inputs(monkeypatch, [""])
    prompt_for_device(DEVICES, default_name="Mic B")
    out = capsys.readouterr().out
    assert "No microphone matching" not in out


def _input_should_not_be_called(monkeypatch) -> None:
    def boom(*_args):
        raise AssertionError("input() should not be called when an explicit device name is given")

    monkeypatch.setattr(builtins, "input", boom)


def test_resolve_device_with_matching_explicit_name_skips_prompt_entirely(monkeypatch):
    _input_should_not_be_called(monkeypatch)
    chosen = resolve_device(DEVICES, explicit_name="Mic B", remembered_name=None)
    assert chosen == DEVICES[1]


def test_resolve_device_with_nonmatching_explicit_name_returns_none_without_prompting(monkeypatch):
    _input_should_not_be_called(monkeypatch)
    chosen = resolve_device(DEVICES, explicit_name="nonexistent-device", remembered_name=None)
    assert chosen is None


def test_resolve_device_without_explicit_name_falls_back_to_interactive_prompt(monkeypatch):
    _script_inputs(monkeypatch, [""])
    chosen = resolve_device(DEVICES, explicit_name=None, remembered_name="Mic B")
    assert chosen == DEVICES[1]
