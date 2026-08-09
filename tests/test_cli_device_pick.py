from __future__ import annotations

import builtins

from mictranscribe.devices import prompt_for_device
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
