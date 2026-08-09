from __future__ import annotations

import queue

from mictranscribe import audio_capture
from mictranscribe.audio_capture import AudioCapture
from mictranscribe.types import InputDevice

# Simulates the real-world Windows case: the same physical mic enumerated
# under multiple host APIs (MME, DirectSound, WASAPI, WDM-KS) with an
# identical name string.
_DUPLICATE_NAME_DEVICES = [
    InputDevice(index=1, name="Microphone (USB audio CODEC)", max_input_channels=1),
    InputDevice(index=6, name="Microphone (USB audio CODEC)", max_input_channels=2),
    InputDevice(index=8, name="Microphone (USB audio CODEC)", max_input_channels=1),
]


def test_resolve_device_index_uses_known_index_when_given(monkeypatch):
    # Even though name-based lookup would resolve to index 1 (first match),
    # the caller already picked index 6 explicitly (e.g. via the interactive
    # picker) and that must be honored, not silently overridden.
    monkeypatch.setattr(audio_capture, "list_input_devices", lambda: _DUPLICATE_NAME_DEVICES)
    capture = AudioCapture(
        "Microphone (USB audio CODEC)", queue.Queue(), device_index=6
    )
    assert capture._resolve_device_index() == 6


def test_resolve_device_index_falls_back_to_name_lookup_without_known_index(monkeypatch):
    monkeypatch.setattr(audio_capture, "list_input_devices", lambda: _DUPLICATE_NAME_DEVICES)
    capture = AudioCapture("Microphone (USB audio CODEC)", queue.Queue(), device_index=None)
    assert capture._resolve_device_index() == 1  # first match, deterministic


def test_reconnect_clears_known_index_forcing_fresh_name_lookup(monkeypatch):
    monkeypatch.setattr(audio_capture, "list_input_devices", lambda: _DUPLICATE_NAME_DEVICES)
    capture = AudioCapture("Microphone (USB audio CODEC)", queue.Queue(), device_index=6)
    assert capture._resolve_device_index() == 6

    # Simulate what _reopen_with_backoff does before attempting to reopen:
    # a replugged device may have a genuinely different index now, so it
    # must re-resolve by name rather than trust the stale cached index.
    capture._known_index = None
    assert capture._resolve_device_index() == 1
