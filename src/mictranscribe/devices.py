from __future__ import annotations

import sounddevice as sd

from .types import InputDevice


def list_input_devices() -> list[InputDevice]:
    devices = sd.query_devices()
    return [
        InputDevice(index=i, name=d["name"], max_input_channels=d["max_input_channels"])
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


def find_device_by_name(devices: list[InputDevice], name: str) -> InputDevice | None:
    """Exact match first, then substring match, to survive minor name drift."""
    for d in devices:
        if d.name == name:
            return d
    for d in devices:
        if name in d.name or d.name in name:
            return d
    return None


def print_devices(devices: list[InputDevice]) -> None:
    for i, d in enumerate(devices):
        print(f"[{i}] {d.name} ({d.max_input_channels} ch)")


def prompt_for_device(devices: list[InputDevice], default_name: str | None) -> InputDevice:
    if not devices:
        raise RuntimeError("No audio input devices found.")

    default_device = find_device_by_name(devices, default_name) if default_name else None

    print("Available microphones:")
    print_devices(devices)
    prompt = "Select a microphone"
    prompt += f" [default: {default_device.name}]: " if default_device else ": "

    while True:
        choice = input(prompt).strip()
        if choice == "" and default_device is not None:
            return default_device
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(devices):
                return devices[idx]
        print("Invalid selection, please enter a listed number.")
