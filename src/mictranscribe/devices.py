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
    if default_name and default_device is None:
        # --device/config gave a name but it matched nothing in the list below
        # (a common mix-up: PulseAudio source names like "alsa_input.usb-...
        # Mpow_HC..." aren't what's matched here — this matches the exact
        # device names shown by --list-devices, e.g. "hassio_mic").
        print(f"No microphone matching '{default_name}' was found; showing the full list.")

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


def resolve_device(
    devices: list[InputDevice], explicit_name: str | None, remembered_name: str | None
) -> InputDevice | None:
    """Pick the device to use, without ever prompting when the caller
    explicitly named one (e.g. via --device).

    An explicit name is resolved directly and must match something —
    returns None on failure, which the caller should treat as a hard
    error rather than falling back to the interactive picker. Silently
    prompting anyway would defeat the entire point of --device: skipping
    human interaction, including in headless/unattended contexts (a
    systemd service, cron, no TTY at all) where there's no one to answer
    a prompt in the first place. With no explicit name, falls back to the
    interactive picker, offering remembered_name as its Enter-to-accept
    default.
    """
    if explicit_name is not None:
        return find_device_by_name(devices, explicit_name)
    return prompt_for_device(devices, default_name=remembered_name)
