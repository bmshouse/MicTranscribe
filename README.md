# MicTranscribe

A long-running, local-only microphone transcription service. It listens to
a microphone, detects speech with VAD, transcribes each utterance with a
local Whisper model, and appends timestamped lines to a daily plain-text
transcript file. No audio is ever written to disk — only the resulting
text.

## Requirements

- Python 3.10+ on Linux, macOS, or Windows.
- Linux/macOS need the PortAudio system library (see below). Windows needs
  nothing extra.

## Install

```bash
python3 -m venv .venv     # Windows: py -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -e .
```

**Linux** — install PortAudio first:

```bash
sudo apt-get install libportaudio2   # Debian/Ubuntu
sudo dnf install portaudio           # Fedora
sudo pacman -S portaudio             # Arch
```

If the mic fails to open with a permissions error, add yourself to the
`audio` group: `sudo usermod -aG audio $USER` (then log out/in).

**macOS** — install PortAudio via Homebrew: `brew install portaudio`. The
first run will prompt for microphone permission — approve it.

**Windows** — no system package needed. If `--list-devices` lists the same
mic several times, that's normal (PortAudio exposes one entry per host API:
MME, DirectSound, WASAPI, WDM-KS). Pick any by number. If your pick fails
with `Invalid sample rate`, re-run and pick a different entry with the same
name — some host-API variants (often WASAPI) reject 16kHz directly.

**GPU acceleration** (optional, Linux/Windows with an Nvidia GPU):

```bash
pip install -e ".[gpu]"
```

`--whisper-device` defaults to `auto` (GPU if available, CPU otherwise), so
this is safe to skip — the same install works either way.

## Usage

```bash
mictranscribe --list-devices   # see available microphones
mictranscribe                  # pick a mic, start listening
```

Transcripts append to `<output-dir>/YYYY-MM-DD.txt` (default: current
directory), one line per utterance:

```
[14:32:07 - 14:32:11] hello there, this is a test
```

Logs go to `<output-dir>/mictranscribe.log`. Config persists to a
platform config directory (e.g. `~/.config/mictranscribe/config.toml` on
Linux) — your mic choice is remembered automatically; other settings can be
edited there directly or passed as flags each run.

## Choosing a model

`--model-size` accepts any Whisper model name `faster-whisper` supports.
Default is `large-v3-turbo`, a good fit for most machines with a few GB of
VRAM (or a modern CPU):

| Model | Best for | Approx. VRAM (int8) | Notes |
|---|---|---|---|
| `large-v3-turbo` | **Default.** Best accuracy-per-resource tradeoff | ~1.5-2 GB | Multilingual, near-`large-v3` accuracy, 6x faster |
| `large-v3` | Maximum accuracy, VRAM no object | ~5-6 GB | Multilingual, slowest |
| `medium` | Mid-range GPU, turbo unavailable | ~2.5-3 GB | Noticeably behind turbo now |
| `small` | Very limited hardware / CPU-only | ~1 GB | Old default, still fine for CPU |
| `distil-large-v3` | English-only, want more speed | ~1.5 GB | **English only** — don't use with `--allowed-languages` covering other languages |
| `tiny` / `base` | Quick tests only | <1 GB | Not recommended for real use |

`--compute-type` (`default`/`int8`/`int8_float16`/`float16`) trades
precision for VRAM — `int8_float16` is a good default on a GPU with limited
memory. Models are downloaded once from Hugging Face and cached locally
(`~/.cache/huggingface/hub`); nothing re-downloads on later runs unless you
switch to a model you haven't used before.

## Configuration reference

| Flag | Meaning | Default |
|---|---|---|
| `--device NAME` | Microphone to use (substring match) | last selected, else prompts |
| `--model-size` | Whisper model, see table above | `large-v3-turbo` |
| `--whisper-device` | Inference backend: `auto`/`cpu`/`cuda` | `auto` |
| `--compute-type` | `default`/`int8`/`int8_float16`/`float16` | `default` |
| `--output-dir` | Where transcripts/logs are written | current directory |
| `--silence-ms` | Trailing silence (ms) that ends an utterance | `1200` |
| `--gain-db` | Digital mic gain in dB, applied after capture | `0.0` |
| `--vad-aggressiveness` | WebRTC VAD aggressiveness `0`-`3` | `3` |
| `--allowed-languages` | Comma-separated language codes to accept, e.g. `en,ja,ko,fr,es` | none (accept any) |
| `--no-speech-threshold` | Discard a segment above this no-speech probability | `0.6` |
| `--log-prob-threshold` | Discard a segment below this avg. token log-probability | `-1.0` |
| `--compression-ratio-threshold` | Discard a segment above this compression ratio | `2.4` |
| `--pulse-source` | PulseAudio source to set the volume of at startup (Linux) | none |
| `--pulse-volume` | PulseAudio source volume percent to set at startup | none |
| `--pulse-server` | PulseAudio server address override | default socket |
| `--list-devices` | Print input devices and exit | |
| `--list-pulse-sources` | Print PulseAudio sources (Linux) and exit | |
| `--check-levels` | Sample the mic, report peak/RMS level, and exit | |
| `--config PATH` | Use a specific config file | platform default |
| `--verbose` | Debug-level console logging | off |

## Tips

**Quiet or noisy mic** — run `mictranscribe --check-levels` to see peak/RMS
without starting the full pipeline. If it's too quiet, prefer fixing it at
the OS/PulseAudio mixer level over `--gain-db`, which amplifies noise right
along with your voice.

**Noise/hallucination filtering** — most runtime is silence, and small
models can hallucinate confident-looking garbage from background noise.
Defaults are tuned against this (strict VAD, Whisper's own VAD filter
always on, confidence-based segment filtering, optional language
allowlist). If junk still gets through, tighten `--no-speech-threshold`;
if genuine quiet speech gets dropped, loosen it.

**PulseAudio volume (Linux)** — `--pulse-source`/`--pulse-volume` set the
OS mixer volume automatically at startup, so you don't need to run `pactl`
by hand each time. Find the source name with `--list-pulse-sources`. Use
`--pulse-server` for a non-default socket (e.g. a PulseAudio instance
managed by another service).

**Running twice by accident** — starting `mictranscribe` again against the
same config fails fast with a clear error rather than silently duplicating
every transcript line. Use `--config` for a different config file if you
want two intentionally-independent instances (e.g. two mics).

## Development

```bash
pip install -e ".[dev]"
pytest
```
