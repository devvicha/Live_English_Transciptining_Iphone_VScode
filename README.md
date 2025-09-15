# Live English — Realtime Transcription (faster-whisper)

Realtime microphone transcription using faster-whisper and PortAudio. Tuned for low-latency local transcription on Apple Silicon, but works cross‑platform.

## Features

- Fast local transcription with `faster-whisper`
- Works with default input or a selected device (e.g., iPhone mic via Continuity)
- Adjustable sample rate, block size, and language
- Low-latency streaming loop that keeps recent context

## Requirements

- Python 3.10
- PortAudio (provided via Conda)
- FFmpeg (for `faster-whisper` audio I/O)

Use the provided Conda env for a known-good setup.

## Installation

Using Conda (recommended):

```bash
conda env create -f environment.yml
conda activate live-english
```

If you prefer pip + venv (macOS/Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install faster-whisper numpy sounddevice
# System deps (install once via your package manager):
# - PortAudio
# - FFmpeg
```

## Usage

List input devices:

```bash
python realtime_trascribe.py --list-devices
```

Transcribe from the default input device:

```bash
python realtime_trascribe.py
```

Select a device by name substring (e.g., iPhone/Continuity microphone) and set language:

```bash
python realtime_trascribe.py --device "iPhone" --lang en
```

Select a device by index and adjust latency (smaller block → lower latency):

```bash
python realtime_trascribe.py --device 3 --rate 16000 --block 1024
```

Available flags:

- `--list-devices`: prints available input devices and exits
- `--device`: device index or name substring
- `--rate`: sample rate (default 16000)
- `--block`: block size frames (default 1024)
- `--lang`: language code (default `en`)

## How It Works

- Captures audio from the selected input using `sounddevice` (PortAudio)
- Buffers short windows (~2s) and runs `faster-whisper` transcription locally
- Keeps ~1s of recent audio to maintain some context between chunks

Model/config summary:

- Model size: `base.en` (good balance of speed/quality for realtime)
- Device: CPU with int8 compute for Apple Silicon-friendly performance

You can change the model size in the script if you want higher quality (e.g., `small`, `medium`), at the cost of latency and CPU usage.

## Troubleshooting

- macOS microphone permission: grant Terminal/IDE mic access under System Settings → Privacy & Security → Microphone.
- No audio devices listed or “Error querying device”: ensure PortAudio is installed and accessible; the Conda env includes `portaudio`.
- iPhone microphone not found: ensure the phone is connected/unlocked and the Continuity input is active; try matching substrings like `"iphone"`, `"continuity"`.
- Choppy or delayed text: increase `--block` size or use a smaller model (e.g., `tiny.en`).
- High CPU usage: reduce sample rate or block size, or use `tiny.en`.

## Project Layout

- `realtime_trascribe.py`: realtime transcription CLI
- `environment.yml`: reproducible Conda environment

## Roadmap Ideas

- Voice activity detection (VAD) to skip silence
- Word timestamps and partial/streaming results display
- Optional diarization and punctuation stabilization
- Configurable hotkeys to pause/resume

## Acknowledgements

- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [sounddevice](https://python-sounddevice.readthedocs.io/)

