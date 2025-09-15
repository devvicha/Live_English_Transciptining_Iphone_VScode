import argparse
import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

# ✅ Load model with MPS backend (M2 Apple Silicon)
# Use tiny.en or base.en for real-time
model_size = "base.en"
model = WhisperModel(model_size, device="cpu", compute_type="int8")  
# Note: faster-whisper uses CPU backend, but leverages Apple’s Core ML/Accelerate libs

# 🔹 Audio settings
sample_rate = 16000  # Whisper expects 16kHz
block_size = 1024    # Smaller = lower latency, higher CPU load
channels = 1

# Queue to hold audio chunks
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Callback from sounddevice each time we get a new audio block"""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def list_input_devices():
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        if int(d.get("max_input_channels", 0)) > 0:
            print(f"[{idx}] {d['name']} — {d.get('hostapi', '')}")


def find_device_id(name_substring: str | None) -> int | None:
    if not name_substring:
        return None
    name_substring = name_substring.lower()
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        if int(d.get("max_input_channels", 0)) > 0 and name_substring in d["name"].lower():
            return idx
    return None


def main():
    parser = argparse.ArgumentParser(description="Realtime transcription from microphone using faster-whisper")
    parser.add_argument("--device", type=str, default=None, help="Input device name substring or index (e.g. 'iPhone Microphone')")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--rate", type=int, default=sample_rate, help="Sample rate (default 16000)")
    parser.add_argument("--block", type=int, default=block_size, help="Block size (default 1024)")
    parser.add_argument("--lang", type=str, default="en", help="Language code (default en)")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    device_to_use = None
    # Allow numeric device index
    if args.device is not None:
        try:
            device_to_use = int(args.device)
        except ValueError:
            # Try to match by substring (e.g., 'iPhone')
            device_to_use = find_device_id(args.device)
            if device_to_use is None:
                print(f"⚠️  No input device matching '{args.device}' found. Falling back to system default.")
                device_to_use = None
    else:
        # Auto-pick iPhone mic if present
        for key in ("iphone microphone", "iphone", "continuity"):
            device_to_use = find_device_id(key)
            if device_to_use is not None:
                break

    print("🎤 Listening... Press Ctrl+C to stop.")
    if device_to_use is not None:
        try:
            device_name = sd.query_devices(device_to_use)["name"]
        except Exception:
            device_name = str(device_to_use)
        print(f"📟 Using input device: {device_name} (id {device_to_use})")
    else:
        print("📟 Using system default input device")

    with sd.InputStream(
        device=device_to_use,
        samplerate=args.rate,
        channels=channels,
        blocksize=args.block,
        dtype="float32",
        callback=audio_callback,
        latency="low",
    ):
        buffer = np.zeros(0, dtype=np.float32)

        try:
            while True:
                # Get next chunk from mic
                chunk = audio_queue.get()
                buffer = np.concatenate((buffer, chunk[:, 0]))  # take single channel

                # Process every ~2 seconds of audio
                if len(buffer) > args.rate * 2:
                    # Save temp buffer for transcription
                    temp_audio = buffer.astype(np.float32)

                    # Run Whisper (faster-whisper incremental)
                    segments, _ = model.transcribe(temp_audio, language=args.lang)
                    text = " ".join([seg.text for seg in segments])

                    print("📝", text)

                    # Keep only last 1 second of audio to maintain context
                    buffer = buffer[-args.rate:]
        except KeyboardInterrupt:
            print("Stopped by user")


if __name__ == "__main__":
    main()
