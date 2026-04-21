"""Audio recording — capture mic or system audio, write chunked WAV files."""

import os
import queue
from pathlib import Path

import sounddevice as sd
import soundfile as sf


def find_device_index(name_substring: str) -> int | None:
    """Find an audio input device by name substring (e.g., 'BlackHole')."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if name_substring.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i
    return None


def record_chunks(
    chunks_dir: Path,
    stop_event,
    chunk_duration: int = 30,
    sample_rate: int = 16000,
    device=None,
):
    """Record audio in fixed-duration chunks, each saved as a WAV file.

    Uses atomic rename (.wav.tmp -> .wav) so consumers only see complete files.
    """
    audio_queue = queue.Queue()
    chunk_samples = chunk_duration * sample_rate
    chunk_index = 0

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[recorder] sounddevice: {status}")
        audio_queue.put(indata.copy())

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        device=device,
        dtype="float32",
        callback=callback,
    ):
        while True:
            chunk_name = f"chunk_{chunk_index:04d}"
            tmp_path = chunks_dir / f"{chunk_name}.wav.tmp"
            final_path = chunks_dir / f"{chunk_name}.wav"

            with sf.SoundFile(
                str(tmp_path), mode="w", samplerate=sample_rate,
                channels=1, format="WAV", subtype="PCM_16",
            ) as wav:
                samples_written = 0
                while samples_written < chunk_samples:
                    if stop_event.is_set() and audio_queue.empty():
                        break
                    try:
                        data = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    wav.write(data)
                    samples_written += len(data)

            os.rename(tmp_path, final_path)
            chunk_index += 1

            if stop_event.is_set() and audio_queue.empty():
                break


def recorder_main(
    session_dir: Path,
    audio_source: str,
    audio_device: str,
    chunk_duration: int,
    sample_rate: int,
    stop_event,
):
    """Entry point for the recorder subprocess."""
    chunks_dir = session_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    device = None
    if audio_source == "system":
        device = find_device_index(audio_device if audio_device != "default" else "BlackHole")
        if device is None:
            print("[recorder] ERROR: BlackHole device not found. Install BlackHole for system audio capture.")
            return

    record_chunks(
        chunks_dir=chunks_dir,
        stop_event=stop_event,
        chunk_duration=chunk_duration,
        sample_rate=sample_rate,
        device=device,
    )
