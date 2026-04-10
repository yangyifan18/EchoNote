import numpy as np
import multiprocessing as mp
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from echonote.recorder import record_chunks, find_device_index


def _fake_input_stream(samplerate, channels, device, dtype, callback, stop_event=None):
    """Simulate sounddevice.InputStream by feeding fake audio via callback."""

    class FakeStream:
        def __init__(self):
            self._running = False

        def __enter__(self):
            self._running = True
            for _ in range(5):
                if not self._running:
                    break
                block = np.zeros((1600, 1), dtype=np.float32)  # 0.1s at 16kHz
                callback(block, 1600, None, None)
            if stop_event is not None:
                stop_event.set()
            return self

        def __exit__(self, *args):
            self._running = False

    return FakeStream()


@patch("echonote.recorder.sd")
def test_record_chunks_creates_wav_files(mock_sd, tmp_session_dir):
    """Recording should create .wav files in the chunks directory."""
    stop_event = mp.Event()
    chunks_dir = tmp_session_dir / "chunks"

    def fake_input_stream(**kwargs):
        return _fake_input_stream(**kwargs, stop_event=stop_event)

    mock_sd.InputStream = fake_input_stream

    record_chunks(
        chunks_dir=chunks_dir,
        stop_event=stop_event,
        chunk_duration=1,
        sample_rate=16000,
    )

    wav_files = sorted(chunks_dir.glob("*.wav"))
    assert len(wav_files) >= 1
    tmp_files = list(chunks_dir.glob("*.wav.tmp"))
    assert len(tmp_files) == 0


@patch("echonote.recorder.sd")
def test_find_device_index_for_blackhole(mock_sd):
    mock_sd.query_devices.return_value = [
        {"name": "MacBook Pro Microphone", "max_input_channels": 1},
        {"name": "BlackHole 2ch", "max_input_channels": 2},
    ]
    idx = find_device_index("BlackHole")
    assert idx == 1


@patch("echonote.recorder.sd")
def test_find_device_index_returns_none_when_not_found(mock_sd):
    mock_sd.query_devices.return_value = [
        {"name": "MacBook Pro Microphone", "max_input_channels": 1},
    ]
    idx = find_device_index("BlackHole")
    assert idx is None
