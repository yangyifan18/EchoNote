import os
import time
import multiprocessing as mp
from pathlib import Path
from unittest.mock import patch, MagicMock

from meetfocus.transcriber import transcribe_file, transcriber_main


def _create_fake_wav(path: Path):
    """Create an empty but valid-enough file to represent a WAV chunk."""
    path.write_bytes(b"fake-audio-data")


@patch("meetfocus.transcriber.WhisperModel")
def test_transcribe_file(MockModel):
    """transcribe_file should return concatenated segment text."""
    mock_model = MockModel.return_value
    seg1 = MagicMock()
    seg1.text = "你好，"
    seg2 = MagicMock()
    seg2.text = "今天我们讨论数据清洗。"
    mock_model.transcribe.return_value = ([seg1, seg2], MagicMock())

    result = transcribe_file(mock_model, "/tmp/test.wav", language="zh")
    assert result == "你好，今天我们讨论数据清洗。"
    mock_model.transcribe.assert_called_once_with("/tmp/test.wav", language="zh", beam_size=5)


@patch("meetfocus.transcriber.WhisperModel")
def test_transcriber_processes_existing_chunks(MockModel, tmp_session_dir):
    """On startup, transcriber should process .wav files already in chunks/."""
    mock_model = MockModel.return_value
    seg = MagicMock()
    seg.text = "测试文本。"
    mock_model.transcribe.return_value = ([seg], MagicMock())

    chunks_dir = tmp_session_dir / "chunks"
    _create_fake_wav(chunks_dir / "chunk_0000.wav")
    _create_fake_wav(chunks_dir / "chunk_0001.wav")

    stop_event = mp.Event()
    stop_event.set()  # Stop immediately after processing existing chunks

    transcriber_main(
        session_dir=tmp_session_dir,
        whisper_model="tiny",
        whisper_device="cpu",
        whisper_language="zh",
        stop_event=stop_event,
    )

    transcript = (tmp_session_dir / "transcript.txt").read_text()
    assert transcript.count("测试文本。") == 2


@patch("meetfocus.transcriber.WhisperModel")
def test_transcriber_ignores_tmp_files(MockModel, tmp_session_dir):
    """Transcriber should not process .wav.tmp files."""
    mock_model = MockModel.return_value

    chunks_dir = tmp_session_dir / "chunks"
    _create_fake_wav(chunks_dir / "chunk_0000.wav.tmp")

    stop_event = mp.Event()
    stop_event.set()

    transcriber_main(
        session_dir=tmp_session_dir,
        whisper_model="tiny",
        whisper_device="cpu",
        whisper_language="zh",
        stop_event=stop_event,
    )

    assert not (tmp_session_dir / "transcript.txt").exists() or \
        (tmp_session_dir / "transcript.txt").read_text() == ""
    mock_model.transcribe.assert_not_called()
