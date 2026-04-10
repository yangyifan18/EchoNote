"""Integration test — full flow using mocked audio and LLM."""

import multiprocessing as mp
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import soundfile as sf

from meetfocus.config import Config
from meetfocus.session import SessionManager
from meetfocus.recorder import record_chunks
from meetfocus.transcriber import transcriber_main
from meetfocus.summarizer import summarize
from meetfocus.writer import write_summary


def _create_test_wav(path: Path, duration_s: float = 1.0, sample_rate: int = 16000):
    """Create a real WAV file with silence (valid audio for Whisper to process)."""
    samples = np.zeros(int(duration_s * sample_rate), dtype=np.float32)
    sf.write(str(path), samples, sample_rate)


@patch("meetfocus.transcriber.WhisperModel")
def test_full_flow_session_to_summary(MockWhisper, tmp_path):
    """Test the full pipeline: session creation -> chunks -> transcription -> summary -> output."""

    # --- Setup ---
    mock_model = MockWhisper.return_value
    seg1 = MagicMock()
    seg1.text = "今天我们来讨论数据清洗的最佳实践。"
    seg2 = MagicMock()
    seg2.text = "第一步是格式统一化。"
    mock_model.transcribe.return_value = ([seg1, seg2], MagicMock())

    # Create session
    sessions_dir = tmp_path / "sessions"
    mgr = SessionManager(sessions_dir=sessions_dir)
    session = mgr.create(audio_source="mic")

    # --- Simulate recording (create chunk files directly) ---
    chunks_dir = session.path / "chunks"
    _create_test_wav(chunks_dir / "chunk_0000.wav")
    _create_test_wav(chunks_dir / "chunk_0001.wav")

    # --- Run transcriber ---
    stop_event = mp.Event()
    stop_event.set()  # Process existing chunks then stop

    transcriber_main(
        session_dir=session.path,
        whisper_model="tiny",
        whisper_device="cpu",
        whisper_language="zh",
        stop_event=stop_event,
    )

    # Verify transcript
    transcript_path = session.path / "transcript.txt"
    assert transcript_path.exists()
    transcript = transcript_path.read_text()
    assert "数据清洗" in transcript
    assert "格式统一化" in transcript

    # --- Mock LLM summarization ---
    summary_text = """---
speaker: 张研究员 (XX实验室)
topic: 数据清洗最佳实践
tags: [meeting, data-cleaning]
---

## 摘要
张研究员分享了数据清洗的最佳实践，重点介绍了格式统一化流程。

## 详细内容
### 数据清洗流程
- 第一步是格式统一化，处理不同数据源的格式差异
- 📌 论文外：实际操作中需要处理15种不同来源格式

## 行动项
- [ ] 调研格式统一化工具"""

    with patch("meetfocus.summarizer.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=summary_text)]
        mock_client.messages.create.return_value = mock_response

        result = summarize(
            transcript=transcript,
            provider_name="claude",
            model="claude-sonnet-4-6",
            api_key="test-key",
        )

    assert "数据清洗最佳实践" in result

    # --- Write to Obsidian ---
    vault_path = tmp_path / "vault"
    output_path = write_summary(
        summary=result,
        vault_path=str(vault_path),
        folder="Meetings",
        date="2026-04-10",
        title="数据清洗最佳实践",
        filename_format="{date}-{title}",
    )

    assert output_path.exists()
    content = output_path.read_text()
    assert "date: 2026-04-10" in content
    assert "数据清洗" in content
    assert "行动项" in content

    # Finish session
    mgr.finish(session.session_id)
    finished = mgr.get(session.session_id)
    assert finished.status == "completed"
