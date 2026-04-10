import json
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from echonote.cli import app

runner = CliRunner()


def test_list_no_sessions(tmp_path):
    with patch("echonote.cli._get_session_manager") as mock_mgr:
        mock_mgr.return_value.list_sessions.return_value = []
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No sessions" in result.stdout or "没有" in result.stdout


def test_config_show(tmp_path):
    with patch("echonote.cli._get_config") as mock_cfg:
        from echonote.config import Config
        mock_cfg.return_value = Config()
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "audio" in result.stdout.lower() or "whisper" in result.stdout.lower()


@patch("echonote.cli._get_config")
@patch("echonote.cli._get_session_manager")
@patch("echonote.cli.run_summarize")
def test_summarize_command(mock_summarize, mock_mgr, mock_cfg, monkeypatch):
    from echonote.config import Config
    from echonote.session import SessionMeta
    from pathlib import Path
    import tempfile

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    with tempfile.TemporaryDirectory() as tmpdir:
        session_dir = Path(tmpdir) / "test-session"
        session_dir.mkdir()
        (session_dir / "transcript.txt").write_text("转写文本内容")

        mock_session = SessionMeta(
            session_id="test-session",
            path=session_dir,
            audio_source="mic",
            status="completed",
            started_at="2026-04-10T14:00:00",
        )
        mock_mgr.return_value.get_latest.return_value = mock_session

        config = Config()
        config.output.obsidian_vault = tmpdir
        mock_cfg.return_value = config

        mock_summarize.return_value = "---\ntopic: test\n---\n\n## 摘要\n测试"

        result = runner.invoke(app, ["summarize"])
        assert result.exit_code == 0
        mock_summarize.assert_called_once()
