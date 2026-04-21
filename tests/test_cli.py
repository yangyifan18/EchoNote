from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from echonote.cli import _resolve_audio_source, app

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


def test_resolve_audio_source_uses_config_when_flag_missing():
    assert _resolve_audio_source(None, "system") == "system"
    assert _resolve_audio_source(None, "mic") == "mic"


def test_resolve_audio_source_cli_override_wins():
    assert _resolve_audio_source(True, "mic") == "system"
    assert _resolve_audio_source(False, "system") == "mic"


@patch("echonote.cli.signal.signal")
@patch("echonote.cli.mp.Process")
@patch("echonote.cli._get_session_manager")
@patch("echonote.cli._get_config")
def test_start_allows_missing_obsidian_config(mock_cfg, mock_mgr, mock_process, mock_signal):
    from echonote.config import Config
    from echonote.session import SessionMeta
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        session_dir = Path(tmpdir) / "test-session"
        session_dir.mkdir()
        (session_dir / "chunks").mkdir()

        config = Config()
        config.audio.source = "system"
        config.output.obsidian_vault = ""
        mock_cfg.return_value = config
        mock_mgr.return_value.create.return_value = SessionMeta(
            session_id="test-session",
            path=session_dir,
            audio_source="system",
            status="recording",
            started_at="2026-04-21T10:00:00",
        )

        rec_proc = MagicMock()
        rec_proc.is_alive.return_value = False
        trans_proc = MagicMock()
        trans_proc.is_alive.return_value = False
        mock_process.side_effect = [rec_proc, trans_proc]

        result = runner.invoke(app, ["start"])

        assert result.exit_code == 0
        mock_mgr.return_value.create.assert_called_once_with(audio_source="system")
        assert "Recording started" in result.stdout


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
