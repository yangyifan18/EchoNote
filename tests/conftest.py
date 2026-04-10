import pytest
from pathlib import Path


@pytest.fixture
def tmp_session_dir(tmp_path):
    """A temporary session directory with chunks/ subdirectory."""
    session_dir = tmp_path / "test-session"
    session_dir.mkdir()
    (session_dir / "chunks").mkdir()
    return session_dir


@pytest.fixture
def tmp_config_dir(tmp_path):
    """A temporary config directory for testing config load/save."""
    config_dir = tmp_path / ".meetfocus"
    config_dir.mkdir()
    return config_dir
