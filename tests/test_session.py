import json
from pathlib import Path
from meetfocus.session import SessionManager, SessionMeta


def test_create_session(tmp_path):
    mgr = SessionManager(sessions_dir=tmp_path)
    session = mgr.create(audio_source="mic")
    assert session.path.exists()
    assert (session.path / "chunks").exists()
    assert (session.path / "meta.json").exists()
    meta = json.loads((session.path / "meta.json").read_text())
    assert meta["audio_source"] == "mic"
    assert meta["status"] == "recording"
    assert "started_at" in meta


def test_list_sessions(tmp_path):
    mgr = SessionManager(sessions_dir=tmp_path)
    s1 = mgr.create(audio_source="mic")
    s2 = mgr.create(audio_source="system")
    sessions = mgr.list_sessions()
    assert len(sessions) == 2
    ids = [s.session_id for s in sessions]
    assert s1.session_id in ids
    assert s2.session_id in ids


def test_get_session(tmp_path):
    mgr = SessionManager(sessions_dir=tmp_path)
    created = mgr.create(audio_source="mic")
    found = mgr.get(created.session_id)
    assert found is not None
    assert found.session_id == created.session_id
    assert found.path == created.path


def test_get_nonexistent_session_returns_none(tmp_path):
    mgr = SessionManager(sessions_dir=tmp_path)
    assert mgr.get("nonexistent") is None


def test_finish_session_updates_meta(tmp_path):
    mgr = SessionManager(sessions_dir=tmp_path)
    session = mgr.create(audio_source="mic")
    mgr.finish(session.session_id)
    meta = json.loads((session.path / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert "finished_at" in meta


def test_get_latest_session(tmp_path):
    mgr = SessionManager(sessions_dir=tmp_path)
    mgr.create(audio_source="mic")
    s2 = mgr.create(audio_source="mic")
    latest = mgr.get_latest()
    assert latest is not None
    assert latest.session_id == s2.session_id
