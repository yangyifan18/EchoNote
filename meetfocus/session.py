"""Session lifecycle management — create, list, query, and finalize sessions."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SessionMeta:
    session_id: str
    path: Path
    audio_source: str
    status: str
    started_at: str
    finished_at: str | None = None


class SessionManager:
    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create(self, audio_source: str) -> SessionMeta:
        now = datetime.now()
        session_id = now.strftime("%Y-%m-%d_%H%M%S")
        session_path = self.sessions_dir / session_id
        counter = 1
        while session_path.exists():
            session_id = now.strftime("%Y-%m-%d_%H%M%S") + f"_{counter}"
            session_path = self.sessions_dir / session_id
            counter += 1
        session_path.mkdir(parents=True)
        (session_path / "chunks").mkdir()
        meta = SessionMeta(
            session_id=session_id, path=session_path, audio_source=audio_source,
            status="recording", started_at=now.isoformat(),
        )
        self._write_meta(meta)
        return meta

    def list_sessions(self) -> list[SessionMeta]:
        sessions = []
        for d in sorted(self.sessions_dir.iterdir(), reverse=True):
            if d.is_dir() and (d / "meta.json").exists():
                sessions.append(self._read_meta(d))
        return sessions

    def get(self, session_id: str) -> SessionMeta | None:
        session_path = self.sessions_dir / session_id
        if not session_path.exists() or not (session_path / "meta.json").exists():
            return None
        return self._read_meta(session_path)

    def get_latest(self) -> SessionMeta | None:
        sessions = self.list_sessions()
        return sessions[0] if sessions else None

    def finish(self, session_id: str) -> None:
        session_path = self.sessions_dir / session_id
        meta = self._read_meta(session_path)
        meta.status = "completed"
        meta.finished_at = datetime.now().isoformat()
        self._write_meta(meta)

    def _write_meta(self, meta: SessionMeta) -> None:
        data = {"session_id": meta.session_id, "audio_source": meta.audio_source,
                "status": meta.status, "started_at": meta.started_at, "finished_at": meta.finished_at}
        (meta.path / "meta.json").write_text(json.dumps(data, indent=2))

    def _read_meta(self, session_path: Path) -> SessionMeta:
        data = json.loads((session_path / "meta.json").read_text())
        return SessionMeta(
            session_id=data["session_id"], path=session_path,
            audio_source=data["audio_source"], status=data["status"],
            started_at=data["started_at"], finished_at=data.get("finished_at"),
        )
