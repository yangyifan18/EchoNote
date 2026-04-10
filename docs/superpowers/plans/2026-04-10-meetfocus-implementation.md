# MeetingFocuser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI tool that records meeting audio, transcribes locally with Whisper, summarizes with a configurable LLM (Claude/OpenAI), and writes structured markdown notes to an Obsidian vault.

**Architecture:** Dual-process (recorder + transcriber) communicating via filesystem (.wav chunks). Recorder writes audio chunks to disk; transcriber watches for new chunks and transcribes them. A separate `summarize` command reads the full transcript and calls an LLM. Output is Obsidian-compatible markdown with frontmatter.

**Tech Stack:** Python 3.11+, typer, sounddevice, faster-whisper, watchdog, anthropic SDK, openai SDK, tomli-w, rich

---

## File Structure

```
MeetingFocuser/
├── pyproject.toml
├── meetfocus/
│   ├── __init__.py
│   ├── cli.py              # typer app, commands, process orchestration
│   ├── config.py            # Config dataclasses, TOML load/save
│   ├── session.py           # Session lifecycle, metadata, listing
│   ├── recorder.py          # Audio capture, chunked WAV writing
│   ├── transcriber.py       # Whisper transcription, watchdog file watcher
│   ├── summarizer.py        # LLM provider abstraction, prompt, summarization
│   └── writer.py            # Obsidian markdown generation, file output
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Shared fixtures (tmp dirs, config, etc.)
│   ├── test_config.py
│   ├── test_session.py
│   ├── test_recorder.py
│   ├── test_transcriber.py
│   ├── test_summarizer.py
│   ├── test_writer.py
│   └── test_cli.py
└── docs/
    └── superpowers/
        ├── specs/
        │   └── 2026-04-10-meetfocus-design.md
        └── plans/
            └── 2026-04-10-meetfocus-implementation.md
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `meetfocus/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Initialize git repo**

```bash
cd /Users/young/Code/MeetingFocuser
git init
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "meetfocus"
version = "0.1.0"
description = "Meeting audio recorder, transcriber, and AI summarizer for Obsidian"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.12",
    "sounddevice>=0.4",
    "soundfile>=0.12",
    "numpy>=1.24",
    "faster-whisper>=1.0",
    "watchdog>=4.0",
    "tomli-w>=1.0",
    "anthropic>=0.40",
    "openai>=1.0",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.12",
]

[project.scripts]
meetfocus = "meetfocus.cli:run"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create package init**

`meetfocus/__init__.py`:
```python
"""MeetingFocuser — meeting audio recorder, transcriber, and AI summarizer."""
```

- [ ] **Step 4: Create test infrastructure**

`tests/__init__.py`:
```python
```

`tests/conftest.py`:
```python
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
```

- [ ] **Step 5: Create .gitignore**

`.gitignore`:
```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.venv/
.env
.superpowers/
```

- [ ] **Step 6: Install in dev mode and verify**

```bash
cd /Users/young/Code/MeetingFocuser
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest --co -q
```

Expected: `no tests ran` (no test files with tests yet, but pytest finds the test directory)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml meetfocus/__init__.py tests/__init__.py tests/conftest.py .gitignore
git commit -m "chore: scaffold project with pyproject.toml and test infrastructure"
```

---

## Task 2: Config Module

**Files:**
- Create: `meetfocus/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

`tests/test_config.py`:
```python
import tomli_w
from pathlib import Path
from meetfocus.config import (
    Config,
    AudioConfig,
    WhisperConfig,
    LLMConfig,
    OutputConfig,
    load_config,
    save_config,
)


def test_default_config_has_expected_values():
    config = Config()
    assert config.audio.source == "mic"
    assert config.audio.sample_rate == 16000
    assert config.audio.chunk_duration == 30
    assert config.whisper.model == "large-v3"
    assert config.whisper.language == "zh"
    assert config.whisper.device == "auto"
    assert config.llm.provider == "claude"
    assert config.output.folder == "Meetings"
    assert config.output.keep_transcript is True
    assert config.output.keep_audio is False


def test_load_config_returns_defaults_when_no_file(tmp_config_dir):
    config = load_config(config_dir=tmp_config_dir)
    assert config.audio.source == "mic"
    assert config.whisper.model == "large-v3"


def test_save_and_load_config_roundtrip(tmp_config_dir):
    config = Config()
    config.audio.source = "system"
    config.llm.provider = "openai"
    config.llm.model = "gpt-4o"
    config.output.obsidian_vault = "/Users/test/vault"

    save_config(config, config_dir=tmp_config_dir)

    loaded = load_config(config_dir=tmp_config_dir)
    assert loaded.audio.source == "system"
    assert loaded.llm.provider == "openai"
    assert loaded.llm.model == "gpt-4o"
    assert loaded.output.obsidian_vault == "/Users/test/vault"


def test_load_config_merges_partial_file(tmp_config_dir):
    """A config file with only [audio] should leave other sections as defaults."""
    config_path = tmp_config_dir / "config.toml"
    partial = {"audio": {"source": "system", "chunk_duration": 60}}
    with open(config_path, "wb") as f:
        tomli_w.dump(partial, f)

    config = load_config(config_dir=tmp_config_dir)
    assert config.audio.source == "system"
    assert config.audio.chunk_duration == 60
    assert config.audio.sample_rate == 16000  # default preserved
    assert config.whisper.model == "large-v3"  # default preserved
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'meetfocus.config'`

- [ ] **Step 3: Implement config module**

`meetfocus/config.py`:
```python
"""Configuration management — load/save TOML config with sensible defaults."""

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
import tomllib
import tomli_w

DEFAULT_CONFIG_DIR = Path.home() / ".meetfocus"


@dataclass
class AudioConfig:
    device: str = "default"
    source: str = "mic"
    chunk_duration: int = 30
    sample_rate: int = 16000


@dataclass
class WhisperConfig:
    model: str = "large-v3"
    language: str = "zh"
    device: str = "auto"


@dataclass
class LLMConfig:
    provider: str = "claude"
    model: str = "claude-sonnet-4-6"
    api_key_env: str = "ANTHROPIC_API_KEY"


@dataclass
class OutputConfig:
    obsidian_vault: str = ""
    folder: str = "Meetings"
    keep_transcript: bool = True
    keep_audio: bool = False
    filename_format: str = "{date}-{title}"


@dataclass
class SessionsConfig:
    dir: str = ""

    def __post_init__(self):
        if not self.dir:
            self.dir = str(DEFAULT_CONFIG_DIR / "sessions")


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    sessions: SessionsConfig = field(default_factory=SessionsConfig)


def _merge_section(section_cls, data: dict):
    """Create a section dataclass, applying only keys that exist in the dataclass."""
    valid_keys = {f.name for f in fields(section_cls)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return section_cls(**filtered)


def load_config(config_dir: Path = DEFAULT_CONFIG_DIR) -> Config:
    """Load config from TOML file, falling back to defaults for missing keys."""
    config_path = config_dir / "config.toml"
    if not config_path.exists():
        return Config()

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    config = Config()
    if "audio" in data:
        config.audio = _merge_section(AudioConfig, data["audio"])
    if "whisper" in data:
        config.whisper = _merge_section(WhisperConfig, data["whisper"])
    if "llm" in data:
        config.llm = _merge_section(LLMConfig, data["llm"])
    if "output" in data:
        config.output = _merge_section(OutputConfig, data["output"])
    if "sessions" in data:
        config.sessions = _merge_section(SessionsConfig, data["sessions"])
    return config


def save_config(config: Config, config_dir: Path = DEFAULT_CONFIG_DIR) -> None:
    """Save config to TOML file."""
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"
    data = asdict(config)
    with open(config_path, "wb") as f:
        tomli_w.dump(data, f)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_config.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add meetfocus/config.py tests/test_config.py
git commit -m "feat: add config module with TOML load/save and sensible defaults"
```

---

## Task 3: Session Manager

**Files:**
- Create: `meetfocus/session.py`
- Create: `tests/test_session.py`

- [ ] **Step 1: Write failing tests**

`tests/test_session.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_session.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'meetfocus.session'`

- [ ] **Step 3: Implement session manager**

`meetfocus/session.py`:
```python
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
        """Create a new session directory with metadata."""
        now = datetime.now()
        session_id = now.strftime("%Y-%m-%d_%H%M%S")
        session_path = self.sessions_dir / session_id

        # Ensure unique directory (if two sessions created in same second)
        counter = 1
        while session_path.exists():
            session_id = now.strftime("%Y-%m-%d_%H%M%S") + f"_{counter}"
            session_path = self.sessions_dir / session_id
            counter += 1

        session_path.mkdir(parents=True)
        (session_path / "chunks").mkdir()

        meta = SessionMeta(
            session_id=session_id,
            path=session_path,
            audio_source=audio_source,
            status="recording",
            started_at=now.isoformat(),
        )
        self._write_meta(meta)
        return meta

    def list_sessions(self) -> list[SessionMeta]:
        """List all sessions, sorted by creation time (newest first)."""
        sessions = []
        for d in sorted(self.sessions_dir.iterdir(), reverse=True):
            if d.is_dir() and (d / "meta.json").exists():
                sessions.append(self._read_meta(d))
        return sessions

    def get(self, session_id: str) -> SessionMeta | None:
        """Get a session by ID. Returns None if not found."""
        session_path = self.sessions_dir / session_id
        if not session_path.exists() or not (session_path / "meta.json").exists():
            return None
        return self._read_meta(session_path)

    def get_latest(self) -> SessionMeta | None:
        """Get the most recent session."""
        sessions = self.list_sessions()
        return sessions[0] if sessions else None

    def finish(self, session_id: str) -> None:
        """Mark a session as completed."""
        session_path = self.sessions_dir / session_id
        meta = self._read_meta(session_path)
        meta.status = "completed"
        meta.finished_at = datetime.now().isoformat()
        self._write_meta(meta)

    def _write_meta(self, meta: SessionMeta) -> None:
        data = {
            "session_id": meta.session_id,
            "audio_source": meta.audio_source,
            "status": meta.status,
            "started_at": meta.started_at,
            "finished_at": meta.finished_at,
        }
        (meta.path / "meta.json").write_text(json.dumps(data, indent=2))

    def _read_meta(self, session_path: Path) -> SessionMeta:
        data = json.loads((session_path / "meta.json").read_text())
        return SessionMeta(
            session_id=data["session_id"],
            path=session_path,
            audio_source=data["audio_source"],
            status=data["status"],
            started_at=data["started_at"],
            finished_at=data.get("finished_at"),
        )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_session.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add meetfocus/session.py tests/test_session.py
git commit -m "feat: add session manager with create, list, get, and finish"
```

---

## Task 4: Recorder

**Files:**
- Create: `meetfocus/recorder.py`
- Create: `tests/test_recorder.py`

- [ ] **Step 1: Write failing tests**

`tests/test_recorder.py`:
```python
import numpy as np
import multiprocessing as mp
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from meetfocus.recorder import record_chunks, find_device_index


def _fake_input_stream(samplerate, channels, device, dtype, callback):
    """Simulate sounddevice.InputStream by feeding fake audio via callback."""

    class FakeStream:
        def __init__(self):
            self._running = False

        def __enter__(self):
            self._running = True
            # Feed a few blocks of fake audio data
            for _ in range(5):
                if not self._running:
                    break
                block = np.zeros((1600, 1), dtype=np.float32)  # 0.1s at 16kHz
                callback(block, 1600, None, None)
            return self

        def __exit__(self, *args):
            self._running = False

    return FakeStream()


@patch("meetfocus.recorder.sd")
def test_record_chunks_creates_wav_files(mock_sd, tmp_session_dir):
    """Recording should create .wav files in the chunks directory."""
    stop_event = mp.Event()
    chunks_dir = tmp_session_dir / "chunks"

    # Mock InputStream to produce fake audio, then set stop_event
    def fake_input_stream(**kwargs):
        stream = _fake_input_stream(**kwargs)
        # Schedule stop after stream context is entered
        original_enter = stream.__enter__

        def patched_enter():
            result = original_enter()
            stop_event.set()
            return result

        stream.__enter__ = patched_enter
        return stream

    mock_sd.InputStream = fake_input_stream

    record_chunks(
        chunks_dir=chunks_dir,
        stop_event=stop_event,
        chunk_duration=1,  # 1 second chunks for fast test
        sample_rate=16000,
    )

    wav_files = sorted(chunks_dir.glob("*.wav"))
    assert len(wav_files) >= 1
    # Verify no .tmp files remain
    tmp_files = list(chunks_dir.glob("*.wav.tmp"))
    assert len(tmp_files) == 0


@patch("meetfocus.recorder.sd")
def test_find_device_index_for_blackhole(mock_sd):
    mock_sd.query_devices.return_value = [
        {"name": "MacBook Pro Microphone", "max_input_channels": 1},
        {"name": "BlackHole 2ch", "max_input_channels": 2},
    ]
    idx = find_device_index("BlackHole")
    assert idx == 1


@patch("meetfocus.recorder.sd")
def test_find_device_index_returns_none_when_not_found(mock_sd):
    mock_sd.query_devices.return_value = [
        {"name": "MacBook Pro Microphone", "max_input_channels": 1},
    ]
    idx = find_device_index("BlackHole")
    assert idx is None
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_recorder.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'meetfocus.recorder'`

- [ ] **Step 3: Implement recorder**

`meetfocus/recorder.py`:
```python
"""Audio recording — capture mic or system audio, write chunked WAV files."""

import os
import queue
from pathlib import Path

import numpy as np
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

    Uses atomic rename (.wav.tmp → .wav) so consumers only see complete files.
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
        while not stop_event.is_set():
            chunk_name = f"chunk_{chunk_index:04d}"
            tmp_path = chunks_dir / f"{chunk_name}.wav.tmp"
            final_path = chunks_dir / f"{chunk_name}.wav"

            with sf.SoundFile(
                str(tmp_path),
                mode="w",
                samplerate=sample_rate,
                channels=1,
                subtype="PCM_16",
            ) as wav:
                samples_written = 0
                while samples_written < chunk_samples:
                    if stop_event.is_set():
                        break
                    try:
                        data = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    wav.write(data)
                    samples_written += len(data)

            # Atomic rename signals chunk is complete
            os.rename(tmp_path, final_path)
            chunk_index += 1


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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_recorder.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add meetfocus/recorder.py tests/test_recorder.py
git commit -m "feat: add recorder with chunked WAV writing and atomic rename"
```

---

## Task 5: Transcriber

**Files:**
- Create: `meetfocus/transcriber.py`
- Create: `tests/test_transcriber.py`

- [ ] **Step 1: Write failing tests**

`tests/test_transcriber.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_transcriber.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'meetfocus.transcriber'`

- [ ] **Step 3: Implement transcriber**

`meetfocus/transcriber.py`:
```python
"""Transcription — watch for audio chunks and transcribe with faster-whisper."""

import time
from pathlib import Path

from faster_whisper import WhisperModel
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler, FileCreatedEvent


def transcribe_file(model: WhisperModel, audio_path: str, language: str = "zh") -> str:
    """Transcribe a single audio file, returning the full text."""
    segments, _info = model.transcribe(audio_path, language=language, beam_size=5)
    return "".join(seg.text for seg in segments)


class _ChunkHandler(PatternMatchingEventHandler):
    """Watchdog handler that transcribes new .wav chunks on creation."""

    def __init__(self, model: WhisperModel, language: str, transcript_path: Path, processed: set):
        super().__init__(
            patterns=["*.wav"],
            ignore_patterns=["*.wav.tmp"],
            ignore_directories=True,
        )
        self.model = model
        self.language = language
        self.transcript_path = transcript_path
        self.processed = processed

    def on_created(self, event: FileCreatedEvent) -> None:
        path = Path(event.src_path)
        if path.name in self.processed:
            return
        self.processed.add(path.name)
        text = transcribe_file(self.model, str(path), self.language)
        with open(self.transcript_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def _has_unprocessed(chunks_dir: Path, processed: set) -> bool:
    """Check if there are .wav files not yet processed."""
    for p in chunks_dir.glob("*.wav"):
        if p.name not in processed:
            return True
    return False


def transcriber_main(
    session_dir: Path,
    whisper_model: str,
    whisper_device: str,
    whisper_language: str,
    stop_event,
):
    """Entry point for the transcriber subprocess."""
    model = WhisperModel(whisper_model, device=whisper_device, compute_type="auto")

    chunks_dir = session_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    transcript_path = session_dir / "transcript.txt"
    processed: set[str] = set()

    # Process any existing chunks (crash recovery)
    for existing in sorted(chunks_dir.glob("*.wav")):
        processed.add(existing.name)
        text = transcribe_file(model, str(existing), whisper_language)
        with open(transcript_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    # Watch for new chunks
    handler = _ChunkHandler(model, whisper_language, transcript_path, processed)
    observer = Observer()
    observer.schedule(handler, str(chunks_dir), recursive=False)
    observer.start()

    try:
        while not stop_event.is_set() or _has_unprocessed(chunks_dir, processed):
            time.sleep(0.5)
        # After stop, give a moment for any final events
        time.sleep(1.0)
        if _has_unprocessed(chunks_dir, processed):
            for p in sorted(chunks_dir.glob("*.wav")):
                if p.name not in processed:
                    processed.add(p.name)
                    text = transcribe_file(model, str(p), whisper_language)
                    with open(transcript_path, "a", encoding="utf-8") as f:
                        f.write(text + "\n")
    finally:
        observer.stop()
        observer.join()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_transcriber.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add meetfocus/transcriber.py tests/test_transcriber.py
git commit -m "feat: add transcriber with watchdog file watcher and crash recovery"
```

---

## Task 6: Summarizer

**Files:**
- Create: `meetfocus/summarizer.py`
- Create: `tests/test_summarizer.py`

- [ ] **Step 1: Write failing tests**

`tests/test_summarizer.py`:
```python
from unittest.mock import patch, MagicMock
from meetfocus.summarizer import (
    build_prompt,
    summarize,
    ClaudeProvider,
    OpenAIProvider,
)


def test_build_prompt_includes_transcript():
    transcript = "这是一段测试转写文本。"
    prompt = build_prompt(transcript)
    assert "这是一段测试转写文本。" in prompt
    assert "摘要" in prompt
    assert "详细内容" in prompt
    assert "Q&A" in prompt
    assert "行动项" in prompt


def test_build_prompt_includes_structural_instructions():
    prompt = build_prompt("test")
    assert "跟随演讲者的原始组织方式" in prompt
    assert "不要过度压缩" in prompt


@patch("meetfocus.summarizer.anthropic")
def test_claude_provider_calls_api(mock_anthropic):
    mock_client = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="## 摘要\n测试总结")]
    mock_client.messages.create.return_value = mock_response

    provider = ClaudeProvider(api_key="test-key", model="claude-sonnet-4-6")
    result = provider.generate("测试 prompt")

    assert result == "## 摘要\n测试总结"
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["messages"][0]["content"] == "测试 prompt"


@patch("meetfocus.summarizer.openai")
def test_openai_provider_calls_api(mock_openai):
    mock_client = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="## 摘要\n测试总结"))]
    mock_client.chat.completions.create.return_value = mock_response

    provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
    result = provider.generate("测试 prompt")

    assert result == "## 摘要\n测试总结"
    mock_client.chat.completions.create.assert_called_once()


@patch("meetfocus.summarizer.anthropic")
def test_summarize_end_to_end(mock_anthropic):
    mock_client = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="---\nspeaker: 张教授\n---\n\n## 摘要\n测试")]
    mock_client.messages.create.return_value = mock_response

    result = summarize(
        transcript="完整的转写文本",
        provider_name="claude",
        model="claude-sonnet-4-6",
        api_key="test-key",
    )

    assert "张教授" in result
    assert "摘要" in result
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_summarizer.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'meetfocus.summarizer'`

- [ ] **Step 3: Implement summarizer**

`meetfocus/summarizer.py`:
```python
"""LLM summarization — provider abstraction and prompt construction."""

from abc import ABC, abstractmethod

import anthropic
import openai


SUMMARIZE_PROMPT = """你是一个专业的学术会议记录整理助手。请根据以下会议转写文本，生成结构化的会议笔记。

## 要求

### 元数据
请从转写内容中提取以下信息（如无法确定，标记为"未知"）：
- 演讲者姓名和机构
- 报告主题
- 相关标签（3-5个，用于知识库分类）

### 摘要
用2-3句话概括本次报告的核心内容和主要贡献。

### 详细内容
这是最重要的部分。请遵循以下原则：
1. **跟随演讲者的原始组织方式** — 如果演讲者按项目/论文时间线讲解，就按项目分块；如果按专题讲解，就按专题分块
2. **不要过度压缩** — 保留具体的数字、方法名、工具名、论文名、实验结果等细节
3. **标注论文外信息** — 用「📌 论文外」标记演讲者分享的、在已发表论文中未涉及的实践经验
4. **保持足够上下文** — 使后续读者（或AI助手）能基于这份笔记进行深入问答

每个分块用 ### 标题标记，内容用 bullet points 组织。

### Q&A
如果转写中包含提问环节，按 Q1/Q2/... 格式整理每个问答对。如果没有明显的提问环节，省略此部分。

### 行动项
提取可执行的TODO事项（如需要调研的工具、需要联系的人、需要阅读的论文等），用 `- [ ]` 格式。如果没有明确的行动项，省略此部分。

## 输出格式

请直接输出 Markdown 格式。以 YAML frontmatter 开头：

---
speaker: 姓名 (机构)
topic: 主题
tags: [tag1, tag2, ...]
---

## 摘要
...

## 详细内容
### 分块标题1
...

## Q&A
### Q1: 问题
...

## 行动项
- [ ] ...

---

## 转写文本

{transcript}"""


def build_prompt(transcript: str) -> str:
    """Construct the summarization prompt with the transcript inserted."""
    return SUMMARIZE_PROMPT.replace("{transcript}", transcript)


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt to LLM and return the response text."""


class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


def _make_provider(provider_name: str, model: str, api_key: str) -> LLMProvider:
    """Create an LLM provider instance by name."""
    if provider_name == "claude":
        return ClaudeProvider(api_key=api_key, model=model)
    elif provider_name == "openai":
        return OpenAIProvider(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")


def summarize(
    transcript: str,
    provider_name: str,
    model: str,
    api_key: str,
) -> str:
    """Summarize a transcript using the configured LLM provider."""
    provider = _make_provider(provider_name, model, api_key)
    prompt = build_prompt(transcript)
    return provider.generate(prompt)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_summarizer.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add meetfocus/summarizer.py tests/test_summarizer.py
git commit -m "feat: add summarizer with Claude/OpenAI providers and structured prompt"
```

---

## Task 7: Writer

**Files:**
- Create: `meetfocus/writer.py`
- Create: `tests/test_writer.py`

- [ ] **Step 1: Write failing tests**

`tests/test_writer.py`:
```python
from pathlib import Path
from meetfocus.writer import write_summary, write_transcript


def test_write_summary_creates_markdown_file(tmp_path):
    vault_dir = tmp_path / "vault" / "Meetings"
    summary_content = "---\nspeaker: 张教授 (XX研究院)\ntopic: 数据工程\ntags: [meeting, data]\n---\n\n## 摘要\n测试总结"

    result_path = write_summary(
        summary=summary_content,
        vault_path=str(tmp_path / "vault"),
        folder="Meetings",
        date="2026-04-10",
        title="数据工程报告",
        filename_format="{date}-{title}",
    )

    assert result_path.exists()
    assert result_path.name == "2026-04-10-数据工程报告.md"
    content = result_path.read_text(encoding="utf-8")
    assert "张教授" in content
    assert "测试总结" in content


def test_write_summary_adds_date_frontmatter(tmp_path):
    summary_content = "---\nspeaker: 测试\ntopic: 测试\ntags: [test]\n---\n\n## 摘要\n内容"

    result_path = write_summary(
        summary=summary_content,
        vault_path=str(tmp_path / "vault"),
        folder="Meetings",
        date="2026-04-10",
        title="测试",
        filename_format="{date}-{title}",
    )

    content = result_path.read_text(encoding="utf-8")
    assert "date: 2026-04-10" in content


def test_write_summary_creates_folder_if_missing(tmp_path):
    vault_dir = tmp_path / "vault"
    # Don't pre-create Meetings folder

    result_path = write_summary(
        summary="---\ntopic: test\n---\n\n## 摘要\ntest",
        vault_path=str(vault_dir),
        folder="Meetings",
        date="2026-04-10",
        title="test",
        filename_format="{date}-{title}",
    )

    assert result_path.exists()
    assert (vault_dir / "Meetings").is_dir()


def test_write_transcript(tmp_path):
    vault_dir = tmp_path / "vault" / "Meetings"
    transcript = "这是完整的转写文本。\n第二段内容。"

    result_path = write_transcript(
        transcript=transcript,
        vault_path=str(tmp_path / "vault"),
        folder="Meetings",
        date="2026-04-10",
        title="测试",
        filename_format="{date}-{title}",
    )

    assert result_path.exists()
    assert result_path.name == "2026-04-10-测试-transcript.md"
    content = result_path.read_text(encoding="utf-8")
    assert "转写文本" in content
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_writer.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'meetfocus.writer'`

- [ ] **Step 3: Implement writer**

`meetfocus/writer.py`:
```python
"""Obsidian writer — generate and write markdown files to vault."""

from pathlib import Path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _inject_date_frontmatter(summary: str, date: str) -> str:
    """Inject date into existing YAML frontmatter if not already present."""
    if summary.startswith("---"):
        # Find the closing ---
        end_idx = summary.index("---", 3)
        frontmatter = summary[3:end_idx]
        if "date:" not in frontmatter:
            frontmatter = f"\ndate: {date}" + frontmatter
        return "---" + frontmatter + summary[end_idx:]
    else:
        # No frontmatter, add one
        return f"---\ndate: {date}\n---\n\n{summary}"


def write_summary(
    summary: str,
    vault_path: str,
    folder: str,
    date: str,
    title: str,
    filename_format: str,
) -> Path:
    """Write the AI summary to the Obsidian vault as a markdown file."""
    vault = Path(vault_path).expanduser()
    output_dir = vault / folder
    _ensure_dir(output_dir)

    filename = filename_format.format(date=date, title=title) + ".md"
    output_path = output_dir / filename

    content = _inject_date_frontmatter(summary, date)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def write_transcript(
    transcript: str,
    vault_path: str,
    folder: str,
    date: str,
    title: str,
    filename_format: str,
) -> Path:
    """Write the raw transcript to the Obsidian vault."""
    vault = Path(vault_path).expanduser()
    output_dir = vault / folder
    _ensure_dir(output_dir)

    filename = filename_format.format(date=date, title=title) + "-transcript.md"
    output_path = output_dir / filename

    content = f"---\ndate: {date}\ntype: transcript\n---\n\n{transcript}"
    output_path.write_text(content, encoding="utf-8")
    return output_path
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_writer.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add meetfocus/writer.py tests/test_writer.py
git commit -m "feat: add Obsidian writer with frontmatter injection and transcript output"
```

---

## Task 8: CLI Commands

**Files:**
- Create: `meetfocus/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

`tests/test_cli.py`:
```python
import json
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from meetfocus.cli import app

runner = CliRunner()


def test_list_no_sessions(tmp_path):
    with patch("meetfocus.cli._get_session_manager") as mock_mgr:
        mock_mgr.return_value.list_sessions.return_value = []
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No sessions" in result.stdout or "没有" in result.stdout


def test_config_show(tmp_path):
    with patch("meetfocus.cli._get_config") as mock_cfg:
        from meetfocus.config import Config
        mock_cfg.return_value = Config()
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "audio" in result.stdout.lower() or "whisper" in result.stdout.lower()


@patch("meetfocus.cli._get_config")
@patch("meetfocus.cli._get_session_manager")
@patch("meetfocus.cli.run_summarize")
def test_summarize_command(mock_summarize, mock_mgr, mock_cfg, monkeypatch):
    from meetfocus.config import Config
    from meetfocus.session import SessionMeta
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
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'meetfocus.cli'`

- [ ] **Step 3: Implement CLI**

`meetfocus/cli.py`:
```python
"""CLI entry point — commands and process orchestration."""

import multiprocessing as mp
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.text import Text

from meetfocus.config import Config, load_config, save_config, DEFAULT_CONFIG_DIR
from meetfocus.session import SessionManager
from meetfocus.recorder import recorder_main
from meetfocus.transcriber import transcriber_main
from meetfocus.summarizer import summarize as run_summarize
from meetfocus.writer import write_summary, write_transcript

app = typer.Typer(help="MeetingFocuser — record, transcribe, and summarize meetings.")
console = Console()


def _get_config() -> Config:
    return load_config()


def _get_session_manager() -> SessionManager:
    config = _get_config()
    return SessionManager(sessions_dir=Path(config.sessions.dir).expanduser())


@app.command()
def start(
    system: bool = typer.Option(False, "--system", help="Capture system audio via BlackHole"),
):
    """Start recording and transcribing a meeting."""
    config = _get_config()

    if not config.output.obsidian_vault:
        console.print("[red]Error:[/red] Obsidian vault path not configured. Run: meetfocus config --set output.obsidian_vault=/path/to/vault")
        raise typer.Exit(1)

    mgr = _get_session_manager()
    audio_source = "system" if system else "mic"
    session = mgr.create(audio_source=audio_source)

    console.print(f"[green]Recording started[/green] (session: {session.session_id})")
    console.print(f"  Audio source: {'System (BlackHole)' if system else 'Microphone'}")
    console.print(f"  Session dir: {session.path}")
    console.print("  Press Ctrl+C to stop.\n")

    stop_event = mp.Event()

    rec_proc = mp.Process(
        target=recorder_main,
        args=(
            session.path,
            audio_source,
            config.audio.device,
            config.audio.chunk_duration,
            config.audio.sample_rate,
            stop_event,
        ),
        name="meetfocus-recorder",
    )
    trans_proc = mp.Process(
        target=transcriber_main,
        args=(
            session.path,
            config.whisper.model,
            config.whisper.device,
            config.whisper.language,
            stop_event,
        ),
        name="meetfocus-transcriber",
    )

    rec_proc.start()
    trans_proc.start()

    def on_sigint(sig, frame):
        console.print("\n[yellow]Stopping...[/yellow]")
        stop_event.set()

    signal.signal(signal.SIGINT, on_sigint)

    # Display live status
    start_time = datetime.now()
    try:
        while rec_proc.is_alive():
            elapsed = datetime.now() - start_time
            minutes, seconds = divmod(int(elapsed.total_seconds()), 60)
            chunks = len(list((session.path / "chunks").glob("*.wav")))
            status = f"  Recording: {minutes:02d}:{seconds:02d} | Chunks: {chunks}"
            sys.stdout.write(f"\r{status}")
            sys.stdout.flush()
            rec_proc.join(timeout=1.0)
    except KeyboardInterrupt:
        stop_event.set()

    console.print("\n[yellow]Waiting for transcription to finish...[/yellow]")
    trans_proc.join(timeout=120)
    if trans_proc.is_alive():
        trans_proc.terminate()

    mgr.finish(session.session_id)

    transcript_path = session.path / "transcript.txt"
    if transcript_path.exists():
        lines = len(transcript_path.read_text().strip().splitlines())
        console.print(f"[green]Done![/green] Transcript: {lines} lines")
    else:
        console.print("[yellow]Done.[/yellow] No transcript generated (too short?)")

    console.print(f"  Run [bold]meetfocus summarize[/bold] to generate AI summary.")


@app.command()
def summarize(
    session_id: str = typer.Argument(None, help="Session ID (defaults to latest)"),
):
    """Summarize a recorded session using an LLM."""
    config = _get_config()
    mgr = _get_session_manager()

    session = mgr.get(session_id) if session_id else mgr.get_latest()
    if session is None:
        console.print("[red]Error:[/red] No session found.")
        raise typer.Exit(1)

    transcript_path = session.path / "transcript.txt"
    if not transcript_path.exists():
        console.print("[red]Error:[/red] No transcript found for this session.")
        raise typer.Exit(1)

    transcript = transcript_path.read_text(encoding="utf-8")
    if not transcript.strip():
        console.print("[red]Error:[/red] Transcript is empty.")
        raise typer.Exit(1)

    # Resolve API key
    api_key = os.environ.get(config.llm.api_key_env, "")
    if not api_key:
        console.print(f"[red]Error:[/red] API key not found. Set the {config.llm.api_key_env} environment variable.")
        raise typer.Exit(1)

    console.print(f"Summarizing with {config.llm.provider} ({config.llm.model})...")

    summary = run_summarize(
        transcript=transcript,
        provider_name=config.llm.provider,
        model=config.llm.model,
        api_key=api_key,
    )

    # Save summary to session dir
    (session.path / "summary.md").write_text(summary, encoding="utf-8")

    # Extract date and title from summary/session
    date = session.started_at[:10]
    # Try to extract topic from frontmatter
    title = "meeting"
    if "topic:" in summary:
        for line in summary.splitlines():
            if line.strip().startswith("topic:"):
                title = line.split(":", 1)[1].strip()
                break

    # Write to Obsidian vault
    if config.output.obsidian_vault:
        result_path = write_summary(
            summary=summary,
            vault_path=config.output.obsidian_vault,
            folder=config.output.folder,
            date=date,
            title=title,
            filename_format=config.output.filename_format,
        )
        console.print(f"[green]Summary written to:[/green] {result_path}")

        if config.output.keep_transcript:
            t_path = write_transcript(
                transcript=transcript,
                vault_path=config.output.obsidian_vault,
                folder=config.output.folder,
                date=date,
                title=title,
                filename_format=config.output.filename_format,
            )
            console.print(f"[green]Transcript written to:[/green] {t_path}")
    else:
        console.print("[yellow]No Obsidian vault configured. Summary saved to session dir only.[/yellow]")

    console.print("[green]Done![/green]")


@app.command("list")
def list_sessions():
    """List all recorded sessions."""
    mgr = _get_session_manager()
    sessions = mgr.list_sessions()

    if not sessions:
        console.print("No sessions found.")
        return

    for s in sessions:
        status_color = "green" if s.status == "completed" else "yellow"
        console.print(
            f"  [{status_color}]{s.status:<12}[/{status_color}] "
            f"{s.session_id}  ({s.audio_source})"
        )


@app.command()
def show(session_id: str = typer.Argument(None, help="Session ID (defaults to latest)")):
    """Show details of a session."""
    mgr = _get_session_manager()
    session = mgr.get(session_id) if session_id else mgr.get_latest()

    if session is None:
        console.print("[red]Error:[/red] Session not found.")
        raise typer.Exit(1)

    console.print(f"[bold]Session:[/bold] {session.session_id}")
    console.print(f"  Status: {session.status}")
    console.print(f"  Audio: {session.audio_source}")
    console.print(f"  Started: {session.started_at}")
    if session.finished_at:
        console.print(f"  Finished: {session.finished_at}")
    console.print(f"  Path: {session.path}")

    chunks = list((session.path / "chunks").glob("*.wav"))
    console.print(f"  Audio chunks: {len(chunks)}")

    transcript_path = session.path / "transcript.txt"
    if transcript_path.exists():
        text = transcript_path.read_text()
        console.print(f"  Transcript: {len(text)} chars")
    else:
        console.print("  Transcript: (none)")

    summary_path = session.path / "summary.md"
    if summary_path.exists():
        console.print("  Summary: [green]available[/green]")
    else:
        console.print("  Summary: (none — run meetfocus summarize)")


@app.command()
def config():
    """Show current configuration."""
    cfg = _get_config()
    console.print("[bold]Current configuration:[/bold]\n")
    console.print(f"[audio]")
    console.print(f"  device = {cfg.audio.device}")
    console.print(f"  source = {cfg.audio.source}")
    console.print(f"  chunk_duration = {cfg.audio.chunk_duration}")
    console.print(f"  sample_rate = {cfg.audio.sample_rate}")
    console.print(f"\n[whisper]")
    console.print(f"  model = {cfg.whisper.model}")
    console.print(f"  language = {cfg.whisper.language}")
    console.print(f"  device = {cfg.whisper.device}")
    console.print(f"\n[llm]")
    console.print(f"  provider = {cfg.llm.provider}")
    console.print(f"  model = {cfg.llm.model}")
    console.print(f"  api_key_env = {cfg.llm.api_key_env}")
    console.print(f"\n[output]")
    console.print(f"  obsidian_vault = {cfg.output.obsidian_vault}")
    console.print(f"  folder = {cfg.output.folder}")
    console.print(f"  keep_transcript = {cfg.output.keep_transcript}")
    console.print(f"  keep_audio = {cfg.output.keep_audio}")
    console.print(f"\n[sessions]")
    console.print(f"  dir = {cfg.sessions.dir}")
    console.print(f"\nConfig file: {DEFAULT_CONFIG_DIR / 'config.toml'}")


@app.command()
def status():
    """Show current recording status."""
    mgr = _get_session_manager()
    latest = mgr.get_latest()

    if latest and latest.status == "recording":
        console.print(f"[green]Recording in progress[/green]: {latest.session_id}")
        chunks = len(list((latest.path / "chunks").glob("*.wav")))
        console.print(f"  Chunks recorded: {chunks}")
    else:
        console.print("No active recording.")


def run():
    """Entry point for the CLI (called by pyproject.toml script)."""
    mp.freeze_support()
    app()


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_cli.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add meetfocus/cli.py tests/test_cli.py
git commit -m "feat: add CLI with start, summarize, list, show, config, and status commands"
```

---

## Task 9: Integration Test and Final Verification

**Files:**
- Create: `tests/test_integration.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write integration test**

`tests/test_integration.py`:
```python
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
    """Test the full pipeline: session creation → chunks → transcription → summary → output."""

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
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/test_integration.py -v
```

Expected: 1 passed

- [ ] **Step 3: Run all tests to verify nothing is broken**

```bash
pytest -v
```

Expected: All tests pass (config: 4, session: 6, recorder: 3, transcriber: 3, summarizer: 5, writer: 4, cli: 3, integration: 1 = ~29 tests)

- [ ] **Step 4: Verify CLI installs and runs**

```bash
pip install -e .
meetfocus --help
meetfocus config
meetfocus list
```

Expected: Help text shows all commands. Config shows defaults. List shows no sessions.

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test covering full session→transcription→summary→output flow"
```

---

## Task 10: Fix Spec and Final Cleanup

- [ ] **Step 1: Update spec to remove MPS from config options**

In `docs/superpowers/specs/2026-04-10-meetfocus-design.md`, the `[whisper]` config section should show:
```toml
device = "auto"           # "cpu" | "cuda" | "auto" (no MPS — CTranslate2 不支持 Metal)
```

(This was already done during planning.)

- [ ] **Step 2: Run final test suite**

```bash
pytest -v --tb=short
```

Expected: All tests pass.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: finalize project setup and documentation"
```

---

## Verification

After completing all tasks, verify the full workflow:

1. **Install and check CLI**:
   ```bash
   pip install -e ".[dev]"
   meetfocus --help
   ```

2. **Run all tests**:
   ```bash
   pytest -v
   ```

3. **Manual smoke test** (requires actual audio hardware + Whisper model download):
   ```bash
   # Configure
   meetfocus config  # Verify defaults are shown
   # Edit ~/.meetfocus/config.toml to set obsidian_vault path

   # Record a short test (speak for ~10 seconds, then Ctrl+C)
   meetfocus start

   # Check status
   meetfocus list
   meetfocus show

   # Summarize (requires LLM API key)
   export ANTHROPIC_API_KEY=sk-...
   meetfocus summarize

   # Verify output in Obsidian vault
   ```

4. **Verify Obsidian output** — open the generated .md file in Obsidian and check:
   - Frontmatter renders correctly
   - Sections are properly structured
   - Tags are clickable
