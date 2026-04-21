"""Transcription — watch for audio chunks and transcribe with faster-whisper."""

import time
from pathlib import Path

from faster_whisper import WhisperModel
from watchdog.events import FileCreatedEvent, PatternMatchingEventHandler
from watchdog.observers import Observer


def transcribe_file(model: WhisperModel, audio_path: str, language: str = "zh") -> str:
    """Transcribe a single audio file, returning the full text."""
    segments, _info = model.transcribe(audio_path, language=language, beam_size=5)
    return "".join(seg.text for seg in segments)


class _ChunkHandler(PatternMatchingEventHandler):
    """Watchdog handler that transcribes new .wav chunks on creation."""

    def __init__(
        self,
        model: WhisperModel,
        language: str,
        transcript_path: Path,
        processed: set,
        keep_audio: bool,
    ):
        super().__init__(
            patterns=["*.wav"],
            ignore_patterns=["*.wav.tmp"],
            ignore_directories=True,
        )
        self.model = model
        self.language = language
        self.transcript_path = transcript_path
        self.processed = processed
        self.keep_audio = keep_audio

    def on_created(self, event: FileCreatedEvent) -> None:
        path = Path(event.src_path)
        if path.name in self.processed:
            return
        _process_chunk(
            model=self.model,
            chunk_path=path,
            language=self.language,
            transcript_path=self.transcript_path,
            keep_audio=self.keep_audio,
        )
        self.processed.add(path.name)


def _has_unprocessed(chunks_dir: Path, processed: set) -> bool:
    """Check if there are .wav files not yet processed."""
    for p in chunks_dir.glob("*.wav"):
        if p.name not in processed:
            return True
    return False


def _process_chunk(
    model: WhisperModel,
    chunk_path: Path,
    language: str,
    transcript_path: Path,
    keep_audio: bool,
) -> None:
    """Transcribe one chunk, append it to the transcript, and optionally delete audio."""
    text = transcribe_file(model, str(chunk_path), language)
    with open(transcript_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    if not keep_audio and chunk_path.exists():
        chunk_path.unlink()


def transcriber_main(
    session_dir: Path,
    whisper_model: str,
    whisper_device: str,
    whisper_language: str,
    stop_event,
    keep_audio: bool = False,
):
    """Entry point for the transcriber subprocess."""
    model = WhisperModel(whisper_model, device=whisper_device, compute_type="auto")

    chunks_dir = session_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    transcript_path = session_dir / "transcript.txt"
    processed: set[str] = set()

    # Process any existing chunks (crash recovery)
    for existing in sorted(chunks_dir.glob("*.wav")):
        _process_chunk(
            model=model,
            chunk_path=existing,
            language=whisper_language,
            transcript_path=transcript_path,
            keep_audio=keep_audio,
        )
        processed.add(existing.name)

    # Watch for new chunks
    handler = _ChunkHandler(model, whisper_language, transcript_path, processed, keep_audio)
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
                    _process_chunk(
                        model=model,
                        chunk_path=p,
                        language=whisper_language,
                        transcript_path=transcript_path,
                        keep_audio=keep_audio,
                    )
                    processed.add(p.name)
    finally:
        observer.stop()
        observer.join()
