"""CLI entry point — commands and process orchestration."""

import multiprocessing as mp
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

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
        console.print("[red]Error:[/red] Obsidian vault path not configured.")
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
        args=(session.path, audio_source, config.audio.device,
              config.audio.chunk_duration, config.audio.sample_rate, stop_event),
        name="meetfocus-recorder",
    )
    trans_proc = mp.Process(
        target=transcriber_main,
        args=(session.path, config.whisper.model, config.whisper.device,
              config.whisper.language, stop_event),
        name="meetfocus-transcriber",
    )

    rec_proc.start()
    trans_proc.start()

    def on_sigint(sig, frame):
        console.print("\n[yellow]Stopping...[/yellow]")
        stop_event.set()

    signal.signal(signal.SIGINT, on_sigint)

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

    (session.path / "summary.md").write_text(summary, encoding="utf-8")

    date = session.started_at[:10]
    title = "meeting"
    if "topic:" in summary:
        for line in summary.splitlines():
            if line.strip().startswith("topic:"):
                title = line.split(":", 1)[1].strip()
                break

    if config.output.obsidian_vault:
        result_path = write_summary(
            summary=summary, vault_path=config.output.obsidian_vault,
            folder=config.output.folder, date=date, title=title,
            filename_format=config.output.filename_format,
        )
        console.print(f"[green]Summary written to:[/green] {result_path}")

        if config.output.keep_transcript:
            t_path = write_transcript(
                transcript=transcript, vault_path=config.output.obsidian_vault,
                folder=config.output.folder, date=date, title=title,
                filename_format=config.output.filename_format,
            )
            console.print(f"[green]Transcript written to:[/green] {t_path}")
    else:
        console.print("[yellow]No Obsidian vault configured.[/yellow]")

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
