"""Obsidian writer — generate and write markdown files to vault."""

import re
from pathlib import Path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_title(title: str, fallback: str = "meeting", max_length: int = 80) -> str:
    """Normalize a title into a filesystem-safe filename component."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", title).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip(" .-")
    if not cleaned:
        return fallback
    truncated = cleaned[:max_length].rstrip(" .-")
    return truncated or fallback


def _inject_date_frontmatter(summary: str, date: str) -> str:
    """Inject date into existing YAML frontmatter if not already present."""
    if summary.startswith("---"):
        end_idx = summary.find("---", 3)
        if end_idx == -1:
            return f"---\ndate: {date}\n---\n\n{summary}"
        frontmatter = summary[3:end_idx]
        if "date:" not in frontmatter:
            frontmatter = f"\ndate: {date}" + frontmatter
        return "---" + frontmatter + summary[end_idx:]
    else:
        return f"---\ndate: {date}\n---\n\n{summary}"


def write_summary(
    summary: str,
    vault_path: str,
    folder: str,
    date: str,
    title: str,
    filename_format: str,
) -> Path:
    vault = Path(vault_path).expanduser()
    output_dir = vault / folder
    _ensure_dir(output_dir)
    safe_title = sanitize_title(title)
    filename = filename_format.format(date=date, title=safe_title) + ".md"
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
    vault = Path(vault_path).expanduser()
    output_dir = vault / folder
    _ensure_dir(output_dir)
    safe_title = sanitize_title(title)
    filename = filename_format.format(date=date, title=safe_title) + "-transcript.md"
    output_path = output_dir / filename
    content = f"---\ndate: {date}\ntype: transcript\n---\n\n{transcript}"
    output_path.write_text(content, encoding="utf-8")
    return output_path
