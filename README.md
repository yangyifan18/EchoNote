<div align="center">

# EchoNote

**Your meetings, transcribed and understood — automatically.**

Record, transcribe with local Whisper, summarize with frontier LLMs, and archive into Obsidian. All in one command.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Whisper](https://img.shields.io/badge/whisper-large--v3-orange.svg)
![Tests](https://img.shields.io/badge/tests-29%20passing-brightgreen.svg)

</div>

---

## Why EchoNote

Academic talks and technical reports are full of details that never make it into the paper — data collection pipelines, engineering tricks, failure stories. Taking notes by hand forces you to choose between listening and writing. Photos don't work for private slides. And a week later, all you remember is "something about data cleaning."

EchoNote runs silently alongside your meeting, captures everything, and hands you back a structured, AI-curated note in your knowledge base — so you can focus on **listening and asking questions**, not on transcription.

## Features

- **Local-first transcription** — Whisper runs on your machine. No audio ever leaves your laptop.
- **Dual-process architecture** — Recording and transcription run in isolated subprocesses. If one crashes, the other survives. Your audio is safe.
- **Structured summaries, not bullet-point compression** — The LLM follows the speaker's original logic (by project, by topic, by timeline), preserves concrete details (numbers, tool names, paper references), and flags information not in the published work. Notes stay queryable for downstream AI agents.
- **Obsidian-native output** — Markdown with YAML frontmatter, ready to drop into your vault. Tags, dates, speakers all wired up.
- **Pluggable LLMs** — Claude and OpenAI out of the box. Swap providers with a config change.
- **Mic or system audio** — Capture in-person talks via microphone, or online meetings via BlackHole virtual audio.
- **Decoupled summarization** — Stop recording now, summarize later. Or re-summarize the same transcript with a different prompt.

## Quick Start

```bash
# 1. Install
git clone git@github.com:yangyifan18/EchoNote.git
cd EchoNote
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Configure your vault + API key
mkdir -p ~/.echonote
cat > ~/.echonote/config.toml <<EOF
[output]
obsidian_vault = "~/Documents/ObsidianVault"
folder = "Meetings"

[llm]
provider = "claude"
model = "claude-sonnet-4-6"
api_key_env = "ANTHROPIC_API_KEY"
EOF

export ANTHROPIC_API_KEY=sk-ant-...

# 3. Record a meeting
echonote start            # Ctrl+C to stop

# 4. Generate the summary
echonote summarize        # Writes to your Obsidian vault
```

## How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                     echonote start                           │
│                                                              │
│  ┌─────────────┐    .wav chunks    ┌──────────────────┐    │
│  │  Recorder    │───── (disk) ────▶│   Transcriber     │    │
│  │  subprocess  │                  │   subprocess       │    │
│  │              │                  │                    │    │
│  │  sounddevice │                  │  faster-whisper    │    │
│  │  atomic I/O  │                  │  watchdog          │    │
│  └─────────────┘                  └──────────────────┘    │
│         │                                   │               │
│         └───────────┬───────────────────────┘               │
│                     ▼                                        │
│         ~/.echonote/sessions/{id}/                           │
│           ├── chunks/       (audio)                          │
│           ├── transcript.txt                                 │
│           └── meta.json                                      │
└──────────────────────────────────────────────────────────────┘
                      │
                      │  echonote summarize
                      ▼
         ┌────────────────────────┐
         │   LLM Provider          │
         │   (Claude / OpenAI)     │
         └────────────┬───────────┘
                      ▼
         ┌────────────────────────┐
         │   Obsidian Vault        │
         │   vault/Meetings/*.md   │
         └────────────────────────┘
```

**Key design decisions:**

| Decision | Rationale |
|---|---|
| Dual-process (not threads) | Fault isolation — a segfault in Whisper's C++ backend can't kill the recorder |
| Atomic WAV writes (`.wav.tmp` → `.wav`) | Transcriber sees only complete chunks, never partial files |
| Local Whisper via `faster-whisper` | 4× faster than the reference implementation, privacy-preserving |
| Decoupled `summarize` command | Re-run with different prompts, or delay summarization until later |
| Filesystem as IPC | No message brokers, no shared memory — just durable `.wav` files |

## Commands

```
echonote start [--system]    Start recording (Ctrl+C to stop)
                             --system  captures system audio via BlackHole
echonote stop                Stop recording (for daemon mode)
echonote summarize [id]      Generate AI summary → Obsidian (default: latest session)
echonote list                List all recorded sessions
echonote show [id]           Show session details, transcript length, summary status
echonote status              Show current recording status
echonote config              Show current configuration
```

## Output Format

EchoNote produces Obsidian-native markdown that respects the speaker's structure:

```markdown
---
date: 2026-04-10
speaker: Dr. Li (XX Research Institute)
topic: Four Years of Multimodal Data Engineering
tags: [meeting, multimodal, data-pipeline]
---

## 摘要
Dr. Li reviewed four years of work on multimodal data processing,
focusing on engineering details not published in the papers...

## 详细内容

### 1. VisualQA (CVPR 2023)
- Motivation: existing VQA datasets lack real-world coverage
- Data collection: 120K medical images from 3 hospitals, 6-month labeling cycle
- 📌 论文外: Initially tried crowdsourcing but quality was uncontrollable,
  switched to in-house team
- Key result: 12% accuracy gain driven by data quality, not model changes

### 2. DataCleaner (NeurIPS 2024)
- Three-stage pipeline: format normalization → semantic dedup → quality scoring
- 📌 论文外: Pipeline uses Airflow, processes 1M records in ~4 hours
- Open-sourced code but cleaning rules are internal

## Q&A
### Q1: Annotation cost breakdown?
~¥15 per image labeling, additional ¥5 for QC review...

## 行动项
- [ ] Investigate the DataCleaner open-source repo
- [ ] Evaluate three-stage cleaning on our datasets
```

## Configuration

All settings live in `~/.echonote/config.toml`:

```toml
[audio]
device = "default"           # or a specific device name
source = "mic"               # "mic" | "system"
chunk_duration = 30          # seconds per audio chunk
sample_rate = 16000

[whisper]
model = "large-v3"           # tiny | base | small | medium | large-v3
language = "zh"              # ISO 639-1
device = "auto"              # "cpu" | "cuda" | "auto" (no MPS — CTranslate2 limitation)

[llm]
provider = "claude"          # "claude" | "openai"
model = "claude-sonnet-4-6"
api_key_env = "ANTHROPIC_API_KEY"

[output]
obsidian_vault = "~/Documents/ObsidianVault"
folder = "Meetings"
keep_transcript = true       # save raw transcript alongside summary
keep_audio = false           # discard .wav chunks after transcription
filename_format = "{date}-{title}"

[sessions]
dir = "~/.echonote/sessions"
```

## Requirements

- **Python 3.11+**
- **macOS / Linux** (Windows untested)
- **Microphone** — any input device `sounddevice` recognizes
- **BlackHole** (optional) — for system audio capture on macOS ([install guide](https://existential.audio/blackhole/))
- **Anthropic or OpenAI API key** — for summarization
- **~3 GB disk** — for the `large-v3` Whisper model (auto-downloaded on first use)

## Architecture Notes

**Why two processes instead of two threads?**

`faster-whisper` runs on a C++ backend (CTranslate2). A segmentation fault in native code would kill the entire Python process, including the audio capture. With separate subprocesses, a transcription crash leaves the recorder untouched — and because audio is persisted to disk immediately, nothing is lost.

**Why atomic renames?**

The recorder writes each audio chunk as `chunk_0001.wav.tmp`, then atomically renames to `chunk_0001.wav` when the chunk is complete. The transcriber's `watchdog` handler filters for `*.wav` and ignores `*.wav.tmp`, so it never reads a half-written file. This is a POSIX guarantee, no locking required.

**Why is `summarize` a separate command?**

Two reasons:
1. **Cost control** — You might not want to burn API credits on every recording. Summarize selectively.
2. **Iteration** — If the first summary misses the point, re-run `echonote summarize` with a different prompt. The transcript is already on disk; you pay only for the LLM call.

## Development

```bash
pip install -e ".[dev]"
pytest -v                    # 29 tests
```

Project structure:
```
echonote/
├── cli.py           # typer commands + process orchestration
├── config.py        # TOML config with dataclasses
├── session.py       # session lifecycle + metadata
├── recorder.py      # sounddevice + chunked WAV writing
├── transcriber.py   # faster-whisper + watchdog
├── summarizer.py    # LLM provider abstraction
└── writer.py        # Obsidian markdown output
```

## Roadmap

- [ ] Live scrolling transcript display during recording
- [ ] Speaker diarization (who said what)
- [ ] Custom prompt templates per meeting type
- [ ] Export to other knowledge bases (Logseq, Notion, plain markdown)
- [ ] Incremental summarization (every N minutes)
- [ ] Windows support

## License

MIT
