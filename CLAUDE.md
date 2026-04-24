# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**EchoNote** — local-first meeting transcription + LLM summarization + Obsidian export. Records audio locally, transcribes with `faster-whisper` in an isolated subprocess, summarizes with Claude / OpenAI, and writes Obsidian-native markdown (YAML frontmatter + structured body) to the user's vault.

See `README.md` for full feature list, design rationale, and architecture notes.

## Build & Test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```

## Module Layout

```
echonote/
├── cli.py          # typer commands + process orchestration
├── config.py       # TOML config with dataclasses
├── session.py      # session lifecycle + metadata
├── recorder.py     # sounddevice + chunked WAV writing (atomic renames)
├── transcriber.py  # faster-whisper + watchdog file watcher + crash recovery
├── summarizer.py   # LLM provider abstraction (Claude / OpenAI)
└── writer.py       # Obsidian markdown output + frontmatter injection
```

## Key Design Decisions (don't break these)

- **Dual-process, not dual-thread.** Whisper runs on a C++ backend (CTranslate2). A segfault there must not kill the recorder. Keep recorder + transcriber as separate subprocesses communicating via the filesystem.
- **Atomic WAV writes.** Recorder writes `chunk_NNNN.wav.tmp`, renames to `.wav` only when the chunk is complete. Transcriber's `watchdog` handler must filter for `*.wav` and ignore `*.wav.tmp`.
- **`summarize` is its own command.** Don't auto-summarize after recording. The user chooses when to pay the API cost and may re-run with a different prompt.
- **LLM provider is pluggable via config**, not hardcoded. Keep `summarizer.py` provider-agnostic.

## 知识库同步

本项目在 Obsidian 知识库中的记录映射为仓库根目录的 `obsidian-docs.md`（symlink 到 vault 里的对应文件）。读写它即读写知识库。

完成以下工作时，在 `obsidian-docs.md` 的"开发历程"section 追加 `### Phase N: 标题（日期范围）` + 要点：
- 新架构 / 新模块集成 / 方向转向
- 重大技术决策（换模型、换 pipeline）

不更新：日常 bug fix、小 refactor、配置调整。

遇到真 pivot（旧方案被新方案替代）时，按 vault 里 `AGENTS.md` 的 Supersession 约定处理：老版本归档到文件底部的"归档"section，加 `^block-id` + `superseded_by` callout，写明废弃原因。
