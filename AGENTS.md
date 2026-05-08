# AGENTS.md

This repo also has `CLAUDE.md`. Read it first; it contains the project overview, key design decisions, commands, and knowledge-base sync rules. This file adds Codex-facing collaboration defaults.

## Agent Collaboration Defaults

- Treat `AGENTS.md` as durable collaboration memory; keep current-task details in the prompt.
- Before writing substantial code, read `CLAUDE.md`, inspect the relevant source and tests, then state a minimal implementation plan.
- Do not rely only on internal model memory when external docs, library behavior, API details, or prior implementations matter; inspect those sources first.
- Prefer the smallest runnable slice, localized diffs, and verification before broad refactors.
- Use subagents only for clearly separable research, implementation, or verification tasks. Do not use them for small edits.
- When a task reveals a stable command, risk, convention, or completion rule, update this file or `CLAUDE.md`.

## Project Guardrails

- Keep recorder and transcriber as separate subprocesses; do not collapse them into threads.
- Preserve atomic WAV writes: recorder writes `.wav.tmp` and renames to `.wav` only after completion.
- Keep `summarize` as an explicit command; do not auto-summarize after recording unless the user asks.
- Keep LLM providers pluggable through config, not hardcoded.
- For major architecture, module integration, or direction changes, update `obsidian-docs.md` following the knowledge-base sync rule in `CLAUDE.md`.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```
