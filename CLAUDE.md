# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal dotfiles + Claude Code configuration repo. It serves two purposes:
1. Editor/terminal dotfiles (vim, nvim, tmux, iTerm color schemes), deployed via `./setup.sh`.
2. The source of truth for Claude Code skills, subagents, and global rules — deployed into `~/.claude/` via symlinks in `./setup.sh`. Nothing under `agent/` is live for Claude Code until `setup.sh` has been run at least once on a given machine.

## Structure

- `agent/AGENT.md` — global development rules, symlinked to `~/.claude/CLAUDE.md` so they apply to *every* project on this machine, not just this repo.
- `agent/skills/` — Claude Code skills; each subdirectory is symlinked individually into `~/.claude/skills/`.
- `agent/agents/` — Claude Code subagents; each subdirectory is symlinked individually into `~/.claude/agents/`.
- `claude/GEMINI.md` — the equivalent global rules file for Gemini CLI. Kept as a separate file, not auto-synced with `agent/AGENT.md` — if a Python/tooling rule changes, update both.
- `nvim/`, `.vim/`, `.vimrc`, `.tmux.conf`, `*.itermcolors` — editor/terminal dotfiles.
- `tg-notify.sh` / `tg_notify.py` — Telegram notification CLI, installed to `~/.local/bin/tg-notify`.
- `charile/` — gitignored scratch space, unrelated to this repo's tracked purpose.

## Deploying changes

Run `./setup.sh` after adding or editing anything under `agent/` or the dotfiles. It rsyncs vim/nvim/tmux configs, installs `tg-notify`, and symlinks `agent/skills/*`, `agent/agents/*`, and `agent/AGENT.md` into `~/.claude/`. A **new** skill or agent directory requires a `setup.sh` run to go live; edits to files inside an already-symlinked skill/agent take effect immediately.

## Adding a new skill or agent

- New skill: create `agent/skills/<name>/SKILL.md` with YAML frontmatter (`name`, `description` at minimum), then run `./setup.sh`.
- New agent: create `agent/agents/<name>/AGENT.md` with `name`, `description`, `tools`, `model`, `permissionMode` frontmatter.
- Before naming a skill, check it won't shadow a Claude Code built-in or an official plugin skill (e.g. `code-review`, `skill-creator` are already covered by built-ins/plugins — don't duplicate them here).

## Gotchas

- No `gh` CLI installed on this machine. Remote is `git@github.com:tangerine1202/cli-config.git`.
