---
name: global_development_rules
description: Global development rules.
---

## Skills

- Before starting a multi-step task (more than one edit or command), load the `karpathy-guidelines` skill.

## Notification

- When a `run_in_background` task or other long-running job finishes, run `tg-notify "<msg>"` with a one-line result (what ran, succeeded or failed). Skip it for quick foreground work.

## Python

- Use `uv` for the environment and packages. Run Python through `uv run <command>`. If the project's `.venv` cannot be found, ask the user before continuing.
- Never use system-level `pip` or install packages outside the `.venv` folder.
- Use `pathlib` for all path handling, never `os.path` (e.g. `Path("models") / "config.yaml"`, not `os.path.join(...)`). The only exception is a legacy third-party API that requires it.
- Prefer `Path.read_text()`, `Path.write_bytes()`, and `Path.mkdir(parents=True)` for file I/O, and check `path.exists()` before destructive file operations.
