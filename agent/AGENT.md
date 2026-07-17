---
name: global_development_rules
description: Global development rules.
---

## Skills

- If the task involve multiple steps, then load the `karpathy-guidelines` at the beginning of any conversation

## Notification

- If the tg-notify <msg> command is provided, use it to notify when the background task finishes.

## Python

- Always use the `uv` venv. If you cannot find it, ask the user first before continue. Never use system-level `pip` or install package outside of the `.venv` folder
- Always use `pathlib` to handle paths, instead of `os.path`
