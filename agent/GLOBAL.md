---
name: global_development_rules
description: Global development rules.
---

## Skills

- Always load the `using-skills` at the beginning of any conversation

## Python

- Always use the `uv` venv. If you cannot find it, ask the user first before continue. Never use system-level `pip` or install package outside of the `.venv` folder
- Always use `pathlib` to handle paths, instead of `os.path`
