---
name: global_python_development_standards
description: Global standards for Python environment management and coding style.
priority: critical
---

# Global Development Rules

## 1. Python Environment & Package Management
* **Tooling**: Always use `uv` as the primary package and virtual environment manager.
* **Activation**: 
    * Before executing any Python script or command, ensure the virtual environment is activated via `source .venv/bin/activate`.
    * Prefer using `uv run <command>` for consistency across different environments.
* **Constraint**: Never use system-level `pip` or install packages outside of the `.venv` directory.

## 2. File & Path Handling
* **Library Requirement**: Use the `pathlib` module for all filesystem path manipulations.
* **Prohibited Modules**: Do not use `os.path` unless explicitly required by a legacy third-party library.
* **Implementation Patterns**:
    * **Correct**: `Path("models") / "robot_arm" / "config.yaml"`
    * **Incorrect**: `os.path.join("models", "robot_arm", "config.yaml")`
* **Utility Methods**: Use `Path.read_text()`, `Path.write_bytes()`, and `Path.mkdir(parents=True)` for cleaner file operations.
* **Robustness**: Always include existence checks (`path.exists()`) when performing destructive file operations.
