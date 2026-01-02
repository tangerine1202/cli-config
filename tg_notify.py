#!/usr/bin/env python3
"""
Simple Python wrapper for tg-notify
Usage:
    from tg_notify import notify, wrap
    notify("Experiment completed!")
    wrap("exp-1", ["python3", "train.py", "--epochs", "10"])
"""

import subprocess
import os
import sys
from pathlib import Path
from typing import List, Union, Optional

def notify(message: str, script_path: str = None) -> bool:
    """
    Send a Telegram notification.

    Args:
        message: The message to send
        script_path: Path to tg-notify (default: looks in PATH, ~/.local/bin, current dir)

    Returns:
        True if successful, False otherwise
    """
    if script_path is None:
        # Try to find the script
        possible_paths = [
            Path.home() / ".local" / "bin" / "tg-notify",  # User bin (preferred)
            Path(__file__).parent / "tg-notify.sh",  # Same directory as this file
            Path.home() / "tg-notify.sh",  # Home directory
            "/usr/local/bin/tg-notify",  # System-wide installation
        ]

        # First check if it's in PATH
        import shutil
        if shutil.which("tg-notify"):
            script_path = "tg-notify"
        else:
            # Check known locations
            for path in possible_paths:
                if path.exists():
                    script_path = str(path)
                    break
            else:
                raise FileNotFoundError(
                    "tg-notify not found. Please run setup.sh or install it to ~/.local/bin"
                )

    try:
        result = subprocess.run(
            [script_path, message],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to send notification: {e}")
        return False


def notify_decorator(func):
    """
    Decorator to send notifications when a function completes.

    Example:
        @notify_decorator
        def train_model():
            # your training code
            return accuracy

        result = train_model()  # Will send notification when done
    """
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        try:
            result = func(*args, **kwargs)
            notify(f"✓ {func_name} completed successfully. Result: {result}")
            return result
        except Exception as e:
            notify(f"✗ {func_name} failed with error: {str(e)}")
            raise
    return wrapper


def wrap(label: str, command: Union[str, List[str]], script_path: Optional[str] = None) -> int:
    """
    Execute a command and send notification based on success/failure.

    Args:
        label: Experiment name/label for the notification
        command: Command to execute (string or list of args)
        script_path: Path to tg-notify (default: auto-detect)

    Returns:
        Exit code of the command

    Example:
        # Using list (recommended)
        exit_code = wrap("exp-1", ["python3", "train.py", "--epochs", "10"])

        # Using string
        exit_code = wrap("exp-1", "python3 train.py --epochs 10")
    """
    if script_path is None:
        # Try to find the script
        import shutil
        if shutil.which("tg-notify"):
            script_path = "tg-notify"
        else:
            possible_paths = [
                Path.home() / ".local" / "bin" / "tg-notify",
                Path(__file__).parent / "tg-notify.sh",
                Path.home() / "tg-notify.sh",
                "/usr/local/bin/tg-notify",
            ]
            for path in possible_paths:
                if path.exists():
                    script_path = str(path)
                    break
            else:
                raise FileNotFoundError(
                    "tg-notify not found. Please run setup.sh or install it to ~/.local/bin"
                )

    # Build the full command
    if isinstance(command, str):
        full_command = [script_path, label] + command.split()
    else:
        full_command = [script_path, label] + command

    try:
        result = subprocess.run(full_command)
        return result.returncode
    except Exception as e:
        print(f"Failed to execute command: {e}")
        return 1


if __name__ == "__main__":
    # Test the notification
    import sys
    message = sys.argv[1] if len(sys.argv) > 1 else "Test notification from Python"
    success = notify(message)
    sys.exit(0 if success else 1)
