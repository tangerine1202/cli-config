#!/usr/bin/env python3
"""
Simple Python wrapper for tg-notify.sh
Usage:
    from tg_notify import notify
    notify("Experiment completed!")
"""

import subprocess
import os
from pathlib import Path

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


if __name__ == "__main__":
    # Test the notification
    import sys
    message = sys.argv[1] if len(sys.argv) > 1 else "Test notification from Python"
    success = notify(message)
    sys.exit(0 if success else 1)
