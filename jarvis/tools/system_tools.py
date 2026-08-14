"""System tools — open applications, execute shell commands, read files."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

from jarvis.config import get_config

logger = logging.getLogger(__name__)


def open_application(app_name: str) -> dict:
    """Open an application by name on the host system.

    Args:
        app_name: The name or path of the application to open.
                  Examples: "Chrome", "Notepad", "Calculator", "C:/path/to/app.exe"

    Returns:
        A dict with status and any error message.
    """
    logger.info("open_application: %s", app_name)

    # Common Windows application aliases
    WIN_ALIASES: dict[str, str] = {
        "chrome": "chrome.exe",
        "google chrome": "chrome.exe",
        "firefox": "firefox.exe",
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "calc": "calc.exe",
        "paint": "mspaint.exe",
        "explorer": "explorer.exe",
        "file explorer": "explorer.exe",
        "task manager": "taskmgr.exe",
        "cmd": "cmd.exe",
        "powershell": "powershell.exe",
        "word": "winword.exe",
        "excel": "excel.exe",
        "vscode": "code",
        "vs code": "code",
        "visual studio code": "code",
        "spotify": "spotify.exe",
        "vlc": "vlc.exe",
    }

    resolved = WIN_ALIASES.get(app_name.strip().lower(), app_name)

    try:
        if os.name == "nt":
            os.startfile(resolved)  # type: ignore[attr-defined]
        else:
            subprocess.Popen([resolved], close_fds=True)
        return {"action": "open_application", "app": app_name, "status": "launched"}
    except FileNotFoundError:
        return {
            "error": f"Application '{app_name}' not found. Check if it's installed and on your PATH."
        }
    except Exception as exc:
        return {"error": f"Failed to open '{app_name}': {exc}"}


def execute_command(command: str) -> dict:
    """Execute a shell command and return its output.

    Args:
        command: The shell command string to run.
                 Examples: "dir", "ipconfig", "python --version"

    Returns:
        A dict with stdout, stderr, returncode, and status.
        Returns an error dict if the command is blocked by permissions.
    """
    cfg = get_config().permissions

    if not cfg.allow_shell:
        return {"error": "Shell execution is disabled. Set permissions.allow_shell: true in config.yaml."}

    # Check blocked patterns
    for pattern in cfg.blocked_commands:
        if re.search(pattern, command, re.IGNORECASE):
            return {"error": f"Command blocked by permissions policy: matched pattern '{pattern}'."}

    logger.info("execute_command: %s", command)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "status": "ok" if result.returncode == 0 else "error",
        }
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out after 30 seconds."}
    except Exception as exc:
        return {"error": f"Command execution failed: {exc}"}


def read_file(path: str) -> dict:
    """Read and return the contents of a file.

    Args:
        path: Absolute or relative path to the file to read.

    Returns:
        A dict with the file content, or an error message.
    """
    cfg = get_config().permissions

    if not cfg.allow_file_read:
        return {"error": "File reading is disabled. Set permissions.allow_file_read: true in config.yaml."}

    resolved = Path(path).expanduser().resolve()

    # Check allowed paths if configured
    if cfg.allowed_paths:
        allowed = [Path(p).expanduser().resolve() for p in cfg.allowed_paths]
        if not any(str(resolved).startswith(str(a)) for a in allowed):
            return {
                "error": (
                    f"Access denied: '{resolved}' is outside the allowed paths. "
                    f"Allowed: {[str(a) for a in allowed]}"
                )
            }

    logger.info("read_file: %s", resolved)

    if not resolved.exists():
        return {"error": f"File not found: {resolved}"}
    if not resolved.is_file():
        return {"error": f"Path is not a file: {resolved}"}

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
        # Truncate very large files to avoid flooding the context
        MAX_CHARS = 8000
        truncated = len(content) > MAX_CHARS
        return {
            "path": str(resolved),
            "content": content[:MAX_CHARS],
            "truncated": truncated,
            "total_chars": len(content),
            "status": "ok",
        }
    except Exception as exc:
        return {"error": f"Could not read '{resolved}': {exc}"}
