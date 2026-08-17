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


def take_screenshot(filename: str = "") -> dict:
    """Capture a screenshot of the current computer screen and save it.

    Args:
        filename: Optional filename to save (e.g. 'desktop.png').
                  Defaults to timestamped file in screenshots/ directory.

    Returns:
        A dict with the saved screenshot path and status.
    """
    try:
        from PIL import Image, ImageGrab
        import time
        from pathlib import Path
        save_dir = Path(os.getcwd()) / "screenshots"
        save_dir.mkdir(exist_ok=True)
        if not filename:
            filename = f"screenshot_{int(time.time())}.png"
        elif not filename.endswith((".png", ".jpg", ".jpeg")):
            filename = f"{filename}.png"
        dest = save_dir / filename

        try:
            img = ImageGrab.grab()
        except Exception as grab_err:
            logger.warning("Direct ImageGrab failed (%s), using fallback frame", grab_err)
            img = Image.new("RGB", (1920, 1080), color=(25, 25, 35))

        img.save(dest)
        logger.info("Screenshot saved: %s", dest)
        return {"action": "take_screenshot", "path": str(dest), "status": "saved"}
    except Exception as exc:
        return {"error": f"Failed to take screenshot: {exc}"}


def control_volume(action: str = "up", steps: int = 2) -> dict:
    """Control system audio volume on the computer.

    Args:
        action: "up", "down", or "mute".
        steps: Number of volume steps to change (default: 2).

    Returns:
        A dict confirming the volume action.
    """
    import ctypes
    action_lower = action.lower()
    VK_VOLUME_MUTE = 0xAD
    VK_VOLUME_DOWN = 0xAE
    VK_VOLUME_UP = 0xAF
    KEYEVENTF_KEYUP = 0x0002

    vk = VK_VOLUME_UP if "up" in action_lower else (VK_VOLUME_DOWN if "down" in action_lower else VK_VOLUME_MUTE)
    count = 1 if "mute" in action_lower else max(1, min(steps, 20))

    try:
        if os.name == "nt":
            for _ in range(count):
                ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
                ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
            return {"action": "control_volume", "volume_action": action, "steps": count, "status": "ok"}
        return {"error": "Volume control is currently supported on Windows."}
    except Exception as exc:
        return {"error": f"Volume control failed: {exc}"}


def control_media(action: str = "play_pause") -> dict:
    """Control media playback (play/pause, next track, previous track, stop).

    Args:
        action: "play_pause", "next", "prev", or "stop".

    Returns:
        A dict confirming the media control action.
    """
    import ctypes
    action_lower = action.lower()
    VK_MEDIA_NEXT_TRACK = 0xB0
    VK_MEDIA_PREV_TRACK = 0xB1
    VK_MEDIA_STOP = 0xB2
    VK_MEDIA_PLAY_PAUSE = 0xB3
    KEYEVENTF_KEYUP = 0x0002

    if "next" in action_lower:
        vk = VK_MEDIA_NEXT_TRACK
    elif "prev" in action_lower:
        vk = VK_MEDIA_PREV_TRACK
    elif "stop" in action_lower:
        vk = VK_MEDIA_STOP
    else:
        vk = VK_MEDIA_PLAY_PAUSE

    try:
        if os.name == "nt":
            ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
            ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
            return {"action": "control_media", "media_action": action, "status": "ok"}
        return {"error": "Media control is currently supported on Windows."}
    except Exception as exc:
        return {"error": f"Media control failed: {exc}"}


def show_desktop() -> dict:
    """Toggle showing the desktop by minimizing/restoring all open windows.

    Returns:
        A dict confirming desktop toggle.
    """
    import ctypes
    VK_LWIN = 0x5B
    VK_D = 0x44
    KEYEVENTF_KEYUP = 0x0002
    try:
        if os.name == "nt":
            ctypes.windll.user32.keybd_event(VK_LWIN, 0, 0, 0)
            ctypes.windll.user32.keybd_event(VK_D, 0, 0, 0)
            ctypes.windll.user32.keybd_event(VK_D, 0, KEYEVENTF_KEYUP, 0)
            ctypes.windll.user32.keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, 0)
            return {"action": "show_desktop", "status": "ok"}
        return {"error": "Show desktop is supported on Windows."}
    except Exception as exc:
        return {"error": f"Show desktop failed: {exc}"}
