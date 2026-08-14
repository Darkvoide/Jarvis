"""Tool registry — single source of truth for all JARVIS tools.

Import ALL_TOOLS and pass it directly to the Orchestrator::

    from jarvis.tools import ALL_TOOLS
    orch = Orchestrator(ctx, tools=ALL_TOOLS)
"""

from jarvis.tools.manipulation_tools import (
    delete_object,
    duplicate_object,
    move_object,
    rotate_object,
)
from jarvis.tools.speech_tools import speak
from jarvis.tools.system_tools import execute_command, open_application, read_file
from jarvis.tools.vision_tools import analyze_object, detect_object, select_object, zoom_object
from jarvis.tools.web_tools import search_web

ALL_TOOLS: list = [
    # Vision
    detect_object,
    select_object,
    zoom_object,
    analyze_object,
    # Manipulation
    move_object,
    rotate_object,
    duplicate_object,
    delete_object,
    # System
    open_application,
    execute_command,
    read_file,
    # Web
    search_web,
    # Speech
    speak,
]

__all__ = ["ALL_TOOLS"]
