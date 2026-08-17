"""Unit tests for JARVIS core components."""

from __future__ import annotations

import pytest


# ── Context tests ──────────────────────────────────────────────────────────────


def test_context_add_message():
    from jarvis.context import JarvisContext
    ctx = JarvisContext()
    ctx.add_message("user", "Hello")
    ctx.add_message("assistant", "Hi there")
    msgs = ctx.get_messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["content"] == "Hi there"


def test_context_turn_count():
    from jarvis.context import JarvisContext
    ctx = JarvisContext()
    ctx.add_message("user", "first")
    ctx.add_message("user", "second")
    ctx.add_message("assistant", "reply")
    assert ctx.turn_count == 2


def test_context_object_tracking():
    from jarvis.context import JarvisContext, DetectedObject
    ctx = JarvisContext()
    objs = [
        DetectedObject(label="cat", confidence=0.9),
        DetectedObject(label="dog", confidence=0.8),
    ]
    ctx.set_detected_objects(objs)
    assert len(ctx.last_detected) == 2


def test_context_select_by_label():
    from jarvis.context import JarvisContext, DetectedObject
    ctx = JarvisContext()
    ctx.set_detected_objects([
        DetectedObject(label="red cup"),
        DetectedObject(label="blue pen"),
    ])
    result = ctx.select_by_label("cup")
    assert result is not None
    assert result.label == "red cup"
    assert ctx.selected_object is result


def test_context_select_missing():
    from jarvis.context import JarvisContext, DetectedObject
    ctx = JarvisContext()
    ctx.set_detected_objects([DetectedObject(label="apple")])
    result = ctx.select_by_label("banana")
    assert result is None


def test_context_reset():
    from jarvis.context import JarvisContext
    ctx = JarvisContext()
    ctx.add_message("user", "hello")
    ctx.reset()
    assert ctx.get_messages() == []
    assert ctx.turn_count == 0


# ── Config tests ───────────────────────────────────────────────────────────────


def test_config_defaults():
    from jarvis.config import JarvisConfig
    cfg = JarvisConfig()
    assert cfg.ollama.model == "qwen2.5:7b"
    assert cfg.stt.model_size == "small"   # upgraded from base for better accuracy
    assert cfg.tts.backend == "pyttsx3"
    assert cfg.input.ptt_key == "F9"


def test_config_permissions_defaults():
    from jarvis.config import JarvisConfig
    cfg = JarvisConfig()
    assert cfg.permissions.allow_shell is True
    assert cfg.permissions.allow_file_read is True
    assert cfg.permissions.allowed_paths == []


# ── System prompt tests ────────────────────────────────────────────────────────


def test_system_prompt_contains_tools():
    from jarvis.system_prompt import build_system_prompt
    tools = ["search_web", "speak", "detect_object"]
    prompt = build_system_prompt(tools)
    assert "search_web" in prompt
    assert "speak" in prompt
    assert "JARVIS" in prompt


def test_system_prompt_default_tools():
    from jarvis.system_prompt import build_system_prompt
    prompt = build_system_prompt()
    assert "detect_object" in prompt


# ── Tool tests — no external deps ─────────────────────────────────────────────


def test_web_tool_structure():
    """Test that search_web returns expected structure on failure."""
    from jarvis.tools.web_tools import search_web
    # This will either work (if duckduckgo-search is installed) or return an error dict
    result = search_web("Python programming", max_results=1)
    assert isinstance(result, dict)
    assert "query" in result or "error" in result


def test_system_tool_blocked_command():
    from jarvis.tools.system_tools import execute_command
    # "rm -rf" is in the default blocked list
    result = execute_command("rm -rf /tmp/test")
    assert "error" in result
    assert "blocked" in result["error"].lower()


def test_system_tool_read_missing_file():
    from jarvis.tools.system_tools import read_file
    result = read_file("/this/path/does/not/exist/file.txt")
    assert "error" in result


def test_manipulation_no_context():
    from jarvis.tools.manipulation_tools import delete_object
    # Without context, should return error dict
    result = delete_object("nonexistent")
    assert "error" in result


def test_vision_select_no_context():
    from jarvis.tools.vision_tools import select_object
    result = select_object("something")
    assert "error" in result


def test_speak_empty_text():
    from jarvis.tools.speech_tools import speak
    result = speak("")
    assert "error" in result


def test_system_screenshot_tool():
    from jarvis.tools.system_tools import take_screenshot
    result = take_screenshot("test_shot.png")
    assert "status" in result
    assert result["status"] == "saved"
    import os
    if os.path.exists(result.get("path", "")):
        try:
            os.remove(result["path"])
        except OSError:
            pass


def test_system_volume_tool():
    from jarvis.tools.system_tools import control_volume
    result = control_volume(action="up", steps=1)
    assert result.get("status") == "ok" or "error" in result


def test_system_media_tool():
    from jarvis.tools.system_tools import control_media
    result = control_media(action="play_pause")
    assert result.get("status") == "ok" or "error" in result


def test_system_desktop_tool():
    from jarvis.tools.system_tools import show_desktop
    result = show_desktop()
    assert result.get("status") == "ok" or "error" in result


def test_gesture_universal_mappings():
    from jarvis.inputs.gesture_input import DEFAULT_GESTURE_INTENT_MAP
    assert "Victory" in DEFAULT_GESTURE_INTENT_MAP
    assert "Thumb_Up" in DEFAULT_GESTURE_INTENT_MAP
    assert "Open_Palm" in DEFAULT_GESTURE_INTENT_MAP
    assert "Closed_Fist" in DEFAULT_GESTURE_INTENT_MAP
    # Ensure gestures are universal everywhere commands, not restricted to photo object manipulation
    assert "screenshot" in DEFAULT_GESTURE_INTENT_MAP["Victory"].lower()
    assert "volume" in DEFAULT_GESTURE_INTENT_MAP["Thumb_Up"].lower()
    assert "desktop" in DEFAULT_GESTURE_INTENT_MAP["Closed_Fist"].lower()


def test_all_tools_registry_complete():
    from jarvis.tools import ALL_TOOLS
    tool_names = [t.__name__ for t in ALL_TOOLS]
    assert "take_screenshot" in tool_names
    assert "control_volume" in tool_names
    assert "control_media" in tool_names
    assert "show_desktop" in tool_names
    assert "speak" in tool_names
    assert "detect_object" in tool_names

