"""Shared mutable conversation context for JARVIS."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectedObject:
    """Represents a single object that JARVIS has detected or selected."""

    label: str
    confidence: float = 1.0
    bounding_box: dict[str, float] | None = None  # {x, y, w, h} normalized 0–1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.label} (conf={self.confidence:.2f})"


class JarvisContext:
    """Thread-safe shared state for a JARVIS session.

    Holds the conversation history, the most recently detected objects,
    and the currently selected object for pronoun resolution
    ("zoom that", "delete the left one", etc.).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # ── Conversation history (OpenAI-style messages list) ──────────────────
        self.messages: list[dict[str, Any]] = []

        # ── Object tracking ────────────────────────────────────────────────────
        self.last_detected: list[DetectedObject] = []
        self.selected_object: DetectedObject | None = None

        # ── Session metadata ───────────────────────────────────────────────────
        self.turn_count: int = 0
        self.session_start: float = time.time()

        # ── Last gesture ───────────────────────────────────────────────────────
        self.last_gesture: str | None = None
        self.last_gesture_time: float = 0.0

    # ── Messages ───────────────────────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history (thread-safe)."""
        with self._lock:
            self.messages.append({"role": role, "content": content})
            if role == "user":
                self.turn_count += 1

    def get_messages(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the messages list (thread-safe)."""
        with self._lock:
            return list(self.messages)

    def append_raw(self, message: dict[str, Any]) -> None:
        """Append a raw message dict (for tool_call / tool result messages)."""
        with self._lock:
            self.messages.append(message)

    # ── Object tracking ────────────────────────────────────────────────────────

    def set_detected_objects(self, objects: list[DetectedObject]) -> None:
        """Update the list of detected objects."""
        with self._lock:
            self.last_detected = objects
            # Auto-select if there's exactly one
            if len(objects) == 1:
                self.selected_object = objects[0]

    def select_by_label(self, label: str) -> DetectedObject | None:
        """Select the first detected object matching label (case-insensitive)."""
        with self._lock:
            label_lower = label.lower()
            for obj in self.last_detected:
                if label_lower in obj.label.lower():
                    self.selected_object = obj
                    return obj
            return None

    def get_selected(self) -> DetectedObject | None:
        """Return the currently selected object."""
        with self._lock:
            return self.selected_object

    # ── Gesture tracking ───────────────────────────────────────────────────────

    def set_gesture(self, gesture: str) -> None:
        with self._lock:
            self.last_gesture = gesture
            self.last_gesture_time = time.time()

    def get_last_gesture(self) -> tuple[str | None, float]:
        with self._lock:
            return self.last_gesture, self.last_gesture_time

    # ── Utilities ──────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all state (start a fresh session)."""
        with self._lock:
            self.messages.clear()
            self.last_detected.clear()
            self.selected_object = None
            self.turn_count = 0
            self.session_start = time.time()
            self.last_gesture = None
            self.last_gesture_time = 0.0

    def summary(self) -> str:
        with self._lock:
            uptime = int(time.time() - self.session_start)
            return (
                f"Turns: {self.turn_count} | "
                f"Messages: {len(self.messages)} | "
                f"Detected objects: {len(self.last_detected)} | "
                f"Selected: {self.selected_object} | "
                f"Uptime: {uptime}s"
            )
