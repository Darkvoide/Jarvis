"""Gesture input — MediaPipe Tasks API gesture recogniser on a webcam feed.

Runs in a daemon thread and maps hand gestures to JARVIS intent strings
that are fed into the orchestrator.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from jarvis.config import get_config

logger = logging.getLogger(__name__)

# ── Universal Gesture → Action & Intent Mapping (Desktop, Media, System) ──────
# Works everywhere across Windows, any app, browser, desktop or video player.
DEFAULT_GESTURE_INTENT_MAP: dict[str, str] = {
    "Victory":       "take a screenshot of the current screen",
    "Pointing_Up":   "scroll up",
    "Thumb_Up":      "increase the system volume",
    "Thumb_Down":    "decrease the system volume",
    "Open_Palm":     "toggle play pause for media",
    "Closed_Fist":   "minimize all windows and show desktop",
    "ILoveYou":      "go to previous tab",   # Ctrl+Shift+Tab
}

# MediaPipe gesture recogniser model — auto-downloaded on first run
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
_MODEL_DIR = Path(__file__).parent.parent.parent / ".mediapipe_models"
_MODEL_PATH = _MODEL_DIR / "gesture_recognizer.task"


def _ensure_model() -> Path:
    """Download the MediaPipe gesture model if not already present."""
    _MODEL_DIR.mkdir(exist_ok=True)
    if not _MODEL_PATH.exists():
        logger.info("Downloading MediaPipe gesture model (~25 MB)…")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        logger.info("Model saved to %s", _MODEL_PATH)
    return _MODEL_PATH


def execute_direct_gesture_action(gesture_name: str) -> bool:
    """Execute direct zero-latency Windows OS actions for detected gestures.

    Returns True if an instant system action was performed.
    """
    import ctypes

    KEYEVENTF_KEYUP = 0x0002

    try:
        if gesture_name == "Thumb_Up":
            # Volume Up (2 steps)
            for _ in range(2):
                ctypes.windll.user32.keybd_event(0xAF, 0, 0, 0)
                ctypes.windll.user32.keybd_event(0xAF, 0, KEYEVENTF_KEYUP, 0)
            print("🔊 [Gesture: Volume Up 👍]")
            return True

        elif gesture_name == "Thumb_Down":
            # Volume Down (2 steps)
            for _ in range(2):
                ctypes.windll.user32.keybd_event(0xAE, 0, 0, 0)
                ctypes.windll.user32.keybd_event(0xAE, 0, KEYEVENTF_KEYUP, 0)
            print("🔉 [Gesture: Volume Down 👎]")
            return True

        elif gesture_name == "Open_Palm":
            # Media Play/Pause toggle
            ctypes.windll.user32.keybd_event(0xB3, 0, 0, 0)
            ctypes.windll.user32.keybd_event(0xB3, 0, KEYEVENTF_KEYUP, 0)
            print("⏯️  [Gesture: Media Play/Pause 🖐️]")
            return True

        elif gesture_name == "Closed_Fist":
            # Show Desktop (Win + D)
            VK_LWIN = 0x5B
            VK_D = 0x44
            ctypes.windll.user32.keybd_event(VK_LWIN, 0, 0, 0)
            ctypes.windll.user32.keybd_event(VK_D, 0, 0, 0)
            ctypes.windll.user32.keybd_event(VK_D, 0, KEYEVENTF_KEYUP, 0)
            ctypes.windll.user32.keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, 0)
            print("🖥️  [Gesture: Show Desktop ✊]")
            return True

        elif gesture_name == "Pointing_Up":
            # Scroll Up
            ctypes.windll.user32.mouse_event(0x0800, 0, 0, 180, 0)
            print("⬆️  [Gesture: Scroll Up ☝️]")
            return True

        elif gesture_name == "Victory":
            # Capture screenshot
            from PIL import ImageGrab
            save_dir = Path(os.getcwd()) / "screenshots"
            save_dir.mkdir(exist_ok=True)
            shot_file = save_dir / f"gesture_screenshot_{int(time.time())}.png"
            ImageGrab.grab().save(shot_file)
            print(f"[Gesture: Screenshot -> {shot_file.name}]")
            return True

        elif gesture_name == "ILoveYou":
            # Previous Tab: Ctrl + Shift + Tab  (works in all browsers)
            VK_CONTROL = 0x11
            VK_SHIFT   = 0x10
            VK_TAB     = 0x09
            ctypes.windll.user32.keybd_event(VK_CONTROL, 0, 0, 0)
            ctypes.windll.user32.keybd_event(VK_SHIFT,   0, 0, 0)
            ctypes.windll.user32.keybd_event(VK_TAB,     0, 0, 0)
            ctypes.windll.user32.keybd_event(VK_TAB,     0, KEYEVENTF_KEYUP, 0)
            ctypes.windll.user32.keybd_event(VK_SHIFT,   0, KEYEVENTF_KEYUP, 0)
            ctypes.windll.user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
            print("[Gesture: Previous Tab]")
            return True

    except Exception as exc:
        logger.warning("Direct gesture action error (%s): %s", gesture_name, exc)

    return False


class GestureWatcher:
    """Watches the webcam for hand gestures and controls the system everywhere.

    Args:
        on_intent: Callback called with an intent string whenever a
                   recognised gesture is detected.
    """

    def __init__(self, on_intent: Callable[[str], None]) -> None:
        self.on_intent = on_intent
        self._cfg = get_config()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_fire: float = 0.0

    def start(self) -> None:
        """Start the gesture watcher in a background daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("GestureWatcher started.")

    def stop(self) -> None:
        """Signal the watcher to stop."""
        self._stop_event.set()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            logger.error("Gesture input dependency missing: %s", exc)
            return

        try:
            model_path = _ensure_model()
        except Exception as exc:
            logger.error("Could not download gesture model: %s", exc)
            return

        # Build recogniser
        base_opts = mp_python.BaseOptions(model_asset_path=str(model_path))
        opts = mp_vision.GestureRecognizerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
        )

        cam_idx = self._cfg.input.camera_index
        debounce = getattr(self._cfg.input, "gesture_debounce", 1.2)
        direct_actions = getattr(self._cfg.input, "direct_gestures", True)
        min_confidence = getattr(self._cfg.input, "gesture_min_confidence", 0.80)

        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            logger.error("GestureWatcher: cannot open camera %d.", cam_idx)
            return

        logger.info("Gesture watcher camera open at index %d.", cam_idx)

        # Merge default mappings with any user custom mappings
        gesture_map = dict(DEFAULT_GESTURE_INTENT_MAP)
        if hasattr(self._cfg.input, "gesture_mapping") and self._cfg.input.gesture_mapping:
            gesture_map.update(self._cfg.input.gesture_mapping)

        # Create recognizer manually (avoid with-statement so we can control shutdown order)
        recogniser = mp_vision.GestureRecognizer.create_from_options(opts)
        try:
            timestamp_ms = 0
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.04)
                    continue

                timestamp_ms += 33  # ~30 fps

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                try:
                    result = recogniser.recognize_for_video(mp_image, timestamp_ms)
                except Exception as rec_exc:
                    # Suppress MediaPipe shutdown race condition
                    if "cannot schedule new futures after shutdown" in str(rec_exc):
                        break
                    logger.warning("Recogniser error: %s", rec_exc)
                    continue

                if result.gestures:
                    top_gesture = result.gestures[0][0]
                    gesture_name = top_gesture.category_name
                    score = top_gesture.score

                    if score < min_confidence or gesture_name == "None":
                        continue  # low confidence or none

                    now = time.time()
                    if now - self._last_fire < debounce:
                        continue  # debounce

                    self._last_fire = now
                    logger.info("Hand gesture detected: %s (confidence: %.2f)", gesture_name, score)

                    # 1. First attempt instant zero-latency direct OS action if enabled
                    executed = False
                    if direct_actions and os.name == "nt":
                        executed = execute_direct_gesture_action(gesture_name)

                    # 2. Also emit to conversational orchestrator if mapped
                    intent = gesture_map.get(gesture_name)
                    if intent and not executed:
                        self.on_intent(intent)

        finally:
            # Safely release camera
            cap.release()
            # Safely close recognizer — suppress harmless Python shutdown race
            try:
                recogniser.close()
            except Exception:
                pass
            logger.info("GestureWatcher stopped.")
