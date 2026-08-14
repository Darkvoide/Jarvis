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

# ── Gesture → JARVIS intent mapping ────────────────────────────────────────────
GESTURE_INTENT_MAP: dict[str, str] = {
    "Victory":       "detect_object from the current camera view",
    "Pointing_Up":   "select the most prominent object in the scene",
    "ILoveYou":      "zoom into the selected object by factor 2",
    "Open_Palm":     "cancel the current action",
    "Thumb_Up":      "confirm the last proposed action",
    "Closed_Fist":   "delete the selected object",
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


class GestureWatcher:
    """Watches the webcam for hand gestures and fires intent callbacks.

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
        debounce = self._cfg.input.gesture_debounce

        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            logger.error("GestureWatcher: cannot open camera %d.", cam_idx)
            return

        logger.info("Gesture watcher camera open at index %d.", cam_idx)

        with mp_vision.GestureRecognizer.create_from_options(opts) as recogniser:
            timestamp_ms = 0
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                timestamp_ms += 33  # ~30 fps

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                result = recogniser.recognize_for_video(mp_image, timestamp_ms)

                if result.gestures:
                    top_gesture = result.gestures[0][0]
                    gesture_name = top_gesture.category_name
                    score = top_gesture.score

                    if score < 0.75:
                        continue  # low confidence, skip

                    now = time.time()
                    if now - self._last_fire < debounce:
                        continue  # debounce

                    intent = GESTURE_INTENT_MAP.get(gesture_name)
                    if intent:
                        logger.info(
                            "Gesture: %s (%.2f) → intent: %s",
                            gesture_name, score, intent,
                        )
                        self._last_fire = now
                        try:
                            from jarvis.inputs.voice_input import _context as _vc
                        except Exception:
                            pass
                        self.on_intent(intent)

        cap.release()
        logger.info("GestureWatcher stopped.")
