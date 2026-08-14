"""Vision capture — grabs a frame from the webcam and returns it base64-encoded."""

from __future__ import annotations

import base64
import logging
import threading
from typing import Optional

import cv2
import numpy as np

from jarvis.config import get_config

logger = logging.getLogger(__name__)

_cap: Optional[cv2.VideoCapture] = None
_cap_lock = threading.Lock()


def _get_capture() -> cv2.VideoCapture:
    """Return (or initialise) the shared VideoCapture instance."""
    global _cap
    with _cap_lock:
        if _cap is None or not _cap.isOpened():
            idx = get_config().input.camera_index
            _cap = cv2.VideoCapture(idx)
            if not _cap.isOpened():
                raise RuntimeError(
                    f"Cannot open camera at index {idx}. "
                    "Check camera_index in config.yaml and webcam permissions."
                )
            # Warm up — skip a few frames
            for _ in range(3):
                _cap.read()
            logger.info("Camera opened at index %d", idx)
        return _cap


def capture_frame() -> Optional[np.ndarray]:
    """Capture a single frame from the webcam.

    Returns:
        A BGR numpy array (H×W×3), or None if capture failed.
    """
    try:
        cap = _get_capture()
        with _cap_lock:
            ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning("Empty frame from camera.")
            return None
        return frame
    except Exception as exc:
        logger.error("capture_frame error: %s", exc)
        return None


def capture_frame_b64(quality: int = 85) -> Optional[str]:
    """Capture a frame and return it as a base64-encoded JPEG string.

    Args:
        quality: JPEG compression quality, 1–100. Default 85.

    Returns:
        Base64 string, or None if capture failed.
    """
    frame = capture_frame()
    if frame is None:
        return None
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def release_camera() -> None:
    """Release the webcam resource. Call on shutdown."""
    global _cap
    with _cap_lock:
        if _cap is not None and _cap.isOpened():
            _cap.release()
            _cap = None
            logger.info("Camera released.")
