"""Vision tools — detect, select, zoom, and analyse objects via camera + LLM."""

from __future__ import annotations

import base64
import logging

from jarvis.config import get_config

logger = logging.getLogger(__name__)

# Lazily imported to avoid startup cost if vision is disabled
_context = None


def _get_context():
    """Return the global JarvisContext (imported lazily to avoid circular deps)."""
    global _context
    if _context is None:
        # Context is injected at runtime by main.py via _set_context()
        raise RuntimeError("Vision context not initialised. Call vision_tools._set_context(ctx).")
    return _context


def _set_context(ctx) -> None:
    """Inject the shared JarvisContext into this module."""
    global _context
    _context = ctx


# ── Helpers ────────────────────────────────────────────────────────────────────


def _capture_frame_b64() -> str | None:
    """Capture a camera frame and return it as a base64-encoded JPEG string."""
    try:
        from jarvis.inputs.vision_capture import capture_frame_b64
        return capture_frame_b64()
    except Exception as exc:
        logger.error("Frame capture failed: %s", exc)
        return None


def _vision_query(prompt: str, image_b64: str) -> str:
    """Send a vision query to Ollama and return the text response."""
    cfg = get_config().ollama
    if cfg.vision_model is None:
        return "[Vision model not configured. Set ollama.vision_model in config.yaml]"

    try:
        import ollama
        response = ollama.chat(
            model=cfg.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
        )
        return response.message.content or ""
    except Exception as exc:
        logger.error("Vision query failed: %s", exc)
        return f"[Vision query error: {exc}]"


# ── Tool functions ─────────────────────────────────────────────────────────────


def detect_object(scene_description: str = "") -> dict:
    """Capture the current camera view and detect all visible objects in the scene.

    Args:
        scene_description: Optional hint to narrow detection
                           (e.g. "red objects", "people").

    Returns:
        A dict with keys:
          - objects: list of detected object labels with confidence scores.
          - raw_response: full model description of the scene.
          - error: present only if something went wrong.
    """
    frame_b64 = _capture_frame_b64()
    if frame_b64 is None:
        return {"error": "Camera unavailable — cannot capture frame."}

    prompt = (
        "List every distinct object you can see in this image. "
        "For each, give: label, approximate location (left/centre/right, near/far), "
        "and confidence (high/medium/low). "
        f"{'Focus on: ' + scene_description + '.' if scene_description else ''}"
        "Respond as a JSON array of objects with keys: label, location, confidence."
    )

    raw = _vision_query(prompt, frame_b64)

    # Try to parse JSON from the response
    import json, re
    detected = []
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            detected = json.loads(match.group())
    except Exception:
        pass

    # Update shared context
    try:
        from jarvis.context import DetectedObject
        ctx = _get_context()
        objs = [
            DetectedObject(
                label=o.get("label", "unknown"),
                confidence={"high": 0.9, "medium": 0.6, "low": 0.3}.get(
                    o.get("confidence", "medium"), 0.6
                ),
                metadata={"location": o.get("location", "")},
            )
            for o in detected
            if isinstance(o, dict)
        ]
        ctx.set_detected_objects(objs)
    except RuntimeError:
        pass  # Context not set up (e.g., during unit tests)

    return {"objects": detected, "raw_response": raw}


def select_object(label: str) -> dict:
    """Select a specific object from the most recently detected objects.

    Args:
        label: The label of the object to select (partial match allowed).

    Returns:
        A dict with:
          - selected: the matched object label, or null if not found.
          - available: list of currently detected object labels.
    """
    try:
        ctx = _get_context()
        obj = ctx.select_by_label(label)
        available = [o.label for o in ctx.last_detected]
        if obj:
            return {"selected": obj.label, "available": available}
        return {
            "selected": None,
            "available": available,
            "error": f"No object matching '{label}' in the current scene.",
        }
    except RuntimeError as exc:
        return {"error": str(exc)}


def zoom_object(label: str, factor: float = 2.0) -> dict:
    """Zoom into a detected object by a given factor.

    Args:
        label: The object label to zoom into (uses last selected if empty).
        factor: Zoom multiplier — e.g. 2.0 = 2× zoom. Range: 1.1 – 10.0.

    Returns:
        A dict confirming the zoom action or reporting an error.
    """
    factor = max(1.1, min(factor, 10.0))

    try:
        ctx = _get_context()
        # Resolve label to detected object, or use already-selected
        target = ctx.select_by_label(label) if label else ctx.get_selected()
        if target is None:
            return {"error": f"Cannot zoom — no object named '{label}' found. Try detect_object first."}

        # In a real system this would drive a display / camera zoom.
        # Here we update metadata and report the action.
        target.metadata["zoom_factor"] = factor
        return {
            "action": "zoom",
            "object": target.label,
            "factor": factor,
            "status": "ok",
        }
    except RuntimeError as exc:
        return {"error": str(exc)}


def analyze_object(label: str = "") -> dict:
    """Perform a detailed visual analysis of the selected or named object.

    Args:
        label: Object label to analyse. Uses the currently selected object if empty.

    Returns:
        A dict with a detailed description of the object.
    """
    frame_b64 = _capture_frame_b64()
    if frame_b64 is None:
        return {"error": "Camera unavailable — cannot analyse object."}

    try:
        ctx = _get_context()
        target = ctx.select_by_label(label) if label else ctx.get_selected()
        target_label = target.label if target else (label or "the main object in the scene")
    except RuntimeError:
        target_label = label or "the main object in the scene"

    prompt = (
        f"Focus on {target_label} in this image. "
        "Describe it in detail: colour, size, material, condition, any text visible, "
        "and anything else that would help identify or interact with it. "
        "Be concise and factual."
    )

    raw = _vision_query(prompt, frame_b64)
    return {"object": target_label, "analysis": raw}
