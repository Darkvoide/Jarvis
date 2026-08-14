"""Object manipulation tools — move, rotate, duplicate, delete."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_context = None


def _set_context(ctx) -> None:
    global _context
    _context = ctx


def _get_context():
    global _context
    if _context is None:
        raise RuntimeError("Manipulation context not initialised.")
    return _context


def move_object(label: str, direction: str, amount: float = 10.0) -> dict:
    """Move a detected object in the specified direction by a given amount.

    Args:
        label: The object label to move.
        direction: Cardinal direction — "left", "right", "up", "down",
                   "forward", or "backward".
        amount: Distance / units to move. Default is 10.0.

    Returns:
        A dict confirming the move action.
    """
    valid_directions = {"left", "right", "up", "down", "forward", "backward"}
    if direction.lower() not in valid_directions:
        return {
            "error": f"Invalid direction '{direction}'. Choose from: {', '.join(valid_directions)}."
        }

    try:
        ctx = _get_context()
        target = ctx.select_by_label(label) if label else ctx.get_selected()
        if target is None:
            return {"error": f"Object '{label}' not found. Run detect_object first."}
        target.metadata.setdefault("position", {"x": 0.0, "y": 0.0, "z": 0.0})
        pos = target.metadata["position"]
        if direction == "left":
            pos["x"] -= amount
        elif direction == "right":
            pos["x"] += amount
        elif direction == "up":
            pos["y"] += amount
        elif direction == "down":
            pos["y"] -= amount
        elif direction == "forward":
            pos["z"] -= amount
        elif direction == "backward":
            pos["z"] += amount
        return {"action": "move", "object": target.label, "direction": direction,
                "amount": amount, "position": pos, "status": "ok"}
    except RuntimeError as exc:
        return {"error": str(exc)}


def rotate_object(label: str, degrees: float, axis: str = "y") -> dict:
    """Rotate a detected object by the given angle.

    Args:
        label: The object label to rotate.
        degrees: Rotation angle in degrees (positive = clockwise).
        axis: Rotation axis — "x", "y", or "z". Default is "y" (vertical).

    Returns:
        A dict confirming the rotation.
    """
    if axis not in {"x", "y", "z"}:
        return {"error": f"Invalid axis '{axis}'. Choose x, y, or z."}

    try:
        ctx = _get_context()
        target = ctx.select_by_label(label) if label else ctx.get_selected()
        if target is None:
            return {"error": f"Object '{label}' not found."}
        rot = target.metadata.setdefault("rotation", {"x": 0.0, "y": 0.0, "z": 0.0})
        rot[axis] = (rot[axis] + degrees) % 360
        return {"action": "rotate", "object": target.label, "axis": axis,
                "degrees": degrees, "rotation": rot, "status": "ok"}
    except RuntimeError as exc:
        return {"error": str(exc)}


def duplicate_object(label: str) -> dict:
    """Duplicate a detected object, creating a copy alongside it.

    Args:
        label: The object label to duplicate.

    Returns:
        A dict with the new object label and confirmation.
    """
    try:
        ctx = _get_context()
        target = ctx.select_by_label(label) if label else ctx.get_selected()
        if target is None:
            return {"error": f"Object '{label}' not found."}

        from jarvis.context import DetectedObject
        import copy
        new_obj = DetectedObject(
            label=f"{target.label}_copy",
            confidence=target.confidence,
            bounding_box=copy.deepcopy(target.bounding_box),
            metadata=copy.deepcopy(target.metadata),
        )
        ctx.last_detected.append(new_obj)
        return {"action": "duplicate", "original": target.label,
                "copy": new_obj.label, "status": "ok"}
    except RuntimeError as exc:
        return {"error": str(exc)}


def delete_object(label: str) -> dict:
    """Delete a detected object from the current scene.

    Args:
        label: The object label to delete.

    Returns:
        A dict confirming deletion, or an error if not found.
    """
    try:
        ctx = _get_context()
        label_lower = label.lower()
        before = len(ctx.last_detected)
        ctx.last_detected = [
            o for o in ctx.last_detected if label_lower not in o.label.lower()
        ]
        after = len(ctx.last_detected)
        removed = before - after
        if removed == 0:
            return {"error": f"No object matching '{label}' to delete."}
        if ctx.selected_object and label_lower in ctx.selected_object.label.lower():
            ctx.selected_object = None
        return {"action": "delete", "object": label, "removed": removed, "status": "ok"}
    except RuntimeError as exc:
        return {"error": str(exc)}
