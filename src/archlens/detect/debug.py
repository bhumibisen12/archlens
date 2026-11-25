"""Debug visualization overlay helpers for component detection."""

from pathlib import Path

import cv2
import numpy as np

from archlens.graph import Node, NodeType

_TYPE_COLORS: dict[NodeType, tuple[int, int, int]] = {
    "box": (210, 90, 20),
    "database": (30, 150, 40),
    "ellipse": (180, 40, 160),
    "component": (20, 130, 220),
}


def draw_detection_overlay(
    image: np.ndarray,
    nodes: list[Node],
    out_path: Path | None = None,
) -> np.ndarray:
    """Render detected node bounding boxes, types, and confidences onto an image.

    Args:
        image: Base BGR image (e.g., ``preprocessed.color``).
        nodes: Detected :class:`Node` instances.
        out_path: Optional file path to save the rendered PNG overlay.

    Returns:
        Copy of ``image`` with detection annotations drawn.
    """
    overlay = image.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    for node in nodes:
        color = _TYPE_COLORS.get(node.type, (0, 165, 255))
        x, y, w, h = node.bbox.x, node.bbox.y, node.bbox.w, node.bbox.h
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        tag = f"{node.id}: {node.type} ({node.confidence:.2f})"
        ty = max(18, y - 6)
        cv2.putText(
            overlay,
            tag,
            (x, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    if out_path is not None:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), overlay)

    return overlay

