"""Debug overlay rendering for extracted connectors and arrowheads."""

import math
from pathlib import Path

import cv2
import numpy as np

from archlens.graph import Edge, Node


def _draw_overlay_arrow_tip(
    canvas: np.ndarray,
    tail_xy: tuple[float, float],
    tip_xy: tuple[float, float],
    color: tuple[int, int, int],
    size: int = 12,
) -> None:
    """Draw a filled triangle arrowhead on the debug overlay."""
    dx = tip_xy[0] - tail_xy[0]
    dy = tip_xy[1] - tail_xy[1]
    length = math.hypot(dx, dy)
    if length <= 1e-3:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    bx = tip_xy[0] - ux * size
    by = tip_xy[1] - uy * size
    half_w = size * 0.45
    tri = np.array(
        [
            [round(tip_xy[0]), round(tip_xy[1])],
            [round(bx + px * half_w), round(by + py * half_w)],
            [round(bx - px * half_w), round(by - py * half_w)],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [tri], color, lineType=cv2.LINE_AA)


def draw_connector_overlay(
    image: np.ndarray,
    nodes: list[Node],
    edges: list[Edge],
    out_path: Path | None = None,
) -> np.ndarray:
    """Render nodes, extracted connector polylines, and arrowheads onto an image."""
    overlay = image.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    for node in nodes:
        b = node.bbox
        cv2.rectangle(overlay, (b.x, b.y), (b.x + b.w, b.y + b.h), (180, 180, 180), 1)

    for edge in edges:
        color = (20, 140, 240) if edge.directed else (160, 110, 40)
        if len(edge.polyline) >= 2:
            pts = np.array(
                [[round(p.x), round(p.y)] for p in edge.polyline],
                dtype=np.int32,
            )
            cv2.polylines(
                overlay, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA
            )
            if edge.directed:
                _draw_overlay_arrow_tip(
                    overlay,
                    (edge.polyline[-2].x, edge.polyline[-2].y),
                    (edge.polyline[-1].x, edge.polyline[-1].y),
                    color,
                )
            mid = edge.polyline[len(edge.polyline) // 2]
            tag = f"{edge.id}: {edge.source}->{edge.target}"
            cv2.putText(
                overlay,
                tag,
                (round(mid.x), max(14, round(mid.y) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    if out_path is not None:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), overlay)

    return overlay
