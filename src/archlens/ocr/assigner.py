"""Assign OCR TextBoxes to component nodes and connector edge-label candidates."""

import math
from collections import defaultdict

from pydantic import BaseModel, Field

from archlens.graph import BBox, Edge, Node, Point
from archlens.ocr.base import TextBox


class EdgeLabelCandidate(BaseModel):
    """Remaining OCR text box that lies outside nodes and near a connector."""

    text: str
    bbox: BBox
    conf: float = Field(default=1.0, ge=0.0, le=1.0)
    edge_id: str | None = None
    distance_px: float | None = None


class TextAssignmentResult(BaseModel):
    """Result of assigning OCR text regions to diagram nodes and edges."""

    nodes: list[Node]
    edge_label_candidates: list[EdgeLabelCandidate] = Field(default_factory=list)
    unassigned_text: list[TextBox] = Field(default_factory=list)


def _overlap_ratio_in_node(text_bbox: BBox, node_bbox: BBox, margin_px: int) -> float:
    """Return fraction of ``text_bbox`` area covered by expanded ``node_bbox``."""
    nx0 = max(0, node_bbox.x - margin_px)
    ny0 = max(0, node_bbox.y - margin_px)
    nx1 = node_bbox.x + node_bbox.w + margin_px
    ny1 = node_bbox.y + node_bbox.h + margin_px

    ix0 = max(text_bbox.x, nx0)
    iy0 = max(text_bbox.y, ny0)
    ix1 = min(text_bbox.x + text_bbox.w, nx1)
    iy1 = min(text_bbox.y + text_bbox.h, ny1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return float((ix1 - ix0) * (iy1 - iy0)) / float(max(1, text_bbox.area))


def _point_to_segment_distance(px: float, py: float, a: Point, b: Point) -> float:
    """Compute Euclidean distance from point ``(px, py)`` to segment ``a-b``."""
    dx = b.x - a.x
    dy = b.y - a.y
    len_sq = dx * dx + dy * dy
    if len_sq <= 1e-9:
        return math.hypot(px - a.x, py - a.y)
    t = max(0.0, min(1.0, ((px - a.x) * dx + (py - a.y) * dy) / len_sq))
    proj_x = a.x + t * dx
    proj_y = a.y + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _point_to_polyline_distance(px: float, py: float, polyline: list[Point]) -> float:
    """Compute minimum distance from ``(px, py)`` to a polyline."""
    if not polyline:
        return float("inf")
    if len(polyline) == 1:
        return math.hypot(px - polyline[0].x, py - polyline[0].y)
    return min(
        _point_to_segment_distance(px, py, polyline[i], polyline[i + 1])
        for i in range(len(polyline) - 1)
    )


def _find_containing_node(
    tb: TextBox,
    nodes: list[Node],
    inside_margin_px: int,
) -> Node | None:
    """Pick the node containing ``tb`` (center inside expanded bbox or >=50% overlap)."""
    cx, cy = tb.bbox.center
    best_node: Node | None = None
    best_score = -1.0

    for node in nodes:
        nb = node.bbox
        center_inside = (
            (nb.x - inside_margin_px) <= cx <= (nb.x + nb.w + inside_margin_px)
            and (nb.y - inside_margin_px) <= cy <= (nb.y + nb.h + inside_margin_px)
        )
        overlap = _overlap_ratio_in_node(tb.bbox, nb, inside_margin_px)
        if not center_inside and overlap < 0.5:
            continue

        ncx, ncy = nb.center
        dist_penalty = math.hypot(cx - ncx, cy - ncy) * 1e-4
        score = overlap - dist_penalty
        if score > best_score:
            best_score = score
            best_node = node

    return best_node


def assign_text_to_diagram(
    text_boxes: list[TextBox],
    nodes: list[Node],
    edges: list[Edge] | None = None,
    inside_margin_px: int = 6,
    max_edge_distance_px: float = 80.0,
) -> TextAssignmentResult:
    """Assign OCR text boxes inside node bounds as labels and others as edge candidates.

    Args:
        text_boxes: Recognized :class:`TextBox` regions from an :class:`OCREngine`.
        nodes: Detected component :class:`Node` instances.
        edges: Optional detected :class:`Edge` instances with polylines. When ``None``,
            all non-node text boxes are returned as unlinked ``EdgeLabelCandidate`` items.
        inside_margin_px: Tolerance around node bounding boxes for text inclusion.
        max_edge_distance_px: Maximum center-to-polyline distance when matching to ``edges``.

    Returns:
        :class:`TextAssignmentResult` with labeled nodes and edge-label candidates.
    """
    node_buckets: dict[str, list[TextBox]] = defaultdict(list)
    remaining: list[TextBox] = []

    for tb in text_boxes:
        if not tb.text.strip():
            continue
        owner = _find_containing_node(tb, nodes, inside_margin_px)
        if owner is not None:
            node_buckets[owner.id].append(tb)
        else:
            remaining.append(tb)

    updated_nodes: list[Node] = []
    for node in nodes:
        bucket = node_buckets.get(node.id, [])
        if bucket:
            bucket.sort(key=lambda item: (item.bbox.y, item.bbox.x))
            joined = " ".join(item.text for item in bucket)
            updated_nodes.append(node.model_copy(update={"label": joined}))
        else:
            updated_nodes.append(node)

    candidates: list[EdgeLabelCandidate] = []
    unassigned: list[TextBox] = []

    for tb in remaining:
        if edges is None:
            candidates.append(EdgeLabelCandidate(text=tb.text, bbox=tb.bbox, conf=tb.conf))
            continue

        cx, cy = tb.bbox.center
        best_edge_id: str | None = None
        best_dist = float("inf")
        for edge in edges:
            dist = _point_to_polyline_distance(cx, cy, edge.polyline)
            if dist < best_dist:
                best_dist = dist
                best_edge_id = edge.id

        if best_edge_id is not None and best_dist <= max_edge_distance_px:
            candidates.append(
                EdgeLabelCandidate(
                    text=tb.text,
                    bbox=tb.bbox,
                    conf=tb.conf,
                    edge_id=best_edge_id,
                    distance_px=round(best_dist, 2),
                )
            )
        else:
            unassigned.append(tb)

    return TextAssignmentResult(
        nodes=updated_nodes,
        edge_label_candidates=candidates,
        unassigned_text=unassigned,
    )
