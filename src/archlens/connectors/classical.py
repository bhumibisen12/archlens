"""Classical skeleton-based arrow and connector extractor."""

import math
from pathlib import Path

import cv2
import numpy as np

from archlens.config import ConnectorsConfig
from archlens.connectors.base import ConnectorExtractor
from archlens.connectors.debug import draw_connector_overlay
from archlens.connectors.skeleton import (
    build_pixel_graph,
    prune_short_spurs,
    remove_small_components,
    simplify_polyline,
    skeletonize_ink,
    trace_polylines,
)
from archlens.graph import BBox, Edge, Node, Point
from archlens.ocr import TextBox
from archlens.preprocess import PreprocessedImages


def _mask_nodes_and_text(
    binary: np.ndarray,
    nodes: list[Node],
    text_boxes: list[TextBox] | None,
    node_dilation_px: int,
    text_dilation_px: int,
) -> np.ndarray:
    """Zero out dilated component node regions and OCR text regions from binary ink."""
    remaining = binary.copy()
    h, w = remaining.shape[:2]

    for node in nodes:
        x0 = max(0, node.bbox.x - node_dilation_px)
        y0 = max(0, node.bbox.y - node_dilation_px)
        x1 = min(w, node.bbox.x + node.bbox.w + node_dilation_px)
        y1 = min(h, node.bbox.y + node.bbox.h + node_dilation_px)
        remaining[y0:y1, x0:x1] = 0

    for tb in text_boxes or ():
        x0 = max(0, tb.bbox.x - text_dilation_px)
        y0 = max(0, tb.bbox.y - text_dilation_px)
        x1 = min(w, tb.bbox.x + tb.bbox.w + text_dilation_px)
        y1 = min(h, tb.bbox.y + tb.bbox.h + text_dilation_px)
        remaining[y0:y1, x0:x1] = 0

    return remaining


def _point_to_bbox_distance(pt: Point, bbox: BBox) -> float:
    """Return Euclidean distance from ``pt`` to the boundary/interior of ``bbox``."""
    dx = max(float(bbox.x) - pt.x, 0.0, pt.x - float(bbox.x + bbox.w))
    dy = max(float(bbox.y) - pt.y, 0.0, pt.y - float(bbox.y + bbox.h))
    return math.hypot(dx, dy)


def _snap_to_nearest_node(
    pt: Point,
    nodes: list[Node],
    max_distance_px: float,
) -> tuple[Node, float] | None:
    """Return the closest node within ``max_distance_px`` of ``pt``, or ``None``."""
    best_node: Node | None = None
    best_dist = float("inf")
    for node in nodes:
        dist = _point_to_bbox_distance(pt, node.bbox)
        if dist <= max_distance_px and dist < best_dist:
            best_dist = dist
            best_node = node
    if best_node is None:
        return None
    return best_node, best_dist


def _polyline_length(points: list[Point]) -> float:
    """Compute total Euclidean arc length of a polyline."""
    return sum(
        math.hypot(points[i + 1].x - points[i].x, points[i + 1].y - points[i].y)
        for i in range(len(points) - 1)
    )


def _probe_corridor(
    ink_mask: np.ndarray,
    origin: Point,
    ux: float,
    uy: float,
    start_offset: float,
    end_offset: float,
    half_width: float,
) -> tuple[int, float]:
    """Measure ink pixel count and max distance-transform radius inside an oriented corridor."""
    h, w = ink_mask.shape[:2]
    px, py = -uy, ux
    quad = np.array(
        [
            [
                origin.x + ux * start_offset + px * half_width,
                origin.y + uy * start_offset + py * half_width,
            ],
            [
                origin.x + ux * start_offset - px * half_width,
                origin.y + uy * start_offset - py * half_width,
            ],
            [
                origin.x + ux * end_offset - px * half_width,
                origin.y + uy * end_offset - py * half_width,
            ],
            [
                origin.x + ux * end_offset + px * half_width,
                origin.y + uy * end_offset + py * half_width,
            ],
        ],
        dtype=np.float32,
    )
    x0 = max(0, int(np.floor(quad[:, 0].min())) - 1)
    y0 = max(0, int(np.floor(quad[:, 1].min())) - 1)
    x1 = min(w, int(np.ceil(quad[:, 0].max())) + 1)
    y1 = min(h, int(np.ceil(quad[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0, 0.0

    corridor = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    local_quad = np.round(quad - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
    cv2.fillConvexPoly(corridor, local_quad, 255)

    roi_ink = ink_mask[y0:y1, x0:x1]
    masked_ink = cv2.bitwise_and(roi_ink, corridor)
    ink_count = int(np.count_nonzero(masked_ink))
    if ink_count == 0:
        return 0, 0.0

    dist = cv2.distanceTransform(masked_ink, cv2.DIST_L2, 3)
    return ink_count, float(dist.max())


def _probe_arrowhead_strength(
    ink_mask: np.ndarray,
    tip: Point,
    inward: Point,
    probe_len_px: int,
    probe_half_w_px: int,
) -> tuple[int, float]:
    """Measure arrowhead flare near ``tip`` compared against the trailing shaft."""
    dx = inward.x - tip.x
    dy = inward.y - tip.y
    seg_len = math.hypot(dx, dy)
    if seg_len <= 1e-6:
        return 0, 0.0
    ux, uy = dx / seg_len, dy / seg_len

    tip_ink, tip_radius = _probe_corridor(
        ink_mask,
        origin=tip,
        ux=ux,
        uy=uy,
        start_offset=-10.0,
        end_offset=float(probe_len_px) - 4.0,
        half_width=float(probe_half_w_px),
    )
    if tip_ink == 0:
        return 0, 0.0

    _, shaft_radius = _probe_corridor(
        ink_mask,
        origin=tip,
        ux=ux,
        uy=uy,
        start_offset=float(probe_len_px),
        end_offset=float(probe_len_px) + 18.0,
        half_width=float(probe_half_w_px),
    )
    flare = tip_radius / max(1.0, shaft_radius)
    score = float(tip_ink) * (1.0 + 0.5 * tip_radius) * min(2.5, max(0.75, flare))
    return tip_ink, score


def _detect_direction(
    ink_mask: np.ndarray,
    polyline: list[Point],
    cfg: ConnectorsConfig,
) -> tuple[bool, bool, float]:
    """Determine arrowhead direction along ``polyline``.

    Returns:
        ``(should_reverse, directed, direction_confidence)``
    """
    if len(polyline) < 2:
        return False, False, 0.5

    start_ink, start_score = _probe_arrowhead_strength(
        ink_mask,
        tip=polyline[0],
        inward=polyline[1],
        probe_len_px=cfg.arrowhead_probe_length_px,
        probe_half_w_px=cfg.arrowhead_probe_half_width_px,
    )
    end_ink, end_score = _probe_arrowhead_strength(
        ink_mask,
        tip=polyline[-1],
        inward=polyline[-2],
        probe_len_px=cfg.arrowhead_probe_length_px,
        probe_half_w_px=cfg.arrowhead_probe_half_width_px,
    )

    if (
        end_ink >= cfg.min_arrowhead_ink_pixels
        and end_score >= max(start_score, 1.0) * cfg.arrowhead_dominance_ratio
    ):
        ratio = end_score / max(start_score, 1.0)
        conf = max(0.6, min(0.96, 0.55 + 0.15 * min(ratio, 2.7)))
        return False, True, conf

    if (
        start_ink >= cfg.min_arrowhead_ink_pixels
        and start_score >= max(end_score, 1.0) * cfg.arrowhead_dominance_ratio
    ):
        ratio = start_score / max(end_score, 1.0)
        conf = max(0.6, min(0.96, 0.55 + 0.15 * min(ratio, 2.7)))
        return True, True, conf

    return False, False, 0.55


class ClassicalConnectorExtractor(ConnectorExtractor):
    """Skeleton + pixel-graph connector extractor with arrowhead direction detection."""

    def __init__(
        self,
        config: ConnectorsConfig | None = None,
        debug_overlay_path: Path | None = None,
    ) -> None:
        """Initialize the extractor with optional configuration and debug overlay path."""
        self.config = config or ConnectorsConfig()
        self.debug_overlay_path = debug_overlay_path

    def extract(
        self,
        preprocessed: PreprocessedImages,
        nodes: list[Node],
        text_boxes: list[TextBox] | None = None,
        debug_overlay_path: Path | None = None,
    ) -> list[Edge]:
        """Extract directed/undirected edges connecting ``nodes``."""
        cfg = self.config
        if len(nodes) < 2:
            return []

        remaining = _mask_nodes_and_text(
            binary=preprocessed.binary,
            nodes=nodes,
            text_boxes=text_boxes,
            node_dilation_px=cfg.node_mask_dilation_px,
            text_dilation_px=cfg.text_mask_dilation_px,
        )
        cleaned = remove_small_components(remaining, cfg.min_polyline_length_px)
        skeleton = skeletonize_ink(cleaned)
        adj = build_pixel_graph(skeleton)
        prune_short_spurs(adj, cfg.spur_prune_length_px)
        raw_polylines = trace_polylines(
            adj,
            max_crossing_angle_deg=cfg.max_crossing_angle_deg,
        )

        edges: list[Edge] = []
        edge_counter = 1

        for raw_pts in raw_polylines:
            simplified = simplify_polyline(raw_pts, epsilon_px=cfg.simplify_epsilon_px)
            if _polyline_length(simplified) < cfg.min_polyline_length_px:
                continue

            snap_start = _snap_to_nearest_node(simplified[0], nodes, cfg.snap_distance_px)
            snap_end = _snap_to_nearest_node(simplified[-1], nodes, cfg.snap_distance_px)
            if snap_start is None or snap_end is None:
                continue

            start_node, start_dist = snap_start
            end_node, end_dist = snap_end
            if start_node.id == end_node.id:
                continue

            should_reverse, directed, dir_conf = _detect_direction(remaining, simplified, cfg)
            if should_reverse:
                simplified = list(reversed(simplified))
                src_node, dst_node = end_node, start_node
            else:
                src_node, dst_node = start_node, end_node

            snap_penalty = (start_dist + end_dist) / max(1.0, 2.0 * cfg.snap_distance_px)
            conf = round(max(0.45, min(0.98, dir_conf * (1.0 - 0.25 * snap_penalty))), 4)

            edges.append(
                Edge(
                    id=f"e{edge_counter}",
                    source=src_node.id,
                    target=dst_node.id,
                    directed=directed,
                    label=None,
                    polyline=simplified,
                    confidence=conf,
                )
            )
            edge_counter += 1

        overlay_target = debug_overlay_path or self.debug_overlay_path
        if overlay_target is not None:
            draw_connector_overlay(preprocessed.color, nodes, edges, out_path=overlay_target)

        return edges
