"""Classical OpenCV contour-based component detector."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from archlens.config import DetectConfig
from archlens.detect.base import ComponentDetector
from archlens.detect.debug import draw_detection_overlay
from archlens.graph import BBox, Node, NodeType
from archlens.preprocess import PreprocessedImages


@dataclass(frozen=True)
class _Candidate:
    """Internal contour candidate prior to deduplication and ID assignment."""

    bbox: BBox
    node_type: NodeType
    confidence: float
    area: float
    is_hole: bool = False
    has_interior_ink: bool = True
    parent_area: float = 0.0


def _containment_ratio(inner: BBox, outer: BBox) -> float:
    """Return fraction of ``inner`` area overlapped by ``outer``."""
    ix0 = max(inner.x, outer.x)
    iy0 = max(inner.y, outer.y)
    ix1 = min(inner.x + inner.w, outer.x + outer.w)
    iy1 = min(inner.y + inner.h, outer.y + outer.h)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = float((ix1 - ix0) * (iy1 - iy0))
    return inter / float(max(1, inner.area))


def _iou(a: BBox, b: BBox) -> float:
    """Return Intersection-over-Union between two bounding boxes."""
    ix0 = max(a.x, b.x)
    iy0 = max(a.y, b.y)
    ix1 = min(a.x + a.w, b.x + b.w)
    iy1 = min(a.y + a.h, b.y + b.h)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = float((ix1 - ix0) * (iy1 - iy0))
    union = float(a.area + b.area) - inter
    return inter / max(1.0, union)


def _max_interior_radius(contour: np.ndarray, bbox: BBox) -> float:
    """Measure maximum inscribed radius of a filled contour using distance transform."""
    shifted = contour - np.array([[[bbox.x, bbox.y]]], dtype=contour.dtype)
    mask = np.zeros((bbox.h, bbox.w), dtype=np.uint8)
    cv2.drawContours(mask, [shifted], -1, 255, thickness=cv2.FILLED)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    return float(dist.max())


def _ellipse_fit_residual(contour: np.ndarray, bbox: BBox) -> float:
    """Compute shape residual ``1 - IoU`` between filled contour and ``cv2.fitEllipse``."""
    if len(contour) < 5:
        return 1.0
    try:
        (cx, cy), (ma, mi), angle = cv2.fitEllipse(contour)
    except cv2.error:
        return 1.0

    if ma <= 1.0 or mi <= 1.0 or not np.isfinite([cx, cy, ma, mi, angle]).all():
        return 1.0

    contour_mask = np.zeros((bbox.h, bbox.w), dtype=np.uint8)
    ellipse_mask = np.zeros((bbox.h, bbox.w), dtype=np.uint8)
    shifted = contour - np.array([[[bbox.x, bbox.y]]], dtype=contour.dtype)
    cv2.drawContours(contour_mask, [shifted], -1, 255, thickness=cv2.FILLED)

    local_center = (cx - bbox.x, cy - bbox.y)
    axes = (max(1, int(round(ma / 2.0))), max(1, int(round(mi / 2.0))))
    cv2.ellipse(
        ellipse_mask,
        (int(round(local_center[0])), int(round(local_center[1]))),
        axes,
        angle,
        0,
        360,
        255,
        thickness=cv2.FILLED,
    )

    inter = np.count_nonzero((contour_mask > 0) & (ellipse_mask > 0))
    union = np.count_nonzero((contour_mask > 0) | (ellipse_mask > 0))
    if union == 0:
        return 1.0
    return 1.0 - (float(inter) / float(union))


def _has_cylinder_top_seam(binary: np.ndarray, bbox: BBox) -> bool:
    """Heuristic: check for a horizontal curved ink band in the upper third (cylinder rim)."""
    top_y0 = bbox.y + max(2, int(bbox.h * 0.14))
    top_y1 = bbox.y + max(5, int(bbox.h * 0.42))
    x0 = bbox.x + max(2, int(bbox.w * 0.15))
    x1 = bbox.x + max(4, int(bbox.w * 0.85))
    if top_y1 <= top_y0 or x1 <= x0:
        return False

    band = binary[top_y0:top_y1, x0:x1]
    if band.size == 0:
        return False
    # Lower rim of top ellipse curves across rows, so measure horizontal column coverage.
    col_coverage = float(np.mean(np.any(band > 0, axis=0)))
    return col_coverage >= 0.65


def _classify_contour(
    contour: np.ndarray,
    bbox: BBox,
    area: float,
    solidity: float,
    extent: float,
    binary: np.ndarray,
    cfg: DetectConfig,
) -> tuple[NodeType, float]:
    """Classify a filtered contour into box, ellipse, database, or component."""
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, cfg.polygon_epsilon_ratio * perimeter, True)
    residual = _ellipse_fit_residual(contour, bbox)

    # Cylinder heuristic first: high extent/solidity + horizontal top-cap rim in upper band.
    if (
        solidity >= 0.80
        and extent >= 0.72
        and bbox.h >= 36
        and _has_cylinder_top_seam(binary, bbox)
        and (len(approx) >= 5 or residual > cfg.ellipse_max_residual)
    ):
        conf = max(0.55, min(0.92, 0.5 * solidity + 0.4 * extent))
        return "database", conf

    if residual <= cfg.ellipse_max_residual and len(approx) >= 5:
        conf = max(0.55, min(0.98, 1.0 - residual))
        return "ellipse", conf

    if len(approx) == 4 and cv2.isContourConvex(approx) and extent >= cfg.rect_min_extent:
        conf = max(0.6, min(0.98, extent))
        return "box", conf

    if 4 <= len(approx) <= 6 and solidity >= 0.85 and extent >= cfg.rect_min_extent:
        conf = max(0.55, min(0.88, extent * 0.95))
        return "box", conf

    conf = max(0.4, min(0.85, 0.5 * solidity + 0.35 * extent))
    return "component", conf


def _has_interior_ink(binary: np.ndarray, contour: np.ndarray, bbox: BBox) -> bool:
    """Return True if the eroded interior of ``contour`` contains ink pixels (e.g., text)."""
    shifted = contour - np.array([[[bbox.x, bbox.y]]], dtype=contour.dtype)
    mask = np.zeros((bbox.h, bbox.w), dtype=np.uint8)
    cv2.drawContours(mask, [shifted], -1, 255, thickness=cv2.FILLED)
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    interior = cv2.erode(mask, erode_k)
    interior_area = int(np.count_nonzero(interior))
    if interior_area == 0:
        return False
    roi_bin = binary[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]
    ink_pixels = int(np.count_nonzero((roi_bin > 0) & (interior > 0)))
    return (ink_pixels / float(interior_area)) >= 0.005


def _merge_cylinder_halves(candidates: list[_Candidate]) -> list[_Candidate]:
    """Merge vertically stacked cylinder top-cap and body holes into one database node."""
    ordered = sorted(candidates, key=lambda c: (c.bbox.y, c.bbox.x))
    used: set[int] = set()
    merged: list[_Candidate] = []

    for i, top in enumerate(ordered):
        if i in used:
            continue
        best_j = -1
        for j, body in enumerate(ordered):
            if j == i or j in used:
                continue
            if top.bbox.y >= body.bbox.y:
                continue
            if top.bbox.h > body.bbox.h * 0.55:
                continue
            x_overlap = min(top.bbox.x + top.bbox.w, body.bbox.x + body.bbox.w) - max(
                top.bbox.x, body.bbox.x
            )
            min_w = min(top.bbox.w, body.bbox.w)
            if min_w <= 0 or (x_overlap / float(min_w)) < 0.72:
                continue
            width_ratio = min_w / float(max(top.bbox.w, body.bbox.w))
            if width_ratio < 0.78:
                continue
            gap = body.bbox.y - (top.bbox.y + top.bbox.h)
            if -18 <= gap <= 18:
                best_j = j
                break

        if best_j >= 0:
            body = ordered[best_j]
            used.add(i)
            used.add(best_j)
            x0 = min(top.bbox.x, body.bbox.x)
            y0 = min(top.bbox.y, body.bbox.y)
            x1 = max(top.bbox.x + top.bbox.w, body.bbox.x + body.bbox.w)
            y1 = max(top.bbox.y + top.bbox.h, body.bbox.y + body.bbox.h)
            merged_bbox = BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)
            merged.append(
                _Candidate(
                    bbox=merged_bbox,
                    node_type="database",
                    confidence=round(max(top.confidence, body.confidence, 0.85), 4),
                    area=float(merged_bbox.area),
                    is_hole=body.is_hole,
                    has_interior_ink=top.has_interior_ink or body.has_interior_ink,
                )
            )
        else:
            used.add(i)
            merged.append(top)

    return merged


def _deduplicate_candidates(
    candidates: list[_Candidate],
    containment_threshold: float,
) -> list[_Candidate]:
    """Drop composite parents, empty arrow-crossing gaps, and nested duplicates."""
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda c: c.area, reverse=True)

    # Pass 1: Remove composite parent contours that wrap multiple distinct components.
    non_composite: list[_Candidate] = []
    for idx, parent in enumerate(ordered):
        children = [
            child
            for j, child in enumerate(ordered)
            if j != idx
            and child.area < parent.area * 0.85
            and _containment_ratio(child.bbox, parent.bbox) >= containment_threshold
        ]
        distinct_children: list[_Candidate] = []
        for child in children:
            if not any(_iou(child.bbox, kept.bbox) > 0.3 for kept in distinct_children):
                distinct_children.append(child)

        if len(distinct_children) >= 2:
            continue
        non_composite.append(parent)

    # Pass 2: Merge cylinder top-ellipse + body holes before filtering empty gaps.
    merged = _merge_cylinder_halves(non_composite)

    # Pass 3: Reject empty white-paper pockets trapped inside a large composite arrow network.
    filtered = [
        cand
        for cand in merged
        if not (
            cand.is_hole
            and not cand.has_interior_ink
            and cand.parent_area > cand.area * 2.5
        )
    ]
    filtered.sort(key=lambda c: c.area, reverse=True)

    # Pass 4: Remove nested duplicates (e.g., inner wall vs outer wall of the same shape).
    kept: list[_Candidate] = []
    for cand in filtered:
        is_nested = any(
            _containment_ratio(cand.bbox, existing.bbox) >= containment_threshold
            for existing in kept
        )
        if not is_nested:
            kept.append(cand)

    return kept


class ClassicalDetector(ComponentDetector):
    """OpenCV contour-based detector for boxes, ellipses, cylinders, and components."""

    def __init__(
        self,
        config: DetectConfig | None = None,
        debug_overlay_path: Path | None = None,
    ) -> None:
        """Initialize the detector with optional configuration and overlay path."""
        self.config = config or DetectConfig()
        self.debug_overlay_path = debug_overlay_path

    def detect(
        self,
        preprocessed: PreprocessedImages,
        debug_overlay_path: Path | None = None,
    ) -> list[Node]:
        """Detect components from preprocessed binary ink masks."""
        cfg = self.config
        binary = preprocessed.binary
        img_h, img_w = binary.shape[:2]
        img_area = float(img_h * img_w)
        max_area = img_area * cfg.max_component_area_ratio

        k = (
            cfg.morph_close_kernel
            if cfg.morph_close_kernel % 2 == 1
            else cfg.morph_close_kernel + 1
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hier = hierarchy[0] if hierarchy is not None else np.empty((0, 4), dtype=np.int32)
        contour_areas = [float(cv2.contourArea(c)) for c in contours]

        # Precompute contour depths in RETR_TREE (odd depth => white-paper hole).
        depths: list[int] = [0] * len(contours)
        for idx in range(len(contours)):
            d = 0
            p = int(hier[idx][3]) if idx < len(hier) else -1
            while p >= 0:
                d += 1
                p = int(hier[p][3])
            depths[idx] = d

        candidates: list[_Candidate] = []
        for idx, contour in enumerate(contours):
            area = contour_areas[idx]
            if area < cfg.min_component_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < cfg.min_width or h < cfg.min_height:
                continue

            aspect = max(w / float(h), h / float(w))
            if aspect > cfg.max_aspect_ratio:
                continue

            bbox = BBox(x=x, y=y, w=w, h=h)
            extent = area / float(bbox.area)
            if extent < cfg.min_extent:
                continue

            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / max(1.0, hull_area)
            if solidity < cfg.min_solidity:
                continue

            if _max_interior_radius(contour, bbox) < cfg.min_interior_radius:
                continue

            node_type, confidence = _classify_contour(
                contour=contour,
                bbox=bbox,
                area=area,
                solidity=solidity,
                extent=extent,
                binary=binary,
                cfg=cfg,
            )
            parent_idx = int(hier[idx][3]) if idx < len(hier) else -1
            parent_area = contour_areas[parent_idx] if parent_idx >= 0 else 0.0
            is_hole = depths[idx] % 2 == 1
            interior_ink = _has_interior_ink(binary, contour, bbox) if is_hole else True
            candidates.append(
                _Candidate(
                    bbox=bbox,
                    node_type=node_type,
                    confidence=round(confidence, 4),
                    area=area,
                    is_hole=is_hole,
                    has_interior_ink=interior_ink,
                    parent_area=parent_area,
                )
            )

        deduped = _deduplicate_candidates(candidates, cfg.nested_containment_threshold)
        deduped.sort(key=lambda c: (c.bbox.y, c.bbox.x))

        nodes = [
            Node(
                id=f"n{idx}",
                type=cand.node_type,
                bbox=cand.bbox,
                label="",
                confidence=cand.confidence,
            )
            for idx, cand in enumerate(deduped, start=1)
        ]

        overlay_target = debug_overlay_path or self.debug_overlay_path
        if overlay_target is not None:
            draw_detection_overlay(preprocessed.color, nodes, out_path=overlay_target)

        return nodes
