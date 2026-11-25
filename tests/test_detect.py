"""Unit and recall tests for src/archlens/detect/."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from archlens.config import DetectConfig, load_config
from archlens.detect import ClassicalDetector, ComponentDetector
from archlens.graph import BBox, Node, load_graph_json
from archlens.preprocess import PreprocessedImages, preprocess_array, preprocess_image

REPO_ROOT = Path(__file__).resolve().parents[1]


def _bbox_iou(a: BBox, b: BBox) -> float:
    """Compute Intersection-over-Union between two bounding boxes."""
    ix0 = max(a.x, b.x)
    iy0 = max(a.y, b.y)
    ix1 = min(a.x + a.w, b.x + b.w)
    iy1 = min(a.y + a.h, b.y + b.h)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = float((ix1 - ix0) * (iy1 - iy0))
    return inter / float(a.area + b.area - inter)


def test_component_detector_is_abstract() -> None:
    """ComponentDetector cannot be instantiated without implementing detect()."""
    with pytest.raises(TypeError):
        ComponentDetector()  # type: ignore[abstract]


def test_classical_detector_classifies_shapes_and_ignores_arrows() -> None:
    """ClassicalDetector classifies box, ellipse, database, component and drops arrows."""
    canvas = np.full((500, 820, 3), 250, dtype=np.uint8)
    ink = (25, 25, 25)

    # Box
    cv2.rectangle(canvas, (40, 50), (190, 140), ink, 3)
    cv2.putText(canvas, "API", (85, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ink, 2)
    # Ellipse
    cv2.ellipse(canvas, (350, 95), (80, 45), 0, 0, 360, ink, 3)
    cv2.putText(canvas, "User", (320, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ink, 2)
    # Cylinder (database)
    cv2.ellipse(canvas, (590, 68), (65, 14), 0, 0, 360, ink, 3)
    cv2.ellipse(canvas, (590, 148), (65, 14), 0, 0, 180, ink, 3)
    cv2.line(canvas, (525, 68), (525, 148), ink, 3)
    cv2.line(canvas, (655, 68), (655, 148), ink, 3)
    cv2.putText(canvas, "DB", (570, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ink, 2)
    # Unknown non-rectangular polygon -> "component" (concave polygon)
    poly = np.array(
        [[120, 260], [220, 290], [270, 250], [250, 380], [160, 360], [100, 330]],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [poly], isClosed=True, color=ink, thickness=3)

    # Thin arrow line that must be excluded
    cv2.line(canvas, (190, 95), (270, 95), ink, 3)
    cv2.fillPoly(
        canvas,
        [np.array([[270, 95], [256, 89], [256, 101]], dtype=np.int32)],
        ink,
    )

    preprocessed = preprocess_array(canvas)
    detector = ClassicalDetector()
    nodes = detector.detect(preprocessed)

    assert all(isinstance(n, Node) for n in nodes)
    types = {n.type for n in nodes}
    assert {"box", "ellipse", "database", "component"}.issubset(types)
    assert len(nodes) == 4


def test_nested_duplicate_removal() -> None:
    """Double-walled box produces a single deduplicated Node."""
    canvas = np.full((300, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (60, 60), (260, 180), (20, 20, 20), 3)
    cv2.rectangle(canvas, (70, 70), (250, 170), (20, 20, 20), 2)

    preprocessed = preprocess_array(canvas)
    nodes = ClassicalDetector().detect(preprocessed)

    assert len(nodes) == 1
    assert nodes[0].type == "box"


def test_debug_overlay_saved(tmp_path: Path) -> None:
    """Providing debug_overlay_path writes a rendered PNG overlay."""
    sample_png = sorted((REPO_ROOT / "sample_images").glob("*.png"))[0]
    preprocessed = preprocess_image(sample_png)
    assert isinstance(preprocessed, PreprocessedImages)

    overlay_path = tmp_path / "overlay" / "detections.png"
    nodes = ClassicalDetector().detect(preprocessed, debug_overlay_path=overlay_path)

    assert len(nodes) >= 1
    assert overlay_path.is_file()
    rendered = cv2.imread(str(overlay_path))
    assert rendered is not None
    assert rendered.shape == preprocessed.color.shape


def test_node_recall_on_sample_synthetic_images() -> None:
    """ClassicalDetector meets the measured recall threshold on sample_images.

    Baseline measured on sample_images (3 diagrams, 20 GT nodes): recall = 1.000 at IoU >= 0.4.
    Baseline measured on data/synthetic (20 diagrams, 147 GT nodes): recall = 0.966 at IoU >= 0.4.
    Test enforces a realistic >= 0.90 recall threshold at IoU >= 0.4.
    """
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    detector = ClassicalDetector(DetectConfig.model_validate(cfg.detect.model_dump()))

    sample_jsons = sorted((REPO_ROOT / "sample_images").glob("*.json"))
    assert len(sample_jsons) >= 3

    total_gt = 0
    matched_gt = 0
    for json_path in sample_jsons:
        gt_graph = load_graph_json(json_path)
        preprocessed = preprocess_image(json_path.with_suffix(".png"), cfg.preprocess)
        preds = detector.detect(preprocessed)

        total_gt += len(gt_graph.nodes)
        used: set[int] = set()
        for gt_node in gt_graph.nodes:
            best_idx = -1
            best_iou = 0.0
            for idx, pred in enumerate(preds):
                if idx in used:
                    continue
                score = _bbox_iou(gt_node.bbox, pred.bbox)
                if score > best_iou:
                    best_idx = idx
                    best_iou = score
            if best_iou >= 0.4 and best_idx >= 0:
                matched_gt += 1
                used.add(best_idx)

    recall = matched_gt / float(total_gt)
    assert recall >= 0.90
