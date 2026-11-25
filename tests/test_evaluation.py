"""Unit and CLI tests for src/archlens/evaluation/ and scripts/evaluate.py."""

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from archlens.evaluation import (
    aggregate_metrics,
    compute_image_metrics,
    evaluate_dataset,
    match_nodes_hungarian,
    normalized_edit_similarity,
)
from archlens.graph import BBox, DiagramGraph, Edge, Node, load_graph_json
from archlens.ocr import OCREngine, TesseractEngine, TextBox
from archlens.ocr.tesseract import resolve_tesseract_cmd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate.py"


def _load_evaluate_module() -> ModuleType:
    """Import scripts/evaluate.py cleanly."""
    spec = importlib.util.spec_from_file_location("evaluate_script", EVAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_script"] = module
    spec.loader.exec_module(module)
    return module


class _EmptyOCREngine(OCREngine):
    """Stub OCR engine for fast deterministic evaluation tests."""

    def read(self, image: np.ndarray) -> list[TextBox]:
        assert image.size > 0
        return []


def test_normalized_edit_similarity() -> None:
    """Normalized edit similarity handles exact, empty, and partial edits."""
    assert normalized_edit_similarity("API Gateway", "  api   gateway ") == pytest.approx(1.0)
    assert normalized_edit_similarity("", "") == pytest.approx(1.0)
    sim = normalized_edit_similarity("AP1 Gateway", "API Gateway")
    assert 0.85 < sim < 1.0
    assert normalized_edit_similarity("abc", "xyz") == pytest.approx(0.0)


def test_hungarian_matching_and_image_metrics() -> None:
    """Hungarian matcher enforces IoU >= 0.5 and separates directed vs undirected F1."""
    gt_nodes = [
        Node(id="g1", type="box", bbox=BBox(x=10, y=10, w=100, h=60), label="API"),
        Node(id="g2", type="database", bbox=BBox(x=200, y=10, w=100, h=60), label="Users DB"),
    ]
    gt_graph = DiagramGraph(
        nodes=gt_nodes,
        edges=[Edge(id="ge1", source="g1", target="g2", directed=True)],
    )

    # Exact match graph
    pred_exact = DiagramGraph(
        nodes=[
            Node(id="p1", type="box", bbox=BBox(x=12, y=12, w=98, h=58), label="API"),
            Node(id="p2", type="database", bbox=BBox(x=202, y=11, w=98, h=58), label="Users DB"),
        ],
        edges=[Edge(id="pe1", source="p1", target="p2", directed=True)],
    )
    matched, un_gt, un_pred = match_nodes_hungarian(gt_nodes, pred_exact.nodes, iou_threshold=0.5)
    assert len(matched) == 2
    assert un_gt == []
    assert un_pred == []

    m_exact = compute_image_metrics("exact.png", gt_graph, pred_exact, timings_ms={"total": 10.0})
    assert m_exact.node_f1 == pytest.approx(1.0)
    assert m_exact.directed_edge_f1 == pytest.approx(1.0)
    assert m_exact.undirected_edge_f1 == pytest.approx(1.0)
    assert m_exact.exact_graph_match is True

    # Reversed edge direction: directed F1 drops to 0.0 while undirected F1 remains 1.0.
    pred_flipped = DiagramGraph(
        nodes=pred_exact.nodes,
        edges=[Edge(id="pe1", source="p2", target="p1", directed=True)],
    )
    m_flipped = compute_image_metrics("flipped.png", gt_graph, pred_flipped)
    assert m_flipped.directed_edge_f1 == pytest.approx(0.0)
    assert m_flipped.undirected_edge_f1 == pytest.approx(1.0)
    assert m_flipped.exact_graph_match is False

    # Box with IoU < 0.5 does not match.
    far_node = Node(id="far", type="box", bbox=BBox(x=400, y=400, w=50, h=50))
    m_none, un_gt2, un_p2 = match_nodes_hungarian(gt_nodes, [far_node], iou_threshold=0.5)
    assert m_none == []
    assert un_gt2 == [0, 1]
    assert un_p2 == [0]

    agg = aggregate_metrics([m_exact, m_flipped])
    assert agg.num_images == 2
    assert agg.exact_graph_match_rate == pytest.approx(0.5)


def test_evaluate_dataset_writes_csv_report_and_worst_overlays(tmp_path: Path) -> None:
    """evaluate_dataset generates metrics.csv, report.md, and worst_overlays/."""
    out_dir = tmp_path / "results"
    per_image, summary = evaluate_dataset(
        data_dir=REPO_ROOT / "sample_images",
        out_dir=out_dir,
        ocr_engine=_EmptyOCREngine(),
        worst_k=2,
    )

    assert len(per_image) == 3
    assert summary.num_images == 3
    assert summary.node_recall > 0.0

    csv_path = out_dir / "metrics.csv"
    report_path = out_dir / "report.md"
    assert csv_path.is_file()
    assert report_path.is_file()

    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3

    overlays = sorted((out_dir / "worst_overlays").glob("*.png"))
    assert len(overlays) == 2

    report_text = report_path.read_text(encoding="utf-8")
    assert "Node F1" in report_text
    assert "Exact Graph Match Rate" in report_text


def test_evaluate_cli_ablation_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """scripts/evaluate.py supports --ocr, --no-perspective, and --no-shadow-removal."""
    eval_mod = _load_evaluate_module()
    if resolve_tesseract_cmd() is None:
        sample_json = sorted((REPO_ROOT / "sample_images").glob("*.json"))[0]
        gt = load_graph_json(sample_json)
        monkeypatch.setattr(
            TesseractEngine,
            "read",
            lambda _self, _img: [
                TextBox(text=n.label, bbox=n.bbox, conf=0.95) for n in gt.nodes if n.label
            ],
        )

    out_dir = tmp_path / "ablation_out"
    rc = eval_mod.main(
        [
            "--data",
            str(REPO_ROOT / "sample_images"),
            "--out",
            str(out_dir),
            "--ocr",
            "tesseract",
            "--no-perspective",
            "--no-shadow-removal",
            "--worst-k",
            "1",
        ]
    )
    assert rc == 0

    report_text = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Perspective correction**: `False`" in report_text
    assert "Shadow removal**: `False`" in report_text
    assert len(list((out_dir / "worst_overlays").glob("*.png"))) == 1

