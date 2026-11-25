"""Unit and headless Streamlit AppTest tests for app/streamlit_app.py."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from archlens.graph import BBox, DiagramGraph, Edge, Node, load_graph_json
from archlens.ocr import TesseractEngine, TextBox
from archlens.ocr import tesseract as tesseract_mod
from archlens.pipeline import PipelineResult
from archlens.review import ReviewEngine

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "app" / "streamlit_app.py"


def _load_app_module() -> ModuleType:
    """Import app/streamlit_app.py cleanly."""
    spec = importlib.util.spec_from_file_location("archlens_streamlit_app", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["archlens_streamlit_app"] = module
    spec.loader.exec_module(module)
    return module


def test_app_helpers_and_friendly_errors() -> None:
    """App helpers format findings, overlays, and missing-OCR errors cleanly."""
    app = _load_app_module()
    samples = app.list_sample_images()
    assert len(samples) >= 3

    tess_msg = app.format_friendly_ocr_error(RuntimeError("binary missing"), "tesseract")
    assert "Tesseract OCR" in tess_msg
    assert "PATH" in tess_msg

    paddle_msg = app.format_friendly_ocr_error(ImportError("no paddleocr"), "paddle")
    assert "PaddleOCR" in paddle_msg
    assert 'pip install -e ".[paddle]"' in paddle_msg

    graph = DiagramGraph(
        nodes=[
            Node(id="u", type="ellipse", bbox=BBox(x=10, y=10, w=80, h=40), label="User"),
            Node(id="db", type="database", bbox=BBox(x=200, y=10, w=80, h=50), label="MySQL DB"),
        ],
        edges=[Edge(id="e1", source="u", target="db", directed=True)],
    )
    report = ReviewEngine().review(graph)
    df = app.build_findings_dataframe(report)
    assert not df.empty
    assert list(df.columns) == ["Severity", "Rule", "Nodes", "Message"]
    styler = app.style_findings_table(df)
    assert styler is not None

    fake_result = PipelineResult(
        image_path="dummy.png",
        graph=graph,
        nodes_detected=2,
        edges_detected=1,
    )
    overlay_rgb = app.build_combined_overlay(
        np.full((120, 320, 3), 255, dtype=np.uint8),
        fake_result,
    )
    assert overlay_rgb.shape == (120, 320, 3)


def test_streamlit_app_renders_with_mocked_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streamlit UI runs headless on a sample image and renders graph, findings, and timings."""
    sample_json = sorted((REPO_ROOT / "sample_images").glob("*.json"))[0]
    gt_graph = load_graph_json(sample_json)

    monkeypatch.setattr(
        TesseractEngine,
        "read",
        lambda _self, _img: [
            TextBox(text=node.label, bbox=node.bbox, conf=0.95)
            for node in gt_graph.nodes
            if node.label
        ],
    )

    at = AppTest.from_file(str(APP_PATH), default_timeout=20)
    at.run()

    assert not at.exception
    assert not at.error
    assert len(at.metric) >= 5
    assert len(at.dataframe) == 1


def test_streamlit_app_shows_friendly_error_when_tesseract_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Tesseract binary shows a friendly st.error rather than an exception."""
    monkeypatch.setattr(tesseract_mod, "resolve_tesseract_cmd", lambda _cmd=None: None)

    at = AppTest.from_file(str(APP_PATH), default_timeout=20)
    at.run()

    assert not at.exception
    assert len(at.error) == 1
    assert "Tesseract OCR" in at.error[0].value

