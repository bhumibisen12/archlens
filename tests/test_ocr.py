"""Unit and conditional integration tests for src/archlens/ocr/."""

import builtins
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import pytesseract
import pytest

from archlens.config import OCRConfig
from archlens.graph import BBox, Edge, Node, Point
from archlens.ocr import (
    OCREngine,
    PaddleOCREngine,
    TesseractEngine,
    TextBox,
    assign_text_to_diagram,
    create_ocr_engine,
    normalize_text,
)
from archlens.ocr import tesseract as tesseract_mod

REPO_ROOT = Path(__file__).resolve().parents[1]


class _MockOCREngine(OCREngine):
    """Deterministic mock OCREngine for unit testing."""

    def __init__(self, boxes: list[TextBox]) -> None:
        self._boxes = boxes

    def read(self, image: np.ndarray) -> list[TextBox]:
        assert image.size > 0
        return list(self._boxes)


def test_ocr_engine_is_abstract() -> None:
    """OCREngine cannot be instantiated without implementing read()."""
    with pytest.raises(TypeError):
        OCREngine()  # type: ignore[abstract]


def test_normalize_text_configurable() -> None:
    """normalize_text strips whitespace and applies char/token replacements."""
    cfg = OCRConfig()
    raw = "  |nbound   AP1  —  R3dis   D8  "
    cleaned = normalize_text(
        raw,
        char_replacements=cfg.char_replacements,
        token_replacements=cfg.token_replacements,
    )
    assert cleaned == "Inbound API - Redis DB"
    assert normalize_text("   \n\t  ") == ""


def test_tesseract_engine_mocked_line_merging(monkeypatch: pytest.MonkeyPatch) -> None:
    """TesseractEngine merges same-line adjacent words and filters low confidence."""
    monkeypatch.setattr(
        tesseract_mod, "resolve_tesseract_cmd", lambda _cmd=None: "/usr/bin/tesseract"
    )
    fake_output = {
        "text": ["API", "Gateway", "noise", "Orders", "DB"],
        "conf": ["92", "89", "12", "95", "91"],
        "left": [100, 145, 20, 400, 470],
        "top": [50, 51, 10, 50, 50],
        "width": [38, 80, 30, 60, 30],
        "height": [20, 20, 15, 20, 20],
        "block_num": [1, 1, 2, 1, 1],
        "par_num": [1, 1, 1, 1, 1],
        "line_num": [1, 1, 1, 1, 1],
    }
    monkeypatch.setattr(pytesseract, "image_to_data", lambda *_args, **_kwargs: fake_output)

    engine = TesseractEngine(OCRConfig(min_confidence=0.4))
    img = np.full((200, 600, 3), 255, dtype=np.uint8)
    boxes = engine.read(img)

    assert len(boxes) == 2
    assert boxes[0].text == "API Gateway"
    assert boxes[0].bbox == BBox(x=100, y=50, w=125, h=21)
    assert boxes[1].text == "Orders DB"


def test_tesseract_engine_missing_binary_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing tesseract binary produces an actionable RuntimeError."""
    monkeypatch.setattr(tesseract_mod, "resolve_tesseract_cmd", lambda _cmd=None: None)
    engine = TesseractEngine()
    with pytest.raises(RuntimeError, match="Tesseract binary"):
        engine.read(np.zeros((50, 50), dtype=np.uint8))


def test_paddle_engine_missing_extra_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PaddleOCREngine raises an ImportError instructing how to install [paddle]."""
    monkeypatch.delitem(sys.modules, "paddleocr", raising=False)
    real_import = builtins.__import__

    def _block_paddle(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "paddleocr":
            raise ImportError("No module named 'paddleocr'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_paddle)

    with pytest.raises(ImportError, match=r'pip install -e "\.\[paddle\]"'):
        PaddleOCREngine()


def test_paddle_engine_requires_local_cache_and_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """PaddleOCREngine requires local cached model dirs and parses offline results."""
    fake_mod = ModuleType("paddleocr")
    captured_kwargs: dict[str, Any] = {}

    class _FakePaddleOCR:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        def ocr(self, _img: np.ndarray, cls: bool = False) -> list[Any]:
            assert cls is False
            return [
                [
                    [[[10, 20], [110, 20], [110, 45], [10, 45]], ("AP1 Gateway", 0.94)],
                    [[[10, 80], [60, 80], [60, 95], [10, 95]], ("low", 0.15)],
                ]
            ]

    fake_mod.PaddleOCR = _FakePaddleOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", fake_mod)

    # Missing model dir -> RuntimeError (no runtime downloads allowed).
    with pytest.raises(RuntimeError, match="paddle_model_dir"):
        PaddleOCREngine(OCRConfig(engine="paddle", paddle_model_dir=None))

    model_root = tmp_path / "paddle_models"
    (model_root / "det").mkdir(parents=True)
    (model_root / "rec").mkdir(parents=True)

    engine = create_ocr_engine(OCRConfig(engine="paddle", paddle_model_dir=str(model_root)))
    assert isinstance(engine, PaddleOCREngine)
    assert captured_kwargs["use_gpu"] is False
    assert captured_kwargs["det_model_dir"] == str(model_root / "det")
    assert captured_kwargs["rec_model_dir"] == str(model_root / "rec")

    boxes = engine.read(np.full((120, 200), 255, dtype=np.uint8))
    assert len(boxes) == 1
    assert boxes[0].text == "API Gateway"
    assert boxes[0].bbox == BBox(x=10, y=20, w=100, h=25)


def test_assign_text_to_diagram_nodes_and_edges() -> None:
    """Mock OCR output assigns interior text to nodes and nearby text to edges."""
    nodes = [
        Node(id="n1", type="box", bbox=BBox(x=20, y=50, w=140, h=80)),
        Node(id="n2", type="database", bbox=BBox(x=320, y=50, w=140, h=80)),
    ]
    edges = [
        Edge(
            id="e1",
            source="n1",
            target="n2",
            polyline=[Point(x=160, y=90), Point(x=320, y=90)],
        )
    ]

    mock_engine = _MockOCREngine(
        [
            TextBox(text="Payment", bbox=BBox(x=45, y=68, w=90, h=20), conf=0.95),
            TextBox(text="Service", bbox=BBox(x=48, y=92, w=84, h=20), conf=0.92),
            TextBox(text="Users DB", bbox=BBox(x=345, y=82, w=90, h=20), conf=0.90),
            TextBox(text="SQL", bbox=BBox(x=220, y=65, w=40, h=18), conf=0.88),
            TextBox(text="Watermark", bbox=BBox(x=10, y=400, w=100, h=20), conf=0.70),
        ]
    )
    boxes = mock_engine.read(np.zeros((450, 500), dtype=np.uint8))

    result = assign_text_to_diagram(boxes, nodes, edges=edges, max_edge_distance_px=50.0)
    assert result.nodes[0].label == "Payment Service"
    assert result.nodes[1].label == "Users DB"
    assert len(result.edge_label_candidates) == 1
    assert result.edge_label_candidates[0].text == "SQL"
    assert result.edge_label_candidates[0].edge_id == "e1"
    assert len(result.unassigned_text) == 1
    assert result.unassigned_text[0].text == "Watermark"

    # Without edges provided, all non-node texts become unlinked edge-label candidates.
    pre_conn = assign_text_to_diagram(boxes, nodes, edges=None)
    assert len(pre_conn.edge_label_candidates) == 2
    assert all(cand.edge_id is None for cand in pre_conn.edge_label_candidates)


@pytest.mark.skipif(
    tesseract_mod.resolve_tesseract_cmd() is None,
    reason="Tesseract binary not installed",
)
def test_tesseract_integration_on_synthetic_sample() -> None:
    """Integration test: read text from a synthetic sample image with Tesseract."""
    sample_png = sorted((REPO_ROOT / "sample_images").glob("*.png"))[0]
    image = cv2.imread(str(sample_png))
    assert image is not None

    engine = TesseractEngine(OCRConfig(min_confidence=0.3))
    boxes = engine.read(image)
    assert len(boxes) >= 1
