"""End-to-end pipeline and CLI tests for ArchLens."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from archlens.cli import main as cli_main
from archlens.config import ArchLensConfig, load_config
from archlens.graph import DiagramGraph, load_graph_json
from archlens.ocr import OCREngine, TesseractEngine, TextBox
from archlens.ocr.tesseract import resolve_tesseract_cmd
from archlens.pipeline import ArchLensPipeline, PipelineResult

REPO_ROOT = Path(__file__).resolve().parents[1]


class _SyntheticGroundTruthOCR(OCREngine):
    """Deterministic local OCR stub returning ground-truth TextBoxes for a DiagramGraph."""

    def __init__(self, graph: DiagramGraph) -> None:
        self._graph = graph

    def read(self, image: np.ndarray) -> list[TextBox]:
        assert image.size > 0
        boxes: list[TextBox] = []
        for node in self._graph.nodes:
            if node.label:
                boxes.append(TextBox(text=node.label, bbox=node.bbox, conf=0.95))
        return boxes


def test_default_config_loads_and_validates(tmp_path: Path) -> None:
    """load_config() loads configs/default.yaml and validates invalid inputs."""
    cfg = load_config()
    assert isinstance(cfg, ArchLensConfig)
    assert cfg.ocr.engine == "tesseract"

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("ocr:\n  engine: nonexistent_engine\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad_yaml)


def test_pipeline_end_to_end_on_three_synthetic_images(tmp_path: Path) -> None:
    """End-to-end pipeline returns valid non-empty DiagramGraph on 3 synthetic images."""
    sample_jsons = sorted((REPO_ROOT / "sample_images").glob("*.json"))[:3]
    assert len(sample_jsons) == 3

    cfg = load_config()
    for idx, json_path in enumerate(sample_jsons):
        gt_graph = load_graph_json(json_path)
        png_path = json_path.with_suffix(".png")
        debug_dir = tmp_path / f"debug_{idx}"

        ocr_stub = _SyntheticGroundTruthOCR(gt_graph)
        pipeline = ArchLensPipeline(config=cfg, ocr_engine=ocr_stub)
        result = pipeline.run(png_path, debug_dir=debug_dir)

        assert isinstance(result, PipelineResult)
        assert result.status == "completed"
        assert isinstance(result.graph, DiagramGraph)
        assert len(result.graph.nodes) >= 3
        assert len(result.graph.edges) >= 1
        assert any(node.label for node in result.graph.nodes)

        expected_stages = {"preprocess", "detect", "ocr", "connectors", "graph_assembly"}
        assert expected_stages.issubset(result.timings_ms.keys())
        assert result.graph.metadata.timings_ms == result.timings_ms
        assert result.graph.metadata.image_size is not None

        assert (debug_dir / "detect_overlay.png").is_file()
        assert (debug_dir / "ocr_overlay.png").is_file()
        assert (debug_dir / "connectors_overlay.png").is_file()
        assert (debug_dir / "preprocess" / "07_binary.png").is_file()


def test_cli_runs_and_writes_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """archlens.cli writes JSON, Mermaid, DOT, and --debug artifacts."""
    sample_png = sorted((REPO_ROOT / "sample_images").glob("*.png"))[0]
    gt_graph = load_graph_json(sample_png.with_suffix(".json"))
    out_dir = tmp_path / "result"

    if resolve_tesseract_cmd() is None:
        monkeypatch.setattr(
            TesseractEngine,
            "read",
            lambda _self, img: _SyntheticGroundTruthOCR(gt_graph).read(img),
        )

    exit_code = cli_main(
        [str(sample_png), "--out", str(out_dir), "--ocr", "tesseract", "--debug"]
    )
    assert exit_code == 0

    stem = sample_png.stem
    assert (out_dir / f"{stem}.json").is_file()
    assert (out_dir / f"{stem}.mmd").is_file()
    assert (out_dir / f"{stem}.dot").is_file()
    assert (out_dir / "debug" / "detect_overlay.png").is_file()
    assert (out_dir / "debug" / "connectors_overlay.png").is_file()

    restored = load_graph_json(out_dir / f"{stem}.json")
    assert len(restored.nodes) >= 3
    assert len(restored.edges) >= 1


def test_cli_module_help_invocation() -> None:
    """python -m archlens.cli --help exits cleanly."""
    proc = subprocess.run(
        [sys.executable, "-m", "archlens.cli", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--ocr" in proc.stdout
    assert "--debug" in proc.stdout
