"""Smoke tests for the ArchLens repository skeleton."""

import importlib
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest
import tomllib

from archlens import ArchLensConfig, ArchLensPipeline, __version__, load_config
from archlens.logging_utils import configure_logging
from archlens.ocr import OCREngine, TextBox

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_subpackages_importable() -> None:
    """All core ArchLens subpackages and modules should import cleanly."""
    modules = [
        "archlens",
        "archlens.config",
        "archlens.logging_utils",
        "archlens.pipeline",
        "archlens.preprocess",
        "archlens.detect",
        "archlens.ocr",
        "archlens.connectors",
        "archlens.graph",
        "archlens.review",
        "archlens.export",
    ]
    for module_name in modules:
        mod = importlib.import_module(module_name)
        assert mod is not None


def test_default_yaml_config_loads() -> None:
    """Default configuration YAML should parse into an ArchLensConfig."""
    config_path = REPO_ROOT / "configs" / "default.yaml"
    assert config_path.is_file()

    cfg = load_config(config_path)
    assert isinstance(cfg, ArchLensConfig)
    assert cfg.ocr.engine == "tesseract"
    assert cfg.review.enable_cycle_check is True


class _EmptyOCR(OCREngine):
    """Minimal stub OCREngine for smoke testing without external binaries."""

    def read(self, image: np.ndarray) -> list[TextBox]:
        return []


def test_pipeline_skeleton_runs(tmp_path: Path) -> None:
    """ArchLensPipeline should initialize and return a PipelineResult."""
    sample = tmp_path / "diagram.png"
    canvas = np.full((240, 320, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (30, 50), (140, 140), (20, 20, 20), 3)
    assert cv2.imwrite(str(sample), canvas)

    pipeline = ArchLensPipeline(ocr_engine=_EmptyOCR())
    result = pipeline.run(sample)

    assert result.image_path == str(sample)
    assert result.status == "completed"
    assert result.nodes_detected >= 1
    assert __version__ == "0.1.0"
    assert pytest.approx(0.1) == 0.1


def test_logging_configuration() -> None:
    """configure_logging should return a logger under the archlens namespace."""
    logger = configure_logging("DEBUG")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "archlens"
    assert logger.level == logging.DEBUG


def test_pyproject_paddle_optional_extra() -> None:
    """PaddleOCR must be declared as an optional extra rather than a base dependency."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject["project"]["dependencies"]
    extras = pyproject["project"]["optional-dependencies"]

    assert not any("paddleocr" in dep.lower() for dep in deps)
    assert "paddle" in extras
    assert any("paddleocr" in dep.lower() for dep in extras["paddle"])

