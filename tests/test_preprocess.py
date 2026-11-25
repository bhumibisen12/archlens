"""Unit and synthetic-image tests for src/archlens/preprocess/."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from archlens.config import PreprocessConfig, load_config
from archlens.preprocess import PreprocessedImages, preprocess_array, preprocess_image

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_preprocess_on_sample_synthetic_images() -> None:
    """Preprocessing sample synthetic diagrams produces matching shapes and ink pixels."""
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml").preprocess
    sample_pngs = sorted((REPO_ROOT / "sample_images").glob("*.png"))
    assert len(sample_pngs) >= 3

    for png_path in sample_pngs:
        result = preprocess_image(png_path, config=cfg)
        assert isinstance(result, PreprocessedImages)
        assert result.color.ndim == 3
        assert result.color.shape[2] == 3
        assert result.gray.shape == result.color.shape[:2]
        assert result.binary.shape == result.color.shape[:2]
        assert result.color.dtype == np.uint8
        assert result.gray.dtype == np.uint8
        assert result.binary.dtype == np.uint8

        unique_vals = set(np.unique(result.binary).tolist())
        assert unique_vals.issubset({0, 255})

        ink_ratio = float(np.count_nonzero(result.binary)) / result.binary.size
        assert 0.002 < ink_ratio < 0.45


def test_preprocess_resize_respects_max_dimension() -> None:
    """Images larger than max_dimension are downscaled proportionally."""
    sample_png = sorted((REPO_ROOT / "sample_images").glob("*.png"))[0]
    cfg = PreprocessConfig(max_dimension=640, enable_perspective_correction=False)

    result = preprocess_image(sample_png, config=cfg)
    assert max(result.color.shape[:2]) == 640
    assert result.gray.shape == result.color.shape[:2]
    assert result.binary.shape == result.color.shape[:2]
    assert result.scale_factor == pytest.approx(0.5)
    assert result.perspective_applied is False


def test_preprocess_perspective_correction_on_paper_quad() -> None:
    """A prominent white paper quadrilateral on dark background is rectified."""
    scene = np.full((600, 800, 3), 25, dtype=np.uint8)
    quad = np.array([[90, 70], [710, 50], [740, 540], [60, 520]], dtype=np.int32)
    cv2.fillConvexPoly(scene, quad, (245, 245, 245))
    cv2.rectangle(scene, (220, 180), (420, 300), (20, 20, 20), 3)

    cfg = PreprocessConfig(
        enable_perspective_correction=True,
        paper_min_area_ratio=0.45,
        denoise_method="gaussian",
    )
    result = preprocess_array(scene, config=cfg)

    assert result.perspective_applied is True
    assert result.binary.shape == result.gray.shape
    assert np.count_nonzero(result.binary) > 100


def test_preprocess_saves_debug_intermediates(tmp_path: Path) -> None:
    """Passing debug_dir saves every intermediate step PNG."""
    sample_png = sorted((REPO_ROOT / "sample_images").glob("*.png"))[0]
    debug_dir = tmp_path / "debug_steps"

    preprocess_image(sample_png, debug_dir=debug_dir)

    expected = [
        "01_loaded.png",
        "02_resized.png",
        "03_perspective.png",
        "04_gray.png",
        "05_shadow_removed.png",
        "06_denoised.png",
        "07_binary.png",
    ]
    for name in expected:
        step_file = debug_dir / name
        assert step_file.is_file()
        assert step_file.stat().st_size > 0


def test_preprocess_error_handling(tmp_path: Path) -> None:
    """Missing or corrupt files raise clear errors."""
    with pytest.raises(FileNotFoundError):
        preprocess_image(tmp_path / "missing.png")

    bad_file = tmp_path / "corrupt.png"
    bad_file.write_bytes(b"not-an-image")
    with pytest.raises(ValueError, match="Unable to decode"):
        preprocess_image(bad_file)

