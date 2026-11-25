"""High-level preprocessing pipeline orchestrator."""

from pathlib import Path

import cv2
import numpy as np

from archlens.config import PreprocessConfig
from archlens.preprocess.filters import (
    binarize_image,
    denoise_image,
    load_image,
    remove_shadows,
    resize_to_max_side,
)
from archlens.preprocess.perspective import correct_perspective
from archlens.preprocess.types import PreprocessedImages


def _save_debug_step(debug_dir: Path | None, filename: str, image: np.ndarray) -> None:
    """Write an intermediate preprocessing image when ``debug_dir`` is enabled."""
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / filename), image)


def preprocess_array(
    image: np.ndarray,
    config: PreprocessConfig | None = None,
    debug_dir: Path | None = None,
) -> PreprocessedImages:
    """Run resizing, optional perspective warp, shadow removal, denoise, and binarization.

    Args:
        image: Input BGR ``uint8`` image array.
        config: Optional :class:`PreprocessConfig`. Uses defaults when omitted.
        debug_dir: Optional directory where intermediate step PNGs are saved.

    Returns:
        :class:`PreprocessedImages` holding ``color``, ``gray``, and ``binary`` images.
    """
    cfg = config or PreprocessConfig()
    debug_path = Path(debug_dir) if debug_dir is not None else None

    _save_debug_step(debug_path, "01_loaded.png", image)

    resized, scale_factor = resize_to_max_side(image, cfg.max_dimension)
    _save_debug_step(debug_path, "02_resized.png", resized)

    perspective_applied = False
    color = resized
    if cfg.enable_perspective_correction:
        color, perspective_applied = correct_perspective(
            resized,
            min_area_ratio=cfg.paper_min_area_ratio,
            canny_low=cfg.paper_canny_low,
            canny_high=cfg.paper_canny_high,
            approx_epsilon=cfg.paper_approx_epsilon,
        )
    _save_debug_step(debug_path, "03_perspective.png", color)

    raw_gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) if color.ndim == 3 else color.copy()
    _save_debug_step(debug_path, "04_gray.png", raw_gray)

    shadow_free = (
        remove_shadows(raw_gray, cfg.shadow_kernel_size)
        if cfg.enable_shadow_removal
        else raw_gray.copy()
    )
    _save_debug_step(debug_path, "05_shadow_removed.png", shadow_free)

    denoised = denoise_image(shadow_free, cfg.denoise_kernel, cfg.denoise_method)
    _save_debug_step(debug_path, "06_denoised.png", denoised)

    binary = binarize_image(
        denoised,
        block_size=cfg.adaptive_block_size,
        c_value=cfg.adaptive_c,
        invert=cfg.binary_invert,
    )
    _save_debug_step(debug_path, "07_binary.png", binary)

    return PreprocessedImages(
        color=color,
        gray=denoised,
        binary=binary,
        scale_factor=scale_factor,
        perspective_applied=perspective_applied,
    )


def preprocess_image(
    image_path: Path,
    config: PreprocessConfig | None = None,
    debug_dir: Path | None = None,
) -> PreprocessedImages:
    """Load an image from disk and run :func:`preprocess_array`."""
    image = load_image(image_path)
    return preprocess_array(image=image, config=config, debug_dir=debug_dir)

