"""Data structures returned by the preprocessing stage."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreprocessedImages:
    """Container holding color, grayscale, and thresholded diagram images.

    Attributes:
        color: Normalized BGR ``uint8`` image of shape ``(H, W, 3)``.
        gray: Shadow-corrected, denoised grayscale ``uint8`` image of shape ``(H, W)``.
        binary: Adaptive-thresholded ``uint8`` image of shape ``(H, W)`` with values in
            ``{0, 255}``. When ``binary_invert=True``, ``255`` represents ink pixels.
        scale_factor: Resize ratio applied relative to the original input dimensions.
        perspective_applied: ``True`` if a document quadrilateral warp was performed.
    """

    color: np.ndarray
    gray: np.ndarray
    binary: np.ndarray
    scale_factor: float = 1.0
    perspective_applied: bool = False

