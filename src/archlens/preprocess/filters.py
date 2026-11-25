"""Low-level OpenCV image loading, normalization, shadow removal, and thresholding."""

from pathlib import Path

import cv2
import numpy as np


def ensure_odd(value: int, minimum: int = 3) -> int:
    """Clamp ``value`` to at least ``minimum`` and round up to an odd integer."""
    clamped = max(value, minimum)
    return clamped if clamped % 2 == 1 else clamped + 1


def load_image(image_path: Path) -> np.ndarray:
    """Load an image from disk as a BGR ``uint8`` array.

    Args:
        image_path: Path to an image file readable by OpenCV.

    Returns:
        Loaded BGR image array.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        ValueError: If OpenCV fails to decode the image.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image file: {path}")
    return image


def resize_to_max_side(image: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float]:
    """Resize ``image`` proportionally if its longest side exceeds ``max_dimension``."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return image, 1.0

    scale = max_dimension / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def remove_shadows(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    """Estimate uneven illumination and divide it out to flatten paper shadows.

    Args:
        gray: Single-channel ``uint8`` grayscale image.
        kernel_size: Morphological dilation kernel size for background estimation.

    Returns:
        Illumination-normalized ``uint8`` grayscale image.
    """
    k = ensure_odd(kernel_size, minimum=3)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.dilate(gray, morph_kernel)
    background = cv2.GaussianBlur(background, (15, 15), 0)
    background = np.maximum(background, 1)
    normalized = cv2.divide(gray, background, scale=255)
    return normalized.astype(np.uint8)


def denoise_image(gray: np.ndarray, kernel_size: int, method: str = "median") -> np.ndarray:
    """Apply configurable local denoising to a grayscale image."""
    if kernel_size <= 1:
        return gray.copy()

    k = ensure_odd(kernel_size, minimum=3)
    if method == "gaussian":
        return cv2.GaussianBlur(gray, (k, k), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(gray, d=k, sigmaColor=50, sigmaSpace=50)
    return cv2.medianBlur(gray, k)


def binarize_image(
    gray: np.ndarray,
    block_size: int,
    c_value: int,
    invert: bool = True,
) -> np.ndarray:
    """Convert grayscale diagram to binary ink mask via Gaussian adaptive thresholding."""
    block = ensure_odd(block_size, minimum=3)
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresh_type,
        block,
        c_value,
    )

