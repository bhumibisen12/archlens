"""Document/paper contour detection and perspective correction."""

import cv2
import numpy as np


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order four 2D points as ``(top-left, top-right, bottom-right, bottom-left)``."""
    points = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    coord_sum = points.sum(axis=1)
    ordered[0] = points[np.argmin(coord_sum)]
    ordered[2] = points[np.argmax(coord_sum)]

    coord_diff = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(coord_diff)]
    ordered[3] = points[np.argmax(coord_diff)]
    return ordered


def find_paper_contour(
    image: np.ndarray,
    min_area_ratio: float,
    canny_low: int,
    canny_high: int,
    approx_epsilon: float,
) -> np.ndarray | None:
    """Locate the largest convex quadrilateral contour likely representing a page.

    Args:
        image: Input BGR or grayscale image.
        min_area_ratio: Minimum contour area fraction relative to image area.
        canny_low: Lower Canny edge threshold.
        canny_high: Upper Canny edge threshold.
        approx_epsilon: Polygon approximation ratio relative to contour perimeter.

    Returns:
        Ordered ``(4, 2)`` float32 array of corners, or ``None`` if not found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = float(gray.shape[0] * gray.shape[1])
    min_area = img_area * min_area_ratio

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(contour)
        if area < min_area:
            break
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, approx_epsilon * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_quad_points(approx.reshape(4, 2))

    return None


def correct_perspective(
    image: np.ndarray,
    min_area_ratio: float = 0.45,
    canny_low: int = 50,
    canny_high: int = 150,
    approx_epsilon: float = 0.02,
) -> tuple[np.ndarray, bool]:
    """Warp the largest paper quadrilateral to a top-down rectangular view.

    Returns ``(image, False)`` unchanged when no qualifying quadrilateral is found.
    """
    quad = find_paper_contour(
        image=image,
        min_area_ratio=min_area_ratio,
        canny_low=canny_low,
        canny_high=canny_high,
        approx_epsilon=approx_epsilon,
    )
    if quad is None:
        return image, False

    tl, tr, br, bl = quad
    width_top = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    height_right = float(np.linalg.norm(tr - br))
    height_left = float(np.linalg.norm(tl - bl))

    target_w = max(int(round(max(width_top, width_bottom))), 1)
    target_h = max(int(round(max(height_left, height_right))), 1)
    if target_w < 16 or target_h < 16:
        return image, False

    dst = np.array(
        [
            [0.0, 0.0],
            [target_w - 1.0, 0.0],
            [target_w - 1.0, target_h - 1.0],
            [0.0, target_h - 1.0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(image, matrix, (target_w, target_h), flags=cv2.INTER_LINEAR)
    return warped, True
