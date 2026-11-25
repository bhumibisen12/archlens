"""Component shape detection interfaces and classical OpenCV detectors."""

from archlens.detect.base import ComponentDetector
from archlens.detect.classical import ClassicalDetector
from archlens.detect.debug import draw_detection_overlay

__all__ = [
    "ClassicalDetector",
    "ComponentDetector",
    "draw_detection_overlay",
]
