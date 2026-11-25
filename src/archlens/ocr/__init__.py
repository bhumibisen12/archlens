"""Swappable local OCR interfaces, adapters, and text-to-diagram assignment."""

from archlens.config import OCRConfig
from archlens.ocr.assigner import (
    EdgeLabelCandidate,
    TextAssignmentResult,
    assign_text_to_diagram,
)
from archlens.ocr.base import OCREngine, TextBox
from archlens.ocr.normalize import normalize_text
from archlens.ocr.paddle import PaddleOCREngine
from archlens.ocr.tesseract import TesseractEngine


def create_ocr_engine(config: OCRConfig | None = None) -> OCREngine:
    """Instantiate the configured local :class:`OCREngine` adapter."""
    cfg = config or OCRConfig()
    if cfg.engine == "paddle":
        return PaddleOCREngine(cfg)
    return TesseractEngine(cfg)


__all__ = [
    "EdgeLabelCandidate",
    "OCREngine",
    "PaddleOCREngine",
    "TesseractEngine",
    "TextAssignmentResult",
    "TextBox",
    "assign_text_to_diagram",
    "create_ocr_engine",
    "normalize_text",
]
