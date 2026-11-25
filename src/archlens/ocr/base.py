"""Abstract OCR interface and detected text region model."""

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, Field

from archlens.graph import BBox


class TextBox(BaseModel):
    """A recognized text snippet with its bounding box and confidence score."""

    text: str = Field(..., description="Normalized OCR text.")
    bbox: BBox = Field(..., description="Axis-aligned bounding box in pixel coordinates.")
    conf: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="OCR confidence score normalized to [0.0, 1.0].",
    )


class OCREngine(ABC):
    """Swappable local OCR engine interface."""

    @abstractmethod
    def read(self, image: np.ndarray) -> list[TextBox]:
        """Extract recognized text boxes from a diagram image.

        Args:
            image: Grayscale or BGR ``uint8`` image array.

        Returns:
            List of :class:`TextBox` detections.
        """
