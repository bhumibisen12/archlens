"""Abstract component detector interface for ArchLens."""

from abc import ABC, abstractmethod

from archlens.graph import Node
from archlens.preprocess import PreprocessedImages


class ComponentDetector(ABC):
    """Interface for diagram component detectors (classical or learned)."""

    @abstractmethod
    def detect(self, preprocessed: PreprocessedImages) -> list[Node]:
        """Detect architectural components in a preprocessed diagram image.

        Args:
            preprocessed: Output from :func:`archlens.preprocess.preprocess_image`.

        Returns:
            List of detected :class:`archlens.graph.Node` objects with bounding
            boxes, coarse types, and confidence scores.
        """

