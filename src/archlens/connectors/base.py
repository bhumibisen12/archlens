"""Abstract connector/arrow extractor interface for ArchLens."""

from abc import ABC, abstractmethod

from archlens.graph import Edge, Node
from archlens.ocr import TextBox
from archlens.preprocess import PreprocessedImages


class ConnectorExtractor(ABC):
    """Interface for extracting connectors and arrows between detected nodes."""

    @abstractmethod
    def extract(
        self,
        preprocessed: PreprocessedImages,
        nodes: list[Node],
        text_boxes: list[TextBox] | None = None,
    ) -> list[Edge]:
        """Extract edges linking component nodes from a preprocessed diagram.

        Args:
            preprocessed: Output from :func:`archlens.preprocess.preprocess_image`.
            nodes: Detected component nodes to link against.
            text_boxes: Optional OCR text regions to mask out before skeletonization.

        Returns:
            List of extracted :class:`archlens.graph.Edge` objects.
        """
