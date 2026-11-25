"""End-to-end pipeline orchestration for ArchLens."""

import time
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from archlens.config import ArchLensConfig, load_config
from archlens.connectors import (
    ClassicalConnectorExtractor,
    ConnectorExtractor,
    draw_connector_overlay,
)
from archlens.detect import ClassicalDetector, ComponentDetector, draw_detection_overlay
from archlens.graph import DiagramGraph, Edge, GraphMetadata, ImageSize
from archlens.logging_utils import configure_logging
from archlens.ocr import OCREngine, TextBox, assign_text_to_diagram, create_ocr_engine
from archlens.preprocess import PreprocessedImages, preprocess_image


class PipelineResult(BaseModel):
    """Structured output returned by :class:`ArchLensPipeline`."""

    image_path: str
    status: str = Field(default="completed")
    graph: DiagramGraph
    timings_ms: dict[str, float] = Field(default_factory=dict)
    debug_artifacts: dict[str, str] = Field(default_factory=dict)
    nodes_detected: int = Field(default=0, ge=0)
    edges_detected: int = Field(default=0, ge=0)


def _draw_ocr_overlay(
    image: np.ndarray,
    text_boxes: list[TextBox],
    out_path: Path,
) -> Path:
    """Save an OCR debug overlay showing recognized TextBoxes."""
    overlay = image.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    for tb in text_boxes:
        b = tb.bbox
        cv2.rectangle(overlay, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 140, 60), 2)
        cv2.putText(
            overlay,
            f"{tb.text} ({tb.conf:.2f})",
            (b.x, max(14, b.y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 120, 40),
            1,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    return out_path


def _attach_edge_labels(edges: list[Edge], candidates: list) -> list[Edge]:
    """Attach the closest OCR EdgeLabelCandidate text to each matched edge."""
    best_by_edge: dict[str, tuple[float, str]] = {}
    for cand in candidates:
        if cand.edge_id is None:
            continue
        dist = cand.distance_px if cand.distance_px is not None else float("inf")
        prev = best_by_edge.get(cand.edge_id)
        if prev is None or dist < prev[0]:
            best_by_edge[cand.edge_id] = (dist, cand.text)

    updated: list[Edge] = []
    for edge in edges:
        match = best_by_edge.get(edge.id)
        if match is not None:
            updated.append(edge.model_copy(update={"label": match[1]}))
        else:
            updated.append(edge)
    return updated


class ArchLensPipeline:
    """Coordinates preprocessing, detection, OCR, connector extraction, and graph assembly."""

    def __init__(
        self,
        config: ArchLensConfig | None = None,
        detector: ComponentDetector | None = None,
        ocr_engine: OCREngine | None = None,
        connector_extractor: ConnectorExtractor | None = None,
    ) -> None:
        """Initialize the pipeline from ``config`` (or ``configs/default.yaml``)."""
        self.config = config or load_config()
        self.logger = configure_logging(self.config.logging.level)
        self.detector = detector or ClassicalDetector(self.config.detect)
        self._ocr_engine = ocr_engine
        self.connector_extractor = connector_extractor or ClassicalConnectorExtractor(
            self.config.connectors
        )

    @property
    def ocr_engine(self) -> OCREngine:
        """Lazily construct the configured OCR engine on first access."""
        if self._ocr_engine is None:
            self._ocr_engine = create_ocr_engine(self.config.ocr)
        return self._ocr_engine

    def run(self, image_path: Path, debug_dir: Path | None = None) -> PipelineResult:
        """Run ``preprocess -> detect -> OCR -> connectors -> graph assembly`` on an image."""
        _, result = self.run_with_intermediates(image_path=image_path, debug_dir=debug_dir)
        return result

    def run_with_intermediates(
        self,
        image_path: Path,
        debug_dir: Path | None = None,
    ) -> tuple[PreprocessedImages, PipelineResult]:
        """Run the full pipeline and return both preprocessed arrays and ``PipelineResult``."""
        resolved = Path(image_path)
        debug_root = Path(debug_dir) if debug_dir is not None else None
        timings_ms: dict[str, float] = {}
        debug_artifacts: dict[str, str] = {}

        # 1. Preprocess
        t0 = time.perf_counter()
        pp_debug = debug_root / "preprocess" if debug_root is not None else None
        preprocessed = preprocess_image(
            resolved,
            config=self.config.preprocess,
            debug_dir=pp_debug,
        )
        timings_ms["preprocess"] = round((time.perf_counter() - t0) * 1000.0, 2)
        if pp_debug is not None and pp_debug.is_dir():
            for file_path in sorted(pp_debug.glob("*.png")):
                debug_artifacts[f"preprocess/{file_path.stem}"] = str(file_path)

        # 2. Detect components
        t0 = time.perf_counter()
        nodes = self.detector.detect(preprocessed)
        timings_ms["detect"] = round((time.perf_counter() - t0) * 1000.0, 2)
        if debug_root is not None:
            det_path = debug_root / "detect_overlay.png"
            draw_detection_overlay(preprocessed.color, nodes, out_path=det_path)
            debug_artifacts["detect_overlay"] = str(det_path)

        # 3. OCR
        t0 = time.perf_counter()
        text_boxes = self.ocr_engine.read(preprocessed.gray)
        timings_ms["ocr"] = round((time.perf_counter() - t0) * 1000.0, 2)
        if debug_root is not None:
            ocr_path = _draw_ocr_overlay(
                preprocessed.color,
                text_boxes,
                debug_root / "ocr_overlay.png",
            )
            debug_artifacts["ocr_overlay"] = str(ocr_path)

        # 4. Connectors
        t0 = time.perf_counter()
        edges = self.connector_extractor.extract(
            preprocessed,
            nodes,
            text_boxes=text_boxes,
        )
        timings_ms["connectors"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # 5. Graph assembly & text assignment
        t0 = time.perf_counter()
        assignment = assign_text_to_diagram(
            text_boxes=text_boxes,
            nodes=nodes,
            edges=edges,
            inside_margin_px=self.config.ocr.inside_margin_px,
            max_edge_distance_px=self.config.ocr.max_edge_distance_px,
        )
        labeled_edges = _attach_edge_labels(edges, assignment.edge_label_candidates)
        img_h, img_w = preprocessed.color.shape[:2]
        graph = DiagramGraph(
            nodes=assignment.nodes,
            edges=labeled_edges,
            metadata=GraphMetadata(
                image_size=ImageSize(width=img_w, height=img_h),
                timings_ms=dict(timings_ms),
                source_image=resolved.name,
                ocr_engine=self.config.ocr.engine,
            ),
        )
        timings_ms["graph_assembly"] = round((time.perf_counter() - t0) * 1000.0, 2)
        graph.metadata.timings_ms = dict(timings_ms)

        if debug_root is not None:
            conn_path = debug_root / "connectors_overlay.png"
            draw_connector_overlay(
                preprocessed.color,
                graph.nodes,
                graph.edges,
                out_path=conn_path,
            )
            debug_artifacts["connectors_overlay"] = str(conn_path)

        self.logger.info(
            "Parsed %s: %d node(s), %d edge(s) in %.1f ms",
            resolved.name,
            len(graph.nodes),
            len(graph.edges),
            sum(timings_ms.values()),
        )
        result = PipelineResult(
            image_path=str(resolved),
            status="completed",
            graph=graph,
            timings_ms=timings_ms,
            debug_artifacts=debug_artifacts,
            nodes_detected=len(graph.nodes),
            edges_detected=len(graph.edges),
        )
        return preprocessed, result
