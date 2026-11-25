"""Unit tests for ArchLens graph Pydantic models and JSON persistence."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from archlens.graph import (
    BBox,
    DiagramGraph,
    Edge,
    GraphMetadata,
    ImageSize,
    Node,
    Point,
    load_graph_json,
    save_graph_json,
)


def _sample_graph() -> DiagramGraph:
    """Create a realistic sample DiagramGraph for testing."""
    return DiagramGraph(
        nodes=[
            Node(
                id="web",
                type="ellipse",
                bbox=BBox(x=20, y=50, w=120, h=60),
                label="Web Client",
                confidence=0.95,
            ),
            Node(
                id="api",
                type="box",
                bbox=BBox(x=220, y=45, w=140, h=70),
                label="API Service",
                confidence=0.92,
            ),
            Node(
                id="db",
                type="database",
                bbox=BBox(x=440, y=40, w=100, h=80),
                label="Orders DB",
                confidence=0.88,
            ),
        ],
        edges=[
            Edge(
                id="e1",
                source="web",
                target="api",
                directed=True,
                label="HTTPS",
                polyline=[Point(x=140, y=80), Point(x=220, y=80)],
                confidence=0.9,
            ),
            Edge(
                id="e2",
                source="api",
                target="db",
                directed=False,
                label=None,
                polyline=[(360, 80), (440, 80)],
                confidence=0.85,
            ),
        ],
        metadata=GraphMetadata(
            image_size=ImageSize(width=640, height=480),
            timings_ms={"preprocess": 5.2, "detect": 12.7},
            source_image="diagram.png",
            ocr_engine="tesseract",
        ),
    )


def test_bbox_helpers_and_point_coercion() -> None:
    """BBox geometric properties and Point tuple coercion should work."""
    bbox = BBox(x=10, y=20, w=100, h=40)
    assert bbox.center == (60.0, 40.0)
    assert bbox.area == 4000

    pt = Point.model_validate((12.5, 34.0))
    assert pt.x == 12.5
    assert pt.y == 34.0


def test_json_round_trip(tmp_path: Path) -> None:
    """Saving a DiagramGraph to JSON and loading it back preserves all fields."""
    original = _sample_graph()
    out_file = tmp_path / "nested" / "graph.json"

    saved_path = save_graph_json(original, out_file)
    assert saved_path == out_file
    assert out_file.is_file()

    restored = load_graph_json(out_file)
    assert restored == original
    assert restored.metadata.image_size == ImageSize(width=640, height=480)
    assert restored.metadata.timings_ms["detect"] == pytest.approx(12.7)


def test_node_and_bbox_validation_errors() -> None:
    """Invalid node types, confidences, and bbox dimensions are rejected."""
    with pytest.raises(ValidationError):
        BBox(x=0, y=0, w=0, h=10)

    with pytest.raises(ValidationError):
        Node(
            id="n1",
            type="triangle",  # type: ignore[arg-type]
            bbox=BBox(x=0, y=0, w=10, h=10),
        )

    with pytest.raises(ValidationError):
        Node(
            id="n1",
            type="box",
            bbox=BBox(x=0, y=0, w=10, h=10),
            confidence=1.5,
        )


def test_graph_rejects_duplicate_ids_and_dangling_edges() -> None:
    """DiagramGraph enforces unique IDs and valid edge endpoints."""
    bbox = BBox(x=0, y=0, w=50, h=50)
    n1 = Node(id="n1", type="box", bbox=bbox, label="A")
    n2 = Node(id="n2", type="component", bbox=bbox, label="B")

    with pytest.raises(ValidationError, match="Duplicate node id"):
        DiagramGraph(nodes=[n1, n1])

    with pytest.raises(ValidationError, match="Duplicate edge id"):
        DiagramGraph(
            nodes=[n1, n2],
            edges=[
                Edge(id="e1", source="n1", target="n2"),
                Edge(id="e1", source="n2", target="n1"),
            ],
        )

    with pytest.raises(ValidationError, match="unknown target node"):
        DiagramGraph(
            nodes=[n1],
            edges=[Edge(id="e1", source="n1", target="missing")],
        )

    with pytest.raises(ValidationError, match="unknown source node"):
        DiagramGraph(
            nodes=[n1],
            edges=[Edge(id="e1", source="missing", target="n1")],
        )

