"""Unit and synthetic-image tests for src/archlens/connectors/."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from archlens.connectors import ClassicalConnectorExtractor, ConnectorExtractor
from archlens.graph import BBox, Edge, Node, load_graph_json
from archlens.ocr import TextBox
from archlens.preprocess import preprocess_array, preprocess_image

REPO_ROOT = Path(__file__).resolve().parents[1]


def _draw_filled_arrowhead(
    canvas: np.ndarray,
    tail: tuple[int, int],
    tip: tuple[int, int],
    color: tuple[int, int, int] = (20, 20, 20),
    size: int = 14,
) -> None:
    """Draw a filled triangular arrowhead for test diagrams."""
    dx = float(tip[0] - tail[0])
    dy = float(tip[1] - tail[1])
    length = float(np.hypot(dx, dy))
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    bx = tip[0] - ux * size
    by = tip[1] - uy * size
    half_w = size * 0.5
    tri = np.array(
        [
            [tip[0], tip[1]],
            [round(bx + px * half_w), round(by + py * half_w)],
            [round(bx - px * half_w), round(by - py * half_w)],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [tri], color)


def test_connector_extractor_is_abstract() -> None:
    """ConnectorExtractor cannot be instantiated without implementing extract()."""
    with pytest.raises(TypeError):
        ConnectorExtractor()  # type: ignore[abstract]


def test_straight_bidirectional_arrows() -> None:
    """ClassicalConnectorExtractor detects straight arrows in both directions."""
    canvas = np.full((360, 600, 3), 255, dtype=np.uint8)
    ink = (20, 20, 20)

    n1 = Node(id="n1", type="box", bbox=BBox(x=40, y=50, w=120, h=70), label="Client")
    n2 = Node(id="n2", type="box", bbox=BBox(x=400, y=50, w=120, h=70), label="API")
    n3 = Node(id="n3", type="database", bbox=BBox(x=400, y=220, w=120, h=70), label="DB")
    n4 = Node(id="n4", type="box", bbox=BBox(x=40, y=220, w=120, h=70), label="Worker")

    for node in (n1, n2, n3, n4):
        b = node.bbox
        cv2.rectangle(canvas, (b.x, b.y), (b.x + b.w, b.y + b.h), ink, 3)

    # Left-to-right arrow: n1 -> n2
    cv2.line(canvas, (160, 85), (400, 85), ink, 3)
    _draw_filled_arrowhead(canvas, (160, 85), (400, 85), ink)

    # Right-to-left arrow: n3 -> n4
    cv2.line(canvas, (400, 255), (160, 255), ink, 3)
    _draw_filled_arrowhead(canvas, (400, 255), (160, 255), ink)

    preprocessed = preprocess_array(canvas)
    edges = ClassicalConnectorExtractor().extract(preprocessed, [n1, n2, n3, n4])

    assert all(isinstance(e, Edge) for e in edges)
    directed_pairs = {(e.source, e.target) for e in edges if e.directed}
    assert ("n1", "n2") in directed_pairs
    assert ("n3", "n4") in directed_pairs
    assert all(len(e.polyline) >= 2 and e.confidence > 0.4 for e in edges)


def test_elbow_arrow_and_text_masking() -> None:
    """Elbow (Z-bend) arrow is traced with >=3 polyline vertices while text is masked."""
    canvas = np.full((420, 620, 3), 255, dtype=np.uint8)
    ink = (20, 20, 20)

    n1 = Node(id="n1", type="box", bbox=BBox(x=40, y=60, w=130, h=70), label="Auth")
    n2 = Node(id="n2", type="database", bbox=BBox(x=420, y=260, w=130, h=80), label="Users")

    for node in (n1, n2):
        b = node.bbox
        cv2.rectangle(canvas, (b.x, b.y), (b.x + b.w, b.y + b.h), ink, 3)

    # Draw Z-shaped elbow arrow: n1 -> n2
    elbow_pts = [(170, 95), (295, 95), (295, 300), (420, 300)]
    for i in range(len(elbow_pts) - 1):
        cv2.line(canvas, elbow_pts[i], elbow_pts[i + 1], ink, 3)
    _draw_filled_arrowhead(canvas, (295, 300), (420, 300), ink)

    # Draw floating edge label near the elbow segment and pass it via text_boxes
    cv2.putText(canvas, "SQL", (315, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ink, 2)
    text_boxes = [TextBox(text="SQL", bbox=BBox(x=312, y=266, w=45, h=24), conf=0.95)]

    preprocessed = preprocess_array(canvas)
    edges = ClassicalConnectorExtractor().extract(
        preprocessed,
        [n1, n2],
        text_boxes=text_boxes,
    )

    assert len(edges) == 1
    edge = edges[0]
    assert edge.source == "n1"
    assert edge.target == "n2"
    assert edge.directed is True
    assert len(edge.polyline) >= 3


def test_crossing_arrows_preserved() -> None:
    """Two perpendicular crossing arrows remain resolved as two distinct edges."""
    canvas = np.full((440, 480, 3), 255, dtype=np.uint8)
    ink = (20, 20, 20)

    n_left = Node(id="left", type="box", bbox=BBox(x=30, y=180, w=90, h=60))
    n_right = Node(id="right", type="box", bbox=BBox(x=360, y=180, w=90, h=60))
    n_top = Node(id="top", type="box", bbox=BBox(x=195, y=30, w=90, h=60))
    n_bottom = Node(id="bottom", type="box", bbox=BBox(x=195, y=350, w=90, h=60))

    nodes = [n_left, n_right, n_top, n_bottom]
    for node in nodes:
        b = node.bbox
        cv2.rectangle(canvas, (b.x, b.y), (b.x + b.w, b.y + b.h), ink, 3)

    # Horizontal arrow left -> right
    cv2.line(canvas, (120, 210), (360, 210), ink, 3)
    _draw_filled_arrowhead(canvas, (120, 210), (360, 210), ink)

    # Vertical arrow top -> bottom
    cv2.line(canvas, (240, 90), (240, 350), ink, 3)
    _draw_filled_arrowhead(canvas, (240, 90), (240, 350), ink)

    preprocessed = preprocess_array(canvas)
    edges = ClassicalConnectorExtractor().extract(preprocessed, nodes)

    directed_pairs = {(e.source, e.target) for e in edges if e.directed}
    assert ("left", "right") in directed_pairs
    assert ("top", "bottom") in directed_pairs


def test_connector_debug_overlay_and_synthetic_sample(tmp_path: Path) -> None:
    """Extractor extracts straight + elbow arrows on sample_images/test_0001 and writes overlay."""
    json_path = REPO_ROOT / "sample_images" / "test_0001.json"
    gt_graph = load_graph_json(json_path)
    preprocessed = preprocess_image(json_path.with_suffix(".png"))

    overlay_path = tmp_path / "overlay" / "connectors.png"
    edges = ClassicalConnectorExtractor().extract(
        preprocessed,
        gt_graph.nodes,
        debug_overlay_path=overlay_path,
    )

    directed_pairs = {(e.source, e.target) for e in edges if e.directed}
    expected_pairs = {(e.source, e.target) for e in gt_graph.edges}
    assert expected_pairs.issubset(directed_pairs)

    assert overlay_path.is_file()
    rendered = cv2.imread(str(overlay_path))
    assert rendered is not None
    assert rendered.shape == preprocessed.color.shape
