"""Generate synthetic labelled architecture diagram images and DiagramGraph JSONs.

Usage example:
    python scripts/generate_synthetic.py --n 20 --seed 42 --out data/synthetic --split train
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from archlens.graph import (
    BBox,
    DiagramGraph,
    Edge,
    GraphMetadata,
    ImageSize,
    Node,
    NodeType,
    Point,
    save_graph_json,
)

LABEL_VOCABULARY: dict[NodeType, tuple[str, ...]] = {
    "ellipse": (
        "User",
        "Web Client",
        "Mobile App",
        "Admin Portal",
        "CDN",
        "External Webhook",
    ),
    "box": (
        "API Gateway",
        "Auth Service",
        "Load Balancer",
        "Message Queue",
        "Payment Service",
        "Order Service",
        "Notification Service",
        "Search Worker",
        "Billing Engine",
        "Rate Limiter",
    ),
    "database": (
        "MySQL DB",
        "PostgreSQL",
        "Redis Cache",
        "Users DB",
        "Object Storage",
        "Audit Log DB",
    ),
}

EDGE_LABELS: tuple[str, ...] = (
    "HTTP",
    "HTTPS",
    "gRPC",
    "SQL",
    "REST",
    "Async",
    "Events",
    "TLS",
    "Cache",
)


@dataclass(frozen=True)
class StyleParams:
    """Visual augmentation parameters for a single synthetic diagram."""

    thickness: int
    wobble: float
    noise_std: float
    lighting_strength: float
    rotation_deg: float
    perspective_scale: float
    ink_color: tuple[int, int, int]
    paper_color: tuple[int, int, int]


def _sample_style(rng: random.Random) -> StyleParams:
    """Sample realistic hand-drawn and scanned paper style variations."""
    ink_val = rng.randint(15, 55)
    paper_base = rng.randint(236, 252)
    return StyleParams(
        thickness=rng.randint(2, 4),
        wobble=rng.uniform(0.0, 2.2),
        noise_std=rng.uniform(2.0, 8.0),
        lighting_strength=rng.uniform(0.04, 0.16),
        rotation_deg=rng.uniform(-3.5, 3.5),
        perspective_scale=rng.uniform(0.0, 0.025),
        ink_color=(ink_val, ink_val, ink_val + rng.randint(0, 10)),
        paper_color=(
            paper_base - rng.randint(0, 8),
            paper_base - rng.randint(0, 4),
            paper_base,
        ),
    )


def _partition_into_layers(num_nodes: int, rng: random.Random) -> list[int]:
    """Split ``num_nodes`` across 2 to 4 non-empty horizontal layers."""
    num_layers = min(num_nodes, rng.randint(2, 4))
    counts = [1] * num_layers
    for _ in range(num_nodes - num_layers):
        counts[rng.randrange(num_layers)] += 1
    return counts


def _pick_node_type_and_label(
    layer_idx: int,
    num_layers: int,
    used_labels: set[str],
    rng: random.Random,
) -> tuple[NodeType, str]:
    """Choose a plausible shape type and unique label for a layer position."""
    if layer_idx == 0:
        shape: NodeType = rng.choices(["ellipse", "box"], weights=[0.65, 0.35])[0]
    elif layer_idx == num_layers - 1:
        shape = rng.choices(["database", "box"], weights=[0.7, 0.3])[0]
    else:
        shape = rng.choices(["box", "database", "ellipse"], weights=[0.7, 0.15, 0.15])[0]

    candidates = [lbl for lbl in LABEL_VOCABULARY[shape] if lbl not in used_labels]
    if not candidates:
        candidates = list(LABEL_VOCABULARY[shape])
    label = rng.choice(candidates)
    used_labels.add(label)
    return shape, label


def _generate_layered_nodes(
    width: int,
    height: int,
    num_nodes: int,
    rng: random.Random,
) -> tuple[list[Node], list[list[int]]]:
    """Create non-overlapping nodes arranged in left-to-right layers."""
    layer_counts = _partition_into_layers(num_nodes, rng)
    num_layers = len(layer_counts)
    margin_x = 95
    margin_y = 85
    usable_w = width - 2 * margin_x
    usable_h = height - 2 * margin_y
    col_w = usable_w / num_layers

    nodes: list[Node] = []
    layers: list[list[int]] = []
    used_labels: set[str] = set()
    node_idx = 0

    for layer_idx, count in enumerate(layer_counts):
        row_h = usable_h / count
        layer_indices: list[int] = []
        max_box_w = max(110, int(col_w * 0.62))
        max_box_h = max(58, int(row_h * 0.56))

        for slot_idx in range(count):
            shape, label = _pick_node_type_and_label(layer_idx, num_layers, used_labels, rng)
            box_w = rng.randint(min(120, max_box_w), min(185, max_box_w))
            box_h = rng.randint(min(60, max_box_h), min(92, max_box_h))

            cell_x0 = margin_x + int(layer_idx * col_w)
            cell_y0 = margin_y + int(slot_idx * row_h)
            pad_x = max(12, int((col_w - box_w) * 0.35))
            pad_y = max(12, int((row_h - box_h) * 0.35))

            max_x = max(cell_x0 + 10, cell_x0 + int(col_w) - box_w - pad_x)
            max_y = max(cell_y0 + 10, cell_y0 + int(row_h) - box_h - pad_y)
            x = rng.randint(cell_x0 + pad_x // 2, max_x)
            y = rng.randint(cell_y0 + pad_y // 2, max_y)

            nodes.append(
                Node(
                    id=f"n{node_idx + 1}",
                    type=shape,
                    bbox=BBox(x=x, y=y, w=box_w, h=box_h),
                    label=label,
                    confidence=1.0,
                )
            )
            layer_indices.append(node_idx)
            node_idx += 1
        layers.append(layer_indices)

    return nodes, layers


def _wobbly_segment(
    p1: tuple[float, float],
    p2: tuple[float, float],
    wobble: float,
    rng: random.Random,
    step_px: float = 18.0,
) -> list[tuple[int, int]]:
    """Subdivide a line segment and add perpendicular hand-drawn jitter."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = float(np.hypot(dx, dy))
    steps = max(2, int(length / step_px))
    if wobble <= 0.05 or length < 1.0:
        return [(round(p1[0]), round(p1[1])), (round(p2[0]), round(p2[1]))]

    nx = -dy / length
    ny = dx / length
    points: list[tuple[int, int]] = []
    for i in range(steps + 1):
        t = i / steps
        jitter = 0.0 if i in (0, steps) else rng.gauss(0.0, wobble)
        px = p1[0] + dx * t + nx * jitter
        py = p1[1] + dy * t + ny * jitter
        points.append((round(px), round(py)))
    return points


def _draw_wobbly_poly(
    canvas: np.ndarray,
    vertices: Sequence[tuple[float, float]],
    closed: bool,
    color: tuple[int, int, int],
    thickness: int,
    wobble: float,
    rng: random.Random,
) -> None:
    """Draw a polygon or polyline with optional hand-drawn wobble."""
    count = len(vertices)
    limit = count if closed else count - 1
    for i in range(limit):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % count]
        pts = np.array(_wobbly_segment(p1, p2, wobble, rng), dtype=np.int32)
        cv2.polylines(
            canvas, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA
        )


def _draw_centered_label(
    canvas: np.ndarray,
    bbox: BBox,
    label: str,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw a component label centered inside its bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    text_thickness = max(1, thickness - 1)
    max_text_w = int(bbox.w * 0.84)

    (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
    if tw > max_text_w and tw > 0:
        font_scale = max(0.32, font_scale * (max_text_w / tw))
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_thickness)

    tx = bbox.x + max(4, (bbox.w - tw) // 2)
    ty = bbox.y + (bbox.h + th) // 2
    cv2.putText(canvas, label, (tx, ty), font, font_scale, color, text_thickness, cv2.LINE_AA)


def _draw_node(
    canvas: np.ndarray,
    node: Node,
    style: StyleParams,
    rng: random.Random,
) -> None:
    """Render a box, cylinder (database), or ellipse node onto the canvas."""
    x, y, w, h = node.bbox.x, node.bbox.y, node.bbox.w, node.bbox.h
    color = style.ink_color
    t = style.thickness

    if node.type == "box":
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        _draw_wobbly_poly(canvas, corners, True, color, t, style.wobble, rng)
    elif node.type == "ellipse":
        center = (x + w // 2, y + h // 2)
        axes = (max(10, w // 2), max(10, h // 2))
        cv2.ellipse(canvas, center, axes, 0, 0, 360, color, t, cv2.LINE_AA)
    elif node.type == "database":
        ry = max(7, min(14, h // 5))
        cx = x + w // 2
        rx = max(10, w // 2)
        cv2.ellipse(canvas, (cx, y + ry), (rx, ry), 0, 0, 360, color, t, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, y + h - ry), (rx, ry), 0, 0, 180, color, t, cv2.LINE_AA)
        _draw_wobbly_poly(
            canvas, [(x, y + ry), (x, y + h - ry)], False, color, t, style.wobble * 0.6, rng
        )
        _draw_wobbly_poly(
            canvas,
            [(x + w, y + ry), (x + w, y + h - ry)],
            False,
            color,
            t,
            style.wobble * 0.6,
            rng,
        )

    _draw_centered_label(canvas, node.bbox, node.label, color, t)


def _build_layer_edges(
    nodes: list[Node],
    layers: list[list[int]],
    rng: random.Random,
) -> list[Edge]:
    """Connect nodes across consecutive layers without duplicates."""
    pairs: set[tuple[int, int]] = set()
    for layer_idx in range(len(layers) - 1):
        src_layer = layers[layer_idx]
        dst_layer = layers[layer_idx + 1]
        for s_idx in src_layer:
            targets = rng.sample(dst_layer, k=min(len(dst_layer), rng.randint(1, 2)))
            for d_idx in targets:
                pairs.add((s_idx, d_idx))
        # Ensure every node in the next layer has at least one incoming edge.
        for d_idx in dst_layer:
            if not any(t == d_idx for _, t in pairs):
                pairs.add((rng.choice(src_layer), d_idx))

    edges: list[Edge] = []
    for edge_idx, (s_idx, d_idx) in enumerate(sorted(pairs), start=1):
        src = nodes[s_idx]
        dst = nodes[d_idx]
        sx = float(src.bbox.x + src.bbox.w)
        sy = float(src.bbox.y + src.bbox.h / 2.0)
        dx = float(dst.bbox.x)
        dy = float(dst.bbox.y + dst.bbox.h / 2.0)

        use_elbow = abs(sy - dy) > 25.0 and rng.random() < 0.55
        if use_elbow:
            mid_x = (sx + dx) / 2.0 + rng.uniform(-12.0, 12.0)
            pts = [Point(x=sx, y=sy), Point(x=mid_x, y=sy), Point(x=mid_x, y=dy), Point(x=dx, y=dy)]
        else:
            pts = [Point(x=sx, y=sy), Point(x=dx, y=dy)]

        label = rng.choice(EDGE_LABELS) if rng.random() < 0.5 else None
        edges.append(
            Edge(
                id=f"e{edge_idx}",
                source=src.id,
                target=dst.id,
                directed=True,
                label=label,
                polyline=pts,
                confidence=1.0,
            )
        )
    return edges


def _draw_arrowhead(
    canvas: np.ndarray,
    tail: tuple[float, float],
    tip: tuple[float, float],
    color: tuple[int, int, int],
    size: int = 12,
) -> None:
    """Draw a filled triangular arrowhead pointing from ``tail`` to ``tip``."""
    dx = tip[0] - tail[0]
    dy = tip[1] - tail[1]
    length = float(np.hypot(dx, dy))
    if length < 1e-3:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux

    base_x = tip[0] - ux * size
    base_y = tip[1] - uy * size
    half_w = size * 0.48
    triangle = np.array(
        [
            [round(tip[0]), round(tip[1])],
            [round(base_x + px * half_w), round(base_y + py * half_w)],
            [round(base_x - px * half_w), round(base_y - py * half_w)],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [triangle], color, lineType=cv2.LINE_AA)


def _draw_edge(
    canvas: np.ndarray,
    edge: Edge,
    style: StyleParams,
    rng: random.Random,
) -> None:
    """Render an edge polyline, its filled arrowhead, and optional label."""
    pts = [(p.x, p.y) for p in edge.polyline]
    _draw_wobbly_poly(canvas, pts, False, style.ink_color, style.thickness, style.wobble * 0.7, rng)
    _draw_arrowhead(canvas, pts[-2], pts[-1], style.ink_color, size=11 + style.thickness)

    if edge.label:
        mid_idx = len(pts) // 2
        p_a = pts[mid_idx - 1]
        p_b = pts[mid_idx]
        mx = int((p_a[0] + p_b[0]) / 2.0)
        my = int((p_a[1] + p_b[1]) / 2.0) - 8
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        (tw, th), _ = cv2.getTextSize(edge.label, font, scale, 1)
        x0 = max(2, mx - tw // 2 - 3)
        y0 = max(2, my - th - 2)
        cv2.rectangle(canvas, (x0, y0), (x0 + tw + 6, y0 + th + 4), style.paper_color, -1)
        cv2.putText(
            canvas,
            edge.label,
            (x0 + 3, y0 + th + 1),
            font,
            scale,
            style.ink_color,
            1,
            cv2.LINE_AA,
        )


def _apply_paper_effects(
    canvas: np.ndarray,
    style: StyleParams,
    rng: random.Random,
) -> np.ndarray:
    """Apply uneven lighting/shadow gradient and Gaussian paper noise."""
    h, w = canvas.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = rng.uniform(0.2 * w, 0.8 * w)
    cy = rng.uniform(0.2 * h, 0.8 * h)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    norm_dist = dist / max(1.0, float(np.hypot(w, h)))

    illumination = 1.0 - style.lighting_strength * norm_dist
    lit = canvas.astype(np.float32) * illumination[:, :, np.newaxis]

    np_rng = np.random.default_rng(rng.randrange(0, 2**31 - 1))
    noise = np_rng.normal(0.0, style.noise_std, size=canvas.shape).astype(np.float32)
    noisy = np.clip(lit + noise, 0, 255).astype(np.uint8)
    return noisy


def _compute_homography(
    width: int,
    height: int,
    style: StyleParams,
    rng: random.Random,
) -> np.ndarray:
    """Create a 3x3 homography combining small rotation and mild perspective warp."""
    center = (width / 2.0, height / 2.0)
    rot_2x3 = cv2.getRotationMatrix2D(center, style.rotation_deg, 1.0)
    rot_3x3 = np.vstack([rot_2x3, [0.0, 0.0, 1.0]])

    src_corners = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    dx_max = width * style.perspective_scale
    dy_max = height * style.perspective_scale
    dst_corners = src_corners.copy()
    for i in range(4):
        dst_corners[i, 0] += rng.uniform(-dx_max, dx_max)
        dst_corners[i, 1] += rng.uniform(-dy_max, dy_max)

    persp = cv2.getPerspectiveTransform(src_corners, dst_corners)
    return (persp @ rot_3x3).astype(np.float64)


def _transform_point(pt: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    """Project a single ``(x, y)`` point through a 3x3 homography."""
    arr = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(arr, homography)[0, 0]
    return (float(warped[0]), float(warped[1]))


def _transform_bbox(bbox: BBox, homography: np.ndarray, width: int, height: int) -> BBox:
    """Compute the clipped axis-aligned bounding box enclosing a warped BBox."""
    corners = [
        (float(bbox.x), float(bbox.y)),
        (float(bbox.x + bbox.w), float(bbox.y)),
        (float(bbox.x + bbox.w), float(bbox.y + bbox.h)),
        (float(bbox.x), float(bbox.y + bbox.h)),
    ]
    warped = [_transform_point(c, homography) for c in corners]
    xs = [p[0] for p in warped]
    ys = [p[1] for p in warped]

    x0 = max(0, int(np.floor(min(xs))))
    y0 = max(0, int(np.floor(min(ys))))
    x1 = min(width, int(np.ceil(max(xs))))
    y1 = min(height, int(np.ceil(max(ys))))
    return BBox(x=x0, y=y0, w=max(1, x1 - x0), h=max(1, y1 - y0))


def generate_diagram(
    rng: random.Random,
    width: int = 1280,
    height: int = 800,
    source_name: str | None = None,
) -> tuple[np.ndarray, DiagramGraph]:
    """Generate one synthetic architecture diagram image and its ground-truth graph."""
    style = _sample_style(rng)
    canvas = np.full((height, width, 3), style.paper_color, dtype=np.uint8)

    num_nodes = rng.randint(3, 12)
    nodes, layers = _generate_layered_nodes(width, height, num_nodes, rng)
    edges = _build_layer_edges(nodes, layers, rng)

    for edge in edges:
        _draw_edge(canvas, edge, style, rng)
    for node in nodes:
        _draw_node(canvas, node, style, rng)

    canvas = _apply_paper_effects(canvas, style, rng)
    homography = _compute_homography(width, height, style, rng)
    warped = cv2.warpPerspective(
        canvas,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=style.paper_color,
    )

    warped_nodes = [
        node.model_copy(update={"bbox": _transform_bbox(node.bbox, homography, width, height)})
        for node in nodes
    ]
    warped_edges = [
        edge.model_copy(
            update={
                "polyline": [
                    Point(x=wx, y=wy)
                    for wx, wy in (
                        _transform_point((pt.x, pt.y), homography) for pt in edge.polyline
                    )
                ]
            }
        )
        for edge in edges
    ]

    graph = DiagramGraph(
        nodes=warped_nodes,
        edges=warped_edges,
        metadata=GraphMetadata(
            image_size=ImageSize(width=width, height=height),
            source_image=source_name,
        ),
    )
    return warped, graph


def generate_dataset(
    out_dir: Path,
    count: int,
    seed: int,
    split: str,
) -> list[tuple[Path, Path]]:
    """Generate ``count`` synthetic diagram PNG + JSON pairs into ``out_dir``."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    outputs: list[tuple[Path, Path]] = []

    for idx in range(count):
        stem = f"{split}_{idx:04d}"
        png_path = out_path / f"{stem}.png"
        json_path = out_path / f"{stem}.json"
        image, graph = generate_diagram(rng=rng, source_name=png_path.name)
        if not cv2.imwrite(str(png_path), image):
            raise OSError(f"Failed to write synthetic image to {png_path}")
        save_graph_json(graph, json_path)
        outputs.append((png_path, json_path))

    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for synthetic diagram generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic labelled architecture diagram images and JSON graphs."
    )
    parser.add_argument("--n", type=int, default=10, help="Number of diagrams to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
        help="Dataset split prefix for output filenames.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    outputs = generate_dataset(out_dir=args.out, count=args.n, seed=args.seed, split=args.split)
    print(f"Generated {len(outputs)} diagram(s) in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
