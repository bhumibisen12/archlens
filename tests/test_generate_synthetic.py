"""Unit tests for scripts/generate_synthetic.py."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import cv2

from archlens.graph import BBox, DiagramGraph, load_graph_json

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_synthetic.py"


def _load_generator_module() -> ModuleType:
    """Import scripts/generate_synthetic.py as a module."""
    spec = importlib.util.spec_from_file_location("generate_synthetic", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_synthetic"] = module
    spec.loader.exec_module(module)
    return module


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    """Return True if two bounding boxes have strictly positive intersection area."""
    return (
        a.x < b.x + b.w
        and a.x + a.w > b.x
        and a.y < b.y + b.h
        and a.y + a.h > b.y
    )


def test_generate_synthetic_cli_and_schema(tmp_path: Path) -> None:
    """Generator CLI creates valid PNG+JSON pairs conforming to DiagramGraph."""
    gen = _load_generator_module()
    out_dir = tmp_path / "synthetic"

    exit_code = gen.main(
        ["--n", "6", "--seed", "77", "--out", str(out_dir), "--split", "val"]
    )
    assert exit_code == 0

    json_files = sorted(out_dir.glob("val_*.json"))
    png_files = sorted(out_dir.glob("val_*.png"))
    assert len(json_files) == 6
    assert len(png_files) == 6

    for png_path, json_path in zip(png_files, json_files, strict=True):
        img = cv2.imread(str(png_path))
        assert img is not None
        height, width = img.shape[:2]

        graph = load_graph_json(json_path)
        assert isinstance(graph, DiagramGraph)
        assert graph.metadata.image_size is not None
        assert graph.metadata.image_size.width == width
        assert graph.metadata.image_size.height == height
        assert graph.metadata.source_image == png_path.name

        assert 3 <= len(graph.nodes) <= 12
        for node in graph.nodes:
            assert node.type in {"box", "database", "ellipse"}
            assert node.label != ""

        for i, node_a in enumerate(graph.nodes):
            for node_b in graph.nodes[i + 1 :]:
                assert not _boxes_overlap(node_a.bbox, node_b.bbox)

        assert len(graph.edges) >= len(graph.nodes) - 1 or len(graph.edges) >= 1
        for edge in graph.edges:
            assert edge.directed is True
            assert len(edge.polyline) >= 2


def test_generate_synthetic_is_deterministic(tmp_path: Path) -> None:
    """Same seed produces identical DiagramGraph JSON outputs."""
    gen = _load_generator_module()
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    gen.generate_dataset(out_dir=dir_a, count=2, seed=999, split="test")
    gen.generate_dataset(out_dir=dir_b, count=2, seed=999, split="test")

    for idx in range(2):
        text_a = (dir_a / f"test_{idx:04d}.json").read_text(encoding="utf-8")
        text_b = (dir_b / f"test_{idx:04d}.json").read_text(encoding="utf-8")
        assert text_a == text_b


def test_sample_images_conform_to_schema() -> None:
    """Tracked sample_images JSON files conform to DiagramGraph."""
    sample_jsons = sorted((REPO_ROOT / "sample_images").glob("*.json"))
    assert len(sample_jsons) >= 3
    for json_path in sample_jsons:
        graph = load_graph_json(json_path)
        assert isinstance(graph, DiagramGraph)
        assert 3 <= len(graph.nodes) <= 12
