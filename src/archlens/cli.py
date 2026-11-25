"""Command-line interface for running ArchLens on diagram images."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from archlens.config import load_config
from archlens.export import to_graphviz_dot, to_mermaid
from archlens.graph import save_graph_json
from archlens.pipeline import ArchLensPipeline


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for ``python -m archlens.cli``."""
    parser = argparse.ArgumentParser(
        prog="archlens.cli",
        description="Parse an architecture diagram image into a structured graph and exports.",
    )
    parser.add_argument("image", type=Path, help="Path to the input diagram image.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("result"),
        help="Output directory for JSON, Mermaid, DOT, and optional debug artifacts.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file (defaults to configs/default.yaml).",
    )
    parser.add_argument(
        "--ocr",
        choices=("tesseract", "paddle"),
        default=None,
        help="OCR engine adapter to use.",
    )
    parser.add_argument(
        "--paddle-model-dir",
        type=str,
        default=None,
        help="Local directory with pre-cached PaddleOCR det/rec/cls models.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate preprocessing images and detection/OCR/connector overlays.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ArchLens CLI and write graph + export outputs."""
    args = parse_args(argv)
    config = load_config(args.config)

    ocr_updates: dict[str, str] = {}
    if args.ocr is not None:
        ocr_updates["engine"] = args.ocr
    if args.paddle_model_dir is not None:
        ocr_updates["paddle_model_dir"] = args.paddle_model_dir
    if ocr_updates:
        config = config.model_copy(
            update={"ocr": config.ocr.model_copy(update=ocr_updates)}
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug" if args.debug else None

    pipeline = ArchLensPipeline(config=config)
    result = pipeline.run(args.image, debug_dir=debug_dir)

    stem = Path(args.image).stem
    json_path = save_graph_json(result.graph, out_dir / f"{stem}.json", indent=config.export.indent)
    mmd_path = out_dir / f"{stem}.mmd"
    mmd_path.write_text(to_mermaid(result.graph), encoding="utf-8")
    dot_path = out_dir / f"{stem}.dot"
    dot_path.write_text(to_graphviz_dot(result.graph), encoding="utf-8")

    total_ms = sum(result.timings_ms.values())
    print(
        f"ArchLens processed {args.image} -> "
        f"{result.nodes_detected} nodes, {result.edges_detected} edges ({total_ms:.1f} ms)"
    )
    print(f"Outputs: {json_path}, {mmd_path}, {dot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
