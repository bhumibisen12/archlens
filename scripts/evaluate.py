"""Evaluate ArchLens against a labelled dataset of diagram images and DiagramGraph JSONs.

Usage example:
    python scripts/evaluate.py --data data/synthetic --out results --ocr tesseract
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from archlens.config import load_config
from archlens.evaluation import evaluate_dataset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for ArchLens evaluation and ablation runs."""
    parser = argparse.ArgumentParser(
        description="Evaluate ArchLens on synthetic or real labelled architecture diagrams."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/synthetic"),
        help="Directory containing diagram images and matching ground-truth .json files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results"),
        help="Output directory for metrics.csv, report.md, and worst_overlays/.",
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
        help="Override OCR engine adapter.",
    )
    parser.add_argument(
        "--paddle-model-dir",
        type=str,
        default=None,
        help="Local directory containing pre-cached PaddleOCR det/rec/cls models.",
    )
    parser.add_argument(
        "--no-perspective",
        action="store_true",
        help="Ablation: disable document perspective correction.",
    )
    parser.add_argument(
        "--no-shadow-removal",
        action="store_true",
        help="Ablation: disable background illumination shadow removal.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Minimum IoU threshold for Hungarian node matching (default: 0.5).",
    )
    parser.add_argument(
        "--worst-k",
        type=int,
        default=10,
        help="Number of worst-performing failure overlays to save (default: 10).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run dataset evaluation and write metrics.csv, report.md, and worst overlays."""
    args = parse_args(argv)
    config = load_config(args.config)

    pp_updates: dict[str, object] = {}
    if args.no_perspective:
        pp_updates["enable_perspective_correction"] = False
    if args.no_shadow_removal:
        pp_updates["enable_shadow_removal"] = False

    ocr_updates: dict[str, object] = {}
    if args.ocr is not None:
        ocr_updates["engine"] = args.ocr
    if args.paddle_model_dir is not None:
        ocr_updates["paddle_model_dir"] = args.paddle_model_dir

    if pp_updates:
        config = config.model_copy(
            update={"preprocess": config.preprocess.model_copy(update=pp_updates)}
        )
    if ocr_updates:
        config = config.model_copy(
            update={"ocr": config.ocr.model_copy(update=ocr_updates)}
        )

    _, summary = evaluate_dataset(
        data_dir=args.data,
        out_dir=args.out,
        config=config,
        iou_threshold=args.iou_threshold,
        worst_k=args.worst_k,
    )
    print(
        f"Evaluated {summary.num_images} image(s) | "
        f"Node F1={summary.node_f1:.4f} | "
        f"Dir Edge F1={summary.directed_edge_f1:.4f} | "
        f"Undir Edge F1={summary.undirected_edge_f1:.4f} | "
        f"Exact Match={summary.exact_graph_match_rate:.4f} | "
        f"Mean Latency={summary.mean_total_latency_ms:.1f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

