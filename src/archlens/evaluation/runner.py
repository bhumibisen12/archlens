"""Dataset evaluation runner producing CSV metrics, Markdown report, and worst overlays."""

import csv
from pathlib import Path

import cv2

from archlens.config import ArchLensConfig, load_config
from archlens.connectors import draw_connector_overlay
from archlens.detect import draw_detection_overlay
from archlens.evaluation.metrics import (
    AggregateMetrics,
    ImageMetrics,
    aggregate_metrics,
    compute_image_metrics,
)
from archlens.graph import BBox, DiagramGraph, load_graph_json
from archlens.ocr import OCREngine
from archlens.pipeline import ArchLensPipeline

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def find_dataset_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """Discover ``(image_path, json_path)`` pairs inside ``data_dir``."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluation directory not found: {root}")

    pairs: list[tuple[Path, Path]] = []
    for json_path in sorted(root.glob("*.json")):
        for ext in _IMAGE_EXTENSIONS:
            candidate = json_path.with_suffix(ext)
            if candidate.is_file():
                pairs.append((candidate, json_path))
                break

    if not pairs:
        raise ValueError(f"No image + JSON pairs found in {root}")
    return pairs


def _rescale_graph_nodes(graph: DiagramGraph, scale_factor: float) -> DiagramGraph:
    """Scale predicted node bboxes back to original image pixel space after downscaling."""
    if abs(scale_factor - 1.0) <= 1e-6 or scale_factor <= 0.0:
        return graph
    inv = 1.0 / scale_factor
    scaled_nodes = [
        node.model_copy(
            update={
                "bbox": BBox(
                    x=max(0, round(node.bbox.x * inv)),
                    y=max(0, round(node.bbox.y * inv)),
                    w=max(1, round(node.bbox.w * inv)),
                    h=max(1, round(node.bbox.h * inv)),
                )
            }
        )
        for node in graph.nodes
    ]
    return graph.model_copy(update={"nodes": scaled_nodes})


def _save_failure_overlay(
    color_bgr,
    gt_graph: DiagramGraph,
    pred_graph: DiagramGraph,
    out_path: Path,
) -> None:
    """Render GT boxes (green dashed-style thin) and predicted nodes/edges for failure review."""
    overlay = draw_detection_overlay(color_bgr, pred_graph.nodes)
    overlay = draw_connector_overlay(overlay, pred_graph.nodes, pred_graph.edges)
    for gt_node in gt_graph.nodes:
        b = gt_node.bbox
        cv2.rectangle(overlay, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 200, 0), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def _write_metrics_csv(per_image: list[ImageMetrics], out_csv: Path) -> Path:
    """Write per-image evaluation metrics to ``metrics.csv``."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    stage_names = sorted({stage for m in per_image for stage in m.timings_ms})
    fieldnames = [
        "image_name",
        "gt_nodes",
        "pred_nodes",
        "matched_nodes",
        "node_precision",
        "node_recall",
        "node_f1",
        "mean_label_similarity",
        "gt_edges",
        "pred_edges",
        "directed_edge_matches",
        "directed_edge_precision",
        "directed_edge_recall",
        "directed_edge_f1",
        "undirected_edge_matches",
        "undirected_edge_precision",
        "undirected_edge_recall",
        "undirected_edge_f1",
        "exact_graph_match",
        "total_latency_ms",
        *[f"timing_{stage}_ms" for stage in stage_names],
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in per_image:
            row = {
                "image_name": item.image_name,
                "gt_nodes": item.gt_nodes,
                "pred_nodes": item.pred_nodes,
                "matched_nodes": item.matched_nodes,
                "node_precision": item.node_precision,
                "node_recall": item.node_recall,
                "node_f1": item.node_f1,
                "mean_label_similarity": item.mean_label_similarity,
                "gt_edges": item.gt_edges,
                "pred_edges": item.pred_edges,
                "directed_edge_matches": item.directed_edge_matches,
                "directed_edge_precision": item.directed_edge_precision,
                "directed_edge_recall": item.directed_edge_recall,
                "directed_edge_f1": item.directed_edge_f1,
                "undirected_edge_matches": item.undirected_edge_matches,
                "undirected_edge_precision": item.undirected_edge_precision,
                "undirected_edge_recall": item.undirected_edge_recall,
                "undirected_edge_f1": item.undirected_edge_f1,
                "exact_graph_match": item.exact_graph_match,
                "total_latency_ms": item.total_latency_ms,
            }
            for stage in stage_names:
                row[f"timing_{stage}_ms"] = item.timings_ms.get(stage, 0.0)
            writer.writerow(row)
    return out_csv


def _write_report_md(
    summary: AggregateMetrics,
    worst: list[ImageMetrics],
    config: ArchLensConfig,
    iou_threshold: float,
    out_md: Path,
) -> Path:
    """Write an aggregate Markdown summary and worst-case table to ``report.md``."""
    lines = [
        "# ArchLens Evaluation Report",
        "",
        f"- **Images evaluated**: {summary.num_images}",
        f"- **IoU threshold**: {iou_threshold}",
        f"- **OCR engine**: `{config.ocr.engine}`",
        f"- **Perspective correction**: `{config.preprocess.enable_perspective_correction}`",
        f"- **Shadow removal**: `{config.preprocess.enable_shadow_removal}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Node Precision | {summary.node_precision:.4f} |",
        f"| Node Recall | {summary.node_recall:.4f} |",
        f"| Node F1 | {summary.node_f1:.4f} |",
        f"| Mean Label Similarity | {summary.mean_label_similarity:.4f} |",
        f"| Directed Edge Precision | {summary.directed_edge_precision:.4f} |",
        f"| Directed Edge Recall | {summary.directed_edge_recall:.4f} |",
        f"| Directed Edge F1 | {summary.directed_edge_f1:.4f} |",
        f"| Undirected Edge F1 | {summary.undirected_edge_f1:.4f} |",
        f"| Exact Graph Match Rate | {summary.exact_graph_match_rate:.4f} |",
        f"| Mean Total Latency (ms) | {summary.mean_total_latency_ms:.2f} |",
        "",
        "## Mean Per-Stage Latency (ms)",
        "",
        "| Stage | Mean (ms) |",
        "| --- | ---: |",
    ]
    for stage, ms in summary.mean_stage_timings_ms.items():
        lines.append(f"| {stage} | {ms:.2f} |")

    lines.extend(
        [
            "",
            f"## Worst {len(worst)} Images",
            "",
            (
                "| Image | Node F1 | Directed Edge F1 | Undirected Edge F1 "
                "| Label Sim | Latency (ms) |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in worst:
        lines.append(
            f"| {item.image_name} | {item.node_f1:.4f} | {item.directed_edge_f1:.4f} | "
            f"{item.undirected_edge_f1:.4f} | {item.mean_label_similarity:.4f} | "
            f"{item.total_latency_ms:.2f} |"
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_md


def evaluate_dataset(
    data_dir: Path,
    out_dir: Path,
    config: ArchLensConfig | None = None,
    iou_threshold: float = 0.5,
    ocr_engine: OCREngine | None = None,
    worst_k: int = 10,
) -> tuple[list[ImageMetrics], AggregateMetrics]:
    """Run end-to-end evaluation across all image+JSON pairs in ``data_dir``."""
    cfg = config or load_config()
    pairs = find_dataset_pairs(data_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pipeline = ArchLensPipeline(config=cfg, ocr_engine=ocr_engine)
    per_image: list[ImageMetrics] = []
    cached_for_overlay: list[tuple[ImageMetrics, object, DiagramGraph, DiagramGraph]] = []

    for img_path, json_path in pairs:
        gt_graph = load_graph_json(json_path)
        preprocessed, result = pipeline.run_with_intermediates(img_path)
        pred_graph = result.graph
        if not preprocessed.perspective_applied:
            pred_graph = _rescale_graph_nodes(pred_graph, preprocessed.scale_factor)

        metrics = compute_image_metrics(
            image_name=img_path.name,
            gt_graph=gt_graph,
            pred_graph=pred_graph,
            iou_threshold=iou_threshold,
            timings_ms=result.timings_ms,
        )
        per_image.append(metrics)
        cached_for_overlay.append((metrics, preprocessed.color, gt_graph, pred_graph))

    summary = aggregate_metrics(per_image)
    ranked_worst = sorted(
        cached_for_overlay,
        key=lambda item: (
            (item[0].node_f1 + item[0].directed_edge_f1) / 2.0,
            item[0].image_name,
        ),
    )[:worst_k]

    worst_dir = out_root / "worst_overlays"
    for idx, (metrics, color_bgr, gt_graph, pred_graph) in enumerate(ranked_worst, start=1):
        stem = Path(metrics.image_name).stem
        _save_failure_overlay(
            color_bgr,
            gt_graph=gt_graph,
            pred_graph=pred_graph,
            out_path=worst_dir / f"{idx:02d}_{stem}.png",
        )

    _write_metrics_csv(per_image, out_root / "metrics.csv")
    _write_report_md(
        summary=summary,
        worst=[item[0] for item in ranked_worst],
        config=cfg,
        iou_threshold=iou_threshold,
        out_md=out_root / "report.md",
    )
    return per_image, summary
