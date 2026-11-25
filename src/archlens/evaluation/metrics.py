"""Quantitative evaluation metrics for ArchLens diagram parsing."""

from collections import Counter

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import linear_sum_assignment

from archlens.graph import BBox, DiagramGraph, Node


def bbox_iou(a: BBox, b: BBox) -> float:
    """Compute Intersection-over-Union between two bounding boxes."""
    ix0 = max(a.x, b.x)
    iy0 = max(a.y, b.y)
    ix1 = min(a.x + a.w, b.x + b.w)
    iy1 = min(a.y + a.h, b.y + b.h)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = float((ix1 - ix0) * (iy1 - iy0))
    union = float(a.area + b.area) - inter
    return inter / max(1.0, union)


def levenshtein_distance(a: str, b: str) -> int:
    """Compute character-level Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for ch_a in a:
        curr = [prev[0] + 1]
        for j, ch_b in enumerate(b):
            cost = 0 if ch_a == ch_b else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def normalized_edit_similarity(pred: str, gt: str) -> float:
    """Return normalized edit similarity ``1 - edit_distance / max_len`` in ``[0, 1]``."""
    p = " ".join(pred.strip().lower().split())
    g = " ".join(gt.strip().lower().split())
    max_len = max(len(p), len(g))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(p, g)
    return max(0.0, 1.0 - (float(dist) / float(max_len)))


def _prf(matches: int, pred_total: int, gt_total: int) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 with safe zero handling."""
    if pred_total == 0 and gt_total == 0:
        return 1.0, 1.0, 1.0
    precision = float(matches) / float(pred_total) if pred_total > 0 else 0.0
    recall = float(matches) / float(gt_total) if gt_total > 0 else 0.0
    if precision + recall == 0.0:
        return precision, recall, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def match_nodes_hungarian(
    gt_nodes: list[Node],
    pred_nodes: list[Node],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Globally match predicted nodes to ground-truth nodes using Hungarian assignment.

    Args:
        gt_nodes: Ground-truth diagram nodes.
        pred_nodes: Predicted diagram nodes.
        iou_threshold: Minimum IoU required for a valid assignment.

    Returns:
        ``(matched_pairs, unmatched_gt_indices, unmatched_pred_indices)`` where each
        item in ``matched_pairs`` is ``(gt_idx, pred_idx, iou)``.
    """
    if not gt_nodes or not pred_nodes:
        return [], list(range(len(gt_nodes))), list(range(len(pred_nodes)))

    iou_matrix = np.zeros((len(gt_nodes), len(pred_nodes)), dtype=np.float64)
    for i, gt_node in enumerate(gt_nodes):
        for j, pred_node in enumerate(pred_nodes):
            iou_matrix[i, j] = bbox_iou(gt_node.bbox, pred_node.bbox)

    row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)
    matched: list[tuple[int, int, float]] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    for r, c in zip(row_ind, col_ind, strict=True):
        score = float(iou_matrix[r, c])
        if score >= iou_threshold:
            matched.append((int(r), int(c), score))
            matched_gt.add(int(r))
            matched_pred.add(int(c))

    unmatched_gt = [i for i in range(len(gt_nodes)) if i not in matched_gt]
    unmatched_pred = [j for j in range(len(pred_nodes)) if j not in matched_pred]
    return matched, unmatched_gt, unmatched_pred


class ImageMetrics(BaseModel):
    """Evaluation metrics for a single diagram image."""

    image_name: str
    gt_nodes: int
    pred_nodes: int
    matched_nodes: int
    node_precision: float
    node_recall: float
    node_f1: float
    mean_label_similarity: float
    gt_edges: int
    pred_edges: int
    directed_edge_matches: int
    directed_edge_precision: float
    directed_edge_recall: float
    directed_edge_f1: float
    undirected_edge_matches: int
    undirected_edge_precision: float
    undirected_edge_recall: float
    undirected_edge_f1: float
    exact_graph_match: bool
    timings_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0


class AggregateMetrics(BaseModel):
    """Micro-averaged dataset evaluation metrics and latency summary."""

    num_images: int
    gt_nodes: int
    pred_nodes: int
    matched_nodes: int
    node_precision: float
    node_recall: float
    node_f1: float
    mean_label_similarity: float
    gt_edges: int
    pred_edges: int
    directed_edge_matches: int
    directed_edge_precision: float
    directed_edge_recall: float
    directed_edge_f1: float
    undirected_edge_matches: int
    undirected_edge_precision: float
    undirected_edge_recall: float
    undirected_edge_f1: float
    exact_graph_match_rate: float
    mean_stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    mean_total_latency_ms: float = 0.0


def compute_image_metrics(
    image_name: str,
    gt_graph: DiagramGraph,
    pred_graph: DiagramGraph,
    iou_threshold: float = 0.5,
    timings_ms: dict[str, float] | None = None,
) -> ImageMetrics:
    """Evaluate a predicted :class:`DiagramGraph` against a ground-truth graph."""
    matched, _, _ = match_nodes_hungarian(
        gt_graph.nodes,
        pred_graph.nodes,
        iou_threshold=iou_threshold,
    )
    node_p, node_r, node_f1 = _prf(len(matched), len(pred_graph.nodes), len(gt_graph.nodes))

    if matched:
        sims = [
            normalized_edit_similarity(
                pred_graph.nodes[p_idx].label,
                gt_graph.nodes[g_idx].label,
            )
            for g_idx, p_idx, _ in matched
        ]
        label_sim = sum(sims) / float(len(sims))
    elif not gt_graph.nodes and not pred_graph.nodes:
        label_sim = 1.0
    else:
        label_sim = 0.0

    pred_to_gt = {
        pred_graph.nodes[p_idx].id: gt_graph.nodes[g_idx].id
        for g_idx, p_idx, _ in matched
    }

    gt_dir_counts = Counter((e.source, e.target) for e in gt_graph.edges if e.directed)
    gt_dir_total = sum(gt_dir_counts.values())
    pred_dir_edges = [e for e in pred_graph.edges if e.directed]
    pred_dir_counts = Counter(
        (pred_to_gt[e.source], pred_to_gt[e.target])
        for e in pred_dir_edges
        if e.source in pred_to_gt and e.target in pred_to_gt
    )
    dir_matches = sum(
        min(gt_dir_counts[pair], pred_dir_counts[pair]) for pair in gt_dir_counts
    )
    dir_p, dir_r, dir_f1 = _prf(dir_matches, len(pred_dir_edges), gt_dir_total)

    gt_undir_counts = Counter(frozenset((e.source, e.target)) for e in gt_graph.edges)
    pred_undir_counts = Counter(
        frozenset((pred_to_gt[e.source], pred_to_gt[e.target]))
        for e in pred_graph.edges
        if e.source in pred_to_gt and e.target in pred_to_gt
    )
    undir_matches = sum(
        min(gt_undir_counts[pair], pred_undir_counts[pair]) for pair in gt_undir_counts
    )
    undir_p, undir_r, undir_f1 = _prf(undir_matches, len(pred_graph.edges), len(gt_graph.edges))

    exact_match = (
        len(matched) == len(gt_graph.nodes) == len(pred_graph.nodes)
        and dir_matches == gt_dir_total == len(pred_graph.edges)
    )
    stage_timings = dict(timings_ms if timings_ms is not None else pred_graph.metadata.timings_ms)
    total_ms = round(sum(stage_timings.values()), 2)

    return ImageMetrics(
        image_name=image_name,
        gt_nodes=len(gt_graph.nodes),
        pred_nodes=len(pred_graph.nodes),
        matched_nodes=len(matched),
        node_precision=round(node_p, 4),
        node_recall=round(node_r, 4),
        node_f1=round(node_f1, 4),
        mean_label_similarity=round(label_sim, 4),
        gt_edges=len(gt_graph.edges),
        pred_edges=len(pred_graph.edges),
        directed_edge_matches=dir_matches,
        directed_edge_precision=round(dir_p, 4),
        directed_edge_recall=round(dir_r, 4),
        directed_edge_f1=round(dir_f1, 4),
        undirected_edge_matches=undir_matches,
        undirected_edge_precision=round(undir_p, 4),
        undirected_edge_recall=round(undir_r, 4),
        undirected_edge_f1=round(undir_f1, 4),
        exact_graph_match=exact_match,
        timings_ms=stage_timings,
        total_latency_ms=total_ms,
    )


def aggregate_metrics(per_image: list[ImageMetrics]) -> AggregateMetrics:
    """Compute micro-averaged dataset metrics and mean latencies."""
    if not per_image:
        return AggregateMetrics(
            num_images=0,
            gt_nodes=0,
            pred_nodes=0,
            matched_nodes=0,
            node_precision=0.0,
            node_recall=0.0,
            node_f1=0.0,
            mean_label_similarity=0.0,
            gt_edges=0,
            pred_edges=0,
            directed_edge_matches=0,
            directed_edge_precision=0.0,
            directed_edge_recall=0.0,
            directed_edge_f1=0.0,
            undirected_edge_matches=0,
            undirected_edge_precision=0.0,
            undirected_edge_recall=0.0,
            undirected_edge_f1=0.0,
            exact_graph_match_rate=0.0,
        )

    gt_nodes = sum(m.gt_nodes for m in per_image)
    pred_nodes = sum(m.pred_nodes for m in per_image)
    matched_nodes = sum(m.matched_nodes for m in per_image)
    node_p, node_r, node_f1 = _prf(matched_nodes, pred_nodes, gt_nodes)

    weighted_sim = sum(m.mean_label_similarity * m.matched_nodes for m in per_image)
    mean_label_sim = (
        weighted_sim / float(matched_nodes)
        if matched_nodes > 0
        else sum(m.mean_label_similarity for m in per_image) / float(len(per_image))
    )

    gt_edges = sum(m.gt_edges for m in per_image)
    pred_edges = sum(m.pred_edges for m in per_image)
    dir_matches = sum(m.directed_edge_matches for m in per_image)
    dir_p, dir_r, dir_f1 = _prf(dir_matches, pred_edges, gt_edges)

    undir_matches = sum(m.undirected_edge_matches for m in per_image)
    undir_p, undir_r, undir_f1 = _prf(undir_matches, pred_edges, gt_edges)

    exact_rate = sum(1 for m in per_image if m.exact_graph_match) / float(len(per_image))

    stage_names = sorted({stage for m in per_image for stage in m.timings_ms})
    mean_stages = {
        stage: round(
            sum(m.timings_ms.get(stage, 0.0) for m in per_image) / float(len(per_image)),
            2,
        )
        for stage in stage_names
    }
    mean_total_ms = round(
        sum(m.total_latency_ms for m in per_image) / float(len(per_image)),
        2,
    )

    return AggregateMetrics(
        num_images=len(per_image),
        gt_nodes=gt_nodes,
        pred_nodes=pred_nodes,
        matched_nodes=matched_nodes,
        node_precision=round(node_p, 4),
        node_recall=round(node_r, 4),
        node_f1=round(node_f1, 4),
        mean_label_similarity=round(mean_label_sim, 4),
        gt_edges=gt_edges,
        pred_edges=pred_edges,
        directed_edge_matches=dir_matches,
        directed_edge_precision=round(dir_p, 4),
        directed_edge_recall=round(dir_r, 4),
        directed_edge_f1=round(dir_f1, 4),
        undirected_edge_matches=undir_matches,
        undirected_edge_precision=round(undir_p, 4),
        undirected_edge_recall=round(undir_r, 4),
        undirected_edge_f1=round(undir_f1, 4),
        exact_graph_match_rate=round(exact_rate, 4),
        mean_stage_timings_ms=mean_stages,
        mean_total_latency_ms=mean_total_ms,
    )

