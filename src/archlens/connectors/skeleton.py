"""Skeletonization, pixel-graph construction, spur pruning, and polyline tracing."""

import heapq
import math
from collections import defaultdict, deque

import cv2
import numpy as np
from skimage.morphology import skeletonize

from archlens.graph import Point

Coord = tuple[int, int]  # (y, x) pixel coordinate in skeleton image

_NEIGHBOR_OFFSETS: tuple[Coord, ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def remove_small_components(ink_mask: np.ndarray, min_span_px: float) -> np.ndarray:
    """Drop tiny isolated ink blobs (e.g., stray letters) shorter than ``min_span_px``."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (ink_mask > 0).astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return ink_mask

    keep = np.zeros(num_labels, dtype=bool)
    for label_id in range(1, num_labels):
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if math.hypot(w, h) >= min_span_px:
            keep[label_id] = True

    return np.where(keep[labels], 255, 0).astype(np.uint8)


def skeletonize_ink(ink_mask: np.ndarray) -> np.ndarray:
    """Compute a 1-pixel topological skeleton from a binary ink mask."""
    binary_bool = ink_mask > 0
    if not np.any(binary_bool):
        return np.zeros(ink_mask.shape[:2], dtype=bool)
    return skeletonize(binary_bool)


def build_pixel_graph(skeleton: np.ndarray) -> dict[Coord, set[Coord]]:
    """Convert a boolean skeleton image into an 8-connected pixel adjacency graph."""
    coords = np.argwhere(skeleton)
    pixel_set: set[Coord] = {(int(y), int(x)) for y, x in coords}
    adj: dict[Coord, set[Coord]] = {pt: set() for pt in pixel_set}

    for y, x in pixel_set:
        neighbors = adj[(y, x)]
        for dy, dx in _NEIGHBOR_OFFSETS:
            nb = (y + dy, x + dx)
            if nb in pixel_set:
                neighbors.add(nb)

    return adj


def prune_short_spurs(adj: dict[Coord, set[Coord]], max_length_px: int) -> None:
    """Iteratively prune short degree-1 spurs attached to junctions (degree >= 3)."""
    if max_length_px <= 0:
        return

    changed = True
    while changed:
        changed = False
        leaves = [pt for pt, nbrs in adj.items() if len(nbrs) == 1]
        for leaf in leaves:
            if leaf not in adj or len(adj[leaf]) != 1:
                continue

            branch: list[Coord] = [leaf]
            prev = leaf
            curr = next(iter(adj[leaf]))
            while len(adj.get(curr, ())) == 2 and len(branch) <= max_length_px:
                branch.append(curr)
                n_a, n_b = adj[curr]
                nxt = n_b if n_a == prev else n_a
                prev, curr = curr, nxt

            if curr in adj and len(adj[curr]) >= 3 and len(branch) <= max_length_px:
                for node in branch:
                    for nb in list(adj.get(node, ())):
                        adj[nb].discard(node)
                    adj.pop(node, None)
                changed = True


def _unit_tangent(chain: list[Coord], at_end: bool, lookback: int = 12) -> tuple[float, float]:
    """Estimate unit tangent pointing *into* ``chain`` from its start or end."""
    if len(chain) < 2:
        return (0.0, 0.0)
    if at_end:
        p_tip = chain[-1]
        p_ref = chain[max(0, len(chain) - 1 - lookback)]
    else:
        p_tip = chain[0]
        p_ref = chain[min(len(chain) - 1, lookback)]
    dy = float(p_ref[0] - p_tip[0])
    dx = float(p_ref[1] - p_tip[1])
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return (0.0, 0.0)
    return (dy / norm, dx / norm)


def _contract_junction_clusters(
    adj: dict[Coord, set[Coord]],
) -> tuple[dict[Coord, int], list[list[Coord]], set[int]]:
    """Collapse 8-connected junction pixel clusters into single topological vertices.

    Returns:
        ``(coord_to_vertex_id, chains_with_vertex_endpoints, endpoint_vertex_ids)``
        where each chain is a list of ``(y, x)`` coordinates and endpoints map to vertex IDs.
    """
    if not adj:
        return {}, [], set()

    junction_pixels = {pt for pt, nbrs in adj.items() if len(nbrs) >= 3}
    endpoint_pixels = {pt for pt, nbrs in adj.items() if len(nbrs) == 1}

    coord_to_vid: dict[Coord, int] = {}
    next_vid = 0

    # Cluster adjacent junction pixels into single topological vertices.
    visited_j: set[Coord] = set()
    for jp in sorted(junction_pixels):
        if jp in visited_j:
            continue
        queue: deque[Coord] = deque([jp])
        visited_j.add(jp)
        while queue:
            cur = queue.popleft()
            coord_to_vid[cur] = next_vid
            for nb in adj[cur]:
                if nb in junction_pixels and nb not in visited_j:
                    visited_j.add(nb)
                    queue.append(nb)
        next_vid += 1

    endpoint_vids: set[int] = set()
    for ep in sorted(endpoint_pixels):
        coord_to_vid[ep] = next_vid
        endpoint_vids.add(next_vid)
        next_vid += 1

    # Trace every edge between topological vertices (ignoring internal cluster links).
    visited_edges: set[frozenset[Coord]] = set()
    chains: list[list[Coord]] = []

    for start in sorted(coord_to_vid):
        start_vid = coord_to_vid[start]
        for nb in sorted(adj[start]):
            if coord_to_vid.get(nb) == start_vid:
                continue  # Internal edge inside the same junction cluster.
            edge_key = frozenset((start, nb))
            if edge_key in visited_edges:
                continue

            visited_edges.add(edge_key)
            chain: list[Coord] = [start, nb]
            prev, curr = start, nb
            while curr not in coord_to_vid:
                nxt_candidates = [pt for pt in adj[curr] if pt != prev]
                if not nxt_candidates:
                    break
                nxt = nxt_candidates[0]
                step_key = frozenset((curr, nxt))
                if step_key in visited_edges:
                    break
                visited_edges.add(step_key)
                chain.append(nxt)
                prev, curr = curr, nxt

            if len(chain) >= 2 and chain[-1] in coord_to_vid:
                chains.append(chain)

    # Drop tiny self-loops (e.g., hollow/filled arrowhead micro-loops) before building endpoints.
    filtered_chains = [
        ch
        for ch in chains
        if not (coord_to_vid[ch[0]] == coord_to_vid[ch[-1]] and _chain_arc_length(ch) <= 30.0)
    ]

    # Recompute true endpoint vertices after removing micro-loops.
    incident_counts: dict[int, int] = defaultdict(int)
    for ch in filtered_chains:
        incident_counts[coord_to_vid[ch[0]]] += 1
        incident_counts[coord_to_vid[ch[-1]]] += 1
    endpoint_vids = {vid for vid, cnt in incident_counts.items() if cnt == 1}

    return coord_to_vid, filtered_chains, endpoint_vids


def _bridge_label_gaps(
    coord_to_vid: dict[Coord, int],
    chains: list[list[Coord]],
    endpoint_vids: set[int],
    max_gap_px: float = 42.0,
    max_angle_deg: float = 25.0,
) -> list[list[Coord]]:
    """Add synthetic bridge chains between collinear degree-1 ends cut by a label."""
    if len(endpoint_vids) < 2:
        return chains

    ep_chains: list[tuple[Coord, tuple[float, float]]] = []
    for ch in chains:
        v_start = coord_to_vid[ch[0]]
        v_end = coord_to_vid[ch[-1]]
        if v_start in endpoint_vids:
            ep_chains.append((ch[0], _unit_tangent(ch, False)))
        if v_end in endpoint_vids and v_end != v_start:
            ep_chains.append((ch[-1], _unit_tangent(ch, True)))

    used: set[int] = set()
    bridges: list[list[Coord]] = []
    candidates: list[tuple[float, int, int]] = []
    for i in range(len(ep_chains)):
        pt_a, tang_a = ep_chains[i]
        for j in range(i + 1, len(ep_chains)):
            pt_b, tang_b = ep_chains[j]
            dy = float(pt_b[0] - pt_a[0])
            dx = float(pt_b[1] - pt_a[1])
            dist = math.hypot(dy, dx)
            if dist < 2.0 or dist > max_gap_px:
                continue
            ux, uy = dx / dist, dy / dist
            cos_a = max(-1.0, min(1.0, -(tang_a[0] * uy + tang_a[1] * ux)))
            cos_b = max(-1.0, min(1.0, tang_b[0] * uy + tang_b[1] * ux))
            ang_a = math.degrees(math.acos(cos_a))
            ang_b = math.degrees(math.acos(cos_b))
            if ang_a <= max_angle_deg and ang_b <= max_angle_deg:
                candidates.append((dist + ang_a + ang_b, i, j))

    candidates.sort(key=lambda item: item[0])
    for _, i, j in candidates:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        bridges.append([ep_chains[i][0], ep_chains[j][0]])

    return chains + bridges


def _chain_arc_length(chain: list[Coord]) -> float:
    """Compute pixel arc length of a chain."""
    return sum(
        math.hypot(chain[i + 1][1] - chain[i][1], chain[i + 1][0] - chain[i][0])
        for i in range(len(chain) - 1)
    )


def trace_polylines(
    adj: dict[Coord, set[Coord]],
    max_crossing_angle_deg: float = 35.0,
    terminal_coords: set[Coord] | None = None,
) -> list[list[tuple[int, int]]]:
    """Trace smooth paths through junctions between terminal endpoints.

    Args:
        adj: 8-connected pixel adjacency graph.
        max_crossing_angle_deg: Maximum angular deviation considered a straight crossing.
        terminal_coords: Optional set of pixel coordinates (e.g., snapped near nodes)
            that act as valid path starts/ends. When ``None``, all degree-1 endpoints
            are treated as terminals.

    Returns:
        List of ``(x, y)`` polylines.
    """
    coord_to_vid, chains, endpoint_vids = _contract_junction_clusters(adj)
    if not chains:
        return []

    chains = _bridge_label_gaps(coord_to_vid, chains, endpoint_vids)

    # Build chain-level adjacency: vertex_id -> list of (chain_idx, at_end, other_vid, tangent_in)
    v_adj: dict[int, list[tuple[int, bool, int, tuple[float, float]]]] = defaultdict(list)
    chain_lengths: list[float] = []
    for cid, ch in enumerate(chains):
        v0 = coord_to_vid[ch[0]]
        v1 = coord_to_vid[ch[-1]]
        chain_lengths.append(_chain_arc_length(ch))
        v_adj[v0].append((cid, False, v1, _unit_tangent(ch, False)))
        v_adj[v1].append((cid, True, v0, _unit_tangent(ch, True)))

    if terminal_coords is not None:
        terminals = {coord_to_vid[c] for c in terminal_coords if c in coord_to_vid}
    else:
        terminals = set(endpoint_vids)

    # Search low-turn-penalty paths between pairs of distinct terminal vertices.
    traced: list[list[tuple[int, int]]] = []
    seen_pairs: set[frozenset[int]] = set()
    max_hops = 12

    for start_vid in sorted(terminals):
        # Dijkstra over (cost, hops, curr_vid, incoming_tangent_outward, path_steps)
        # path_steps stores (cid, reverse_chain)
        Step = tuple[int, bool]
        QueueItem = tuple[float, int, int, tuple[float, float] | None, tuple[Step, ...]]
        pq: list[QueueItem] = [(0.0, 0, start_vid, None, ())]
        best_cost: dict[tuple[int, int], float] = {}

        while pq:
            cost, hops, curr_vid, prev_dir, steps = heapq.heappop(pq)
            if hops > 0 and curr_vid in terminals and curr_vid != start_vid:
                pair_key = frozenset((start_vid, curr_vid))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    coords: list[Coord] = []
                    for cid, rev in steps:
                        seg = list(reversed(chains[cid])) if rev else chains[cid]
                        if not coords:
                            coords.extend(seg)
                        else:
                            coords.extend(seg[1:])
                    traced.append([(x, y) for y, x in coords])
                continue

            if hops >= max_hops:
                continue

            used_cids = {cid for cid, _ in steps}
            for cid, at_end, nxt_vid, tang_at_curr in v_adj[curr_vid]:
                if cid in used_cids:
                    continue
                # Tangents point *into* the chain from curr_vid; compare prev_dir with tang_at_curr.
                turn_penalty = 0.0
                if prev_dir is not None:
                    dot = max(
                        -1.0,
                        min(1.0, prev_dir[0] * tang_at_curr[0] + prev_dir[1] * tang_at_curr[1]),
                    )
                    turn_deg = math.degrees(math.acos(dot))
                    is_crossing = len(v_adj[curr_vid]) >= 4
                    max_allowed_turn = max_crossing_angle_deg if is_crossing else 118.0
                    if turn_deg > max_allowed_turn:
                        continue  # Never turn sharply at crossings; forbid U-turns.
                    if turn_deg <= max_crossing_angle_deg:
                        turn_penalty = turn_deg * 0.5
                    else:
                        turn_penalty = 500.0 + turn_deg

                # Outward tangent arriving at nxt_vid.
                exit_in_tang = _unit_tangent(chains[cid], not at_end)
                nxt_prev_dir = (-exit_in_tang[0], -exit_in_tang[1])
                nxt_cost = cost + chain_lengths[cid] + turn_penalty
                state_key = (nxt_vid, cid)
                if nxt_cost < best_cost.get(state_key, float("inf")):
                    best_cost[state_key] = nxt_cost
                    heapq.heappush(
                        pq,
                        (nxt_cost, hops + 1, nxt_vid, nxt_prev_dir, (*steps, (cid, at_end))),
                    )

    return traced


def simplify_polyline(points_xy: list[tuple[int, int]], epsilon_px: float) -> list[Point]:
    """Simplify a dense pixel trace into a clean list of :class:`Point` vertices."""
    if len(points_xy) <= 2:
        return [Point(x=float(x), y=float(y)) for x, y in points_xy]

    arr = np.array(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(arr, epsilon=epsilon_px, closed=False)
    simplified = [Point(x=float(pt[0][0]), y=float(pt[0][1])) for pt in approx]
    if len(simplified) < 2:
        return [
            Point(x=float(points_xy[0][0]), y=float(points_xy[0][1])),
            Point(x=float(points_xy[-1][0]), y=float(points_xy[-1][1])),
        ]
    return simplified
