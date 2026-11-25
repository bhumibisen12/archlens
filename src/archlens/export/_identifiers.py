"""Shared identifier sanitization helpers for text graph exporters."""

import re

from archlens.graph.models import Node


def sanitize_node_ids(
    nodes: list[Node],
    reserved_ids: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Build a collision-free mapping from node IDs to target-safe identifiers.

    Args:
        nodes: Nodes whose IDs should be sanitized.
        reserved_ids: Case-insensitive reserved keywords to prefix with ``n_``.

    Returns:
        Mapping from original ``node.id`` to unique sanitized identifiers.
    """
    mapping: dict[str, str] = {}
    used: set[str] = set()

    for node in nodes:
        candidate = re.sub(r"[^A-Za-z0-9_]", "_", node.id).strip("_")
        if not candidate or candidate[0].isdigit() or candidate.lower() in reserved_ids:
            candidate = f"n_{candidate}" if candidate else "n_node"

        unique = candidate
        suffix = 2
        while unique in used:
            unique = f"{candidate}_{suffix}"
            suffix += 1

        used.add(unique)
        mapping[node.id] = unique

    return mapping

