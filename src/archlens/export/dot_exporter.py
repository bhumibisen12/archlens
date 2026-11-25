"""Graphviz DOT exporter for :class:`DiagramGraph`."""

from archlens.export._identifiers import sanitize_node_ids
from archlens.graph.models import DiagramGraph, NodeType

_DOT_RESERVED_IDS = frozenset(
    {"node", "edge", "graph", "digraph", "subgraph", "strict"}
)

_DOT_SHAPES: dict[NodeType, str] = {
    "box": "box",
    "database": "cylinder",
    "ellipse": "ellipse",
    "component": "component",
}


def escape_dot_label(text: str) -> str:
    """Escape a label string for safe inclusion in a double-quoted DOT attribute.

    Args:
        text: Raw label text.

    Returns:
        Escaped string safe inside DOT ``"..."`` quotes.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def to_graphviz_dot(graph: DiagramGraph) -> str:
    """Export a :class:`DiagramGraph` in Graphviz DOT format.

    Args:
        graph: The diagram graph to export.

    Returns:
        Graphviz ``digraph`` specification string.
    """
    lines: list[str] = ["digraph ArchLens {", "    rankdir=LR;"]
    id_map = sanitize_node_ids(graph.nodes, reserved_ids=_DOT_RESERVED_IDS)

    for node in graph.nodes:
        safe_id = id_map[node.id]
        raw_label = node.label if node.label.strip() else node.id
        safe_label = escape_dot_label(raw_label)
        shape = _DOT_SHAPES[node.type]
        lines.append(f'    {safe_id} [label="{safe_label}", shape={shape}];')

    for edge in graph.edges:
        src = id_map[edge.source]
        dst = id_map[edge.target]
        attrs: list[str] = []
        if edge.label and edge.label.strip():
            attrs.append(f'label="{escape_dot_label(edge.label)}"')
        if not edge.directed:
            attrs.append("dir=none")

        attr_str = f" [{', '.join(attrs)}]" if attrs else ""
        lines.append(f"    {src} -> {dst}{attr_str};")

    lines.append("}")
    return "\n".join(lines) + "\n"
