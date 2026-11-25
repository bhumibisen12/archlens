"""Mermaid ``flowchart LR`` exporter with safe label escaping."""

from archlens.export._identifiers import sanitize_node_ids
from archlens.graph.models import DiagramGraph, NodeType

_RESERVED_IDS = frozenset(
    {"end", "subgraph", "graph", "flowchart", "style", "class", "click", "call", "href"}
)

_MERMAID_ESCAPE_MAP: dict[str, str] = {
    '"': "#quot;",
    "#": "#35;",
    "&": "#amp;",
    "<": "#lt;",
    ">": "#gt;",
    "[": "#91;",
    "]": "#93;",
    "(": "#40;",
    ")": "#41;",
    "{": "#123;",
    "}": "#125;",
    "|": "#124;",
    ";": "#59;",
    "`": "#96;",
}


def escape_mermaid_label(text: str) -> str:
    """Collapse newlines and escape Mermaid control characters in a label.

    Args:
        text: Raw node or edge label text.

    Returns:
        Safe single-line string suitable inside Mermaid quoted labels.
    """
    collapsed = " ".join(text.split())
    return "".join(_MERMAID_ESCAPE_MAP.get(ch, ch) for ch in collapsed)


def _format_node(safe_id: str, node_type: NodeType, safe_label: str) -> str:
    """Format a Mermaid node declaration according to its shape type."""
    if node_type == "database":
        return f'    {safe_id}[("{safe_label}")]'
    if node_type == "ellipse":
        return f'    {safe_id}(["{safe_label}"])'
    return f'    {safe_id}["{safe_label}"]'


def to_mermaid(graph: DiagramGraph) -> str:
    """Export a :class:`DiagramGraph` as a Mermaid ``flowchart LR`` diagram.

    Args:
        graph: The diagram graph to export.

    Returns:
        Mermaid flowchart definition string.
    """
    lines: list[str] = ["flowchart LR"]
    id_map = sanitize_node_ids(graph.nodes, reserved_ids=_RESERVED_IDS)

    for node in graph.nodes:
        raw_label = node.label if node.label.strip() else node.id
        safe_label = escape_mermaid_label(raw_label)
        lines.append(_format_node(id_map[node.id], node.type, safe_label))

    for edge in graph.edges:
        src = id_map[edge.source]
        dst = id_map[edge.target]
        connector = "-->" if edge.directed else "---"
        if edge.label and edge.label.strip():
            safe_edge_label = escape_mermaid_label(edge.label)
            lines.append(f'    {src} {connector}|"{safe_edge_label}"| {dst}')
        else:
            lines.append(f"    {src} {connector} {dst}")

    return "\n".join(lines) + "\n"
