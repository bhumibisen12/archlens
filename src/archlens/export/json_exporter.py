"""JSON exporter for :class:`DiagramGraph`."""

from archlens.graph.models import DiagramGraph


def to_json(graph: DiagramGraph, indent: int = 2) -> str:
    """Serialize a :class:`DiagramGraph` to a formatted JSON string.

    Args:
        graph: The diagram graph to export.
        indent: JSON indentation width.

    Returns:
        UTF-8 JSON string representation of ``graph``.
    """
    return graph.model_dump_json(indent=indent)

