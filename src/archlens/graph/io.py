"""JSON serialization and loading helpers for :class:`DiagramGraph`."""

from pathlib import Path

from archlens.graph.models import DiagramGraph


def save_graph_json(graph: DiagramGraph, path: Path, indent: int = 2) -> Path:
    """Serialize a :class:`DiagramGraph` to a UTF-8 JSON file.

    Args:
        graph: The diagram graph to serialize.
        path: Output JSON file path.
        indent: Indentation width for pretty-printed JSON.

    Returns:
        The resolved output :class:`Path`.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(graph.model_dump_json(indent=indent) + "\n", encoding="utf-8")
    return out_path


def load_graph_json(path: Path) -> DiagramGraph:
    """Load and validate a :class:`DiagramGraph` from a JSON file.

    Args:
        path: Path to a JSON file serialized from :class:`DiagramGraph`.

    Returns:
        A validated :class:`DiagramGraph` instance.
    """
    raw = Path(path).read_text(encoding="utf-8")
    return DiagramGraph.model_validate_json(raw)

