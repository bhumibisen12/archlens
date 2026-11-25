"""Unit tests for ArchLens graph exporters (JSON, Mermaid, Graphviz DOT)."""

from archlens.export import to_graphviz_dot, to_json, to_mermaid
from archlens.graph import BBox, DiagramGraph, Edge, Node


def _build_export_graph() -> DiagramGraph:
    """Build a graph covering all node types, directed/undirected edges, and special chars."""
    return DiagramGraph(
        nodes=[
            Node(
                id="client-1",
                type="ellipse",
                bbox=BBox(x=10, y=10, w=100, h=50),
                label='Browser "Client"\n[v2]',
            ),
            Node(
                id="api gateway",
                type="box",
                bbox=BBox(x=200, y=10, w=120, h=60),
                label="API <Gateway> & Proxy",
            ),
            Node(
                id="1db",
                type="database",
                bbox=BBox(x=400, y=10, w=90, h=70),
                label="Main | DB #1",
            ),
            Node(
                id="end",
                type="component",
                bbox=BBox(x=400, y=120, w=110, h=50),
                label="",
            ),
        ],
        edges=[
            Edge(
                id="e1",
                source="client-1",
                target="api gateway",
                directed=True,
                label='REST [JSON] | "TLS"',
            ),
            Edge(
                id="e2",
                source="api gateway",
                target="1db",
                directed=True,
                label=None,
            ),
            Edge(
                id="e3",
                source="api gateway",
                target="end",
                directed=False,
                label="Async",
            ),
        ],
    )


def test_to_json_round_trip() -> None:
    """to_json output should deserialize cleanly back into an equal DiagramGraph."""
    graph = _build_export_graph()
    payload = to_json(graph)
    restored = DiagramGraph.model_validate_json(payload)
    assert restored == graph


def test_to_mermaid_flowchart_lr_and_safe_escaping() -> None:
    """to_mermaid should emit flowchart LR with shape syntax and escaped labels."""
    graph = _build_export_graph()
    mermaid = to_mermaid(graph)
    lines = mermaid.strip().splitlines()

    assert lines[0] == "flowchart LR"
    # Shapes and sanitized IDs (numeric-leading and reserved keyword 'end')
    assert 'client_1(["Browser #quot;Client#quot; #91;v2#93;"])' in mermaid
    assert 'api_gateway["API #lt;Gateway#gt; #amp; Proxy"]' in mermaid
    assert 'n_1db[("Main #124; DB #35;1")]' in mermaid
    assert 'n_end["end"]' in mermaid

    # Edges: directed with escaped label, directed unlabeled, undirected labeled
    assert 'client_1 -->|"REST #91;JSON#93; #124; #quot;TLS#quot;"| api_gateway' in mermaid
    assert "api_gateway --> n_1db" in mermaid
    assert 'api_gateway ---|"Async"| n_end' in mermaid

    # Ensure raw unescaped quotes/brackets inside labels do not appear
    assert 'Browser "Client"' not in mermaid
    assert 'REST [JSON]' not in mermaid


def test_to_mermaid_handles_id_collisions() -> None:
    """Nodes whose IDs collide after sanitization receive distinct Mermaid IDs."""
    graph = DiagramGraph(
        nodes=[
            Node(id="svc-a", type="box", bbox=BBox(x=0, y=0, w=50, h=50), label="First"),
            Node(id="svc a", type="box", bbox=BBox(x=100, y=0, w=50, h=50), label="Second"),
        ],
        edges=[Edge(id="e1", source="svc-a", target="svc a", directed=True)],
    )
    mermaid = to_mermaid(graph)
    assert 'svc_a["First"]' in mermaid
    assert 'svc_a_2["Second"]' in mermaid
    assert "svc_a --> svc_a_2" in mermaid


def test_to_graphviz_dot_output() -> None:
    """to_graphviz_dot should produce a valid digraph with shapes and escaped labels."""
    graph = _build_export_graph()
    dot = to_graphviz_dot(graph)

    assert dot.startswith("digraph ArchLens {")
    assert "rankdir=LR;" in dot
    assert 'client_1 [label="Browser \\"Client\\"\\n[v2]", shape=ellipse];' in dot
    assert 'api_gateway [label="API <Gateway> & Proxy", shape=box];' in dot
    assert 'n_1db [label="Main | DB #1", shape=cylinder];' in dot
    assert 'n_end [label="end", shape=component];' not in dot
    assert 'end [label="end", shape=component];' in dot
    assert 'client_1 -> api_gateway [label="REST [JSON] | \\"TLS\\""];' in dot
    assert "api_gateway -> n_1db;" in dot
    assert 'api_gateway -> end [label="Async", dir=none];' in dot
    assert dot.strip().endswith("}")
