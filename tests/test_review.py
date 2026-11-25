"""Unit tests for src/archlens/review/ role inference and architectural rules."""

from archlens.config import ReviewConfig
from archlens.graph import BBox, DiagramGraph, Edge, Node, NodeType
from archlens.review import (
    CycleDetectionRule,
    DirectClientToDatabaseRule,
    DirectServiceToServiceRule,
    MissingGatewayOrAuthRule,
    OrphanNodesRule,
    ReviewEngine,
    SinglePointOfFailureRule,
    UnlabeledNodesRule,
    infer_node_role,
    load_roles_config,
)

_BOX = BBox(x=0, y=0, w=100, h=60)


def _node(nid: str, label: str, ntype: NodeType = "box") -> Node:
    """Create a Node helper for hand-built review test graphs."""
    return Node(id=nid, type=ntype, bbox=_BOX, label=label)


def _edge(eid: str, src: str, dst: str, directed: bool = True) -> Edge:
    """Create an Edge helper for hand-built review test graphs."""
    return Edge(id=eid, source=src, target=dst, directed=directed)


def test_role_inference_from_configs_roles_yaml() -> None:
    """configs/roles.yaml infers all 8 expected roles and explains each match."""
    roles_cfg = load_roles_config()
    cases = [
        (_node("n1", "MySQL DB", "database"), "database"),
        (_node("n2", "Redis Cache"), "cache"),
        (_node("n3", "Message Queue"), "queue"),
        (_node("n4", "API Gateway"), "gateway"),
        (_node("n5", "Auth Service"), "auth"),
        (_node("n6", "Load Balancer"), "load_balancer"),
        (_node("n7", "Web Client", "ellipse"), "user_client"),
        (_node("n8", "Payment Service"), "service"),
        (_node("n9", ""), "unknown"),
    ]
    for node, expected_role in cases:
        inferred = infer_node_role(node, roles_cfg)
        assert inferred.role == expected_role
        assert inferred.reason != ""


def test_rule_single_point_of_failure() -> None:
    """SPOF rule flags a bottleneck node on all paths and passes with redundant paths."""
    spof_graph = DiagramGraph(
        nodes=[
            _node("u", "User", "ellipse"),
            _node("gw", "API Gateway"),
            _node("s1", "Order Service"),
            _node("s2", "Billing Service"),
            _node("db", "Orders DB", "database"),
        ],
        edges=[
            _edge("e1", "u", "gw"),
            _edge("e2", "gw", "s1"),
            _edge("e3", "gw", "s2"),
            _edge("e4", "s1", "db"),
            _edge("e5", "s2", "db"),
        ],
    )
    engine = ReviewEngine(rules=[SinglePointOfFailureRule()])
    report = engine.review(spof_graph)
    assert len(report.findings) == 1
    assert report.findings[0].severity == "warning"
    assert report.findings[0].node_ids == ["gw"]

    redundant_graph = DiagramGraph(
        nodes=[
            _node("u", "User", "ellipse"),
            _node("lb1", "Load Balancer A"),
            _node("lb2", "Load Balancer B"),
            _node("s1", "Service A"),
            _node("s2", "Service B"),
            _node("db", "Orders DB", "database"),
        ],
        edges=[
            _edge("e1", "u", "lb1"),
            _edge("e2", "u", "lb2"),
            _edge("e3", "lb1", "s1"),
            _edge("e4", "lb2", "s2"),
            _edge("e5", "s1", "db"),
            _edge("e6", "s2", "db"),
        ],
    )
    assert engine.review(redundant_graph).findings == []


def test_rule_direct_client_to_database() -> None:
    """Direct client-to-database connection produces an error finding."""
    bad_graph = DiagramGraph(
        nodes=[_node("c", "Mobile App", "ellipse"), _node("db", "MySQL DB", "database")],
        edges=[_edge("e1", "c", "db")],
    )
    engine = ReviewEngine(rules=[DirectClientToDatabaseRule()])
    report = engine.review(bad_graph)
    assert len(report.findings) == 1
    assert report.findings[0].severity == "error"
    assert report.findings[0].node_ids == ["c", "db"]

    safe_graph = DiagramGraph(
        nodes=[
            _node("c", "Mobile App", "ellipse"),
            _node("api", "Order Service"),
            _node("db", "MySQL DB", "database"),
        ],
        edges=[_edge("e1", "c", "api"), _edge("e2", "api", "db")],
    )
    assert engine.review(safe_graph).findings == []


def test_rule_missing_gateway_or_auth() -> None:
    """User-to-service path without gateway or auth produces a warning."""
    unguarded = DiagramGraph(
        nodes=[
            _node("u", "Web Client", "ellipse"),
            _node("lb", "Load Balancer"),
            _node("svc", "Payment Service"),
        ],
        edges=[_edge("e1", "u", "lb"), _edge("e2", "lb", "svc")],
    )
    engine = ReviewEngine(rules=[MissingGatewayOrAuthRule()])
    report = engine.review(unguarded)
    assert len(report.findings) == 1
    assert report.findings[0].severity == "warning"
    assert report.findings[0].node_ids == ["u", "svc"]
    assert report.findings[0].evidence["path"] == ["u", "lb", "svc"]

    guarded = DiagramGraph(
        nodes=[
            _node("u", "Web Client", "ellipse"),
            _node("gw", "API Gateway"),
            _node("svc", "Payment Service"),
        ],
        edges=[_edge("e1", "u", "gw"), _edge("e2", "gw", "svc")],
    )
    assert engine.review(guarded).findings == []


def test_rule_direct_service_to_service_respects_config() -> None:
    """Direct service-to-service calls flag when require_queue_between_services=True."""
    graph = DiagramGraph(
        nodes=[_node("s1", "Order Service"), _node("s2", "Payment Service")],
        edges=[_edge("e1", "s1", "s2")],
    )

    enabled_engine = ReviewEngine(
        config=ReviewConfig(require_queue_between_services=True),
        rules=[DirectServiceToServiceRule()],
    )
    report = enabled_engine.review(graph)
    assert len(report.findings) == 1
    assert report.findings[0].severity == "warning"
    assert report.findings[0].node_ids == ["s1", "s2"]

    disabled_engine = ReviewEngine(
        config=ReviewConfig(require_queue_between_services=False),
        rules=[DirectServiceToServiceRule()],
    )
    assert disabled_engine.review(graph).findings == []

    queued_graph = DiagramGraph(
        nodes=[
            _node("s1", "Order Service"),
            _node("q", "Message Queue"),
            _node("s2", "Payment Service"),
        ],
        edges=[_edge("e1", "s1", "q"), _edge("e2", "q", "s2")],
    )
    assert enabled_engine.review(queued_graph).findings == []


def test_rule_orphan_nodes() -> None:
    """Disconnected nodes trigger orphan_node warnings."""
    graph = DiagramGraph(
        nodes=[
            _node("s1", "Auth Service"),
            _node("db", "Users DB", "database"),
            _node("orphan", "Legacy Worker"),
        ],
        edges=[_edge("e1", "s1", "db")],
    )
    engine = ReviewEngine(rules=[OrphanNodesRule()])
    report = engine.review(graph)
    assert len(report.findings) == 1
    assert report.findings[0].severity == "warning"
    assert report.findings[0].node_ids == ["orphan"]


def test_rule_unlabeled_nodes() -> None:
    """Empty-label nodes trigger unlabeled_node info findings."""
    graph = DiagramGraph(
        nodes=[_node("n1", "API Gateway"), _node("n2", "  ")],
        edges=[_edge("e1", "n1", "n2")],
    )
    engine = ReviewEngine(rules=[UnlabeledNodesRule()])
    report = engine.review(graph)
    assert len(report.findings) == 1
    assert report.findings[0].severity == "info"
    assert report.findings[0].node_ids == ["n2"]


def test_rule_cycles_flagged_as_info() -> None:
    """Directed dependency cycles are flagged with severity 'info'."""
    graph = DiagramGraph(
        nodes=[
            _node("a", "Order Service"),
            _node("b", "Billing Service"),
            _node("c", "Audit Service"),
        ],
        edges=[
            _edge("e1", "a", "b"),
            _edge("e2", "b", "c"),
            _edge("e3", "c", "a"),
        ],
    )
    engine = ReviewEngine(rules=[CycleDetectionRule()])
    report = engine.review(graph)
    assert len(report.findings) == 1
    assert report.findings[0].severity == "info"
    assert report.findings[0].node_ids == ["a", "b", "c"]


def test_review_engine_is_deterministic_across_input_ordering() -> None:
    """ReviewEngine emits identical findings regardless of node/edge list ordering."""
    nodes = [
        _node("u", "User", "ellipse"),
        _node("s1", "Order Service"),
        _node("s2", "Payment Service"),
        _node("db", "MySQL DB", "database"),
        _node("blank", ""),
    ]
    edges = [
        _edge("e1", "u", "db"),
        _edge("e2", "u", "s1"),
        _edge("e3", "s1", "s2"),
        _edge("e4", "s2", "s1"),
    ]
    graph_a = DiagramGraph(nodes=nodes, edges=edges)
    graph_b = DiagramGraph(nodes=list(reversed(nodes)), edges=list(reversed(edges)))

    engine = ReviewEngine()
    report_a = engine.review(graph_a)
    report_b = engine.review(graph_b)

    assert report_a.findings == report_b.findings
    assert report_a.summary == report_b.summary
    assert report_a.summary["error"] >= 1
    assert report_a.summary["warning"] >= 1
    assert report_a.summary["info"] >= 2

