"""Rule engine orchestrator for deterministic architectural design review."""

from collections.abc import Sequence
from pathlib import Path

import networkx as nx

from archlens.config import ReviewConfig
from archlens.graph import DiagramGraph
from archlens.review.base import ReviewContext, ReviewRule
from archlens.review.models import Finding, ReviewReport, Severity
from archlens.review.roles import RolesConfig, infer_node_role, load_roles_config
from archlens.review.rules import (
    CycleDetectionRule,
    DirectClientToDatabaseRule,
    DirectServiceToServiceRule,
    MissingGatewayOrAuthRule,
    OrphanNodesRule,
    SinglePointOfFailureRule,
    UnlabeledNodesRule,
)

_SEVERITY_RANK: dict[Severity, int] = {"error": 0, "warning": 1, "info": 2}


def default_rules() -> list[ReviewRule]:
    """Return the default ordered list of ArchLens architectural review rules."""
    return [
        DirectClientToDatabaseRule(),
        MissingGatewayOrAuthRule(),
        SinglePointOfFailureRule(),
        DirectServiceToServiceRule(),
        OrphanNodesRule(),
        UnlabeledNodesRule(),
        CycleDetectionRule(),
    ]


class ReviewEngine:
    """Evaluates a :class:`DiagramGraph` against a deterministic set of rules."""

    def __init__(
        self,
        config: ReviewConfig | None = None,
        roles_config: RolesConfig | None = None,
        rules: Sequence[ReviewRule] | None = None,
    ) -> None:
        """Initialize the review engine with optional config, role map, and rules."""
        self.config = config or ReviewConfig()
        if roles_config is not None:
            self.roles_config = roles_config
        else:
            roles_path = (
                Path(self.config.roles_config_path)
                if self.config.roles_config_path
                else None
            )
            self.roles_config = load_roles_config(roles_path)
        self.rules: list[ReviewRule] = list(rules) if rules is not None else default_rules()

    def build_context(self, graph: DiagramGraph) -> ReviewContext:
        """Build a deterministic :class:`ReviewContext` from ``graph``."""
        sorted_nodes = sorted(graph.nodes, key=lambda n: n.id)
        sorted_edges = sorted(graph.edges, key=lambda e: e.id)

        node_map = {node.id: node for node in sorted_nodes}
        roles = {
            node.id: infer_node_role(node, self.roles_config) for node in sorted_nodes
        }

        nx_graph = nx.DiGraph()
        directed_graph = nx.DiGraph()
        for node in sorted_nodes:
            nx_graph.add_node(node.id)
            directed_graph.add_node(node.id)

        for edge in sorted_edges:
            nx_graph.add_edge(edge.source, edge.target, id=edge.id)
            if edge.directed:
                directed_graph.add_edge(edge.source, edge.target, id=edge.id)
            else:
                nx_graph.add_edge(edge.target, edge.source, id=edge.id)

        return ReviewContext(
            graph=graph,
            node_map=node_map,
            roles=roles,
            nx_graph=nx_graph,
            directed_graph=directed_graph,
            config=self.config,
        )

    def review(self, graph: DiagramGraph) -> ReviewReport:
        """Run all configured rules on ``graph`` and return a sorted :class:`ReviewReport`."""
        context = self.build_context(graph)
        findings: list[Finding] = []
        for rule in self.rules:
            findings.extend(rule.evaluate(context))

        findings.sort(
            key=lambda f: (_SEVERITY_RANK[f.severity], f.rule_id, f.node_ids, f.message)
        )
        summary: dict[Severity, int] = {"error": 0, "warning": 0, "info": 0}
        for finding in findings:
            summary[finding.severity] += 1

        return ReviewReport(
            findings=findings,
            node_roles=context.roles,
            summary=summary,
        )

