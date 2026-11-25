"""Built-in deterministic architectural design check rules."""

import networkx as nx

from archlens.graph import Node
from archlens.review.base import ReviewContext, ReviewRule
from archlens.review.models import Finding


def _node_display(node: Node) -> str:
    """Format a readable node identifier including its label when available."""
    label = node.label.strip()
    return f"{node.id} ({label})" if label else node.id


def _canonical_cycle(cycle: list[str]) -> list[str]:
    """Rotate a cycle so its lexicographically smallest node ID comes first."""
    min_idx = min(range(len(cycle)), key=lambda idx: cycle[idx])
    return cycle[min_idx:] + cycle[:min_idx]


class SinglePointOfFailureRule(ReviewRule):
    """Flag nodes upon which every end-to-end path depends."""

    rule_id = "single_point_of_failure"

    def evaluate(self, context: ReviewContext) -> list[Finding]:
        """Identify cut vertices whose removal breaks all source-to-sink paths."""
        graph = context.nx_graph
        if graph.number_of_nodes() < 3 or graph.number_of_edges() == 0:
            return []

        node_ids = sorted(graph.nodes)
        sources = [nid for nid in node_ids if graph.in_degree(nid) == 0]
        sinks = [nid for nid in node_ids if graph.out_degree(nid) == 0]

        reachable_pairs = [
            (src, dst)
            for src in sources
            for dst in sinks
            if src != dst and nx.has_path(graph, src, dst)
        ]

        findings: list[Finding] = []
        if reachable_pairs:
            candidates = [
                nid
                for nid in node_ids
                if graph.in_degree(nid) >= 1 and graph.out_degree(nid) >= 1
            ]
            for cand in candidates:
                reduced = graph.copy()
                reduced.remove_node(cand)
                surviving = any(
                    src != cand and dst != cand and nx.has_path(reduced, src, dst)
                    for src, dst in reachable_pairs
                )
                if not surviving:
                    node_label = _node_display(context.node_map[cand])
                    findings.append(
                        Finding(
                            rule_id=self.rule_id,
                            severity="warning",
                            message=(
                                f"Single point of failure at {node_label}: removing this node "
                                f"disconnects all {len(reachable_pairs)} source-to-sink path(s)."
                            ),
                            node_ids=[cand],
                            evidence={
                                "affected_pairs": [
                                    [src, dst] for src, dst in reachable_pairs
                                ]
                            },
                        )
                    )
            return findings

        # Fallback for cyclic graphs without degree-0 sources/sinks.
        undirected = graph.to_undirected()
        for cand in sorted(nx.articulation_points(undirected)):
            node_label = _node_display(context.node_map[cand])
            findings.append(
                Finding(
                    rule_id=self.rule_id,
                    severity="warning",
                    message=(
                        f"Single point of failure at {node_label}: removing this node "
                        "disconnects the diagram graph."
                    ),
                    node_ids=[cand],
                )
            )
        return findings


class DirectClientToDatabaseRule(ReviewRule):
    """Flag direct connections between a user/client node and a database."""

    rule_id = "direct_client_to_database"

    def evaluate(self, context: ReviewContext) -> list[Finding]:
        """Check edges for direct ``user_client -> database`` links."""
        findings: list[Finding] = []
        for edge in sorted(context.graph.edges, key=lambda item: item.id):
            src_role = context.roles[edge.source].role
            dst_role = context.roles[edge.target].role

            client_id: str | None = None
            db_id: str | None = None
            if src_role == "user_client" and dst_role == "database":
                client_id, db_id = edge.source, edge.target
            elif not edge.directed and dst_role == "user_client" and src_role == "database":
                client_id, db_id = edge.target, edge.source

            if client_id is not None and db_id is not None:
                client_desc = _node_display(context.node_map[client_id])
                db_desc = _node_display(context.node_map[db_id])
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity="error",
                        message=(
                            f"Client {client_desc} connects directly to database {db_desc} "
                            f"(via edge {edge.id}) without an intervening service or gateway."
                        ),
                        node_ids=[client_id, db_id],
                        evidence={"edge_id": edge.id},
                    )
                )
        return findings


class MissingGatewayOrAuthRule(ReviewRule):
    """Flag paths from user/client to service that lack a gateway or auth node."""

    rule_id = "missing_gateway_or_auth"

    def evaluate(self, context: ReviewContext) -> list[Finding]:
        """Inspect paths from ``user_client`` nodes to ``service`` nodes."""
        node_ids = sorted(context.node_map.keys())
        clients = [nid for nid in node_ids if context.roles[nid].role == "user_client"]
        services = [nid for nid in node_ids if context.roles[nid].role == "service"]
        if not clients or not services:
            return []

        graph = context.nx_graph
        findings: list[Finding] = []
        for client_id in clients:
            for service_id in services:
                if not nx.has_path(graph, client_id, service_id):
                    continue
                paths = sorted(nx.all_simple_paths(graph, client_id, service_id))
                unguarded = next(
                    (
                        path
                        for path in paths
                        if not any(
                            context.roles[mid].role in {"gateway", "auth"}
                            for mid in path[1:-1]
                        )
                    ),
                    None,
                )
                if unguarded is not None:
                    path_str = " -> ".join(
                        _node_display(context.node_map[nid]) for nid in unguarded
                    )
                    findings.append(
                        Finding(
                            rule_id=self.rule_id,
                            severity="warning",
                            message=(
                                "No gateway or auth component sits between user/client and "
                                f"service along path: {path_str}."
                            ),
                            node_ids=[client_id, service_id],
                            evidence={"path": unguarded},
                        )
                    )
        return findings


class DirectServiceToServiceRule(ReviewRule):
    """Flag direct service-to-service edges without a queue when configured."""

    rule_id = "direct_service_to_service"

    def evaluate(self, context: ReviewContext) -> list[Finding]:
        """Check ``service -> service`` edges when ``require_queue_between_services`` is set."""
        if not context.config.require_queue_between_services:
            return []

        findings: list[Finding] = []
        for edge in sorted(context.graph.edges, key=lambda item: item.id):
            src_role = context.roles[edge.source].role
            dst_role = context.roles[edge.target].role
            if src_role == "service" and dst_role == "service":
                src_desc = _node_display(context.node_map[edge.source])
                dst_desc = _node_display(context.node_map[edge.target])
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity="warning",
                        message=(
                            f"Service {src_desc} calls service {dst_desc} directly "
                            f"(edge {edge.id}) without an intervening message queue."
                        ),
                        node_ids=[edge.source, edge.target],
                        evidence={"edge_id": edge.id},
                    )
                )
        return findings


class OrphanNodesRule(ReviewRule):
    """Flag disconnected nodes with no incoming or outgoing edges."""

    rule_id = "orphan_node"

    def evaluate(self, context: ReviewContext) -> list[Finding]:
        """Return a warning for each node with total degree 0."""
        if not context.config.enable_orphan_check:
            return []

        graph = context.nx_graph
        findings: list[Finding] = []
        for nid in sorted(context.node_map):
            if graph.in_degree(nid) == 0 and graph.out_degree(nid) == 0:
                desc = _node_display(context.node_map[nid])
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity="warning",
                        message=f"Orphan node {desc} has no incoming or outgoing connections.",
                        node_ids=[nid],
                    )
                )
        return findings


class UnlabeledNodesRule(ReviewRule):
    """Flag component nodes whose OCR label is empty."""

    rule_id = "unlabeled_node"

    def evaluate(self, context: ReviewContext) -> list[Finding]:
        """Return an info finding for each node without a text label."""
        if not context.config.enable_unlabeled_check:
            return []

        findings: list[Finding] = []
        for nid in sorted(context.node_map):
            node = context.node_map[nid]
            if not node.label.strip():
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity="info",
                        message=f"Node {nid} (type={node.type!r}) has no text label.",
                        node_ids=[nid],
                    )
                )
        return findings


class CycleDetectionRule(ReviewRule):
    """Flag directed dependency cycles as informational findings."""

    rule_id = "dependency_cycle"

    def evaluate(self, context: ReviewContext) -> list[Finding]:
        """Detect directed cycles and emit deterministic ``info`` findings."""
        if not context.config.enable_cycle_check:
            return []

        raw_cycles = list(nx.simple_cycles(context.directed_graph))
        canonical = sorted(
            (_canonical_cycle(cycle) for cycle in raw_cycles),
            key=lambda c: (len(c), c),
        )

        findings: list[Finding] = []
        for cycle in canonical:
            loop_names = [*cycle, cycle[0]]
            cycle_str = " -> ".join(
                _node_display(context.node_map[nid]) for nid in loop_names
            )
            findings.append(
                Finding(
                    rule_id=self.rule_id,
                    severity="info",
                    message=f"Directed dependency cycle detected: {cycle_str}.",
                    node_ids=cycle,
                    evidence={"cycle": loop_names},
                )
            )
        return findings

