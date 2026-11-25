"""Abstract base class and shared context for ArchLens review rules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import networkx as nx

from archlens.config import ReviewConfig
from archlens.graph import DiagramGraph, Node
from archlens.review.models import Finding, RoleInference


@dataclass(frozen=True)
class ReviewContext:
    """Precomputed deterministic graph and role context shared by all review rules."""

    graph: DiagramGraph
    node_map: dict[str, Node]
    roles: dict[str, RoleInference]
    nx_graph: nx.DiGraph
    directed_graph: nx.DiGraph
    config: ReviewConfig


class ReviewRule(ABC):
    """Interface for a deterministic architectural review rule."""

    rule_id: str

    @abstractmethod
    def evaluate(self, context: ReviewContext) -> list[Finding]:
        """Evaluate the rule against ``context`` and return deterministic findings."""

