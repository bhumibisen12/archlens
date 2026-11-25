"""Rule-based architectural design checks, role inference, and review engine."""

from archlens.review.base import ReviewContext, ReviewRule
from archlens.review.engine import ReviewEngine, default_rules
from archlens.review.models import Finding, ReviewReport, RoleInference, RoleName, Severity
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

__all__ = [
    "CycleDetectionRule",
    "DirectClientToDatabaseRule",
    "DirectServiceToServiceRule",
    "Finding",
    "MissingGatewayOrAuthRule",
    "OrphanNodesRule",
    "ReviewContext",
    "ReviewEngine",
    "ReviewReport",
    "ReviewRule",
    "RoleInference",
    "RoleName",
    "RolesConfig",
    "Severity",
    "SinglePointOfFailureRule",
    "UnlabeledNodesRule",
    "default_rules",
    "infer_node_role",
    "load_roles_config",
]
