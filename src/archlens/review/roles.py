"""Keyword-based node role inference loaded from ``configs/roles.yaml``."""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from archlens.graph import Node
from archlens.review.models import RoleInference, RoleName

DEFAULT_ROLES_PATH = Path(__file__).resolve().parents[3] / "configs" / "roles.yaml"


class RolesConfig(BaseModel):
    """Schema for ``configs/roles.yaml``."""

    priority_order: list[RoleName] = Field(
        default_factory=lambda: [
            "load_balancer",
            "gateway",
            "auth",
            "cache",
            "queue",
            "database",
            "user_client",
            "service",
        ]
    )
    roles: dict[RoleName, list[str]] = Field(default_factory=dict)
    default_labeled_role: RoleName = Field(default="service")
    default_unlabeled_role: RoleName = Field(default="unknown")


def load_roles_config(path: Path | None = None) -> RolesConfig:
    """Load role keyword mappings from YAML, falling back to defaults if absent."""
    target = Path(path) if path is not None else DEFAULT_ROLES_PATH
    if not target.is_file():
        if path is not None:
            raise FileNotFoundError(f"Roles config file not found: {target}")
        return RolesConfig()

    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return RolesConfig.model_validate(raw)


def _matches_keyword(label_lower: str, keyword: str) -> bool:
    """Return True if ``keyword`` matches ``label_lower`` at alphanumeric boundaries."""
    pattern = rf"(?<![a-z0-9]){re.escape(keyword.lower())}(?![a-z0-9])"
    return re.search(pattern, label_lower) is not None


def infer_node_role(node: Node, roles_config: RolesConfig) -> RoleInference:
    """Deterministically infer a node's architectural role from its label and shape."""
    label = node.label.strip()
    label_lower = label.lower()

    if label_lower:
        for role in roles_config.priority_order:
            for keyword in roles_config.roles.get(role, ()):
                if _matches_keyword(label_lower, keyword):
                    return RoleInference(
                        role=role,
                        matched_keyword=keyword,
                        reason=f"Label {label!r} matched keyword {keyword!r} for role {role!r}.",
                    )

        if node.type == "database":
            return RoleInference(
                role="database",
                reason=f"Shape type 'database' for label {label!r} with no specialized keyword.",
            )

        return RoleInference(
            role=roles_config.default_labeled_role,
            reason=(
                f"Label {label!r} did not match specialized keywords; "
                f"defaulted to {roles_config.default_labeled_role!r}."
            ),
        )

    if node.type == "database":
        return RoleInference(
            role="database",
            reason="Unlabeled node has shape type 'database'.",
        )

    return RoleInference(
        role=roles_config.default_unlabeled_role,
        reason="Node has no label.",
    )

