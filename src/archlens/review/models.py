"""Typed data models for architectural review findings and reports."""

from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["error", "warning", "info"]
RoleName = Literal[
    "database",
    "cache",
    "queue",
    "gateway",
    "auth",
    "load_balancer",
    "user_client",
    "service",
    "unknown",
]


class RoleInference(BaseModel):
    """Explainable role assignment for a diagram component node."""

    role: RoleName
    matched_keyword: str | None = None
    reason: str


class Finding(BaseModel):
    """A single explainable finding emitted by a :class:`ReviewRule`."""

    rule_id: str = Field(..., min_length=1)
    severity: Severity
    message: str = Field(..., min_length=1)
    node_ids: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


class ReviewReport(BaseModel):
    """Aggregated deterministic output from :class:`ReviewEngine`."""

    findings: list[Finding] = Field(default_factory=list)
    node_roles: dict[str, RoleInference] = Field(default_factory=dict)
    summary: dict[Severity, int] = Field(
        default_factory=lambda: {"error": 0, "warning": 0, "info": 0}
    )

