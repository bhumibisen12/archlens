"""Canonical Pydantic schema models for ArchLens diagram graphs."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

NodeType = Literal["box", "database", "ellipse", "component"]


class BBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""

    x: int = Field(..., ge=0, description="Left edge X coordinate in pixels.")
    y: int = Field(..., ge=0, description="Top edge Y coordinate in pixels.")
    w: int = Field(..., gt=0, description="Width in pixels.")
    h: int = Field(..., gt=0, description="Height in pixels.")

    @property
    def center(self) -> tuple[float, float]:
        """Return the ``(cx, cy)`` center of the bounding box."""
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def area(self) -> int:
        """Return the pixel area of the bounding box."""
        return self.w * self.h


class Point(BaseModel):
    """2D coordinate point along a connector polyline."""

    x: float = Field(..., description="X coordinate in pixels.")
    y: float = Field(..., description="Y coordinate in pixels.")

    @model_validator(mode="before")
    @classmethod
    def _coerce_sequence(cls, value: Any) -> Any:
        """Allow constructing a Point from a 2-item ``(x, y)`` tuple or list."""
        if isinstance(value, tuple | list) and len(value) == 2:
            return {"x": value[0], "y": value[1]}
        return value


class Node(BaseModel):
    """A detected architectural component node."""

    id: str = Field(..., min_length=1, description="Unique node identifier.")
    type: NodeType = Field(default="box", description="Detected shape category.")
    bbox: BBox = Field(..., description="Bounding box of the component.")
    label: str = Field(default="", description="Extracted text label for the component.")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Detection/recognition confidence score in [0.0, 1.0].",
    )


class Edge(BaseModel):
    """A detected connector or arrow between two component nodes."""

    id: str = Field(..., min_length=1, description="Unique edge identifier.")
    source: str = Field(..., min_length=1, description="Identifier of the source node.")
    target: str = Field(..., min_length=1, description="Identifier of the target node.")
    directed: bool = Field(
        default=True,
        description="Whether an arrowhead indicates direction from source to target.",
    )
    label: str | None = Field(
        default=None,
        description="Optional text label associated with the connector.",
    )
    polyline: list[Point] = Field(
        default_factory=list,
        description="Ordered list of points tracing the connector path.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Detection confidence score in [0.0, 1.0].",
    )


class ImageSize(BaseModel):
    """Dimensions of the processed source image in pixels."""

    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class GraphMetadata(BaseModel):
    """Execution metadata recorded alongside a parsed diagram graph."""

    image_size: ImageSize | None = Field(
        default=None,
        description="Width and height of the analyzed image.",
    )
    timings_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Per-stage execution durations in milliseconds.",
    )
    source_image: str | None = Field(
        default=None,
        description="Optional file path or name of the source diagram image.",
    )
    ocr_engine: str | None = Field(
        default=None,
        description="Identifier of the OCR engine used during parsing.",
    )
    schema_version: str = Field(
        default="1.0",
        description="Schema contract version.",
    )


class DiagramGraph(BaseModel):
    """Canonical structured directed graph representing an architecture diagram."""

    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    metadata: GraphMetadata = Field(default_factory=GraphMetadata)

    @model_validator(mode="after")
    def _validate_integrity(self) -> "DiagramGraph":
        """Ensure unique node/edge IDs and valid edge endpoint references."""
        node_ids: set[str] = set()
        for node in self.nodes:
            if node.id in node_ids:
                raise ValueError(f"Duplicate node id: {node.id!r}")
            node_ids.add(node.id)

        edge_ids: set[str] = set()
        for edge in self.edges:
            if edge.id in edge_ids:
                raise ValueError(f"Duplicate edge id: {edge.id!r}")
            edge_ids.add(edge.id)
            if edge.source not in node_ids:
                raise ValueError(
                    f"Edge {edge.id!r} references unknown source node {edge.source!r}"
                )
            if edge.target not in node_ids:
                raise ValueError(
                    f"Edge {edge.id!r} references unknown target node {edge.target!r}"
                )
        return self
