"""Typed configuration models and YAML loader for ArchLens."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """Configuration for local logging."""

    level: str = Field(default="INFO", description="Python logging level name.")


class PreprocessConfig(BaseModel):
    """Configuration for image preprocessing."""

    max_dimension: int = Field(default=2048, ge=128)
    enable_perspective_correction: bool = Field(default=True)
    paper_min_area_ratio: float = Field(default=0.45, gt=0.0, lt=1.0)
    paper_canny_low: int = Field(default=50, ge=0)
    paper_canny_high: int = Field(default=150, gt=0)
    paper_approx_epsilon: float = Field(default=0.02, gt=0.0, lt=0.5)
    enable_shadow_removal: bool = Field(default=True)
    shadow_kernel_size: int = Field(default=51, ge=3)
    denoise_method: Literal["gaussian", "median", "bilateral"] = Field(default="median")
    denoise_kernel: int = Field(default=3, ge=1)
    adaptive_block_size: int = Field(default=31, ge=3)
    adaptive_c: int = Field(default=10)
    binary_invert: bool = Field(default=True)


class DetectConfig(BaseModel):
    """Configuration for component shape detection."""

    min_component_area: int = Field(default=1200, ge=1)
    max_component_area_ratio: float = Field(default=0.45, gt=0.0, lt=1.0)
    min_width: int = Field(default=32, ge=1)
    min_height: int = Field(default=24, ge=1)
    max_aspect_ratio: float = Field(default=6.0, gt=1.0)
    min_solidity: float = Field(default=0.65, ge=0.0, le=1.0)
    min_extent: float = Field(default=0.45, ge=0.0, le=1.0)
    min_interior_radius: float = Field(default=8.0, ge=1.0)
    morph_close_kernel: int = Field(default=3, ge=1)
    polygon_epsilon_ratio: float = Field(default=0.03, gt=0.0, lt=1.0)
    rect_min_extent: float = Field(default=0.72, ge=0.0, le=1.0)
    ellipse_max_residual: float = Field(default=0.18, gt=0.0, le=1.0)
    nested_containment_threshold: float = Field(default=0.8, gt=0.0, le=1.0)


class OCRConfig(BaseModel):
    """Configuration for local OCR adapters and text normalization."""

    engine: Literal["tesseract", "paddle"] = Field(default="tesseract")
    language: str = Field(default="eng")
    min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    tesseract_psm: int = Field(default=11, ge=0, le=13)
    paddle_model_dir: str | None = Field(
        default=None,
        description="Local directory containing pre-downloaded PaddleOCR det/rec/cls models.",
    )
    char_replacements: dict[str, str] = Field(
        default_factory=lambda: {
            "|": "I",
            "—": "-",
            "–": "-",
            "‘": "'",
            "’": "'",
            "“": '"',
            "”": '"',
        }
    )
    token_replacements: dict[str, str] = Field(
        default_factory=lambda: {
            "AP1": "API",
            "D8": "DB",
            "R3dis": "Redis",
        }
    )
    inside_margin_px: int = Field(default=6, ge=0)
    max_edge_distance_px: float = Field(default=80.0, gt=0.0)


class ConnectorsConfig(BaseModel):
    """Configuration for arrow and connector detection."""

    snap_distance_px: int = Field(default=35, ge=1)
    node_mask_dilation_px: int = Field(default=4, ge=0)
    text_mask_dilation_px: int = Field(default=2, ge=0)
    spur_prune_length_px: int = Field(default=14, ge=0)
    min_polyline_length_px: float = Field(default=18.0, gt=0.0)
    simplify_epsilon_px: float = Field(default=3.0, gt=0.0)
    max_crossing_angle_deg: float = Field(default=35.0, gt=0.0, le=90.0)
    arrowhead_probe_length_px: int = Field(default=22, ge=6)
    arrowhead_probe_half_width_px: int = Field(default=12, ge=3)
    arrowhead_dominance_ratio: float = Field(default=1.3, gt=1.0)
    min_arrowhead_ink_pixels: int = Field(default=35, ge=1)


class ReviewConfig(BaseModel):
    """Configuration for rule-based architectural checks."""

    enable_cycle_check: bool = Field(default=True)
    enable_orphan_check: bool = Field(default=True)
    enable_unlabeled_check: bool = Field(default=True)
    require_queue_between_services: bool = Field(default=True)
    max_fan_in: int = Field(default=5, ge=1)
    max_fan_out: int = Field(default=5, ge=1)
    roles_config_path: str | None = Field(default=None)


class ExportConfig(BaseModel):
    """Configuration for structured graph and report export."""

    indent: int = Field(default=2, ge=0)


class ArchLensConfig(BaseModel):
    """Top-level configuration for the ArchLens pipeline."""

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    detect: DetectConfig = Field(default_factory=DetectConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    connectors: ConnectorsConfig = Field(default_factory=ConnectorsConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: Path | None = None) -> ArchLensConfig:
    """Load an :class:`ArchLensConfig` from a YAML file, or ``configs/default.yaml``.

    Args:
        path: Optional path to a YAML configuration file.

    Returns:
        Validated :class:`ArchLensConfig` instance.
    """
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not target.is_file():
        if path is not None:
            raise FileNotFoundError(f"Configuration file not found: {target}")
        return ArchLensConfig()

    raw_text = target.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text) or {}
    return ArchLensConfig.model_validate(data)

