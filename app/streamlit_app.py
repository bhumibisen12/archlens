"""Local Streamlit UI for ArchLens diagram parsing and architectural review."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from archlens import ArchLensConfig, ArchLensPipeline, __version__, load_config
from archlens.connectors import draw_connector_overlay
from archlens.detect import draw_detection_overlay
from archlens.export import to_graphviz_dot, to_json, to_mermaid
from archlens.pipeline import PipelineResult
from archlens.preprocess import PreprocessedImages
from archlens.review import ReviewEngine, ReviewReport

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "sample_images"

_SEVERITY_STYLES: dict[str, str] = {
    "error": "background-color: #fde2e1; color: #8c1d18; font-weight: 600;",
    "warning": "background-color: #fef3c7; color: #92400e; font-weight: 600;",
    "info": "background-color: #dbeafe; color: #1e40af; font-weight: 600;",
}


def list_sample_images(sample_dir: Path = SAMPLE_DIR) -> list[Path]:
    """Return sorted sample diagram PNG paths from ``sample_images/``."""
    if not sample_dir.is_dir():
        return []
    return sorted(sample_dir.glob("*.png"))


def build_combined_overlay(
    color_bgr: np.ndarray,
    result: PipelineResult,
) -> np.ndarray:
    """Draw both detected component boxes and extracted arrows onto an RGB image."""
    det_bgr = draw_detection_overlay(color_bgr, result.graph.nodes)
    combined_bgr = draw_connector_overlay(det_bgr, result.graph.nodes, result.graph.edges)
    return cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR or grayscale OpenCV image to RGB for Streamlit display."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_findings_dataframe(report: ReviewReport) -> pd.DataFrame:
    """Convert :class:`ReviewReport` findings into a tabular DataFrame."""
    rows = [
        {
            "Severity": finding.severity.upper(),
            "Rule": finding.rule_id,
            "Nodes": ", ".join(finding.node_ids) if finding.node_ids else "-",
            "Message": finding.message,
        }
        for finding in report.findings
    ]
    return pd.DataFrame(rows, columns=["Severity", "Rule", "Nodes", "Message"])


def style_findings_table(df: pd.DataFrame) -> Any:
    """Apply severity badge colors to the findings DataFrame."""

    def _color_severity(val: object) -> str:
        return _SEVERITY_STYLES.get(str(val).lower(), "")

    return df.style.map(_color_severity, subset=["Severity"])


def format_friendly_ocr_error(exc: Exception, engine: str) -> str:
    """Return an actionable, user-friendly error message for missing local OCR backends."""
    detail = str(exc)
    if engine == "paddle":
        return (
            "PaddleOCR could not be initialized locally.\n\n"
            f"Details: {detail}\n\n"
            "Make sure the optional extra is installed (`pip install -e \".[paddle]\"`) "
            "and `paddle_model_dir` points to a local directory containing pre-downloaded "
            "`det/` and `rec/` models (runtime downloads are disabled)."
        )
    return (
        "Tesseract OCR could not be executed locally.\n\n"
        f"Details: {detail}\n\n"
        "Install the Tesseract OCR binary and ensure `tesseract` is available on your PATH "
        "(see README.md for Windows, macOS, and Linux install commands)."
    )


def run_analysis(
    image_path: Path,
    config: ArchLensConfig,
    debug_dir: Path | None = None,
) -> tuple[PreprocessedImages, PipelineResult, ReviewReport, dict[str, float]]:
    """Run preprocessing, the ArchLens pipeline, and the architectural review engine."""
    pipeline = ArchLensPipeline(config=config)
    preprocessed, result = pipeline.run_with_intermediates(image_path, debug_dir=debug_dir)

    t0 = time.perf_counter()
    report = ReviewEngine(config=config.review).review(result.graph)
    review_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    timings = dict(result.timings_ms)
    timings["review"] = review_ms
    return preprocessed, result, report, timings


def _render_sidebar(base_cfg: ArchLensConfig) -> tuple[ArchLensConfig, bool]:
    """Render sidebar controls and return updated config plus ``show_debug`` flag."""
    st.sidebar.header("Configuration")

    ocr_engine = st.sidebar.selectbox(
        "OCR Engine",
        options=["tesseract", "paddle"],
        index=0 if base_cfg.ocr.engine == "tesseract" else 1,
    )
    paddle_model_dir = base_cfg.ocr.paddle_model_dir or ""
    if ocr_engine == "paddle":
        paddle_model_dir = st.sidebar.text_input(
            "Local PaddleOCR Model Dir",
            value=paddle_model_dir,
            help="Directory containing local det/ and rec/ models (no network calls).",
        )
    min_ocr_conf = st.sidebar.slider(
        "Min OCR Confidence",
        min_value=0.0,
        max_value=1.0,
        value=float(base_cfg.ocr.min_confidence),
        step=0.05,
    )

    show_debug = st.sidebar.checkbox("Show Debug Views", value=False)

    with st.sidebar.expander("Preprocess & Detection Thresholds", expanded=False):
        enable_persp = st.checkbox(
            "Perspective Correction",
            value=base_cfg.preprocess.enable_perspective_correction,
        )
        block_size = st.slider(
            "Adaptive Threshold Block Size",
            min_value=5,
            max_value=71,
            value=int(base_cfg.preprocess.adaptive_block_size),
            step=2,
        )
        adaptive_c = st.slider(
            "Adaptive Threshold C",
            min_value=-10,
            max_value=30,
            value=int(base_cfg.preprocess.adaptive_c),
            step=1,
        )
        min_area = st.slider(
            "Min Component Area (px²)",
            min_value=200,
            max_value=5000,
            value=int(base_cfg.detect.min_component_area),
            step=100,
        )
        min_solidity = st.slider(
            "Min Shape Solidity",
            min_value=0.3,
            max_value=0.95,
            value=float(base_cfg.detect.min_solidity),
            step=0.05,
        )
        snap_dist = st.slider(
            "Connector Snap Distance (px)",
            min_value=10,
            max_value=80,
            value=int(base_cfg.connectors.snap_distance_px),
            step=5,
        )

    with st.sidebar.expander("Review Rules", expanded=False):
        req_queue = st.checkbox(
            "Require Queue Between Services",
            value=base_cfg.review.require_queue_between_services,
        )
        chk_cycles = st.checkbox(
            "Flag Cycles (Info)",
            value=base_cfg.review.enable_cycle_check,
        )
        chk_orphans = st.checkbox(
            "Flag Orphan Nodes",
            value=base_cfg.review.enable_orphan_check,
        )
        chk_unlabeled = st.checkbox(
            "Flag Unlabeled Nodes",
            value=base_cfg.review.enable_unlabeled_check,
        )

    updated = base_cfg.model_copy(
        update={
            "preprocess": base_cfg.preprocess.model_copy(
                update={
                    "enable_perspective_correction": enable_persp,
                    "adaptive_block_size": block_size,
                    "adaptive_c": adaptive_c,
                }
            ),
            "detect": base_cfg.detect.model_copy(
                update={
                    "min_component_area": min_area,
                    "min_solidity": min_solidity,
                }
            ),
            "ocr": base_cfg.ocr.model_copy(
                update={
                    "engine": ocr_engine,
                    "min_confidence": min_ocr_conf,
                    "paddle_model_dir": paddle_model_dir.strip() or None,
                }
            ),
            "connectors": base_cfg.connectors.model_copy(
                update={"snap_distance_px": snap_dist}
            ),
            "review": base_cfg.review.model_copy(
                update={
                    "require_queue_between_services": req_queue,
                    "enable_cycle_check": chk_cycles,
                    "enable_orphan_check": chk_orphans,
                    "enable_unlabeled_check": chk_unlabeled,
                }
            ),
        }
    )
    return updated, show_debug


def _select_input_image(temp_dir: Path) -> tuple[Path | None, str]:
    """Render input source controls and return ``(image_path, stem)``."""
    samples = list_sample_images()
    mode_options = ["Sample Image", "Upload Image"] if samples else ["Upload Image"]
    source_mode = st.radio("Diagram Source", options=mode_options, horizontal=True)

    if source_mode == "Sample Image" and samples:
        chosen = st.selectbox(
            "Choose a sample from sample_images/",
            options=samples,
            format_func=lambda p: p.name,
        )
        return chosen, chosen.stem

    uploaded = st.file_uploader(
        "Upload a diagram photo or scan",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
    )
    if uploaded is None:
        return None, "diagram"

    suffix = Path(uploaded.name).suffix or ".png"
    target = temp_dir / f"uploaded{suffix}"
    target.write_bytes(uploaded.getvalue())
    return target, Path(uploaded.name).stem


def main() -> None:
    """Render the ArchLens Streamlit application."""
    st.set_page_config(page_title="ArchLens", layout="wide")
    st.title("ArchLens")
    st.caption(
        f"v{__version__} — Local, privacy-first architecture diagram parser & reviewer "
        "(no cloud APIs, no network calls)"
    )

    base_cfg = load_config()
    config, show_debug = _render_sidebar(base_cfg)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        image_path, stem = _select_input_image(tmp_dir)
        if image_path is None:
            st.info("Select a sample diagram or upload an image to begin analysis.")
            return

        debug_dir = tmp_dir / "debug" if show_debug else None
        try:
            preprocessed, result, report, timings = run_analysis(
                image_path=image_path,
                config=config,
                debug_dir=debug_dir,
            )
        except (ImportError, RuntimeError) as exc:
            st.error(format_friendly_ocr_error(exc, config.ocr.engine))
            return
        except Exception as exc:
            st.error(f"Failed to analyze diagram: {exc}")
            return

        # Visual inspection row
        st.subheader("Visual Analysis")
        col1, col2, col3 = st.columns(3)
        orig_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        with col1:
            st.markdown("**Original**")
            if orig_bgr is not None:
                st.image(bgr_to_rgb(orig_bgr), use_container_width=True)
        with col2:
            st.markdown("**Preprocessed (Binary)**")
            st.image(preprocessed.binary, clamp=True, use_container_width=True)
        with col3:
            st.markdown("**Detected Nodes & Arrows**")
            st.image(build_combined_overlay(preprocessed.color, result), use_container_width=True)

        if show_debug and debug_dir is not None:
            with st.expander("Debug Artifacts", expanded=True):
                artifacts = sorted(result.debug_artifacts.items())
                if artifacts:
                    d_cols = st.columns(min(4, len(artifacts)))
                    for idx, (label, path_str) in enumerate(artifacts):
                        img = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            disp = bgr_to_rgb(img) if img.ndim == 3 else img
                            d_cols[idx % len(d_cols)].image(
                                disp,
                                caption=label,
                                clamp=True,
                                use_container_width=True,
                            )

        # Graph rendering + downloads
        st.subheader("Extracted Architecture Graph")
        g_col, d_col = st.columns([3, 1])
        with g_col:
            dot_text = to_graphviz_dot(result.graph)
            st.graphviz_chart(dot_text, use_container_width=True)
        with d_col:
            st.metric("Nodes", result.nodes_detected)
            st.metric("Edges", result.edges_detected)
            st.download_button(
                "Download JSON",
                data=to_json(result.graph, indent=config.export.indent),
                file_name=f"{stem}.json",
                mime="application/json",
                use_container_width=True,
            )
            st.download_button(
                "Download Mermaid",
                data=to_mermaid(result.graph),
                file_name=f"{stem}.mmd",
                mime="text/markdown",
                use_container_width=True,
            )

        # Architectural review findings
        st.subheader("Architectural Review Findings")
        s1, s2, s3 = st.columns(3)
        s1.metric("Errors", report.summary["error"])
        s2.metric("Warnings", report.summary["warning"])
        s3.metric("Info", report.summary["info"])

        findings_df = build_findings_dataframe(report)
        if findings_df.empty:
            st.success("No design rule violations detected.")
        else:
            st.dataframe(
                style_findings_table(findings_df),
                use_container_width=True,
                hide_index=True,
            )

        # Stage timings
        st.subheader("Stage Timings (ms)")
        timing_cols = st.columns(len(timings) + 1)
        for idx, (stage, ms) in enumerate(timings.items()):
            timing_cols[idx].metric(stage.replace("_", " ").title(), f"{ms:.1f}")
        timing_cols[-1].metric("Total", f"{sum(timings.values()):.1f}")


if __name__ == "__main__":
    main()
