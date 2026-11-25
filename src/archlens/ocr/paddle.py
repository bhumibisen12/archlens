"""Optional offline PaddleOCR adapter (lazy-imported from ``[paddle]`` extra)."""

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from archlens.config import OCRConfig
from archlens.graph import BBox
from archlens.ocr.base import OCREngine, TextBox
from archlens.ocr.normalize import normalize_text


class PaddleOCREngine(OCREngine):
    """Local offline PaddleOCR adapter loaded strictly from a local model cache."""

    def __init__(self, config: OCRConfig | None = None) -> None:
        """Lazy-import PaddleOCR and initialize it from pre-cached local model directories.

        Raises:
            ImportError: If the optional ``[paddle]`` extra is not installed.
            RuntimeError: If ``paddle_model_dir`` is unset or missing local model files.
        """
        self.config = config or OCRConfig()

        try:
            from paddleocr import PaddleOCR  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is an optional extra and is not installed. "
                'Install it with: pip install -e ".[paddle]"'
            ) from exc

        model_root = self._resolve_local_model_dir()
        det_dir = model_root / "det"
        rec_dir = model_root / "rec"
        cls_dir = model_root / "cls"
        for subdir in (det_dir, rec_dir):
            if not subdir.is_dir():
                raise RuntimeError(
                    f"Missing local PaddleOCR model directory: {subdir}. "
                    "ArchLens disables runtime downloads; pre-populate local det/rec models."
                )

        # Enforce zero network checks / telemetry at runtime.
        os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("PADDLEOCR_OFFLINE", "1")

        kwargs: dict[str, Any] = {
            "lang": self._map_language(self.config.language),
            "use_angle_cls": cls_dir.is_dir(),
            "use_gpu": False,
            "show_log": False,
            "det_model_dir": str(det_dir),
            "rec_model_dir": str(rec_dir),
        }
        if cls_dir.is_dir():
            kwargs["cls_model_dir"] = str(cls_dir)

        self._ocr: Any = PaddleOCR(**kwargs)

    def _resolve_local_model_dir(self) -> Path:
        """Validate and return the configured local PaddleOCR cache directory."""
        if not self.config.paddle_model_dir:
            raise RuntimeError(
                "OCRConfig.paddle_model_dir must point to a local PaddleOCR model cache "
                "(runtime network downloads are disabled)."
            )
        root = Path(self.config.paddle_model_dir)
        if not root.is_dir():
            raise RuntimeError(
                f"Configured paddle_model_dir does not exist: {root}. "
                "Pre-download PaddleOCR models locally before running."
            )
        return root

    @staticmethod
    def _map_language(lang: str) -> str:
        """Map ISO 639-2 language codes (e.g. ``eng``) to PaddleOCR codes."""
        return "en" if lang == "eng" else lang

    def read(self, image: np.ndarray) -> list[TextBox]:
        """Run offline PaddleOCR on ``image`` and return normalized TextBoxes."""
        cfg = self.config
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
        raw_results = self._ocr.ocr(bgr, cls=False) or []

        boxes: list[TextBox] = []
        for page in raw_results:
            if not page:
                continue
            for entry in page:
                if not entry or len(entry) < 2:
                    continue
                poly, (raw_text, raw_conf) = entry[0], entry[1]
                conf = float(raw_conf)
                if conf < cfg.min_confidence:
                    continue

                norm = normalize_text(
                    str(raw_text),
                    char_replacements=cfg.char_replacements,
                    token_replacements=cfg.token_replacements,
                )
                if not norm:
                    continue

                xs = [float(pt[0]) for pt in poly]
                ys = [float(pt[1]) for pt in poly]
                x0 = max(0, int(np.floor(min(xs))))
                y0 = max(0, int(np.floor(min(ys))))
                x1 = max(x0 + 1, int(np.ceil(max(xs))))
                y1 = max(y0 + 1, int(np.ceil(max(ys))))
                boxes.append(
                    TextBox(
                        text=norm,
                        bbox=BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0),
                        conf=round(max(0.0, min(1.0, conf)), 4),
                    )
                )

        return boxes
