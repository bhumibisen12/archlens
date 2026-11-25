"""Local Tesseract OCR adapter using pytesseract."""

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from archlens.config import OCRConfig
from archlens.graph import BBox
from archlens.ocr.base import OCREngine, TextBox
from archlens.ocr.normalize import normalize_text


@dataclass
class _WordEntry:
    """Intermediate word detection from Tesseract TSV output."""

    block_num: int
    par_num: int
    line_num: int
    x: int
    y: int
    w: int
    h: int
    text: str
    conf: float


def _merge_line_words(words: list[_WordEntry], cfg: OCRConfig) -> list[TextBox]:
    """Merge adjacent words on the same Tesseract line into label-level TextBoxes."""
    if not words:
        return []

    words.sort(key=lambda item: (item.block_num, item.par_num, item.line_num, item.x))
    groups: list[list[_WordEntry]] = []

    for word in words:
        if not groups:
            groups.append([word])
            continue

        prev = groups[-1][-1]
        same_line = (
            word.block_num == prev.block_num
            and word.par_num == prev.par_num
            and word.line_num == prev.line_num
        )
        max_gap = max(24, int(max(prev.h, word.h) * 1.6))
        horiz_gap = word.x - (prev.x + prev.w)
        if same_line and -6 <= horiz_gap <= max_gap:
            groups[-1].append(word)
        else:
            groups.append([word])

    boxes: list[TextBox] = []
    for group in groups:
        raw_text = " ".join(item.text for item in group)
        norm = normalize_text(
            raw_text,
            char_replacements=cfg.char_replacements,
            token_replacements=cfg.token_replacements,
        )
        if not norm:
            continue

        x0 = min(item.x for item in group)
        y0 = min(item.y for item in group)
        x1 = max(item.x + item.w for item in group)
        y1 = max(item.y + item.h for item in group)
        avg_conf = sum(item.conf for item in group) / float(len(group))
        boxes.append(
            TextBox(
                text=norm,
                bbox=BBox(x=max(0, x0), y=max(0, y0), w=max(1, x1 - x0), h=max(1, y1 - y0)),
                conf=round(max(0.0, min(1.0, avg_conf)), 4),
            )
        )
    return boxes


def _windows_tesseract_candidates() -> tuple[Path, ...]:
    """Return common Windows installation paths for ``tesseract.exe``."""
    local_app = Path(os.environ.get("LOCALAPPDATA", ""))
    candidates = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    if str(local_app):
        candidates.extend(
            [
                local_app / "Programs" / "Tesseract-OCR" / "tesseract.exe",
                local_app / "Tesseract-OCR" / "tesseract.exe",
            ]
        )
    return tuple(candidates)


def resolve_tesseract_cmd(cmd: str | None = None) -> str | None:
    """Locate the local Tesseract executable on PATH or standard Windows install paths."""
    target = cmd or pytesseract.pytesseract.tesseract_cmd
    found = shutil.which(target)
    if found is not None:
        return found

    target_path = Path(target)
    if target_path.is_file():
        return str(target_path)

    if sys.platform == "win32":
        for candidate in _windows_tesseract_candidates():
            if candidate.is_file():
                return str(candidate)

    return None


class TesseractEngine(OCREngine):
    """Local OCR adapter backed by the Tesseract CLI via ``pytesseract``."""

    def __init__(self, config: OCRConfig | None = None) -> None:
        """Initialize TesseractEngine with an optional :class:`OCRConfig`."""
        self.config = config or OCRConfig()

    def _ensure_binary_available(self) -> None:
        """Verify that the local ``tesseract`` binary can be invoked."""
        cmd = pytesseract.pytesseract.tesseract_cmd
        resolved = resolve_tesseract_cmd(cmd)
        if resolved is None:
            raise RuntimeError(
                f"Tesseract binary ({cmd!r}) was not found on PATH or standard install paths. "
                "Install Tesseract locally to use TesseractEngine."
            )
        pytesseract.pytesseract.tesseract_cmd = resolved

    def read(self, image: np.ndarray) -> list[TextBox]:
        """Run local Tesseract OCR and return normalized :class:`TextBox` detections."""
        self._ensure_binary_available()
        cfg = self.config

        rgb = (
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.ndim == 3
            else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        )
        tess_cfg = f"--psm {cfg.tesseract_psm}"
        data = pytesseract.image_to_data(
            rgb,
            lang=cfg.language,
            config=tess_cfg,
            output_type=pytesseract.Output.DICT,
        )

        words: list[_WordEntry] = []
        count = len(data.get("text", []))
        for idx in range(count):
            raw_word = str(data["text"][idx]).strip()
            if not raw_word:
                continue
            try:
                conf_raw = float(data["conf"][idx])
            except (TypeError, ValueError):
                continue
            if conf_raw < 0.0:
                continue

            conf_norm = conf_raw / 100.0
            if conf_norm < cfg.min_confidence:
                continue

            w = int(data["width"][idx])
            h = int(data["height"][idx])
            if w <= 0 or h <= 0:
                continue

            words.append(
                _WordEntry(
                    block_num=int(data["block_num"][idx]),
                    par_num=int(data["par_num"][idx]),
                    line_num=int(data["line_num"][idx]),
                    x=int(data["left"][idx]),
                    y=int(data["top"][idx]),
                    w=w,
                    h=h,
                    text=raw_word,
                    conf=conf_norm,
                )
            )

        return _merge_line_words(words, cfg)
