"""Configurable text normalization helpers for OCR outputs."""

from collections.abc import Mapping


def normalize_text(
    text: str,
    char_replacements: Mapping[str, str] | None = None,
    token_replacements: Mapping[str, str] | None = None,
) -> str:
    """Strip whitespace and fix common configurable OCR character/token confusions.

    Args:
        text: Raw OCR string.
        char_replacements: Optional single-character/substring replacements.
        token_replacements: Optional exact whitespace-delimited token replacements.

    Returns:
        Cleaned single-line text string.
    """
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""

    if char_replacements:
        for src, dst in char_replacements.items():
            cleaned = cleaned.replace(src, dst)

    if token_replacements:
        tokens = [token_replacements.get(tok, tok) for tok in cleaned.split()]
        cleaned = " ".join(tokens)

    return " ".join(cleaned.split())
