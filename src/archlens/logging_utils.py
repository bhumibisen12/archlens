"""Structured local logging helpers for ArchLens."""

import logging

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the root ``archlens`` package logger.

    Args:
        level: Logging level name (e.g., ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).

    Returns:
        Configured ``archlens`` :class:`logging.Logger`.
    """
    logger = logging.getLogger("archlens")
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``archlens`` namespace."""
    return logging.getLogger(f"archlens.{name}")

