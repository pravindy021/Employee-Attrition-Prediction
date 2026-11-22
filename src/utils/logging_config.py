"""Logging configuration for scripts and application modules."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with a clear console format."""
    logger = logging.getLogger("employee_attrition")
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

