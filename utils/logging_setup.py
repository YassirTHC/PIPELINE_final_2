"""Console logging helpers with emoji-aware formatting."""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional

_LOGGER_NAME = "pipeline_console"
_HANDLER_NAME = "console"


def _supports_emoji() -> bool:
    encoding = (sys.stdout.encoding or "").lower()
    env_encoding = (os.environ.get("PYTHONIOENCODING") or "").lower()
    if "utf" not in encoding and "utf" not in env_encoding:
        return False
    return True


def _build_formatter(use_emoji: bool) -> logging.Formatter:
    fmt = "[%(asctime)s] %(levelname)s | %(message)s"
    formatter = logging.Formatter(fmt=fmt, datefmt="%H:%M:%S")
    if os.environ.get("PIPELINE_LOG_UTC") == "1":
        formatter.converter = time.gmtime
    return formatter


def get_console_logger(no_emoji: bool = False, name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a configured console logger, reconfiguring the formatter when needed."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = True

    effective_no_emoji = no_emoji or not _supports_emoji()

    handler: Optional[logging.Handler] = None
    for existing in logger.handlers:
        if existing.name == _HANDLER_NAME:
            handler = existing
            break

    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
        handler.name = _HANDLER_NAME
        logger.addHandler(handler)

    handler.setFormatter(_build_formatter(not effective_no_emoji))
    return logger
