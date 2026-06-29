"""
utils/logger.py
===============

Centralised logging. Provides:

1. A configured standard ``logging.Logger`` for human-readable console output.
2. A ``DecisionLogger`` that appends every mission decision and state transition
   to a JSON-lines file, satisfying the safety requirement to "log every
   decision and state transition".

Single responsibility: produce and persist structured log records.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

_DEF_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s"


def get_logger(name: str, level: str = "INFO", console: bool = True) -> logging.Logger:
    """Return a module logger configured once with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers and console:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEF_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


class DecisionLogger:
    """
    Append-only JSON-lines logger for auditable mission decisions.

    Each record carries a timestamp, an event type (e.g. ``state_transition``,
    ``llm_action``, ``safety_abort``), and an arbitrary payload. The file can be
    replayed after a run for analysis or incident review.
    """

    def __init__(self, path: str, also_console: bool = True) -> None:
        self._path = path
        self._console = get_logger("decisions") if also_console else None
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def log(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Persist a single decision/event record."""
        record = {
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": event,
            "payload": payload or {},
        }
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if self._console is not None:
            self._console.info("%s | %s", event, json.dumps(payload or {}))
