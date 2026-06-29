"""
utils/config_loader.py
=======================

Loads YAML configuration files into plain dictionaries and provides safe,
dotted-key access with defaults.

Single responsibility: read configuration from disk and expose it.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml


class Config:
    """
    Thin wrapper around a parsed YAML dict supporting dotted lookups, e.g.::

        cfg.get("navigation.astar.heuristic_weight", 1.0)
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load a YAML file into a :class:`Config`."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls(data)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Return a nested value by dotted key, or ``default`` if absent."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, key: str) -> Dict[str, Any]:
        """Return a top-level section as a dict (empty dict if missing)."""
        value = self._data.get(key, {})
        return value if isinstance(value, dict) else {}

    @property
    def raw(self) -> Dict[str, Any]:
        return self._data
