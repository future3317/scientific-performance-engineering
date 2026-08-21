"""Calibration bundle status classification."""

from __future__ import annotations

from typing import Any

from .state import derive_cell_state


def classify_result(raw: dict[str, Any]) -> str:
    """Classify a persisted cell before resume reuse."""
    state = derive_cell_state(raw)
    if raw.get("failure_class") == "infrastructure" or str(raw.get("failure_stage", "")) in {"executor", "worker", "agent"}:
        return "rerun"
    if state == "resource_blocked":
        return "rerun"
    if state == "invalid":
        return "blocked_requires_revision"
    if state == "eligible":
        return "reusable"
    return "rerun"
