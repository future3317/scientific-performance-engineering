"""Calibration bundle status classification."""

from __future__ import annotations

from typing import Any


def classify_result(raw: dict[str, Any]) -> str:
    """Classify a persisted cell before resume reuse."""
    execution_validity = str(raw.get("execution_validity", ""))
    if bool(raw.get("protocol_failure")):
        return "blocked_requires_revision"
    if execution_validity == "resource_blocked" or bool(raw.get("timeout")):
        return "rerun"
    if raw.get("failure_class") == "infrastructure" or str(raw.get("failure_stage", "")) in {"executor", "worker", "agent"}:
        return "rerun"
    if execution_validity == "invalid":
        return "blocked_requires_revision"
    if execution_validity == "valid" and raw.get("efficacy_eligible") is True:
        return "reusable"
    return "rerun"

