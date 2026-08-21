"""Canonical derived state for one calibration cell."""

from __future__ import annotations

from typing import Any


def derive_cell_state(result: dict[str, Any]) -> str:
    """Derive one state from the persisted execution facts.

    The legacy JSON fields remain as reporting fields, but callers should use
    this value for resume and population decisions instead of combining the
    booleans independently.
    """
    if bool(result.get("protocol_failure")) or result.get("validity") == "invalid":
        return "invalid"
    if result.get("execution_validity") == "resource_blocked" or bool(result.get("timeout")):
        return "resource_blocked"
    if result.get("execution_validity") != "valid":
        return "pending"
    if result.get("efficacy_eligible") is True and result.get("calibration_status") == "eligible":
        return "eligible"
    if result.get("calibration_status") == "blocked" or result.get("efficacy_eligible") is False:
        return "ineligible"
    return "pending"


def serialize_cell_state(result: dict[str, Any]) -> dict[str, Any]:
    """Attach the canonical derived state to a result before persistence."""
    result["calibration_state"] = derive_cell_state(result)
    return result
