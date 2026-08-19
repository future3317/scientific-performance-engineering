"""Hidden verifier helpers for EVOL-COMPILER-DRIFT-20.

Result-shape validation used by benchmark.py; no stored goldens.
"""

from __future__ import annotations

from typing import Any


def validate_result(result: Any) -> tuple[bool, dict[str, Any]]:
    """Check that the solution returned a well-formed episode result."""
    if not isinstance(result, dict):
        return False, {"reason": f"expected dict, got {type(result).__name__}"}
    action = result.get("action")
    if not isinstance(action, dict) or str(action.get("condition", "")).upper() not in {"C", "C_STRESS", "D"}:
        return False, {"reason": "missing valid declarative episode action"}
    return True, {"action": action}
