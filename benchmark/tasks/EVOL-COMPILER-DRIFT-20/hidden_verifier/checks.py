"""Hidden verifier helpers for EVOL-COMPILER-DRIFT-20.

Result-shape validation used by benchmark.py; no stored goldens.
"""

from __future__ import annotations

from typing import Any


def validate_result(result: Any) -> tuple[bool, dict[str, Any]]:
    """Check that the solution returned a well-formed episode result."""
    if not isinstance(result, dict):
        return False, {"reason": f"expected dict, got {type(result).__name__}"}
    score = result.get("episode_score")
    if not isinstance(score, (int, float)):
        return False, {"reason": f"episode_score must be numeric, got {type(score).__name__}"}
    if not (0.0 <= float(score) <= 1.0 + 1e-9):
        return False, {"reason": f"episode_score {score} out of [0,1]"}
    if "episode_metrics" not in result:
        return False, {"reason": "missing episode_metrics"}
    return True, {"episode_score": float(score)}
