"""Versioned utility contract used by replay and evolution metrics."""

from __future__ import annotations

import math


UTILITY_POLICY_ID = "bounded_log_speedup_v1"
UTILITY_LOG_SCALE = 0.5


def utility_effect(
    utility_on: float,
    utility_off: float,
    *,
    higher_is_better: bool = True,
    log_scale: float = UTILITY_LOG_SCALE,
) -> float:
    """Map a positive paired measurement to a bounded dimensionless effect."""
    if not math.isfinite(utility_on) or not math.isfinite(utility_off):
        raise ValueError("utility values must be finite")
    if utility_on <= 0.0 or utility_off <= 0.0 or not math.isfinite(log_scale) or log_scale <= 0.0:
        raise ValueError("log-speedup utility requires positive values and scale")
    speedup = utility_on / utility_off if higher_is_better else utility_off / utility_on
    return math.tanh(math.log(speedup) / log_scale)


def validate_policy(policy_id: str) -> None:
    if policy_id != UTILITY_POLICY_ID:
        raise ValueError(f"unsupported utility_policy_id: {policy_id!r}")
