"""Versioned utility contract used by replay and evolution metrics."""

from __future__ import annotations

import math


UTILITY_POLICY_ID = "normalized_task_utility_v1"


def normalized_delta(utility_on: float, utility_off: float, *, scale: float = 1.0) -> float:
    """Return a bounded paired utility delta in ``[-1, 1]``.

    Replay inputs may retain their native task units, but the value used for a
    promotion decision is always normalized by the declared positive scale.
    """
    if not math.isfinite(utility_on) or not math.isfinite(utility_off) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("utility values must be finite and scale must be positive")
    raw = (utility_on - utility_off) / scale
    return max(-1.0, min(1.0, raw))


def validate_policy(policy_id: str) -> None:
    if policy_id != UTILITY_POLICY_ID:
        raise ValueError(f"unsupported utility_policy_id: {policy_id!r}")
