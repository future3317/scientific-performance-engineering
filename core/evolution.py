"""Small statistical gates for rule maintenance."""

from __future__ import annotations

import math
from collections.abc import Sequence


def anytime_lower_bound(successes: int, trials: int, delta: float = 0.05) -> float:
    if trials < 1 or not 0 < delta < 1 or successes < 0 or successes > trials:
        raise ValueError("invalid confidence-sequence inputs")
    alpha_n = delta / (trials * (trials + 1))
    return max(0.0, successes / trials - math.sqrt(math.log(2.0 / alpha_n) / (2.0 * trials)))


def classify_drift(baseline: Sequence[float], recent: Sequence[float], threshold: float = 0.1, min_samples: int = 5) -> str:
    """Classify an effect stream without deleting or mutating a rule spec."""
    if len(baseline) < min_samples or len(recent) < min_samples:
        return "stale"
    baseline_mean = sum(baseline) / len(baseline)
    recent_mean = sum(recent) / len(recent)
    scale = max(1e-12, abs(baseline_mean), 1.0)
    return "suspected_drift" if abs(recent_mean - baseline_mean) / scale >= threshold else "stable"


def leave_one_source_out(source_effects: dict[str, float], minimum: float) -> dict[str, object]:
    """Return a promotion gate requiring the rule to survive each source removal."""
    if not source_effects:
        return {"passed": False, "checked_sources": [], "minimum_effect": None}
    checks = {source: (sum(value for key, value in source_effects.items() if key != source) / max(1, len(source_effects) - 1)) >= minimum for source in source_effects}
    return {"passed": all(checks.values()), "checked_sources": checks, "minimum_effect": minimum}

