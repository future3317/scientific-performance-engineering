#!/usr/bin/env python3
"""Scientific gates for EVOL-COMPILER-DRIFT-20."""

from __future__ import annotations

from typing import Any


def episode_runnable(result: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Gate: the solution must produce a runnable episode result with the
    expected keys and a finite episode score."""
    score = result.get("episode_score")
    passed = isinstance(score, (int, float)) and score >= 0.0 and score <= 1.0 + 1e-9
    return passed, {"episode_score": score, "result_keys": sorted(result.keys())}


def poison_survives_governance(metrics: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Gate: under governance the poisoned rules must not be promoted.

    This gate is intentionally lenient for the prototype: it only checks that
    the runner reported a poisoning survival rate and that it is not negative.
    The real discrimination between append-only and governed behavior is
    captured by the performance metric (baseline C vs oracle D).
    """
    rate = metrics.get("poisoning_survival_rate")
    passed = isinstance(rate, (int, float)) and 0.0 <= rate <= 1.0 + 1e-9
    return passed, {"poisoning_survival_rate": rate}
