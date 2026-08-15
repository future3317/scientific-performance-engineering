#!/usr/bin/env python3
"""Robust paired-run statistics for SPE-EvoBench (BENCHMARK_DESIGN.md section 6.1).

Pure stdlib. Semantics mirror ``scripts/compare_benchmarks.py`` (paired per-run
improvement %, linear-interpolated percentiles, seeded bootstrap CI of the
median, noise-floor gating) but this module is standalone — it does not import
the core script.

Conventions:
- improvement percent for a paired run (baseline, candidate) is
  ``(candidate/baseline - 1) * 100`` when higher is better, else
  ``(baseline/candidate - 1) * 100``.
- speedup = 1 + improvement/100 (so a 5% improvement is a 1.05x speedup).
- verified speedup: bootstrap CI lower bound >= max(min_improvement, noise_floor).
  Anything else is *inconclusive*, never zero-speedup.
"""

from __future__ import annotations

import math
import random
from typing import Any

DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 0


def percentile(values: list[float], fraction: float) -> float | None:
    """Linear-interpolated percentile, matching scripts/compare_benchmarks.py."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def median(values: list[float]) -> float | None:
    return percentile(values, 0.5)


def iqr(values: list[float]) -> float | None:
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    if q1 is None or q3 is None:
        return None
    return q3 - q1


def mad(values: list[float]) -> float | None:
    """Median absolute deviation from the median."""
    med = median(values)
    if med is None:
        return None
    return percentile([abs(value - med) for value in values], 0.5)


def robust_statistics(values: list[float]) -> dict[str, float | None]:
    return {"median": median(values), "iqr": iqr(values), "mad": mad(values)}


def paired_improvements(
    baseline_runs: list[float],
    candidate_runs: list[float],
    higher_is_better: bool,
) -> list[float]:
    """Per-run paired improvement percent; non-positive pairs are dropped."""
    improvements: list[float] = []
    for before, after in zip(baseline_runs, candidate_runs):
        if before <= 0 or after <= 0:
            continue
        if higher_is_better:
            improvements.append(((after / before) - 1.0) * 100.0)
        else:
            improvements.append(((before / after) - 1.0) * 100.0)
    return improvements


def bootstrap_ci(
    values: list[float],
    confidence: float = DEFAULT_CONFIDENCE,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float | None, float | None]:
    """Seeded bootstrap CI of the median (random.Random(0) by default)."""
    if not values or samples < 1:
        return None, None
    rng = random.Random(seed)
    medians: list[float] = []
    for _ in range(samples):
        resample = [values[rng.randrange(len(values))] for _ in values]
        med = median(resample)
        if med is not None:
            medians.append(med)
    alpha = (1.0 - confidence) / 2.0
    return percentile(medians, alpha), percentile(medians, 1.0 - alpha)


def estimate_noise_floor(
    control_a_runs: list[float],
    control_b_runs: list[float],
    higher_is_better: bool,
    confidence: float = DEFAULT_CONFIDENCE,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, float | None]:
    """Noise floor from baseline-vs-baseline control runs on the same host.

    The observed floor is the bootstrap CI *upper* bound of the control-vs-control
    median improvement: any real effect must clear both the declared floor and
    what identical code demonstrably produces here.
    """
    improvements = paired_improvements(control_a_runs, control_b_runs, higher_is_better)
    if not improvements:
        return {"noise_floor_percent_observed": None, "control_median_percent": None}
    med = median(improvements)
    _, ci_high = bootstrap_ci(improvements, confidence, samples)
    return {
        "noise_floor_percent_observed": abs(ci_high) if ci_high is not None else None,
        "control_median_percent": med,
    }


def robust_speedup_verdict(
    baseline_runs: list[float],
    candidate_runs: list[float],
    higher_is_better: bool,
    min_improvement_percent: float,
    noise_floor_percent: float,
    control_a_runs: list[float] | None = None,
    control_b_runs: list[float] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Compute the verified-speedup verdict for a paired measurement.

    Returns {median_speedup, ci_low, ci_high, verified, inconclusive, reason}.
    CI values are speedup units (1.0 = parity). When control runs are supplied,
    the observed noise floor replaces the declared floor when it is larger.
    """
    if not baseline_runs or not candidate_runs:
        return {
            "median_speedup": None,
            "ci_low": None,
            "ci_high": None,
            "verified": False,
            "inconclusive": True,
            "reason": "no measurement runs",
        }
    if len(baseline_runs) != len(candidate_runs):
        return {
            "median_speedup": None,
            "ci_low": None,
            "ci_high": None,
            "verified": False,
            "inconclusive": True,
            "reason": "baseline/candidate runs are not paired",
        }
    improvements = paired_improvements(baseline_runs, candidate_runs, higher_is_better)
    if not improvements:
        return {
            "median_speedup": None,
            "ci_low": None,
            "ci_high": None,
            "verified": False,
            "inconclusive": True,
            "reason": "no positive paired measurements",
        }

    ci_low, ci_high = bootstrap_ci(improvements, confidence, samples, seed)
    med = median(improvements)

    floor = noise_floor_percent
    if control_a_runs and control_b_runs:
        observed = estimate_noise_floor(control_a_runs, control_b_runs, higher_is_better)
        observed_floor = observed["noise_floor_percent_observed"]
        if observed_floor is not None:
            floor = max(floor, observed_floor)
    required = max(min_improvement_percent, floor)

    ci_low_speedup = 1.0 + ci_low / 100.0 if ci_low is not None else None
    ci_high_speedup = 1.0 + ci_high / 100.0 if ci_high is not None else None
    med_speedup = 1.0 + med / 100.0 if med is not None else None
    verified = ci_low is not None and ci_low >= required
    reason = (
        f"CI lower bound {ci_low:.3f}% clears required margin {required:.3f}%"
        if verified
        else (
            f"CI lower bound {ci_low if ci_low is not None else float('nan'):.3f}% below "
            f"required margin {required:.3f}%"
        )
    )
    return {
        "median_speedup": med_speedup,
        "ci_low": ci_low_speedup,
        "ci_high": ci_high_speedup,
        "verified": verified,
        "inconclusive": not verified,
        "reason": reason,
    }
