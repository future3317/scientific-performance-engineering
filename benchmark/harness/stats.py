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
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .fingerprint import fingerprints_compatible

DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 0
NOISE_CONTROL_SCHEMA_VERSION = 1


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

    The observed floor is the bootstrap CI *upper* bound of the median symmetric
    multiplicative deviation between identical controls.  The construction is
    label-symmetric, so swapping the two control arms cannot change the floor.
    """
    del higher_is_better  # identical controls have no directional improvement
    deviations = [
        (max(float(a) / float(b), float(b) / float(a)) - 1.0) * 100.0
        for a, b in zip(control_a_runs, control_b_runs)
        if float(a) > 0 and float(b) > 0
    ]
    if not deviations:
        return {"noise_floor_percent_observed": None, "control_median_percent": None}
    med = median(deviations)
    _, ci_high = bootstrap_ci(deviations, confidence, samples)
    return {
        "noise_floor_percent_observed": abs(ci_high) if ci_high is not None else None,
        "control_median_percent": med,
    }


def effective_noise_floor(declared: float, observed: float | None) -> float:
    """Return the conservative floor used by every matched verifier cell."""
    declared_value = float(declared)
    if not math.isfinite(declared_value) or declared_value < 0:
        raise ValueError("declared noise floor must be a finite non-negative number")
    if observed is None:
        return declared_value
    observed_value = float(observed)
    if not math.isfinite(observed_value) or observed_value < 0:
        raise ValueError("observed noise floor must be a finite non-negative number")
    return max(declared_value, observed_value)


def _noise_control_digest(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "artifact_digest"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_noise_control(path: str | Path, artifact: dict[str, Any]) -> dict[str, Any]:
    """Write one immutable task×outer-trial same-host control artifact."""
    payload = dict(artifact)
    payload["schema_version"] = NOISE_CONTROL_SCHEMA_VERSION
    payload["effective_noise_floor_percent"] = effective_noise_floor(
        float(payload["declared_noise_floor_percent"]),
        payload.get("observed_noise_floor_percent"),
    )
    payload["artifact_digest"] = _noise_control_digest(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def read_noise_control(path: str | Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load and fail closed on an incompatible or tampered control artifact."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"noise control artifact unavailable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("noise control artifact must be an object")
    if payload.get("schema_version") != NOISE_CONTROL_SCHEMA_VERSION:
        raise ValueError("noise control schema_version mismatch")
    digest = payload.get("artifact_digest")
    if not isinstance(digest, str) or digest != _noise_control_digest(payload):
        raise ValueError("noise control artifact digest mismatch")
    required = {
        "task_id", "outer_trial_id", "benchmark_revision", "task_manifest_digest",
        "hardware_fingerprint", "software_fingerprint", "compile_threads",
        "compiler_cache_policy", "control_a_runs", "control_b_runs",
        "observed_noise_floor_percent", "declared_noise_floor_percent",
        "effective_noise_floor_percent", "expected_speedup_range",
    }
    missing = sorted(key for key in required if key not in payload)
    if missing:
        raise ValueError("noise control artifact missing: " + ", ".join(missing))
    if not isinstance(payload["control_a_runs"], list) or not isinstance(payload["control_b_runs"], list):
        raise ValueError("noise control control runs must be arrays")
    if len(payload["control_a_runs"]) != 5 or len(payload["control_b_runs"]) != 5:
        raise ValueError("noise control artifact requires five control repetitions per arm")
    for key in ("control_a_runs", "control_b_runs"):
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0 for value in payload[key]):
            raise ValueError(f"noise control {key} must contain positive finite measurements")
    expected = expected or {}
    for key in ("task_id", "outer_trial_id", "benchmark_revision", "task_manifest_digest", "compile_threads", "compiler_cache_policy", "primary_metric", "higher_is_better", "expected_speedup_range"):
        if key in expected and payload.get(key) != expected[key]:
            raise ValueError(f"noise control {key} mismatch")
    for key in ("hardware_fingerprint", "software_fingerprint"):
        if key not in expected:
            continue
        observed = payload.get(key)
        reference = expected[key]
        if not isinstance(observed, dict) or not isinstance(reference, dict):
            raise ValueError(f"noise control {key} missing")
        compatible, reasons = fingerprints_compatible(observed, reference)
        if not compatible:
            raise ValueError(f"noise control {key} mismatch: {'; '.join(reasons)}")
    observed_floor = payload.get("observed_noise_floor_percent")
    expected_floor = effective_noise_floor(float(payload["declared_noise_floor_percent"]), observed_floor)
    if float(payload.get("effective_noise_floor_percent")) != expected_floor:
        raise ValueError("noise control effective floor mismatch")
    return payload


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
