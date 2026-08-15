#!/usr/bin/env python3
"""Run a paired rule intervention replay and emit a digest-attested manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.sequential_stats import mixture_lower_bound
from core.utility import UTILITY_POLICY_ID, normalized_delta, validate_policy


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def beta_tail_probability(alpha: int, beta: int, threshold: float) -> float:
    """Exact Beta-binomial tail for integer posterior parameters."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    n = alpha + beta - 1
    tail = 0.0
    for j in range(alpha):
        tail += math.comb(n, j) * threshold**j * (1.0 - threshold) ** (n - j)
    return min(1.0, max(0.0, tail))


def anytime_lower_bound(successes: int, trials: int, delta: float) -> float:
    """A time-uniform Hoeffding bound via a summable alpha schedule.

    Using alpha_n = delta/(n(n+1)) makes the bound valid at every inspected
    sample count by a union bound; it avoids optional-stopping claims based on
    a fixed-n normal interval.
    """
    if trials < 1 or not 0 < delta < 1:
        raise ValueError("trials must be positive and delta must be in (0, 1)")
    alpha_n = delta / (trials * (trials + 1))
    radius = math.sqrt(math.log(2.0 / alpha_n) / (2.0 * trials))
    return max(0.0, successes / trials - radius)


def betting_lower_bound(successes: int, trials: int, delta: float) -> float:
    """Public replay gate using the beta-binomial mixture boundary."""
    return mixture_lower_bound(successes, trials, delta)


def evaluate_cases(
    cases: list[dict[str, Any]],
    epsilon: float,
    p_min: float,
    delta: float,
    utility_policy_id: str = UTILITY_POLICY_ID,
    utility_scale: float = 1.0,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("replay requires at least one paired case")
    validate_policy(utility_policy_id)
    effects = [
        normalized_delta(float(case["utility_on"]), float(case["utility_off"]), scale=utility_scale)
        for case in cases
    ]
    scientific_ok = all(bool(case.get("scientific_ok", False)) and bool(case.get("quality_ok", True)) for case in cases)
    successes = sum(effect > epsilon for effect in effects)
    failures = len(effects) - successes
    alpha = 1 + successes
    beta = 1 + failures
    posterior_probability = beta_tail_probability(alpha, beta, p_min)
    mean_effect = mean(effects)
    if len(effects) > 1:
        variance = sum((effect - mean_effect) ** 2 for effect in effects) / (len(effects) - 1)
        standard_error = math.sqrt(variance / len(effects))
    else:
        standard_error = 0.0
    # The mixture boundary is valid when the maintainer checks the stream after
    # any number of replay cases and is tighter than the old union-bound gate.
    effect_radius = math.sqrt(math.log(2.0 * len(effects) * (len(effects) + 1) / delta) / (2.0 * len(effects)))
    lower_confidence_bound = mean_effect - effect_radius
    promotion_probability_lower_bound = betting_lower_bound(successes, len(effects), delta)
    outcome = "passed" if scientific_ok and mean_effect > epsilon and promotion_probability_lower_bound >= p_min else "failed"
    return {
        "n": len(effects),
        "mean_effect": mean_effect,
        "utility_policy_id": utility_policy_id,
        "utility_scale": utility_scale,
        "lower_confidence_bound": lower_confidence_bound,
        "epsilon": epsilon,
        "successes": successes,
        "failures": failures,
        "prior_alpha": 1,
        "prior_beta": 1,
        "p_min": p_min,
        "delta": delta,
        "posterior_probability": posterior_probability,
        "promotion_probability_lower_bound": promotion_probability_lower_bound,
        "confidence_method": "beta-binomial-mixture-cs",
        "scientific_gates_passed": scientific_ok,
        "outcome": outcome,
    }


def build_evidence_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialize the paired replay as auditable on/off EvidenceEvents."""
    events: list[dict[str, Any]] = []
    for index, case in enumerate(payload["cases"]):
        common = {"revision": case.get("revision", payload.get("revision", "unknown")), "seed_family": case.get("seed_family", payload.get("seed_family", "replay"))}
        for arm in ("on", "off"):
            events.append({
                "event_id": f"{case.get('case_id', index)}-{arm}", "context": case.get("context", common),
                "rule_id": payload["rule_id"], "rule_version": int(payload.get("rule_version", 1)),
                "assignment": {"arm": arm, "propensity": float(case.get("propensity", 0.5))},
                "outcome_vector": {"utility": float(case["utility_on"] if arm == "on" else case["utility_off"])},
                "scientific_gates": {"scientific_ok": bool(case.get("scientific_ok", False)), "quality_ok": bool(case.get("quality_ok", True))},
                "artifacts": case.get("artifacts", {}), "versions": case.get("versions", {}),
                "source_id": case.get("source_id", f"replay-{index}"), "independence_group": case.get("independence_group", "replay"),
                "timestamp": case.get("timestamp", datetime.now(timezone.utc).isoformat()),
            })
    return events


def git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_manifest(payload: dict[str, Any], input_path: Path, output_path: Path, harness_revision: str) -> dict[str, Any]:
    result = evaluate_cases(
        payload["cases"],
        float(payload.get("epsilon", 0.0)),
        float(payload.get("p_min", 0.8)),
        float(payload.get("delta", 0.05)),
        str(payload.get("utility_policy_id", UTILITY_POLICY_ID)),
        float(payload.get("utility_scale", 1.0)),
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "rule_id": payload["rule_id"],
        "command": " ".join(["python", "scripts/run_rule_replay.py", str(input_path), str(output_path)]),
        "case_bundle_path": str(input_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_bundle_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
        "harness_revision": harness_revision,
        "result": result,
        "evidence_events": build_evidence_events(payload),
        "result_digest": hashlib.sha256(canonical_json(result)).hexdigest(),
        "outcome": result["outcome"],
    }
    manifest["attestation"] = {"algorithm": "sha256", "manifest_digest": hashlib.sha256(canonical_json(manifest)).hexdigest()}
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="paired replay input JSON")
    parser.add_argument("output", type=Path, help="manifest output JSON")
    parser.add_argument("--harness-revision", default=None)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list) or not payload.get("rule_id"):
        raise SystemExit("input needs rule_id and a non-empty cases list")
    manifest = build_manifest(payload, args.input, args.output, args.harness_revision or git_revision(Path(__file__).resolve().parents[1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "outcome": manifest["result"]["outcome"], "result_digest": manifest["result_digest"]}))


if __name__ == "__main__":
    main()
