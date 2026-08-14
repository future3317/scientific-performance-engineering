#!/usr/bin/env python3
"""Run a paired rule intervention replay and emit a digest-attested manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


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


def evaluate_cases(cases: list[dict[str, Any]], epsilon: float, p_min: float, delta: float) -> dict[str, Any]:
    if not cases:
        raise ValueError("replay requires at least one paired case")
    effects = [float(case["utility_on"]) - float(case["utility_off"]) for case in cases]
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
    lower_confidence_bound = mean_effect - 1.645 * standard_error
    outcome = "passed" if scientific_ok and lower_confidence_bound > epsilon and posterior_probability > 1.0 - delta else "failed"
    return {
        "n": len(effects),
        "mean_effect": mean_effect,
        "lower_confidence_bound": lower_confidence_bound,
        "epsilon": epsilon,
        "successes": successes,
        "failures": failures,
        "prior_alpha": 1,
        "prior_beta": 1,
        "p_min": p_min,
        "delta": delta,
        "posterior_probability": posterior_probability,
        "scientific_gates_passed": scientific_ok,
        "outcome": outcome,
    }


def git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_manifest(payload: dict[str, Any], input_path: Path, output_path: Path, harness_revision: str) -> dict[str, Any]:
    result = evaluate_cases(payload["cases"], float(payload.get("epsilon", 0.0)), float(payload.get("p_min", 0.8)), float(payload.get("delta", 0.05)))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "rule_id": payload["rule_id"],
        "command": " ".join(["python", "scripts/run_rule_replay.py", str(input_path), str(output_path)]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_bundle_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
        "harness_revision": harness_revision,
        "result": result,
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
    print(json.dumps({"output": str(args.output), "outcome": result["outcome"], "result_digest": manifest["result_digest"]}))


if __name__ == "__main__":
    main()
