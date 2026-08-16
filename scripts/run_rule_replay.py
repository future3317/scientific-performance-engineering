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
from core.sequential_stats import bounded_mean_interval, mixture_lower_bound
from core.models import validate_identifier
from core.utility import UTILITY_LOG_SCALE, UTILITY_POLICY_ID, utility_effect, validate_policy


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
    """Compatibility name for the active Beta-Binomial mixture boundary."""
    if trials < 1 or not 0 < delta < 1:
        raise ValueError("trials must be positive and delta must be in (0, 1)")
    return mixture_lower_bound(successes, trials, delta)


def betting_lower_bound(successes: int, trials: int, delta: float) -> float:
    """Public replay gate using the beta-binomial mixture boundary."""
    return mixture_lower_bound(successes, trials, delta)


def evaluate_cases(
    cases: list[dict[str, Any]],
    epsilon: float,
    p_min: float,
    delta: float,
    utility_policy_id: str = UTILITY_POLICY_ID,
    utility_scale: float = UTILITY_LOG_SCALE,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("replay requires at least one paired case")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("replay cases must be objects")
        if not case.get("paired_replay"):
            raise ValueError("replay cases must be harness-owned paired interventions")
        if case.get("paired_replay") and not case.get("same_fixture_id"):
            raise ValueError("paired replay case must identify its shared fixture")
        measured = case.get("intervention_measurements")
        control = case.get("baseline_measurements")
        if measured is not None or control is not None:
            if not isinstance(measured, list) or not isinstance(control, list) or not measured or not control:
                raise ValueError("paired replay measured arms must be non-empty lists")
            if len(measured) != len(control):
                raise ValueError("paired replay measured arms must have equal length")
            if case.get("control_measured") is not True:
                raise ValueError("paired replay control_measured attestation is required")
        elif "utility_on" not in case or "utility_off" not in case:
            raise ValueError("paired replay case must contain measured arms")
        elif float(case.get("utility_off", 0.0)) == 0.0:
            # Scalar-only legacy payloads cannot establish a control arm.  A
            # measured baseline may legitimately be zero and is accepted above.
            raise ValueError("paired replay requires a non-zero control for legacy scalar cases; measured arms may be zero")
    validate_policy(utility_policy_id)
    case_effects: list[tuple[str, float, bool]] = []
    repetition_count = 0
    for case in cases:
        group_id = str(case.get("independence_group") or case.get("source_id") or case.get("case_id"))
        intervention = case.get("intervention_measurements")
        baseline = case.get("baseline_measurements")
        if isinstance(intervention, list) and isinstance(baseline, list):
            higher_is_better = bool(case.get("higher_is_better", True))
            repetition_effects = []
            for on, off in zip(intervention, baseline):
                on_value, off_value = float(on), float(off)
                repetition_effects.append(utility_effect(
                    on_value,
                    off_value,
                    higher_is_better=higher_is_better,
                    log_scale=float(case.get("utility_scale", utility_scale)),
                ))
            repetition_count += len(repetition_effects)
            case_effect = mean(repetition_effects)
        else:
            repetition_count += 1
            case_effect = utility_effect(
                float(case["utility_on"]),
                float(case["utility_off"]),
                higher_is_better=bool(case.get("higher_is_better", True)),
                log_scale=utility_scale,
            )
        case_effects.append((group_id, case_effect, bool(case.get("scientific_ok", False)) and bool(case.get("quality_ok", True))))
    grouped: dict[str, list[tuple[float, bool]]] = {}
    for group_id, effect, gates_passed in case_effects:
        grouped.setdefault(group_id, []).append((effect, gates_passed))
    effects = [mean(effect for effect, _ in group_cases) for group_cases in grouped.values()]
    group_quality = [all(gates_passed for _, gates_passed in group_cases) for group_cases in grouped.values()]
    scientific_ok = all(group_quality)
    successes = sum(effect > epsilon and gates_passed for effect, gates_passed in zip(effects, group_quality))
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
    # This descriptive interval is not used for promotion; promotion uses the
    # time-uniform Bernoulli boundary below.
    lower_confidence_bound = mean_effect - 1.96 * standard_error
    upper_confidence_bound = mean_effect + 1.96 * standard_error
    utility_effect_lcb, utility_effect_ucb = bounded_mean_interval(effects, delta)
    promotion_probability_lower_bound = betting_lower_bound(successes, len(effects), delta)
    outcome = "passed" if scientific_ok and mean_effect > epsilon and promotion_probability_lower_bound >= p_min else "failed"
    return {
        "n": len(effects),
        "case_count": len(case_effects),
        "repetition_count": repetition_count,
        "independence_group_count": len(effects),
        "mean_effect": mean_effect,
        "utility_policy_id": utility_policy_id,
        "utility_scale": utility_scale,
        "lower_confidence_bound": lower_confidence_bound,
        "upper_confidence_bound": upper_confidence_bound,
        "utility_effect_lcb": utility_effect_lcb,
        "utility_effect_ucb": utility_effect_ucb,
        "epsilon": epsilon,
        "successes": successes,
        "failures": failures,
        "prior_alpha": 1,
        "prior_beta": 1,
        "p_min": p_min,
        "delta": delta,
        "posterior_probability": posterior_probability,
        "promotion_probability_lower_bound": promotion_probability_lower_bound,
        "confidence_method": "beta-binomial-mixture-e-process",
        "scientific_gates_passed": scientific_ok,
        "outcome": outcome,
    }


def build_evidence_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialize paired replay directly as canonical EvidenceEvent v2."""
    events: list[dict[str, Any]] = []
    for index, case in enumerate(payload["cases"]):
        common = dict(case.get("context") or {})
        common.setdefault("revision", case.get("revision", payload.get("revision", "unknown")))
        common.setdefault("seed_family", case.get("seed_family", payload.get("seed_family", "replay")))
        common.setdefault("rule_versions", {})
        versions = dict(common["rule_versions"])
        versions[payload["rule_id"]] = int(payload.get("rule_version", 1))
        common["rule_versions"] = versions
        on_value = float(case["utility_on"])
        off_value = float(case["utility_off"])
        paired_effect = utility_effect(
            on_value,
            off_value,
            higher_is_better=bool(case.get("higher_is_better", True)),
            log_scale=float(case.get("utility_scale", UTILITY_LOG_SCALE)),
        )
        for arm in ("on", "off"):
            events.append({
                "schema_version": 2,
                "event_id": f"{case.get('case_id', index)}-{arm}", "context": common,
                "assignment": {"interventions": {payload["rule_id"]: int(arm == "on")}, "propensity": float(case.get("propensity", 0.5)), "design_id": "paired-replay-v2"},
                "evidence_stream": "representative", "query_id": str(case.get("query_id", case.get("case_id", index))),
                "outcome_vector": {"utility": paired_effect if arm == "on" else 0.0, "paired_effect": paired_effect},
                "scientific_gates": {"scientific_ok": bool(case.get("scientific_ok", False)), "quality_ok": bool(case.get("quality_ok", True))},
                "artifacts": case.get("artifacts", {}), "versions": case.get("versions", {}),
                "source_id": case.get("source_id", f"replay-{index}"), "independence_group": case.get("independence_group", f"replay-{index}"),
                "timestamp": case.get("timestamp", datetime.now(timezone.utc).isoformat()), "trust_zone": "local", "attacker_controlled_fields": [],
            })
    return events


def git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_manifest(payload: dict[str, Any], input_path: Path, output_path: Path, harness_revision: str) -> dict[str, Any]:
    validate_identifier(payload.get("rule_id"), "rule_id")
    result = evaluate_cases(
        payload["cases"],
        float(payload.get("epsilon", 0.0)),
        float(payload.get("p_min", 0.8)),
        float(payload.get("delta", 0.05)),
        str(payload.get("utility_policy_id", UTILITY_POLICY_ID)),
        float(payload.get("utility_scale", UTILITY_LOG_SCALE)),
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
