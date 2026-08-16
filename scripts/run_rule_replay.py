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
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.sequential_stats import bounded_mean_interval, mixture_lower_bound, minimum_all_successes, paired_repetition_interval
from core.models import validate_identifier
from core.utility import UTILITY_LOG_SCALE, UTILITY_POLICY_ID, utility_effect, validate_policy


@dataclass(frozen=True)
class GroupEffectCertificate:
    """Within-context paired certificate used as one promotion Bernoulli trial."""

    independence_group: str
    effect: float
    lcb: float
    ucb: float
    n_repetitions: int
    scientific_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "independence_group": self.independence_group,
            "effect": self.effect,
            "lcb": self.lcb,
            "ucb": self.ucb,
            "n_repetitions": self.n_repetitions,
            "scientific_ok": self.scientific_ok,
        }


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


def paired_group_effects(
    cases: list[dict[str, Any]],
    *,
    utility_scale: float = UTILITY_LOG_SCALE,
) -> list[dict[str, Any]]:
    """Reduce repetitions and cases to one bounded effect per independence group."""
    grouped: dict[str, list[tuple[float, bool, str, int]]] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError("replay cases must be objects")
        group_id = str(case.get("independence_group") or case.get("source_id") or case.get("case_id") or f"group-{index}")
        measured = case.get("intervention_measurements")
        control = case.get("baseline_measurements")
        higher_is_better = bool(case.get("higher_is_better", True))
        if isinstance(measured, list) and isinstance(control, list):
            if not measured or len(measured) != len(control):
                raise ValueError("paired replay measured arms must be non-empty and equal length")
            repetition_effects = [utility_effect(
                float(on), float(off), higher_is_better=higher_is_better,
                log_scale=float(case.get("utility_scale", utility_scale)),
            ) for on, off in zip(measured, control)]
            case_effect = mean(repetition_effects)
            repetition_count = len(repetition_effects)
        else:
            if "utility_on" not in case or "utility_off" not in case:
                raise ValueError("paired replay case must contain measured arms")
            case_effect = utility_effect(
                float(case["utility_on"]), float(case["utility_off"]),
                higher_is_better=higher_is_better, log_scale=utility_scale,
            )
            repetition_count = 1
        gates_passed = bool(case.get("scientific_ok", False)) and bool(case.get("quality_ok", True))
        grouped.setdefault(group_id, []).append((case_effect, gates_passed, str(case.get("case_id", index)), repetition_count))
    results: list[dict[str, Any]] = []
    for group_id, entries in grouped.items():
        effects = [entry[0] for entry in entries]
        effects_for_interval = []
        for entry in entries:
            # Reconstruct repetition effects only when the source case carries
            # paired measurements; scalar cases remain uncertifiable.
            case = next(item for item in cases if str(item.get("case_id", "")) == entry[2])
            measured = case.get("intervention_measurements")
            control = case.get("baseline_measurements")
            if isinstance(measured, list) and isinstance(control, list):
                effects_for_interval.extend(
                    utility_effect(float(on), float(off), higher_is_better=bool(case.get("higher_is_better", True)), log_scale=float(case.get("utility_scale", utility_scale)))
                    for on, off in zip(measured, control)
                )
        if effects_for_interval:
            # Equal observed repetitions are not mathematical proof of zero
            # uncertainty for timing/performance measurements.  Every
            # non-deterministic observable therefore uses the predeclared
            # paired interval, including zero empirical-variance samples.
            lcb, ucb = paired_repetition_interval(effects_for_interval, 0.05)
        else:
            lcb, ucb = -1.0, 1.0
        results.append({
            "independence_group": group_id,
            "effect": mean(effects),
            "lcb": lcb,
            "ucb": ucb,
            "scientific_ok": all(entry[1] for entry in entries),
            "case_ids": [entry[2] for entry in entries],
            "case_count": len(entries),
            "repetition_count": sum(entry[3] for entry in entries),
        })
    return results


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
    group_effects = paired_group_effects(cases, utility_scale=utility_scale)
    effects = [float(item["effect"]) for item in group_effects]
    group_quality = [bool(item["scientific_ok"]) for item in group_effects]
    repetition_count = sum(int(item["repetition_count"]) for item in group_effects)
    scientific_ok = all(group_quality)
    certificates = [item for item in group_effects]
    successes = sum(
        (float(item["effect"]) > epsilon if p_min <= 0.0 else float(item["lcb"]) > epsilon)
        and bool(item["scientific_ok"])
        for item in certificates
    )
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
    minimum_replay_groups = minimum_all_successes(p_min, delta) if p_min > 0.0 else 1
    outcome = "passed" if (
        scientific_ok
        and mean_effect > epsilon
        and len(effects) >= minimum_replay_groups
        and promotion_probability_lower_bound >= p_min
    ) else "failed"
    return {
        "n": len(effects),
        "case_count": len(cases),
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
        "minimum_replay_groups": minimum_replay_groups,
        "replay_groups_sufficient": len(effects) >= minimum_replay_groups,
        "confidence_method": "beta-binomial-mixture-e-process",
        "scientific_gates_passed": scientific_ok,
        "outcome": outcome,
    }


def build_evidence_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialize paired replay directly as canonical EvidenceEvent v2."""
    events: list[dict[str, Any]] = []
    grouped = paired_group_effects(
        payload["cases"],
        utility_scale=float(payload.get("utility_scale", UTILITY_LOG_SCALE)),
    )
    cases_by_group = {
        str(case.get("independence_group") or case.get("source_id") or case.get("case_id") or f"group-{index}"): case
        for index, case in enumerate(payload["cases"])
    }
    for index, group in enumerate(grouped):
        case = cases_by_group[group["independence_group"]]
        common = dict(case.get("context") or {})
        common.setdefault("revision", case.get("revision", payload.get("revision", "unknown")))
        common.setdefault("seed_family", case.get("seed_family", payload.get("seed_family", "replay")))
        common.setdefault("rule_versions", {})
        versions = dict(common["rule_versions"])
        versions[payload["rule_id"]] = int(payload.get("rule_version", 1))
        common["rule_versions"] = versions
        for arm in ("on", "off"):
            events.append({
                "schema_version": 2,
                "event_id": f"{group['independence_group']}-{arm}", "context": common,
                "assignment": {"interventions": {payload["rule_id"]: int(arm == "on")}, "propensity": float(case.get("propensity", 0.5)), "design_id": "paired-replay-v2"},
                "evidence_stream": "representative", "query_id": str(case.get("query_id", group["independence_group"])),
                "outcome_vector": {"utility": float(case.get("utility_on", 0.0)) if arm == "on" else float(case.get("utility_off", 0.0)), "paired_effect": float(group["effect"]), "contrast": "on-minus-off"},
                "scientific_gates": {"scientific_ok": bool(group["scientific_ok"]), "quality_ok": bool(group["scientific_ok"])},
                "artifacts": {**dict(case.get("artifacts", {})), "paired_contrast": {"effect": float(group["effect"]), "lcb": float(group["lcb"]), "ucb": float(group["ucb"]), "n_repetitions": int(group["repetition_count"])}}, "versions": case.get("versions", {}),
                "source_id": case.get("source_id", f"replay-{index}"), "independence_group": group["independence_group"],
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
        "execution_source": "synthetic_family",
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
