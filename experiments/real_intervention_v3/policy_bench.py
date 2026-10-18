from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .analyze_real import _time


POLICIES = ("NEVER", "ALWAYS", "HORIZON_ONLY", "SIMPLE_SUPERVISED", "ORIGINAL_ACRE", "REGIME_AWARE_ACRE", "ORACLE")
_PREEXEC_FEATURES = {"horizon", "batch", "parameter_count", "input_height", "input_width", "input_channels", "input_numel",
                     "sequence_length", "hidden_size", "num_heads", "depth", "width", "num_parameters",
                     "num_nodes", "num_edges", "num_features", "num_classes"}
_FORBIDDEN_FEATURE_PARTS = ("time", "latency", "benefit", "speedup", "saving", "output", "receipt", "status", "error", "candidate", "compile", "correctness", "oom", "session", "order", "split")


def _features(pair: Mapping[str, Any]) -> dict[str, Any]:
    raw = pair.get("features")
    values = dict(raw) if isinstance(raw, Mapping) else {}
    values.setdefault("horizon", pair.get("horizon"))
    for key in ("family", "model", "source", "split", "session", "order"):
        if key not in values and key in pair:
            values[key] = pair[key]
    return values


def _valid_candidate(pair: Mapping[str, Any]) -> tuple[bool, float | None, float | None]:
    s0, s1 = pair.get("S0"), pair.get("S1")
    if pair.get("status") != "VALID_PAIR" or not isinstance(s0, Mapping) or not isinstance(s1, Mapping):
        return False, None, None
    if s0.get("status") != "valid" or s1.get("status") != "valid":
        return False, None, None
    a = _time(s0, ("total_cold_inclusive_s", "total_cold_inclusive_seconds", "total_cold_inclusive_timing_seconds"))
    b = _time(s1, ("total_cold_inclusive_s", "total_cold_inclusive_seconds", "total_cold_inclusive_timing_seconds"))
    return a is not None and b is not None, a, b


def _candidate_failure(pair: Mapping[str, Any]) -> bool:
    return pair.get("status") in {"INTERVENTION_MODEL_UNSUPPORTED", "INTERVENTION_NUMERIC_MISMATCH"}


def _label(pair: Mapping[str, Any]) -> int | None:
    valid, baseline, candidate = _valid_candidate(pair)
    if valid and baseline is not None and candidate is not None and baseline > 0 and candidate > 0:
        return int(candidate < baseline)
    if _candidate_failure(pair):
        return 0
    return None


def _fit_horizon(dev: Sequence[Mapping[str, Any]]) -> float:
    samples = [(float(_features(row)["horizon"]), row, str(row.get("family", "UNKNOWN")))
               for row in dev if isinstance(_features(row).get("horizon"), (int, float)) and not isinstance(_features(row).get("horizon"), bool)]
    samples = [(x, row, family) for x, row, family in samples if _label(row) is not None and math.isfinite(x)]
    if not samples:
        return math.inf
    horizons = sorted({x for x, _, _ in samples})
    thresholds = horizons + [math.nextafter(horizons[-1], math.inf)]
    def score(threshold: float) -> tuple[float, float]:
        family_failures: dict[str, list[float]] = defaultdict(list)
        family_savings: dict[str, list[float]] = defaultdict(list)
        for horizon, row, family in samples:
            deploy = horizon >= threshold
            valid, baseline, candidate = _valid_candidate(row)
            family_failures[family].append(float(deploy and _candidate_failure(row)))
            if valid and baseline is not None and candidate is not None:
                family_savings[family].append(baseline - candidate if deploy else 0.0)
        failure_rate = statistics.fmean(statistics.fmean(values) for values in family_failures.values())
        family_means = [statistics.fmean(values) for values in family_savings.values() if values]
        mean_saving = statistics.fmean(family_means) if family_means else 0.0
        return -failure_rate, mean_saving
    return max(thresholds, key=lambda threshold: (score(threshold), threshold))


def _numeric_preexec(pair: Mapping[str, Any]) -> dict[str, float]:
    output = {}
    for key, value in _features(pair).items():
        lowered = str(key).lower()
        if key not in _PREEXEC_FEATURES or any(part in lowered for part in _FORBIDDEN_FEATURE_PARTS):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            output[str(key)] = float(value)
    return output


def _fit_stump(dev: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    samples = [(row, _label(row)) for row in dev]
    samples = [(row, label) for row, label in samples if label is not None]
    if samples and all(label == 0 for _, label in samples):
        return {"feature": None, "threshold": None, "positive_above": True, "fit_status": "NO_POSITIVE_OUTCOME"}
    feature_names = sorted({key for row, _ in samples for key in _numeric_preexec(row)})
    if not samples or not feature_names:
        return {"feature": None, "threshold": None, "positive_above": True, "fit_status": "NO_FIT_DATA"}
    family_n: dict[str, int] = defaultdict(int)
    for row, _ in samples:
        family_n[str(row.get("family", "UNKNOWN"))] += 1
    best: tuple[float, str, float, bool] | None = None
    for feature in feature_names:
        values = [(numeric[feature], label, str(row.get("family", "UNKNOWN")))
                  for row, label in samples if (numeric := _numeric_preexec(row)).get(feature) is not None]
        unique = sorted({value for value, _, _ in values})
        if len(unique) < 2:
            continue
        thresholds = [math.nextafter(unique[0], -math.inf)]
        thresholds.extend((left + right) / 2 for left, right in zip(unique, unique[1:]))
        thresholds.append(math.nextafter(unique[-1], math.inf))
        for threshold in thresholds:
            for positive_above in (True, False):
                group_scores: dict[str, list[float]] = defaultdict(list)
                for value, label, family in values:
                    prediction = int(value >= threshold) if positive_above else int(value < threshold)
                    group_scores[family].append(float(prediction == label))
                score = statistics.fmean(statistics.fmean(items) for items in group_scores.values())
                candidate = (score, feature, threshold, positive_above)
                if best is None or (candidate[0], -feature_names.index(candidate[1]), -candidate[2], candidate[3]) > (
                    best[0], -feature_names.index(best[1]), -best[2], best[3]
                ):
                    best = candidate
    if best is None:
        return {"feature": None, "threshold": None, "positive_above": True, "fit_status": "NO_SPLIT"}
    return {"feature": best[1], "threshold": best[2], "positive_above": best[3], "fit_status": "FITTED",
            "development_family_weighted_accuracy": best[0]}


def _acre_store(root: Path) -> Path | None:
    candidate = root / "acre_store"
    if candidate.is_dir():
        return candidate.resolve()
    return None


def _admitted_family_count(root: Path, dev: Sequence[Mapping[str, Any]], cal: Sequence[Mapping[str, Any]]) -> int:
    manifest_path = root / "FROZEN_MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        family_split = manifest.get("family_split")
        if isinstance(family_split, Mapping):
            return len(family_split)
    return 0


def freeze_policies(dev: list[dict[str, Any]], cal: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    """Fit deployable rules on development only and record calibration qualification."""
    root = Path(root)
    for row in dev:
        if str(row.get("split", "")).lower() in {"final", "test", "evaluation"}:
            raise ValueError("development input contains a held-out split")
    for row in cal:
        if str(row.get("split", "")).lower() in {"final", "test", "evaluation"}:
            raise ValueError("calibration input contains a held-out split")
    horizon = _fit_horizon(dev)
    stump = _fit_stump(dev)
    calibrated_families = {str(row.get("family", "UNKNOWN")) for row in cal if _label(row) is not None}
    valid_cal_families = {str(row.get("family", "UNKNOWN")) for row in cal if _valid_candidate(row)[0]}
    valid_dev_families = {str(row.get("family", "UNKNOWN")) for row in dev if _valid_candidate(row)[0]}
    store = _acre_store(root)
    regime_file = root / "acre_regimes.json"
    admitted_families = _admitted_family_count(root, dev, cal)
    final_eligible = bool(valid_dev_families) and admitted_families >= 18 and len(valid_cal_families) >= 5
    plan = {
        "schema": "evospe_v3_policy_freeze_v1",
        "status": "FROZEN" if final_eligible else "UNDERPOWERED_NOT_FINAL",
        "policies": list(POLICIES),
        "development": {"pairs": len(dev), "families": len({str(row.get("family", "UNKNOWN")) for row in dev}),
                        "horizon_threshold": horizon if math.isfinite(horizon) else None,
                        "simple_supervised": stump,
                        "fit_labels": "valid measured log benefit > 0; candidate failure is ineligible, never assigned synthetic seconds"},
        "calibration": {"pairs": len(cal), "families_with_eligible_outcome": len(calibrated_families),
                        "valid_candidate_families": len(valid_cal_families), "minimum_valid_families_for_final": 5},
        "release_qualification": {"development_nonempty": bool(valid_dev_families), "valid_development_families": len(valid_dev_families), "admitted_families": admitted_families,
                                  "minimum_admitted_families": 18, "minimum_valid_calibration_families": 5,
                                  "valid_calibration_families": len(valid_cal_families), "final_eligible": final_eligible},
        "preexecution_feature_exclusions": list(_FORBIDDEN_FEATURE_PARTS),
        "preexecution_feature_schema": sorted(_PREEXEC_FEATURES),
        "horizon_objective": "minimize family-equal candidate-failure deployment rate, then maximize family-equal measured net seconds; failures have no invented duration",
        "deployment_risk": "candidate runtime or numerical failure after deployment is unsafe; missing deployed outcomes make total net seconds unknown",
        "acre": {"store": str(store) if store else None, "regime_file": str(regime_file.resolve()) if regime_file.is_file() else None,
                 "configured": store is not None, "regimes_configured": regime_file.is_file(),
                 "status": "CANONICAL_STORE_CONFIGURED" if store else "ACRE_NOT_FITTED",
                 "reason": None if store else "paired timing rows do not provide canonical RuleSpec/RuleState or support certificates; no ACRE artifacts were synthesized",
                 "unconfigured_behavior": "abstain"},
        "fit_split": "development",
        "provenance": {
            "run_identity": json.loads((root / "RUN_IDENTITY.json").read_text(encoding="utf-8")) if (root / "RUN_IDENTITY.json").is_file() else None,
            "manifest_digest": json.loads((root / "FROZEN_MANIFEST.json").read_text(encoding="utf-8")).get("manifest_digest") if (root / "FROZEN_MANIFEST.json").is_file() else None,
        },
        "calibration_use": "independent-family qualification only; no fit threshold is changed using calibration outcomes",
        "held_out_policy": "no final labels are read by freeze_policies",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "POLICY_FREEZE.json").write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return plan


def _acre_route(pair: Mapping[str, Any], frozen: Mapping[str, Any], *, regime_aware: bool) -> tuple[bool, float, str]:
    from core.acre.engine import AcreEngine
    from core.acre.regimes import SupportRegime, SupportRegimeDecision, SupportScope

    acre = frozen.get("acre", {})
    store = acre.get("store") if isinstance(acre, Mapping) else None
    if not store or not acre.get("configured"):
        return False, 0.0, "ACRE_NOT_FITTED"
    engine = AcreEngine.from_store(store)
    features = _features(pair)
    context = pair.get("acre_context", features.get("acre_context"))
    if not isinstance(context, Mapping):
        return False, 0.0, "missing_preexecution_context"
    regimes = None
    versions = None
    if regime_aware:
        regime_path = acre.get("regime_file") if isinstance(acre, Mapping) else None
        if not regime_path:
            return False, 0.0, "regimes_not_configured"
        raw = json.loads(Path(regime_path).read_text(encoding="utf-8"))
        regimes = {}
        for rule_id, value in raw.items():
            scope_raw = dict(value["scope"])
            scope_raw["required_strata"] = tuple(scope_raw.get("required_strata", ()))
            scope = SupportScope(**scope_raw)
            regimes[rule_id] = SupportRegimeDecision(
                regime=SupportRegime(value["regime"]), action=value["action"], reason_code=value["reason_code"], scope=scope,
                **{key: value[key] for key in ("predicate", "independent_groups", "net_benefit_lcb", "harm_ucb", "stratum_bounds",
                                                "source_certificate_digest", "certificate_scope_digest", "certificate_digest",
                                                "decision_features", "decision_contexts") if key in value},
            )
        versions = next(iter(regimes.values())).scope.versions if regimes else {}
    started = time.perf_counter()
    decision = engine.route(context, support_regimes=regimes, runtime_versions=versions)
    elapsed = time.perf_counter() - started
    return bool(decision.selected_rule_ids), elapsed, "routed" if decision.selected_rule_ids else "canonical_gate_abstain"


def _deploy(policy: str, pair: Mapping[str, Any], frozen: Mapping[str, Any]) -> tuple[bool, float, str]:
    features = _features(pair)
    for key in ("static_compatible", "intervention_applicable", "correctness_policy_allowed"):
        if features.get(key) is False:
            return False, 0.0, f"{key}_false"
    if policy == "NEVER":
        return False, 0.0, "never"
    if policy == "ALWAYS":
        return True, 0.0, "always"
    if policy == "HORIZON_ONLY":
        threshold = frozen["development"].get("horizon_threshold")
        value = features.get("horizon")
        return bool(threshold is not None and isinstance(value, (int, float)) and not isinstance(value, bool) and value >= threshold), 0.0, "horizon_rule"
    if policy == "SIMPLE_SUPERVISED":
        stump = frozen["development"]["simple_supervised"]
        feature = stump.get("feature")
        values = _numeric_preexec(pair)
        if not feature or feature not in values or stump.get("threshold") is None:
            return False, 0.0, "stump_abstain"
        above = values[feature] >= float(stump["threshold"])
        return bool(above == bool(stump["positive_above"])), 0.0, "stump_rule"
    if policy == "ORIGINAL_ACRE":
        return _acre_route(pair, frozen, regime_aware=False)
    if policy == "REGIME_AWARE_ACRE":
        return _acre_route(pair, frozen, regime_aware=True)
    if policy == "ORACLE":
        valid, baseline, candidate = _valid_candidate(pair)
        return bool(valid and baseline is not None and candidate is not None and candidate < baseline), 0.0, "scoring_only_oracle"
    raise ValueError(f"unknown policy: {policy}")


def _family_ci(rows: Sequence[Mapping[str, Any]], *, replicates: int = 10_000, seed: int = 101) -> dict[str, Any] | None:
    import random
    by_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(float(row["observed_saving_seconds"]))
    if not by_family:
        return None
    means = {family: statistics.fmean(values) for family, values in by_family.items()}
    if len(means) == 1:
        return {"lower": None, "upper": None, "status": "INSUFFICIENT_INDEPENDENT_FAMILIES",
                "inferential": False, "unit": "family", "families": 1, "seed": seed, "replicates": replicates}
    names = sorted(means)
    rng = random.Random(seed)
    draws = sorted(statistics.fmean(means[names[rng.randrange(len(names))]] for _ in names) for _ in range(replicates))
    return {"lower": draws[int(0.025 * (replicates - 1))], "upper": draws[int(0.975 * (replicates - 1))],
            "unit": "family", "estimand": "family-equal mean net seconds per pair",
            "families": len(names), "seed": seed, "replicates": replicates}


def evaluate_policies(pairs: list[dict[str, Any]], frozen: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate frozen decisions by split; outcome fields are read only after routing."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[str(pair.get("split", "UNKNOWN"))].append(pair)
    results = []
    for split, rows in sorted(groups.items()):
        for policy in POLICIES:
            routed = []
            for pair in rows:
                deploy, overhead, reason = _deploy(policy, pair, frozen)
                valid, baseline, candidate = _valid_candidate(pair)
                failure = _candidate_failure(pair)
                observed = baseline - candidate if valid and baseline is not None and candidate is not None else None
                harm = failure or (observed is not None and observed < 0)
                routed.append({"pair_id": pair.get("pair_id"), "family": str(pair.get("family", "UNKNOWN")),
                               "deploy": bool(deploy), "decision_reason": reason, "gate_overhead_seconds": overhead,
                               "harm": bool(deploy and harm), "unsafe_deployment": bool(deploy and failure),
                               "missing": not valid, "candidate_failure": failure,
                               "observed_saving_seconds": observed if deploy and observed is not None else 0.0,
                "measured_saving_seconds": observed,
                               "deployed_outcome_missing": bool(deploy and not valid),
                               "candidate_correct": pair.get("S1", {}).get("status") == "valid" if isinstance(pair.get("S1"), Mapping) else False,
                               "observed_log_benefit": math.log(baseline / candidate) if deploy and valid and baseline and candidate else None})
            n = len(routed)
            deployed = [row for row in routed if row["deploy"]]
            missing_deployed = any(row["deployed_outcome_missing"] for row in deployed)
            unknown_deployed = sum(row["deployed_outcome_missing"] for row in deployed)
            unknown_harm = sum(row["deployed_outcome_missing"] and not row["candidate_failure"] for row in deployed)
            observed_deployed_harm = sum(row["harm"] for row in deployed)
            harm_known = unknown_deployed == 0
            conditional_harm_rate = observed_deployed_harm / len(deployed) if deployed and harm_known else None
            unconditional_harm_rate = sum(row["harm"] for row in routed) / n if n and harm_known else None
            observed_net = sum(row["observed_saving_seconds"] for row in routed)
            all_known = not missing_deployed
            ci_rows = [dict(row, observed_saving_seconds=row["observed_saving_seconds"] - row["gate_overhead_seconds"])
                       for row in routed]
            ci = _family_ci(ci_rows) if all_known else None
            results.append({
                "policy": policy, "split": split, "role": "scoring_only" if policy == "ORACLE" else "deployable",
                "n_pairs": n,
                "coverage": len(deployed) / n if n else None,
                "deployed": len(deployed), "harm_unconditional_count": sum(row["harm"] for row in routed),
                "harm_unconditional_rate": unconditional_harm_rate,
                "harm_unconditional_rate_lower": sum(row["harm"] for row in routed) / n if n else None,
                "harm_unconditional_rate_upper": (sum(row["harm"] for row in routed) + unknown_harm) / n if n else None,
                "harm_deployed_count": observed_deployed_harm,
                "unknown_deployed_count": unknown_deployed,
                "conditional_harm_rate": conditional_harm_rate,
                "conditional_harm_rate_lower": observed_deployed_harm / len(deployed) if deployed else None,
                "conditional_harm_rate_upper": (observed_deployed_harm + unknown_harm) / len(deployed) if deployed else None,
                "unsafe_deployment_count": sum(row["unsafe_deployment"] for row in routed),
                "candidate_failure_count": sum(row["candidate_failure"] for row in routed),
                "candidate_correct_count": sum(row["candidate_correct"] for row in routed),
                "missing_count": sum(row["missing"] for row in routed),
                "missed_benefit_seconds": sum(max(0.0, float(row["measured_saving_seconds"])) for row in routed
                                               if not row["deploy"] and row["measured_saving_seconds"] is not None),
                "observed_net_seconds_vs_never": observed_net,
                "total_net_seconds_vs_never": 0.0 if policy == "NEVER" else (observed_net if all_known else None),
                "net_after_gate_seconds": 0.0 if policy == "NEVER" else (observed_net - sum(row["gate_overhead_seconds"] for row in routed) if all_known else None),
                "gate_overhead_seconds": sum(row["gate_overhead_seconds"] for row in routed),
                "family_bootstrap_95_ci_seconds": ci,
                "log_benefit": [row["observed_log_benefit"] for row in routed if row["observed_log_benefit"] is not None],
                "rows": routed,
            })
    return results


def write_policy_csv(path: Path, policy_results: Sequence[Mapping[str, Any]]) -> None:
    fields = ("policy", "split", "n_pairs", "coverage", "deployed", "harm_unconditional_count", "harm_unconditional_rate",
              "harm_unconditional_rate_lower", "harm_unconditional_rate_upper", "harm_deployed_count", "unknown_deployed_count",
              "conditional_harm_rate", "conditional_harm_rate_lower", "conditional_harm_rate_upper", "unsafe_deployment_count", "candidate_failure_count",
              "candidate_correct_count", "missing_count", "missed_benefit_seconds", "observed_net_seconds_vs_never",
              "total_net_seconds_vs_never", "net_after_gate_seconds", "gate_overhead_seconds", "family_bootstrap_95_ci_seconds")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in policy_results:
            writer.writerow({**result, "family_bootstrap_95_ci_seconds": json.dumps(result.get("family_bootstrap_95_ci_seconds"), sort_keys=True)})
