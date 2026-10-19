"""Development-frozen simple policies and held-out policy scoring."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .effect_dataset import family_cluster_bootstrap_ci

_SIMPLE_POLICIES = (
    "NEVER", "ALWAYS", "APPLICABILITY_RULE", "HORIZON_THRESHOLD",
    "FAMILY_OR_SHAPE_HEURISTIC", "SIMPLE_SUPERVISED",
)


def _context(pair: Mapping[str, Any]) -> Mapping[str, Any]:
    value = pair.get("context")
    return value if isinstance(value, Mapping) else {}


def _input_dims(pair: Mapping[str, Any]) -> list[Any] | None:
    context = _context(pair)
    qualification = context.get("qualification")
    qualification = qualification if isinstance(qualification, Mapping) else {}
    summary = qualification.get("model_summary")
    summary = summary if isinstance(summary, Mapping) else {}
    tensors = context.get("actual_input_shapes") or summary.get("input_shapes") or {}
    if not isinstance(tensors, Mapping):
        return None
    return next((dims for dims in tensors.values()
                 if isinstance(dims, list) and len(dims) == 4), None)


def _feature(pair: Mapping[str, Any], name: str) -> float | None:
    context = _context(pair)
    qualification = context.get("qualification")
    qualification = qualification if isinstance(qualification, Mapping) else {}
    summary = qualification.get("model_summary")
    summary = summary if isinstance(summary, Mapping) else {}
    shape = context.get("shape") or summary.get("input_shapes")
    if name == "horizon":
        value = context.get("horizon")
    elif name == "batch":
        value = context.get("batch")
    elif name == "conv2d_module_count":
        value = context.get("conv2d_module_count", qualification.get("conv2d_count"))
    elif name == "input_channels":
        dims = _input_dims(pair)
        value = context.get("input_channels", dims[1] if dims else None)
    elif name == "input_height":
        dims = _input_dims(pair)
        value = dims[-2] if dims else None
    elif name == "input_width":
        dims = _input_dims(pair)
        value = dims[-1] if dims else None
    else:
        raise ValueError(f"unknown frozen policy feature: {name}")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _eligible(pair: Mapping[str, Any]) -> bool:
    context = _context(pair)
    qualification = context.get("qualification")
    if not isinstance(qualification, Mapping):
        return False
    return qualification.get("eligibility") == "ELIGIBLE_CHANNELS_LAST_CORRECTNESS_ONLY"


def _development_rows(pairs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [pair for pair in pairs
            if pair.get("split") == "development"
            and str(pair.get("family_id", "")).strip()]


def _training_score(rows: Sequence[Mapping[str, Any]], deploy) -> tuple[float | None, int]:
    by_family: dict[str, list[float]] = defaultdict(list)
    for pair in rows:
        family = str(pair.get("family_id", ""))
        if not family:
            continue
        if not deploy(pair):
            s0 = pair.get("S0") if isinstance(pair.get("S0"), Mapping) else {}
            if (s0.get("arm_execution") != "COMPLETE"
                    or s0.get("scientific_correctness") != "PASS"):
                return None, len(by_family)
            by_family[family].append(0.0)
            continue
        s1 = pair.get("S1") if isinstance(pair.get("S1"), Mapping) else {}
        if (pair.get("effect_status") == "unsafe"
                or s1.get("scientific_correctness") == "FAIL"):
            return None, len(by_family)
        benefit = pair.get("log_benefit")
        if (pair.get("effect_status") != "valid"
                or isinstance(benefit, bool)
                or not isinstance(benefit, (int, float))
                or not math.isfinite(float(benefit))):
            return None, len(by_family)
        by_family[family].append(float(benefit))
    means = [sum(values) / len(values) for values in by_family.values() if values]
    return (sum(means) / len(means) if means else None, len(by_family))


def _fit_supervised_grid(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select a one-feature rule using only the supplied development families."""
    features = ("horizon", "batch", "conv2d_module_count", "input_channels",
                "input_height", "input_width")
    candidates = []
    for feature in features:
        observed = sorted({value for pair in rows
                           if (value := _feature(pair, feature)) is not None})
        thresholds = [observed[0] - 1.0, *observed, observed[-1] + 1.0] if observed else []
        for threshold in thresholds:
            for direction in ("ge", "lt"):
                def deploy(pair, f=feature, t=threshold, d=direction):
                    value = _feature(pair, f)
                    return (_eligible(pair) and value is not None
                            and (value >= t if d == "ge" else value < t))
                score, families = _training_score(rows, deploy)
                if score is not None:
                    deployed = sum(bool(deploy(pair)) for pair in rows)
                    candidates.append((score, -deployed, families, feature, threshold, direction))
    if not candidates:
        return {"status": "NOT_FITTED", "feature": None, "threshold": None,
                "direction": None, "development_families": 0}
    score, _, families, feature, threshold, direction = max(
        candidates, key=lambda item: (item[0], item[1], item[2], item[3], -item[4], item[5]),
    )
    return {"status": "FITTED", "feature": feature, "threshold": threshold,
            "direction": direction, "development_families": families,
            "development_family_equal_log_benefit": score}


def _fit_threshold(rows: Sequence[Mapping[str, Any]], feature: str) -> dict[str, Any]:
    values = sorted({value for pair in rows if (value := _feature(pair, feature)) is not None})
    candidates = [values[0] - 1.0, *values, values[-1] + 1.0] if values else []
    scored = []
    for threshold in candidates:
        deploy = lambda pair, t=threshold: (
            _eligible(pair) and (_feature(pair, feature) or 0.0) >= t
        )
        score, families = _training_score(
            rows, deploy,
        )
        if score is not None:
            deployed = sum(bool(deploy(pair)) for pair in rows)
            scored.append((score, -deployed, threshold, families))
    if not scored:
        return {"status": "NOT_FITTED", "feature": feature, "threshold": None,
                "development_families": 0, "selection_scope": "development_only"}
    score, _, threshold, families = max(scored)
    return {"status": "FITTED", "feature": feature, "threshold": threshold,
            "development_families": families,
            "development_family_equal_log_benefit": score,
            "selection_scope": "development_only"}


def _fit_supervised_stump(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # Keep this boundary even when called directly: only development outcomes
    # may be used, while every planned development context remains in scope.
    rows = _development_rows(rows)
    families = sorted({str(pair.get("family_id", "")) for pair in rows} - {""})
    folds = []
    for held_out in families:
        training = [pair for pair in rows if str(pair.get("family_id", "")) != held_out]
        held_rows = [pair for pair in rows if str(pair.get("family_id", "")) == held_out]
        fitted = _fit_supervised_grid(training)
        if fitted["status"] != "FITTED":
            folds.append({"held_out_family": held_out, "status": "NOT_FITTED"})
            continue
        threshold = float(fitted["threshold"])
        feature = str(fitted["feature"])
        direction = str(fitted["direction"])
        def deploy(pair):
            feature_value = _feature(pair, feature)
            return (_eligible(pair) and feature_value is not None
                    and (feature_value >= threshold if direction == "ge" else feature_value < threshold))

        fold_score, _ = _training_score(held_rows, deploy)
        folds.append({
            "held_out_family": held_out,
            "status": "SCORED" if fold_score is not None else "HARD_QUALIFICATION_FAILED",
            "family_mean_log_benefit": fold_score,
            "training_families": fitted["development_families"],
            "training_family_ids": sorted({str(pair["family_id"]) for pair in training}),
            "selected_feature": feature, "selected_threshold": threshold,
            "selected_direction": direction,
        })
    final_fit = _fit_supervised_grid(rows)
    if final_fit["status"] != "FITTED":
        return {"status": "NOT_FITTED", "feature": None, "threshold": None,
                "direction": None, "development_families": 0,
                "training_contract": "pre-execution-only; leave-one-family-out CV"}
    cv_scores = [fold["family_mean_log_benefit"] for fold in folds
                 if fold.get("family_mean_log_benefit") is not None]
    return {
        **final_fit,
        "training_contract": "pre-execution-only; leave-one-family-out family-aware CV",
        "cross_validation": {
            "method": "leave_one_independent_family_out",
            "fold_count": len(folds),
            "scored_fold_count": len(cv_scores),
            "family_equal_mean_log_benefit": sum(cv_scores) / len(cv_scores) if cv_scores else None,
            "folds": folds,
        },
    }


def freeze_policy_set(development_pairs: Sequence[Mapping[str, Any]], *,
                      calibration_pairs: Sequence[Mapping[str, Any]],
                      acre_status: str, acre_rule_id: str | None = None,
                      acre_route_decisions: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Fit and select policies on development; calibration/final outcomes are not read."""
    dev = _development_rows(development_pairs)
    horizon = _fit_threshold(dev, "horizon")
    supervised = _fit_supervised_stump(dev)
    models = {
        "APPLICABILITY_RULE": {"status": "FROZEN",
                               "predicate": "prequalified 4D Conv2d channels-last path"},
        "HORIZON_THRESHOLD": horizon,
        "FAMILY_OR_SHAPE_HEURISTIC": {
            "status": "FROZEN", "predicate": "eligible and batch >= 8",
            "source": "predeclared_static_heuristic",
        },
        "SIMPLE_SUPERVISED": supervised,
        "ORIGINAL_ACRE": {
            "status": acre_status, "rule_id": acre_rule_id,
            "route_decisions": dict((acre_route_decisions or {}).get("ORIGINAL_ACRE", {})),
        },
        "REGIME_AWARE_ACRE": {
            "status": acre_status, "rule_id": acre_rule_id,
            "route_decisions": dict((acre_route_decisions or {}).get("REGIME_AWARE_ACRE", {})),
        },
    }
    baseline_cv = _cross_validated_simple_baselines(dev, acre_status=acre_status)
    strongest = baseline_cv["selected_policy"]
    return {
        "schema_version": 1,
        "training_split": "development",
        "calibration_role": "not_used_for_policy_or_comparator_selection",
        "policies": models,
        "acre_status": acre_status,
        "acre_rule_id": acre_rule_id,
        "strongest_simple_baseline": strongest,
        "strongest_simple_selection_split": "development",
        "strongest_simple_selection_method": "leave_one_independent_family_out",
        "strongest_simple_cross_validation": baseline_cv,
        "calibration_outcomes_read": False,
        "final_outcomes_read": False,
        "strongest_simple_selection_rule": (
            "development family-CV only; every fold must have zero known unsafe deployments, "
            "zero missing deployed outcomes, zero missing non-deployed baselines, "
            "zero unknown decisions, and a score; then maximize "
            "family-equal mean log benefit"
        ),
    }


def _deploy(policy: str, pair: Mapping[str, Any], models: Mapping[str, Any]) -> bool | None:
    if policy == "NEVER":
        return False
    if policy == "ALWAYS":
        return _eligible(pair)
    if policy == "APPLICABILITY_RULE":
        context = _context(pair)
        qualification = context.get("qualification") or {}
        return _eligible(pair) and int(qualification.get("conv2d_count", 0)) > 0
    if policy == "FAMILY_OR_SHAPE_HEURISTIC":
        return _eligible(pair) and (_feature(pair, "batch") or 0) >= 8
    if policy in {"HORIZON_THRESHOLD", "SIMPLE_SUPERVISED"}:
        model = models.get(policy, {})
        if model.get("status") != "FITTED":
            return None
        value = _feature(pair, str(model["feature"]))
        if value is None:
            return False
        if policy == "HORIZON_THRESHOLD" or model.get("direction") == "ge":
            return _eligible(pair) and value >= float(model["threshold"])
        return _eligible(pair) and value < float(model["threshold"])
    if policy in {"ORIGINAL_ACRE", "REGIME_AWARE_ACRE"}:
        model = models.get(policy, {})
        context_id = str(pair.get("context_id", ""))
        decisions = model.get("route_decisions", {})
        decision = decisions.get(context_id) if isinstance(decisions, Mapping) else None
        if not isinstance(decision, Mapping):
            return None
        rule_id = model.get("rule_id")
        selected = decision.get("selected_rule_ids")
        return bool(rule_id and isinstance(selected, (list, tuple)) and rule_id in selected)
    if policy == "ORACLE":
        return pair.get("effect_status") == "valid" and pair.get("log_benefit", 0) > 0
    raise ValueError(f"unknown policy: {policy}")


def _score(policy: str, pairs: Sequence[Mapping[str, Any]], *,
           models: Mapping[str, Any], acre_status: str) -> dict[str, Any]:
    if policy in {"ORIGINAL_ACRE", "REGIME_AWARE_ACRE"} and acre_status != "FITTED":
        return {
            "policy": policy, "status": "ACRE_NOT_FITTED",
            "deployment_coverage": None, "known_unsafe_deployments": None,
            "missing_deployed_outcomes": None, "family_equal_mean_net_log_benefit": None,
            "family_cluster_bootstrap_ci95": None,
            "reason": "no canonical rule and support certificate were fitted; abstention is not efficacy",
        }
    start = time.perf_counter()
    decisions = [(pair, _deploy(policy, pair, models)) for pair in pairs]
    routing_seconds = time.perf_counter() - start
    routed = [(pair, decision) for pair, decision in decisions if decision is True]
    family_utility: dict[str, list[float]] = defaultdict(list)
    family_bounded_utility: dict[str, list[float]] = defaultdict(list)
    family_bounds: dict[str, list[tuple[float, float]]] = defaultdict(list)
    family_seconds: dict[str, list[float]] = defaultdict(list)
    incomplete_families: set[str] = set()
    all_families = {str(pair.get("family_id", "")) for pair in pairs} - {""}
    unsafe = missing = unknown_decisions = missing_baseline = missing_family = 0
    not_deployed = 0

    def add_bound(family: str, lower: float = -1.0, upper: float = 1.0) -> None:
        if family:
            family_bounds[family].append((lower, upper))
            incomplete_families.add(family)

    for pair, decision in decisions:
        family = str(pair.get("family_id", ""))
        if not family:
            missing_family += 1
        if decision is None:
            unknown_decisions += 1
            add_bound(family)
            continue
        s0 = pair.get("S0") if isinstance(pair.get("S0"), Mapping) else {}
        s1 = pair.get("S1") if isinstance(pair.get("S1"), Mapping) else {}
        if decision is False:
            # Not deploying S1 has zero incremental benefit when the baseline ran safely,
            # even if the unused candidate outcome is missing.
            if s0.get("arm_execution") == "COMPLETE" and s0.get("scientific_correctness") == "PASS":
                not_deployed += 1
                if family:
                    family_utility[family].append(0.0)
                    family_bounded_utility[family].append(0.0)
                    family_bounds[family].append((0.0, 0.0))
                    family_seconds[family].append(0.0)
            else:
                missing_baseline += 1
                add_bound(family)
            continue
        if (s1.get("scientific_correctness") == "FAIL"
                or pair.get("effect_status") == "unsafe"):
            unsafe += 1
            add_bound(family)
            continue
        if pair.get("effect_status") != "valid":
            missing += 1
            add_bound(family)
            continue
        value = pair.get("log_benefit")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            missing += 1
            add_bound(family)
            continue
        log_benefit = float(value)
        bounded_utility = math.tanh(log_benefit / math.log(2.0))
        if family:
            family_utility[family].append(log_benefit)
            family_bounded_utility[family].append(bounded_utility)
            family_bounds[family].append((bounded_utility, bounded_utility))
            seconds_saved = pair.get("cold_seconds_saved_per_pair")
            if (not isinstance(seconds_saved, bool)
                    and isinstance(seconds_saved, (int, float))
                    and math.isfinite(float(seconds_saved))):
                family_seconds[family].append(float(seconds_saved))
    family_logs = {
        family: sum(values) / len(values) for family, values in family_utility.items()
        if values and family not in incomplete_families
    }
    family_saved = {
        family: sum(values) / len(values) for family, values in family_seconds.items()
        if values and family not in incomplete_families
    }
    family_bounded = {
        family: sum(values) / len(values) for family, values in family_bounded_utility.items()
        if values and family not in incomplete_families
    }
    family_bound_means = {
        family: (
            sum(lower for lower, _ in values) / len(values),
            sum(upper for _, upper in values) / len(values),
        )
        for family, values in family_bounds.items() if values
    }
    log_effect = (sum(family_logs.values()) / len(family_logs)) if family_logs else None
    seconds_effect = (sum(family_saved.values()) / len(family_saved)) if family_saved else None
    bounded_effect = (sum(family_bounded.values()) / len(family_bounded)) if family_bounded else None
    unresolved_bounds = (
        [sum(value[0] for value in family_bound_means.values()) / len(family_bound_means),
         sum(value[1] for value in family_bound_means.values()) / len(family_bound_means)]
        if family_bound_means else None
    )
    interval = family_cluster_bootstrap_ci(family_logs, seed=101, replicates=10_000)
    status = "SCORED" if pairs else "NO_CONTEXTS"
    return {
        "policy": policy, "status": status,
        "planned_contexts": len(pairs),
        "deployment_coverage": len(routed) / len(pairs) if pairs else None,
        "deployed_contexts": len(routed), "known_unsafe_deployments": unsafe,
        "missing_deployed_outcomes": missing,
        "missing_non_deployed_baseline_outcomes": missing_baseline,
        "known_valid_policy_families": len(family_logs),
        "planned_independent_families": len(all_families),
        "family_equal_mean_net_log_benefit": log_effect,
        "family_cluster_bootstrap_ci95": interval,
        "bounded_utility_scale": "tanh(log(T_S0/T_S1) / ln(2)); interval [-1,1]",
        "family_equal_mean_net_bounded_utility_complete_case": bounded_effect,
        "family_cluster_bootstrap_ci95_bounded_utility_complete_case":
            family_cluster_bootstrap_ci(family_bounded, seed=101, replicates=10_000),
        "family_equal_mean_net_seconds_saved_per_pair": seconds_effect,
        "complete_case_families": len(family_logs),
        "incomplete_families": sorted(incomplete_families),
        "known_complete_case_only": True,
        "unknown_policy_decisions": unknown_decisions,
        "known_not_deployed_contexts": not_deployed,
        "contexts_without_family_id": missing_family,
        "unresolved_benefit_bounds": {
            "scale": "family-equal mean bounded utility",
            "interval": unresolved_bounds,
            "independent_families": len(family_bound_means),
            "contexts_without_family_id_excluded": missing_family,
            "predeclared_context_bounds": {
                "missing_deployed_outcome": [-1.0, 1.0],
                "unsafe_deployment": [-1.0, 1.0],
                "unknown_policy_decision": [-1.0, 1.0],
                "missing_non_deployed_baseline_outcome": [-1.0, 1.0],
            },
            "context_counts": {
                "missing_deployed_outcome": missing,
                "unsafe_deployment": unsafe,
                "unknown_policy_decision": unknown_decisions,
                "missing_non_deployed_baseline_outcome": missing_baseline,
                "known_not_deployed": not_deployed,
            },
            "unsafe_deployments_remain_a_separate_hard_gate": True,
        },
        "policy_routing_overhead_seconds": routing_seconds,
    }


def evaluate_policy_set(pairs: Sequence[Mapping[str, Any]], *,
                        models: Mapping[str, Any] | None = None,
                        acre_status: str = "ACRE_NOT_FITTED",
                        acre_rule_id: str | None = None) -> list[dict[str, Any]]:
    frozen = models or {}
    policies = (
        "NEVER", "ALWAYS", "APPLICABILITY_RULE", "HORIZON_THRESHOLD",
        "FAMILY_OR_SHAPE_HEURISTIC", "SIMPLE_SUPERVISED", "ORIGINAL_ACRE",
        "REGIME_AWARE_ACRE", "ORACLE",
    )
    rows = [_score(policy, pairs, models=frozen, acre_status=acre_status)
            for policy in policies]
    for row in rows:
        row["acre_rule_id"] = acre_rule_id if "ACRE" in row["policy"] else None
    return rows


def _cross_validated_simple_baselines(
    rows: Sequence[Mapping[str, Any]], *, acre_status: str,
) -> dict[str, Any]:
    """Choose a comparator from leave-one-independent-family-out development scores."""
    rows = _development_rows(rows)
    families = sorted({str(pair["family_id"]) for pair in rows if pair.get("family_id")})
    fold_results: dict[str, list[dict[str, Any]]] = {policy: [] for policy in _SIMPLE_POLICIES}
    folds = []
    for held_out in families:
        training = [pair for pair in rows if str(pair.get("family_id", "")) != held_out]
        held_rows = [pair for pair in rows if str(pair.get("family_id", "")) == held_out]
        fold_models = {
            "APPLICABILITY_RULE": {
                "status": "FROZEN",
                "predicate": "prequalified 4D Conv2d channels-last path",
            },
            "HORIZON_THRESHOLD": _fit_threshold(training, "horizon"),
            "FAMILY_OR_SHAPE_HEURISTIC": {
                "status": "FROZEN", "predicate": "eligible and batch >= 8",
                "source": "predeclared_static_heuristic",
            },
            "SIMPLE_SUPERVISED": _fit_supervised_grid(training),
        }
        fold_scores = {}
        for policy in _SIMPLE_POLICIES:
            score = _score(policy, held_rows, models=fold_models, acre_status=acre_status)
            fold_results[policy].append(score)
            fold_scores[policy] = {
                "status": score["status"],
                "family_equal_mean_net_log_benefit": score.get("family_equal_mean_net_log_benefit"),
                "known_unsafe_deployments": score.get("known_unsafe_deployments"),
                "missing_deployed_outcomes": score.get("missing_deployed_outcomes"),
                "missing_non_deployed_baseline_outcomes":
                    score.get("missing_non_deployed_baseline_outcomes"),
                "unknown_policy_decisions": score.get("unknown_policy_decisions"),
            }
        folds.append({
            "held_out_family": held_out,
            "training_family_ids": sorted(set(families) - {held_out}),
            "scores": fold_scores,
        })

    summaries = []
    for policy, scores in fold_results.items():
        safe = bool(scores) and all(
            score.get("status") == "SCORED"
            and score.get("known_unsafe_deployments") == 0
            and score.get("missing_deployed_outcomes") == 0
            and score.get("missing_non_deployed_baseline_outcomes") == 0
            and score.get("unknown_policy_decisions") == 0
            and score.get("family_equal_mean_net_log_benefit") is not None
            for score in scores
        )
        family_scores = [float(score["family_equal_mean_net_log_benefit"])
                         for score in scores if score.get("family_equal_mean_net_log_benefit") is not None]
        coverages = [float(score["deployment_coverage"]) for score in scores
                     if score.get("deployment_coverage") is not None]
        summaries.append({
            "policy": policy,
            "eligible_for_selection": safe,
            "scored_folds": len(family_scores),
            "family_equal_mean_log_benefit": (
                sum(family_scores) / len(family_scores) if safe else None
            ),
            "mean_deployment_coverage": (
                sum(coverages) / len(coverages) if safe and coverages else None
            ),
        })
    eligible = [row for row in summaries if row["eligible_for_selection"]]
    selected = max(
        eligible,
        key=lambda row: (row["family_equal_mean_log_benefit"],
                         row["mean_deployment_coverage"], row["policy"]),
    )["policy"] if eligible else None
    return {
        "method": "leave_one_independent_family_out",
        "selection_split": "development",
        "fold_count": len(families),
        "selected_policy": selected,
        "policies": summaries,
        "folds": folds,
    }
