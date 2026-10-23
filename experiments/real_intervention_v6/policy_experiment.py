"""Development/calibration policy freezing and family-level effect summaries."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


BOOTSTRAP_SEED = 20261023
BOOTSTRAP_REPLICATES = 10_000
HORIZON_THRESHOLDS: tuple[int | str, ...] = (128, 1024, 4096, "never", "always")
RESEARCH_THRESHOLDS = (0.0, 0.01, 0.05, 0.1)
RESEARCH_FEATURES = ("logH", "logR", "logparams", "conv_fraction")
SIMPLE_FEATURES = ("horizon", "reuse_count", "conv_fraction")
_OUTCOME_FIELDS = {"effect", "log_benefit", "s0_s", "s1_s", "outcome", "outcomes"}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _family(row: Mapping[str, Any]) -> str:
    value = str(row.get("family_id") or "")
    if not value:
        raise ValueError("each policy row requires a non-empty family_id")
    return value


def _context(row: Mapping[str, Any]) -> str:
    value = str(row.get("context_id") or "")
    if not value:
        raise ValueError("each policy row requires a non-empty context_id")
    return value


def _feature(row: Mapping[str, Any], name: str) -> float | None:
    features = row.get("features")
    features = features if isinstance(features, Mapping) else {}
    aliases = {
        "horizon": (row.get("horizon"), features.get("horizon")),
        "reuse_count": (row.get("reuse_count"), features.get("reuse_count"), features.get("reuse")),
        "parameters": (features.get("parameters"), features.get("parameter_count")),
        "conv_fraction": (features.get("conv_fraction"),),
    }
    for value in aliases.get(name, (features.get(name),)):
        parsed = _number(value)
        if parsed is not None:
            return parsed
    return None


def _model_feature(row: Mapping[str, Any], name: str) -> float | None:
    if name == "logH":
        value = _feature(row, "horizon")
    elif name == "logR":
        value = _feature(row, "reuse_count")
    elif name == "logparams":
        value = _feature(row, "parameters")
    else:
        return _feature(row, "conv_fraction")
    return math.log(value) if value is not None and value > 0 else None


def _effect(row: Mapping[str, Any]) -> float | None:
    value = _number(row.get("log_benefit"))
    if value is not None:
        return value
    s0, s1 = _number(row.get("s0_s")), _number(row.get("s1_s"))
    if s0 is None or s1 is None or s0 <= 0 or s1 <= 0:
        return None
    return math.log(s0 / s1)


def _family_means(rows: Sequence[Mapping[str, Any]], values: Sequence[float | None]) -> dict[str, float]:
    family_context: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row, value in zip(rows, values):
        if value is not None:
            family_context[_family(row)][_context(row)].append(value)
    result = {}
    for family, contexts in family_context.items():
        context_means = [sum(items) / len(items) for items in contexts.values()]
        if context_means:
            result[family] = sum(context_means) / len(context_means)
    return result


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _family_utility(rows: Sequence[Mapping[str, Any]], actions: Sequence[bool]) -> tuple[float | None, dict[str, float], bool]:
    values: list[float | None] = []
    incomplete_deployment = False
    for row, deploy in zip(rows, actions):
        outcome = _effect(row)
        if not deploy:
            values.append(0.0)
        else:
            values.append(outcome)
            incomplete_deployment |= outcome is None
    means = _family_means(rows, values)
    return _mean(list(means.values())) if not incomplete_deployment else None, means, incomplete_deployment


def _score(rows: Sequence[Mapping[str, Any]], actions: Sequence[bool]) -> float | None:
    score, _, _ = _family_utility(rows, actions)
    return score


def _threshold_actions(rows: Sequence[Mapping[str, Any]], feature: str, threshold: float | str) -> list[bool]:
    if threshold == "never":
        return [False] * len(rows)
    if threshold == "always":
        return [True] * len(rows)
    return [(_feature(row, feature) is not None and _feature(row, feature) >= float(threshold)) for row in rows]


def _feature_thresholds(rows: Sequence[Mapping[str, Any]], feature: str) -> list[float | str]:
    values = sorted({value for row in rows if (value := _feature(row, feature)) is not None})
    candidates: list[float | str] = ["never", "always"]
    if len(values) == 1:
        candidates.append(values[0])
    else:
        candidates.extend((left + right) / 2 for left, right in zip(values, values[1:]))
    return candidates


def _fit_ridge(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    feature_rows = [
        (row, [_model_feature(row, name) for name in RESEARCH_FEATURES])
        for row in rows
    ]
    supported = [(row, values) for row, values in feature_rows
                 if all(value is not None for value in values)]
    usable = [(row, values) for row, values in supported if _effect(row) is not None]
    selected = usable
    if not selected:
        return None
    families: dict[str, int] = defaultdict(int)
    for row, _ in selected:
        families[_family(row)] += 1
    x = np.asarray([values for _, values in selected], dtype=float)
    y = np.asarray([_effect(row) for row, _ in selected], dtype=float)
    weights = np.asarray([1.0 / families[_family(row)] for row, _ in selected], dtype=float)
    weights /= weights.sum()
    center = (x * weights[:, None]).sum(axis=0)
    scale = np.sqrt(((x - center) ** 2 * weights[:, None]).sum(axis=0))
    scale[scale == 0] = 1.0
    design = np.column_stack((np.ones(len(x)), (x - center) / scale))
    penalty = np.eye(design.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    weighted_design = design * np.sqrt(weights[:, None])
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ (y * np.sqrt(weights)),
    )
    return {
        "feature_names": list(RESEARCH_FEATURES),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "ridge_penalty": 1.0,
        "training_feature_min": np.asarray([values for _, values in supported], dtype=float).min(axis=0).tolist(),
        "training_feature_max": np.asarray([values for _, values in supported], dtype=float).max(axis=0).tolist(),
        "training_rows": len(rows),
        "supported_feature_rows": len(supported),
        "fit_rows_with_observed_outcome": len(selected),
        "missing_outcome_rows": sum(_effect(row) is None for row in rows),
    }


def _research_feature_status(row: Mapping[str, Any], model: Mapping[str, Any]) -> str:
    features = [_model_feature(row, name) for name in model["feature_names"]]
    if any(value is None for value in features):
        return "MISSING_FEATURE_NEVER"
    if any(value < lower or value > upper for value, lower, upper in zip(
        features, model["training_feature_min"], model["training_feature_max"]
    )):
        return "OOD_NEVER"
    return "IN_SUPPORT"


def _ridge_values(rows: Sequence[Mapping[str, Any]], model: Mapping[str, Any]) -> list[float | None]:
    values = []
    for row in rows:
        if _research_feature_status(row, model) != "IN_SUPPORT":
            values.append(None)
            continue
        features = [_model_feature(row, name) for name in model["feature_names"]]
        centered = [(value - center) / scale for value, center, scale in zip(features, model["center"], model["scale"])]
        values.append(float(model["coefficients"][0] + sum(
            coefficient * value for coefficient, value in zip(model["coefficients"][1:], centered)
        )))
    return values


def _reject_final_outcomes(value: Any) -> None:
    if isinstance(value, Mapping):
        bad = _OUTCOME_FIELDS.intersection(value)
        if bad:
            raise ValueError(f"final contexts must be pre-execution only; found {sorted(bad)}")
        for child in value.values():
            _reject_final_outcomes(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_final_outcomes(child)


def _check_disjoint(dev_rows: Sequence[Mapping[str, Any]], cal_rows: Sequence[Mapping[str, Any]], final_rows: Sequence[Mapping[str, Any]]) -> None:
    dev_families = {_family(row) for row in dev_rows}
    cal_families = {_family(row) for row in cal_rows}
    final_families = {_family(row) for row in final_rows}
    if dev_families & cal_families or dev_families & final_families or cal_families & final_families:
        raise ValueError("development, calibration, and final families must be disjoint")


def freeze_policies(
    dev_rows: Sequence[Mapping[str, Any]],
    cal_rows: Sequence[Mapping[str, Any]],
    final_contexts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fit on development, tune only the research cutoff on calibration, freeze final actions."""
    dev, cal, final = list(dev_rows), list(cal_rows), list(final_contexts)
    for rows, expected in ((dev, "development"), (cal, "calibration")):
        for row in rows:
            _family(row)
            _context(row)
            if "split" in row and row["split"] != expected:
                raise ValueError(f"row passed as {expected} has split={row['split']!r}")
    for row in final:
        _family(row)
        _context(row)
        _reject_final_outcomes(row)
    _check_disjoint(dev, cal, final)
    final_keys = [(_family(row), _context(row)) for row in final]
    if len(set(final_keys)) != len(final_keys):
        raise ValueError("final family/context pairs must be unique")

    policies: dict[str, dict[str, Any]] = {
        "NEVER": {"kind": "constant", "deploy": False},
        "ALWAYS": {"kind": "constant", "deploy": True},
    }
    horizon_candidates = [("H_ONLY", threshold) for threshold in HORIZON_THRESHOLDS]
    simple_candidates = [
        (feature, threshold)
        for feature in SIMPLE_FEATURES
        for threshold in _feature_thresholds(dev, feature)
    ]
    for name, candidates in (("H_ONLY", horizon_candidates), ("SIMPLE", simple_candidates)):
        scored = []
        for candidate in candidates:
            feature, threshold = ("horizon", candidate[1]) if name == "H_ONLY" else candidate
            actions = _threshold_actions(dev, feature, threshold)
            score = _score(dev, actions)
            if score is not None:
                scored.append((score, sum(actions), feature, threshold))
        if scored:
            _, _, feature, threshold = max(scored, key=lambda item: (item[0], -item[1], str(item[3])))
            policies[name] = {"kind": "threshold", "feature": feature, "threshold": threshold}
        else:
            policies[name] = {"kind": "not_estimable", "fallback": "NEVER"}

    model = _fit_ridge(dev)
    research: dict[str, Any] = {"kind": "research_ridge", "status": "NOT_FITTED", "fallback": "NEVER"}
    if model is not None:
        research.update({"status": "FITTED", "model": model, "threshold": RESEARCH_THRESHOLDS[-1]})
        cal_predictions = _ridge_values(cal, model)
        candidates = []
        for threshold in RESEARCH_THRESHOLDS:
            actions = [value is not None and value >= threshold for value in cal_predictions]
            score = _score(cal, actions)
            if score is not None:
                candidates.append((score, threshold))
        if candidates:
            research["threshold"] = max(candidates, key=lambda item: (item[0], item[1]))[1]
            research["calibration_status"] = "TUNED"
        else:
            research["status"] = "NOT_TUNED"
            research["calibration_status"] = "NO_COMPLETE_CALIBRATION_CANDIDATE"
    policies["ACRE_SYS_RESEARCH"] = research

    dev_simple = []
    for name in ("NEVER", "ALWAYS", "H_ONLY", "SIMPLE"):
        spec = policies[name]
        actions = _predict_rows(dev, spec)
        score = _score(dev, actions)
        if score is not None:
            dev_simple.append((score, name))
    best_simple = max(dev_simple, key=lambda item: (item[0], -("NEVER", "ALWAYS", "H_ONLY", "SIMPLE").index(item[1])))[1] if dev_simple else None

    predictions = []
    for row in final:
        research_spec = policies["ACRE_SYS_RESEARCH"]
        research_status = (
            _research_feature_status(row, research_spec["model"])
            if research_spec.get("status") in ("FITTED", "NOT_TUNED")
            else "NOT_FITTED_NEVER"
        )
        predictions.append({
            "family_id": _family(row),
            "context_id": _context(row),
            "research_context_status": research_status,
            "predictions": {name: action for name, spec in policies.items()
                            if (action := _predict_row(row, spec)) is not None},
        })
    return {
        "schema_version": 1,
        "policies": policies,
        "strongest_simple_policy_on_development": best_simple,
        "calibration_role": "research threshold selection only; no model fitting",
        "final_outcomes_accessed": False,
        "original_acre_status": "NOT_CERTIFIED",
        "final_predictions": predictions,
    }


def _predict_row(row: Mapping[str, Any], spec: Mapping[str, Any]) -> bool | None:
    kind = spec.get("kind")
    if kind == "constant":
        return bool(spec["deploy"])
    if kind == "threshold":
        value = _feature(row, str(spec["feature"]))
        threshold = spec["threshold"]
        if threshold == "never":
            return False
        if threshold == "always":
            return True
        return value is not None and value >= float(threshold)
    if kind == "research_ridge":
        if spec.get("status") != "FITTED":
            return False
        values = _ridge_values([row], spec["model"])
        return values[0] is not None and values[0] >= float(spec["threshold"])
    if kind == "not_estimable":
        return False
    return None


def _predict_rows(rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> list[bool]:
    return [_predict_row(row, spec) is True for row in rows]


def _bootstrap_ci(values: Sequence[float], *, seed: int = BOOTSTRAP_SEED) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    sample = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(sample), size=(BOOTSTRAP_REPLICATES, len(sample)))
    means = sample[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _routing_cost(
    policy: str,
    context_id: str,
    routing_overhead_s: float | Mapping[str, Any] | None,
) -> float | None:
    if policy in {"NEVER", "ALWAYS", "ORACLE"}:
        return 0.0
    if routing_overhead_s is None:
        return None
    value: Any = routing_overhead_s
    if isinstance(routing_overhead_s, Mapping):
        value = routing_overhead_s.get(policy)
        if isinstance(value, Mapping):
            value = value.get(context_id)
    if value is None:
        return None
    parsed = _number(value)
    if parsed is None or parsed < 0:
        raise ValueError("routing overhead must be a finite, non-negative number of seconds")
    return parsed


def _policy_log_value(row: Mapping[str, Any], deploy: bool | None) -> float | None:
    if deploy is None:
        return None
    if not deploy:
        return 0.0
    return _effect(row)


def _policy_seconds_value(
    row: Mapping[str, Any],
    policy: str,
    deploy: bool | None,
    routing_overhead_s: float | Mapping[str, Any] | None,
) -> tuple[float | None, float | None, float | None]:
    if deploy is None:
        return None, None, None
    gross = 0.0
    if deploy:
        s0, s1 = _number(row.get("s0_s")), _number(row.get("s1_s"))
        if s0 is None or s1 is None or s0 < 0 or s1 < 0:
            return None, None, None
        gross = s0 - s1
    cost = _routing_cost(policy, _context(row), routing_overhead_s)
    return gross, cost, gross - cost if cost is not None else None


def _seconds_summary(
    rows: Sequence[Mapping[str, Any]],
    policy: str,
    actions: Sequence[bool | None],
    routing_overhead_s: float | Mapping[str, Any] | None,
) -> dict[str, Any]:
    values = [_policy_seconds_value(row, policy, action, routing_overhead_s)
              for row, action in zip(rows, actions)]
    gross = [item[0] for item in values]
    costs = [item[1] for item in values]
    net = [item[2] for item in values]
    complete = bool(rows) and all(value is not None for value in net)
    family_context: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    incomplete_contexts: set[tuple[str, str]] = set()
    for row, value in zip(rows, net):
        key = (_family(row), _context(row))
        if value is None:
            incomplete_contexts.add(key)
        else:
            family_context[key[0]][key[1]].append(value)
    family_means = {}
    for family, contexts in family_context.items():
        context_means = [sum(context_values) / len(context_values)
                         for context, context_values in contexts.items()
                         if (family, context) not in incomplete_contexts]
        if context_means:
            family_means[family] = sum(context_means) / len(context_means)
    family_mean = _mean(list(family_means.values()))
    return {
        "status": "COMPLETE" if complete else "INCOMPLETE_OR_MISSING_ROUTING_OVERHEAD",
        "planned_contexts": len(rows),
        "observed_net_contexts": sum(value is not None for value in net),
        "missing_net_contexts": sum(value is None for value in net),
        "gross_savings_s_observed_sum": sum(value for value in gross if value is not None),
        "gross_savings_s": sum(gross) if rows and all(value is not None for value in gross) else None,
        "routing_overhead_s_observed_sum": sum(value for value in costs if value is not None),
        "routing_overhead_s": sum(costs) if rows and all(value is not None for value in costs) else None,
        "net_benefit_s_observed_sum": sum(value for value in net if value is not None),
        "net_benefit_s": sum(net) if complete else None,
        "family_equal_net_benefit_s": family_mean,
        "family_mean_net_benefit_s": family_means,
        "family_bootstrap_ci95_net_benefit_s": _bootstrap_ci(list(family_means.values())),
        "missingness_bounds_s": (
            {"lower": "-unbounded", "upper": "+unbounded"}
            if any(value is None for value in net)
            else {"lower": family_mean, "upper": family_mean}
        ),
    }


def _paired_difference(
    rows: Sequence[Mapping[str, Any]],
    left: Sequence[bool | None],
    right: Sequence[bool | None],
    *,
    metric: str,
    left_policy: str,
    right_policy: str,
    routing_overhead_s: float | Mapping[str, Any] | None,
) -> dict[str, Any]:
    context_differences: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    incomplete_contexts: set[tuple[str, str]] = set()
    planned_contexts = {(_family(row), _context(row)) for row in rows}
    for row, left_action, right_action in zip(rows, left, right):
        if metric == "log_benefit":
            left_value = _policy_log_value(row, left_action)
            right_value = _policy_log_value(row, right_action)
        else:
            left_value = _policy_seconds_value(row, left_policy, left_action, routing_overhead_s)[2]
            right_value = _policy_seconds_value(row, right_policy, right_action, routing_overhead_s)[2]
        key = (_family(row), _context(row))
        if left_value is None or right_value is None:
            incomplete_contexts.add(key)
        else:
            context_differences[key[0]][key[1]].append(left_value - right_value)

    family_differences = {}
    observed_contexts = set()
    for family, contexts in context_differences.items():
        context_means = []
        for context, values in contexts.items():
            key = (family, context)
            if key in incomplete_contexts:
                continue
            context_means.append(sum(values) / len(values))
            observed_contexts.add(key)
        if context_means:
            family_differences[family] = sum(context_means) / len(context_means)
    values = list(family_differences.values())
    mean_difference = _mean(values)
    return {
        "estimate_status": "COMPLETE" if observed_contexts == planned_contexts else "INCOMPLETE_OBSERVED_ONLY",
        "planned_family_contexts": len(planned_contexts),
        "paired_family_contexts": len(observed_contexts),
        "missing_family_contexts": len(planned_contexts - observed_contexts),
        "paired_families": len(family_differences),
        "family_equal_mean_difference": mean_difference,
        "family_bootstrap_ci95": _bootstrap_ci(values),
        "family_mean_differences": family_differences,
        "missingness_bounds": (
            {"lower": "-unbounded", "upper": "+unbounded"}
            if observed_contexts != planned_contexts
            else {"lower": mean_difference, "upper": mean_difference}
        ),
    }


def _summarize_values(rows: Sequence[Mapping[str, Any]], values: Sequence[float | None]) -> dict[str, Any]:
    means = _family_means(rows, values)
    observed = list(means.values())
    raw_s0 = [value for row in rows if (value := _number(row.get("s0_s"))) is not None and value >= 0]
    raw_s1 = [value for row in rows if (value := _number(row.get("s1_s"))) is not None and value >= 0]
    return {
        "estimate_status": (
            "NOT_ESTIMABLE_EMPTY" if not rows
            else "COMPLETE" if not any(value is None for value in values)
            else "INCOMPLETE_OBSERVED_ONLY"
        ),
        "planned_rows": len(rows),
        "observed_rows": sum(value is not None for value in values),
        "missing_rows": sum(value is None for value in values),
        "planned_families": len({_family(row) for row in rows}) if rows else 0,
        "observed_families": len(means),
        "missing_families": len({_family(row) for row in rows} - means.keys()) if rows else 0,
        "family_equal_mean_log_benefit": _mean(observed),
        "family_mean_log_benefit": means,
        "family_bootstrap_ci95": _bootstrap_ci(observed),
        "missingness_bounds": (
            {"lower": "-unbounded", "upper": "+unbounded"}
            if any(value is None for value in values)
            else {"lower": _mean(observed), "upper": _mean(observed)}
        ),
        "positive_families": sum(value > 0 for value in observed),
        "negative_families": sum(value < 0 for value in observed),
        "zero_families": sum(value == 0 for value in observed),
        "raw_s0_seconds": sum(raw_s0),
        "raw_s1_seconds": sum(raw_s1),
        "raw_seconds_paired_rows": sum(
            (s0 := _number(row.get("s0_s"))) is not None and s0 >= 0
            and (s1 := _number(row.get("s1_s"))) is not None and s1 >= 0
            for row in rows
        ),
    }


def _prediction_map(freeze: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result = {}
    for item in freeze.get("final_predictions", []):
        key = (str(item.get("family_id") or ""), str(item.get("context_id") or ""))
        if not all(key) or key in result:
            raise ValueError("frozen predictions require unique family/context identities")
        result[key] = item.get("predictions", {})
    return result


def summarize(
    rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Any] | None = None,
    routing_overhead_s: float | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize outcomes and frozen policy utility, charging measured routing seconds.

    ``routing_overhead_s`` is a shared per-context cost, or a mapping from
    policy names to per-context costs (a scalar per policy or a context-id map).
    Missing routed-policy overhead stays missing; it is never treated as zero.
    """
    observed_rows = list(rows)
    direct = [_effect(row) for row in observed_rows]
    result: dict[str, Any] = {"overall": _summarize_values(observed_rows, direct)}
    if predictions is None:
        return result

    frozen = _prediction_map(predictions)
    policy_summaries: dict[str, dict[str, Any]] = {}
    family_by_policy: dict[str, dict[str, float]] = {}
    actions_by_policy: dict[str, list[bool | None]] = {}
    names = list(predictions.get("policies", {}))
    for name in names:
        values: list[float | None] = []
        actions_for_policy: list[bool | None] = []
        for row in observed_rows:
            actions = frozen.get((_family(row), _context(row)), {})
            deploy = actions.get(name)
            actions_for_policy.append(deploy if isinstance(deploy, bool) else None)
            if deploy is False:
                values.append(0.0)
            elif deploy is True:
                values.append(_effect(row))
            else:
                values.append(None)
        summary = _summarize_values(observed_rows, values)
        summary["seconds"] = _seconds_summary(
            observed_rows, name, actions_for_policy, routing_overhead_s
        )
        policy_summaries[name] = summary
        family_by_policy[name] = _family_means(observed_rows, values)
        actions_by_policy[name] = actions_for_policy

    # The hindsight oracle is an analysis upper bound, never a frozen action.
    oracle_values = []
    oracle_actions = []
    for row in observed_rows:
        effect = _effect(row)
        oracle_actions.append(effect > 0 if effect is not None else None)
        oracle_values.append(max(effect, 0.0) if effect is not None else None)
    oracle_summary = _summarize_values(observed_rows, oracle_values)
    oracle_summary["seconds"] = _seconds_summary(
        observed_rows, "ORACLE", oracle_actions, routing_overhead_s
    )
    policy_summaries["ORACLE"] = oracle_summary
    family_by_policy["ORACLE"] = _family_means(observed_rows, oracle_values)
    actions_by_policy["ORACLE"] = oracle_actions
    result["policies"] = policy_summaries

    research_name = "ACRE_SYS_RESEARCH"
    simple_name = predictions.get("strongest_simple_policy_on_development")
    research_actions = actions_by_policy.get(research_name, [None] * len(observed_rows))
    simple_actions = actions_by_policy.get(str(simple_name), [None] * len(observed_rows))
    log_difference = _paired_difference(
        observed_rows, research_actions, simple_actions,
        metric="log_benefit", left_policy=research_name,
        right_policy=str(simple_name), routing_overhead_s=routing_overhead_s,
    )
    seconds_difference = _paired_difference(
        observed_rows, research_actions, simple_actions,
        metric="seconds", left_policy=research_name,
        right_policy=str(simple_name), routing_overhead_s=routing_overhead_s,
    )
    result["research_minus_best_simple"] = {
        "policy": research_name,
        "best_simple_policy": simple_name,
        "log_benefit_difference": log_difference,
        "raw_seconds_net_benefit_difference": seconds_difference,
    }
    return result
