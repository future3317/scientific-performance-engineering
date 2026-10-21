"""Family-level effect summaries and outcome-blind frozen policy predictions."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .certificate_feasibility import independent_group_count


BOOTSTRAP_SEED = 20261021
BOOTSTRAP_REPLICATES = 10_000
HORIZON_THRESHOLDS = (1, 16, 128, 1024, 4096)
CONV_COUNT_THRESHOLDS = (0, 5, 20, 50)
_PREDICTIVE_POLICIES = (
    "HORIZON_THRESHOLD",
    "STRUCTURE_CONV_COUNT",
    "SIMPLE_SUPERVISED_RIDGE",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _first_number(row: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        value = _number(row.get(name))
        if value is not None:
            return value
    return None


def _context_key(pair: Mapping[str, Any]) -> tuple[str, str, str]:
    return tuple(str(pair.get(key, "")) for key in ("batch", "horizon", "contrast"))


def _raw_seconds(pair: Mapping[str, Any], arm_name: str) -> float | None:
    flat = _number(pair.get(f"{arm_name.lower()}_s"))
    if flat is not None and flat >= 0:
        return flat
    arms = pair.get("arms") if isinstance(pair.get("arms"), Mapping) else pair
    arm = arms.get(arm_name)
    timing = arm.get("timing") if isinstance(arm, Mapping) else None
    value = _number(timing.get("cold_seconds")) if isinstance(timing, Mapping) else None
    return value if value is not None and value >= 0 else None


def _effect(pair: Mapping[str, Any]) -> float | None:
    value = _number(pair.get("log_benefit"))
    if value is not None:
        return value
    s0 = _first_number(pair, ("s0_s",))
    s1 = _first_number(pair, ("s1_s",))
    arms = pair.get("arms") if isinstance(pair.get("arms"), Mapping) else pair
    for arm_name, target in (("S0", "s0"), ("S1", "s1")):
        if (target == "s0" and s0 is None) or (target == "s1" and s1 is None):
            arm = arms.get(arm_name)
            timing = arm.get("timing") if isinstance(arm, Mapping) else None
            seconds = _number(timing.get("cold_seconds")) if isinstance(timing, Mapping) else None
            if target == "s0":
                s0 = seconds
            else:
                s1 = seconds
    if s0 is None or s1 is None or s0 <= 0 or s1 <= 0:
        return None
    return math.log(s0 / s1)


def _scientifically_usable(pair: Mapping[str, Any]) -> bool:
    if pair.get("analysis_status") == "VALID_PAIRED":
        return (
            pair.get("execution_status") == "COMPLETE"
            and pair.get("scientific_status") == "PASS"
            and pair.get("platform_status") == "HEALTHY"
            and pair.get("exposure_status") == "MATCHED"
        )
    return pair.get("effect_status") == "valid"


def _percentile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _bootstrap_ci(family_means: Mapping[str, float]) -> list[float] | None:
    values = list(family_means.values())
    if len(values) < 2:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    draws = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    return [_percentile(draws, 0.025), _percentile(draws, 0.975)]


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_family_context: dict[str, dict[tuple[str, str, str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    planned_by_family: dict[str, int] = defaultdict(int)
    unsafe_families: set[str] = set()
    valid_pairs = 0
    unsafe_pairs = 0
    s0_seconds = []
    s1_seconds = []
    raw_seconds_paired_count = 0
    for pair in rows:
        family = str(pair.get("family_id") or "")
        if not family:
            continue
        planned_by_family[family] += 1
        if pair.get("scientific_status") == "NUMERICAL_UNSAFE" or pair.get("effect_status") == "unsafe":
            unsafe_pairs += 1
            unsafe_families.add(family)
            continue
        effect = _effect(pair)
        if _scientifically_usable(pair) and effect is not None:
            by_family_context[family][_context_key(pair)].append(effect)
            valid_pairs += 1
            s0 = _raw_seconds(pair, "S0")
            s1 = _raw_seconds(pair, "S1")
            if s0 is not None:
                s0_seconds.append(s0)
            if s1 is not None:
                s1_seconds.append(s1)
            if s0 is not None and s1 is not None:
                raw_seconds_paired_count += 1

    family_means = {
        family: sum(sum(values) / len(values) for values in contexts.values()) / len(contexts)
        for family, contexts in by_family_context.items() if contexts
    }
    planned_families = set(planned_by_family)
    missing_families = planned_families - set(family_means)
    missing_pairs = len(rows) - valid_pairs
    means = list(family_means.values())
    mean = sum(means) / len(means) if means else None
    ordered = sorted(means)
    leave_one_out = {
        family: sum(value for other, value in family_means.items() if other != family)
        / (len(family_means) - 1)
        for family in family_means
        if len(family_means) > 1
    }
    return {
        "planned_pairs": len(rows),
        "valid_pairs": valid_pairs,
        "missing_pairs": missing_pairs,
        "raw_s0_seconds_sum": sum(s0_seconds),
        "raw_s1_seconds_sum": sum(s1_seconds),
        "raw_net_saved_seconds_sum": sum(s0_seconds) - sum(s1_seconds),
        "raw_seconds_paired_count": raw_seconds_paired_count,
        "raw_seconds_scope": "descriptive sum over valid observations with recorded arm seconds",
        "unsafe_pairs": unsafe_pairs,
        "planned_independent_families": len(planned_families),
        "observed_independent_families": len(family_means),
        "missing_independent_families": len(missing_families),
        "positive_families": sum(value > 0 for value in means),
        "negative_families": sum(value < 0 for value in means),
        "zero_families": sum(value == 0 for value in means),
        "family_equal_mean_log_benefit": mean,
        "median_family_mean_log_benefit": (
            _percentile(ordered, 0.5) if ordered else None
        ),
        "median_family_speedup": math.exp(_percentile(ordered, .5)) if ordered else None,
        "family_cluster_bootstrap_ci95": _bootstrap_ci(family_means),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "family_mean_log_benefit": family_means,
        "leave_one_family_out_mean_log_benefit": leave_one_out,
        "leave_one_family_out_range": (
            [min(leave_one_out.values()), max(leave_one_out.values())]
            if leave_one_out else None
        ),
        "missingness_bounds": (
            {"lower": "-unbounded", "upper": "+unbounded"}
            if missing_pairs else {"lower": mean, "upper": mean}
        ),
        "unsafe_families": sorted(unsafe_families),
    }


def summarize(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize valid effects with families as the only independent units."""
    rows = list(pairs)
    contrast_values = sorted({str(pair.get("contrast")) for pair in rows if pair.get("contrast") is not None})
    has_unspecified = any(pair.get("contrast") is None for pair in rows)
    if contrast_values and has_unspecified:
        contrast_values.append("UNSPECIFIED")
    partitions = (
        {contrast: [pair for pair in rows if str(pair.get("contrast", "UNSPECIFIED")) == contrast]
         for contrast in contrast_values}
        if contrast_values else {None: rows}
    )

    def one_partition(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        by_horizon: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        by_batch: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        by_joint: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for pair in selected:
            split = str(pair.get("split", "UNSPECIFIED"))
            horizon = str(pair.get("horizon", "UNSPECIFIED"))
            batch = str(pair.get("batch", "UNSPECIFIED"))
            by_split[split].append(pair)
            by_horizon[horizon].append(pair)
            by_batch[batch].append(pair)
            by_joint[f"{split}|H={horizon}|B={batch}"].append(pair)
        return {
            "overall": _summarize_rows(selected),
            "by_split": {key: _summarize_rows(value) for key, value in sorted(by_split.items())},
            "by_horizon": {key: _summarize_rows(value) for key, value in sorted(by_horizon.items())},
            "by_batch": {key: _summarize_rows(value) for key, value in sorted(by_batch.items())},
            "by_split_horizon_batch": {
                key: _summarize_rows(value) for key, value in sorted(by_joint.items())
            },
        }

    separated = {str(key): one_partition(value) for key, value in partitions.items() if key is not None}
    if len(partitions) > 1:
        return {
            "overall": {
                "status": "NOT_POOLED_DISTINCT_CONTRASTS",
                "planned_pairs": len(rows),
                "contrasts": list(partitions),
            },
            "by_contrast": separated,
        }
    single = one_partition(rows)
    if contrast_values:
        single["by_contrast"] = separated
    return single


def _feature(pair: Mapping[str, Any], name: str) -> float | None:
    fields = {
        "horizon": ("horizon",),
        "batch": ("batch",),
        "conv_count": ("conv_count", "conv2d_count", "conv2d_module_count"),
        "parameter_count": ("parameter_count", "parameters"),
    }
    value = _first_number(pair, fields[name])
    features = pair.get("features")
    features = features if isinstance(features, Mapping) else {}
    nested_name = {
        "batch": "batch",
        "parameter_count": "parameters",
        "conv_count": "conv2d",
    }.get(name)
    if value is None and nested_name is not None:
        nested_value = features.get(nested_name)
        if isinstance(nested_value, Mapping):
            nested_value = next((nested_value[key] for key in ("count", "module_count", "total") if key in nested_value), None)
        value = _number(nested_value)
    if value is None:
        return None
    if name == "conv_count":
        return value if value >= 0 else None
    return math.log(value) if value > 0 else None


def _family_training_rows(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[tuple[str, str, str], list[tuple[Mapping[str, Any], float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pair in pairs:
        family = str(pair.get("family_id") or "")
        effect = _effect(pair)
        if family and pair.get("split") == "development" and _scientifically_usable(pair) and effect is not None:
            grouped[family][_context_key(pair)].append((pair, effect))
    result = []
    for family, contexts in grouped.items():
        feature_names = ("horizon", "batch", "parameter_count", "conv_count")
        family_rows = []
        for context, rows in contexts.items():
            context_features: dict[str, float] = {}
            for name in ("horizon", "batch", "parameter_count", "conv_count"):
                values = [value for pair, _ in rows if (value := _feature(pair, name)) is not None]
                if not values:
                    break
                if name == "horizon" or name == "batch":
                    # These context-defining features should be identical across sessions.
                    values = [values[0]]
                context_features[name] = sum(values) / len(values)
            else:
                family_rows.append({
                    "family_id": family,
                    "context": context,
                    "effect": sum(effect for _, effect in rows) / len(rows),
                    "features": context_features.copy(),
                })
                continue
        result.extend(family_rows)
    return result


def _ridge_fit(training: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    feature_names = ("horizon", "batch", "parameter_count", "conv_count")
    complete = [row for row in training if all(row["features"][name] is not None for name in feature_names)]
    family_effects: dict[str, list[float]] = defaultdict(list)
    for row in complete:
        family_effects[row["family_id"]].append(row["effect"])
    family_means = {
        family: sum(values) / len(values) for family, values in family_effects.items()
    }
    positive = sum(value > 0 for value in family_means.values())
    negative = sum(value < 0 for value in family_means.values())
    if positive < 2 or negative < 2:
        return {
            "status": "NOT_FITTED",
            "reason": "development_requires_at_least_two_positive_and_two_negative_families",
            "positive_families": positive,
            "negative_families": negative,
            "feature_names": list(feature_names),
        }
    matrix = np.asarray([[row["features"][name] for name in feature_names] for row in complete], dtype=float)
    target = np.asarray([row["effect"] for row in complete], dtype=float)
    complete_context_counts: dict[str, int] = defaultdict(int)
    for row in complete:
        complete_context_counts[row["family_id"]] += 1
    weights = np.asarray(
        [1.0 / complete_context_counts[row["family_id"]] for row in complete],
        dtype=float,
    )
    weights /= weights.sum()
    center = (matrix * weights[:, None]).sum(axis=0)
    scale = np.sqrt(((matrix - center) ** 2 * weights[:, None]).sum(axis=0))
    scale[scale == 0] = 1.0
    standardized = (matrix - center) / scale
    design = np.column_stack((np.ones(len(standardized)), standardized))
    penalty = np.eye(design.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    weighted_design = design * np.sqrt(weights[:, None])
    weighted_target = target * np.sqrt(weights)
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    return {
        "status": "FITTED",
        "feature_names": list(feature_names),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "ridge_penalty": 1.0,
        "training_families": len(family_means),
        "training_contexts": len(complete),
        "positive_families": positive,
        "negative_families": negative,
    }


def _predictive_outcome(pair: Mapping[str, Any]) -> float | None:
    return _effect(pair) if _scientifically_usable(pair) else None


def _policy_utility(
    pairs: Sequence[Mapping[str, Any]],
    name: str,
    spec: Mapping[str, Any],
    frozen_predictions: Mapping[str, bool | None] | None = None,
) -> dict[str, Any]:
    by_family_context: dict[str, dict[tuple[str, str, str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    missing_deployments = 0
    unsafe_deployments = 0
    unknown_predictions = 0
    deployed = 0
    negative_deployed_pairs = 0
    deployed_seconds_saved = 0.0
    deployed_seconds_complete = 0
    for pair in pairs:
        family = str(pair.get("family_id") or "")
        context = _context_key(pair)
        if not family:
            unknown_predictions += 1
            continue
        pair_id = str(pair.get("pair_id") or "")
        should_deploy = (
            frozen_predictions[pair_id]
            if frozen_predictions is not None and pair_id in frozen_predictions
            else _predict(name, spec, pair)
        )
        if should_deploy is None:
            unknown_predictions += 1
            continue
        if not should_deploy:
            by_family_context[family][context].append(0.0)
            continue
        deployed += 1
        if pair.get("scientific_status") == "NUMERICAL_UNSAFE" or pair.get("effect_status") == "unsafe":
            unsafe_deployments += 1
            continue
        benefit = _predictive_outcome(pair)
        if benefit is None:
            missing_deployments += 1
            continue
        by_family_context[family][context].append(benefit)
        negative_deployed_pairs += benefit < 0
        s0, s1 = _raw_seconds(pair, "S0"), _raw_seconds(pair, "S1")
        if s0 is not None and s1 is not None:
            deployed_seconds_saved += s0 - s1
            deployed_seconds_complete += 1
    means = {
        family: sum(sum(values) / len(values) for values in contexts.values()) / len(contexts)
        for family, contexts in by_family_context.items() if contexts
    }
    complete = (
        missing_deployments == 0
        and unsafe_deployments == 0
        and unknown_predictions == 0
    )
    score = sum(means.values()) / len(means) if means and complete else None
    return {
        "policy": name,
        "status": "ESTIMABLE" if complete else "NOT_ESTIMABLE_INCOMPLETE_DEPLOYMENT_EVIDENCE",
        "planned_pairs": len(pairs),
        "deployed_pairs": deployed,
        "negative_deployed_pairs": negative_deployed_pairs,
        "negative_deployed_families": sum(value < 0 for value in means.values()),
        "coverage": deployed / len(pairs) if pairs else None,
        "net_seconds_saved": deployed_seconds_saved if deployed_seconds_complete == deployed else None,
        "deployed_pairs_with_seconds": deployed_seconds_complete,
        "unsafe_deployments": unsafe_deployments,
        "missing_deployed_outcomes": missing_deployments,
        "unknown_predictions": unknown_predictions,
        "observed_deployed_families": len(means),
        "family_equal_policy_utility": score,
        "family_cluster_bootstrap_ci95": _bootstrap_ci(means) if complete else None,
        "missingness_bounds": (
            {"lower": "-unbounded", "upper": "+unbounded"}
            if not complete else {"lower": score, "upper": score}
        ),
    }


def _predict(name: str, spec: Mapping[str, Any], pair: Mapping[str, Any]) -> bool | None:
    if spec.get("status") != "FITTED":
        return None
    if name == "NEVER":
        return False
    if name == "ALWAYS":
        return True
    if name == "HORIZON_THRESHOLD":
        value = _first_number(pair, ("horizon",))
        return value >= spec["threshold"] if value is not None else None
    if name == "STRUCTURE_CONV_COUNT":
        value = _feature(pair, "conv_count")
        return value >= spec["threshold"] if value is not None else None
    if name == "SIMPLE_SUPERVISED_RIDGE":
        names = spec["feature_names"]
        values = [_feature(pair, feature) for feature in names]
        if any(value is None for value in values):
            return None
        centered = [(value - mean) / scale for value, mean, scale in zip(values, spec["center"], spec["scale"])]
        prediction = spec["coefficients"][0] + sum(
            coefficient * value for coefficient, value in zip(spec["coefficients"][1:], centered)
        )
        return prediction > 0.0
    return None


def _spec_identity(spec: Mapping[str, Any], index: int) -> str:
    pair_id = spec.get("pair_id")
    if pair_id is None or not str(pair_id):
        raise ValueError(f"planned spec {index} is missing its frozen pair_id")
    return str(pair_id)


def _feature_identity(pair: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contrast": pair.get("contrast"),
        "horizon": _feature(pair, "horizon"),
        "batch": _feature(pair, "batch"),
        "parameter_count": _feature(pair, "parameter_count"),
        "conv_count": _feature(pair, "conv_count"),
    }


def _single_contrast(*groups: Sequence[Mapping[str, Any]]) -> str | None:
    contrasts = {
        str(pair["contrast"])
        for group in groups
        for pair in group
        if pair.get("contrast") is not None
    }
    if len(contrasts) > 1:
        raise ValueError("analysis and policy fitting must run separately for each contrast")
    return next(iter(contrasts), None)


def freeze_policies(
    dev_pairs: Sequence[Mapping[str, Any]],
    cal_pairs: Sequence[Mapping[str, Any]],
    all_specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select rules on development, validate them on calibration, freeze final predictions.

    ``all_specs`` contributes only identifiers and pre-execution features. Any
    outcome fields attached to those specifications are deliberately ignored.
    """
    contrast = _single_contrast(dev_pairs, cal_pairs, all_specs)
    dev = [pair for pair in dev_pairs if pair.get("split") == "development"]
    cal = [pair for pair in cal_pairs if pair.get("split") == "calibration"]
    dev_summary = _summarize_rows(dev)
    positive = dev_summary["positive_families"]
    negative = dev_summary["negative_families"]
    enough_heterogeneity = positive >= 2 and negative >= 2
    policies: dict[str, dict[str, Any]] = {
        "NEVER": {"status": "FITTED", "rule": "always abstain"},
        "ALWAYS": {"status": "FITTED", "rule": "always deploy"},
        "ORIGINAL_ACRE": {
            "status": "ORIGINAL_ACRE_NOT_FITTED",
            "reason": "this sidecar does not invoke the canonical engine or create a certificate",
            "canonical_certificate_fitted": False,
        },
    }
    if enough_heterogeneity:
        candidate_rules: list[tuple[str, dict[str, Any]]] = []
        for threshold in HORIZON_THRESHOLDS:
            candidate_rules.append((
                "HORIZON_THRESHOLD",
                {"status": "FITTED", "threshold": threshold, "selection_split": "development"},
            ))
        for threshold in CONV_COUNT_THRESHOLDS:
            candidate_rules.append((
                "STRUCTURE_CONV_COUNT",
                {"status": "FITTED", "threshold": threshold, "selection_split": "development"},
            ))
        training = _family_training_rows(dev)
        ridge = _ridge_fit(training)
        policies["SIMPLE_SUPERVISED_RIDGE"] = ridge

        selected: dict[str, tuple[float, dict[str, Any]]] = {}
        for name, spec in candidate_rules:
            score = _policy_utility(dev, name, spec)
            if score["status"] != "ESTIMABLE" or score["family_equal_policy_utility"] is None:
                continue
            current = selected.get(name)
            candidate = (score["family_equal_policy_utility"], spec)
            if current is None or (candidate[0], -float(spec.get("threshold", 0))) > (
                current[0], -float(current[1].get("threshold", 0))
            ):
                selected[name] = candidate
        for name in ("HORIZON_THRESHOLD", "STRUCTURE_CONV_COUNT"):
            if name in selected:
                policies[name] = selected[name][1]
            else:
                policies[name] = {
                    "status": "NOT_FITTED",
                    "reason": "no development candidate had complete deployed outcomes",
                    "selection_split": "development",
                }
    else:
        for name in _PREDICTIVE_POLICIES:
            policies[name] = {
                "status": "NOT_FITTED",
                "reason": "development has fewer than two positive and two negative independent families",
            }

    calibration = [
        _policy_utility(cal, name, spec)
        for name, spec in policies.items()
        if spec.get("status") == "FITTED"
    ]
    final_predictions = []
    final_specs = [spec for spec in all_specs if spec.get("split") == "final"]
    seen_ids: set[str] = set()
    for index, spec in enumerate(final_specs):
        identity = _spec_identity(spec, index)
        if identity in seen_ids:
            raise ValueError(f"duplicate planned final pair id: {identity}")
        seen_ids.add(identity)
        final_predictions.append({
            "pair_id": identity,
            "feature_identity": _feature_identity(spec),
            "predictions": {
                name: _predict(name, policy, spec)
                for name, policy in policies.items()
            },
        })

    simple_names = ("NEVER", "ALWAYS", "HORIZON_THRESHOLD", "STRUCTURE_CONV_COUNT")
    dev_simple = [
        (name, _policy_utility(dev, name, policies[name]))
        for name in simple_names
        if policies.get(name, {}).get("status") == "FITTED"
    ]
    dev_simple = [
        item for item in dev_simple
        if item[1]["status"] == "ESTIMABLE"
        and item[1]["family_equal_policy_utility"] is not None
    ]
    strongest_simple = max(
        dev_simple,
        key=lambda item: (
            item[1]["family_equal_policy_utility"]
            if item[1]["family_equal_policy_utility"] is not None else float("-inf"),
            -simple_names.index(item[0]),
        ),
        default=(None, None),
    )[0]

    return {
        "schema_version": 1,
        "contrast": contrast,
        "policy_selection_split": "development",
        "calibration_role": "validation_only_after_policy_freeze",
        "dev_independent_families": independent_group_count(dev),
        "dev_positive_families": positive,
        "dev_negative_families": negative,
        "predictive_policy_eligibility": "FITTED" if enough_heterogeneity else "NOT_FITTED",
        "policies": policies,
        "strongest_simple_policy_on_development": strongest_simple,
        "calibration_validation": calibration,
        "final_predictions": final_predictions,
        "final_outcomes_accessed": False,
        "original_acre_status": "ORIGINAL_ACRE_NOT_FITTED",
    }


def evaluate_policies(pairs: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any]) -> list[dict[str, Any]]:
    splits = {pair.get("split") for pair in pairs}
    if len(splits) > 1:
        raise ValueError("policy evaluation must run separately for each split")
    contrast = _single_contrast(pairs)
    if contrast is not None and freeze.get("contrast") not in (None, contrast):
        raise ValueError("policy freeze contrast does not match evaluation pairs")
    policies = freeze.get("policies", {})
    final = bool(pairs) and all(pair.get("split") == "final" for pair in pairs)
    frozen_by_id: dict[str, Mapping[str, Any]] = {}
    if final:
        for prediction in freeze.get("final_predictions", []):
            pair_id = str(prediction.get("pair_id") or "")
            if not pair_id or pair_id in frozen_by_id:
                raise ValueError("frozen final predictions require unique, non-empty pair_ids")
            frozen_by_id[pair_id] = prediction
        observed_ids = [str(pair.get("pair_id") or "") for pair in pairs]
        if any(not pair_id for pair_id in observed_ids) or len(set(observed_ids)) != len(observed_ids):
            raise ValueError("final evaluation pairs require unique, non-empty pair_ids")
        if set(observed_ids) != set(frozen_by_id):
            raise ValueError("final evaluation pair_ids do not match the frozen final prediction set")
        for pair in pairs:
            identity = str(pair["pair_id"])
            frozen = frozen_by_id[identity]
            if _feature_identity(pair) != frozen.get("feature_identity"):
                raise ValueError(f"final pair feature identity changed after policy freeze: {identity}")
            recomputed = {
                name: _predict(name, policy, pair) if policy.get("status") == "FITTED" else None
                for name, policy in policies.items()
            }
            if recomputed != frozen.get("predictions"):
                raise ValueError(f"final policy prediction changed after freeze: {identity}")
    rows = []
    rows_by_name: dict[str, dict[str, Any]] = {}
    for name, spec in policies.items():
        if not isinstance(spec, Mapping):
            continue
        if spec.get("status") == "NOT_FITTED" or spec.get("status") == "ORIGINAL_ACRE_NOT_FITTED":
            row = {
                "policy": name,
                "status": spec.get("status"),
                "reason": spec.get("reason"),
                "planned_pairs": len(pairs),
                "deployed_pairs": None,
                "coverage": None,
                "family_equal_policy_utility": None,
                "missingness_bounds": {"lower": "NOT_ESTIMABLE", "upper": "NOT_ESTIMABLE"},
            }
            rows.append(row)
            rows_by_name[name] = row
            continue
        predictions = (
            {pair_id: frozen_by_id[pair_id]["predictions"].get(name) for pair_id in frozen_by_id}
            if final else None
        )
        row = _policy_utility(pairs, name, spec, frozen_predictions=predictions)
        rows.append(row)
        rows_by_name[name] = row
    if final:
        benchmark = freeze.get("strongest_simple_policy_on_development")
        benchmark_row = rows_by_name.get(benchmark)
        for row in rows:
            row["strongest_simple_policy_on_development"] = benchmark
            row["final_utility_difference_vs_strongest_simple"] = (
                row["family_equal_policy_utility"] - benchmark_row["family_equal_policy_utility"]
                if benchmark_row is not None
                and row.get("family_equal_policy_utility") is not None
                and benchmark_row.get("family_equal_policy_utility") is not None
                else None
            )
    return rows
