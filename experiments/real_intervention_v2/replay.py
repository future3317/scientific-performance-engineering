from __future__ import annotations

import json
import math
from pathlib import Path
import random
from statistics import median
from typing import Any, Mapping, Sequence

from core.acre.engine import AcreEngine
from core.acre.regimes import SupportRegime, SupportRegimeDecision, SupportScope
from core.models import RuleSpec, RuleState, TaskContext
from .contracts import digest


POLICIES = ("never", "always", "simple_horizon", "simple_break_even", "supervised_small",
            "original_acre", "regime_aware_acre", "pairwise_oracle")


def _benefit(row: Mapping[str, Any]) -> float:
    arms = row["arms"]
    return float(arms["eager"]["total_job_s"] - arms["compile"]["total_job_s"])


def _task_context(context: Mapping[str, Any]) -> TaskContext:
    return TaskContext(
        domain="runtime",
        workload={key: context[key] for key in (
            "benchmark_source", "model_id", "task_mode", "planned_horizon",
            "batch_size", "input_shapes", "dtype", "shape_regime", "model_parameter_count",
        ) if key in context},
        hardware={key: context[key] for key in ("device_type", "gpu_uuid") if key in context},
        software={key: context[key] for key in ("torch_version", "compiler_backend") if key in context},
        evidence={},
    )


def fit_stump(train: Sequence[Mapping[str, Any]], feature: str = "model_parameter_count",
              minimum_effect_log: float = 0.04879016416943205) -> tuple[str, float | None, bool]:
    def value(row):
        context = row["context"]
        for part in feature.split("."):
            context = context.get(part, {}) if isinstance(context, Mapping) else {}
        return float(context) if isinstance(context, (int, float)) else None
    eligible = [row for row in train
                if all(row.get("arms", {}).get(arm, {}).get("status") == "valid"
                       for arm in ("eager", "compile"))
                and row.get("correctness", {}).get("passed", False)]
    samples = []
    for row in eligible:
        ratio = float(row["arms"]["eager"]["total_job_s"]) / float(row["arms"]["compile"]["total_job_s"])
        effect = math.log(ratio)
        if effect > minimum_effect_log:
            label = 1
        elif effect < -minimum_effect_log:
            label = 0
        else:
            continue
        samples.append((value(row), label))
    samples = [(x, y) for x, y in samples if x is not None]
    if not samples:
        return feature, None, False
    candidates = sorted({x for x, _ in samples})
    best = (float("-inf"), candidates[0], False)
    for threshold in candidates:
        for above in (False, True):
            correct = sum((int((x >= threshold) == above) == y) for x, y in samples)
            if correct > best[0]:
                best = (correct, threshold, above)
    return feature, best[1], best[2]


def _acre(manifest: Mapping[str, Any], context: Mapping[str, Any], regime: bool = False) -> bool:
    rules = [RuleSpec.from_dict(item) for item in manifest.get("acre_rules", [])]
    states = {item["rule_id"]: RuleState.from_dict(item) for item in manifest.get("acre_states", [])}
    engine = AcreEngine(rule_specs=rules, rule_states=states)
    task_context = _task_context(context)
    if not regime:
        return bool(engine.route(task_context).selected_rule_ids)
    decisions = {}
    for rule_id, raw in manifest.get("acre_regimes", {}).items():
        scope = SupportScope(**{**raw["scope"], "required_strata": tuple(raw["scope"].get("required_strata", []))})
        decisions[rule_id] = SupportRegimeDecision(
            regime=SupportRegime(raw["regime"]), action=raw["action"], reason_code=raw["reason_code"], scope=scope,
            **{key: raw[key] for key in ("predicate", "independent_groups", "net_benefit_lcb", "harm_ucb", "stratum_bounds", "source_certificate_digest", "certificate_scope_digest", "certificate_digest", "decision_features", "decision_contexts") if key in raw},
        )
    versions = next(iter(decisions.values())).scope.versions if decisions else {}
    return bool(engine.route(task_context, support_regimes=decisions, runtime_versions=versions).selected_rule_ids)


def _benefit_if_valid(row: Mapping[str, Any]) -> float | None:
    return (_benefit(row)
            if all(row.get("arms", {}).get(arm, {}).get("status") == "valid"
                   for arm in ("eager", "compile"))
            and row.get("correctness", {}).get("passed", False) else None)


def _fit_horizon_threshold(train: Sequence[Mapping[str, Any]]) -> int:
    candidates = sorted({int(row["context"]["planned_horizon"]) for row in train})
    valid = [(row, _benefit_if_valid(row)) for row in train]
    valid = [(row, benefit) for row, benefit in valid if benefit is not None]
    if not candidates or not valid:
        return 10**18
    return max(candidates, key=lambda threshold: sum(
        benefit for row, benefit in valid
        if int(row["context"]["planned_horizon"]) >= threshold
    ))


def _fit_break_even_threshold(train: Sequence[Mapping[str, Any]]) -> int:
    estimates = []
    for row in train:
        eager, compiled = row.get("arms", {}).get("eager", {}), row.get("arms", {}).get("compile", {})
        if (eager.get("status") != "valid" or compiled.get("status") != "valid"
                or not row.get("correctness", {}).get("passed", False)):
            continue
        h = int(row["context"]["planned_horizon"])
        eager_warm, compile_warm = eager.get("warm_median_s"), compiled.get("warm_median_s")
        if eager_warm is None or compile_warm is None:
            continue
        per_call_saving = float(eager_warm) - float(compile_warm)
        overhead = (float(compiled["total_job_s"]) - h * float(compile_warm)
                    - max(0.0, float(eager["total_job_s"]) - h * float(eager_warm)))
        if per_call_saving > 0:
            estimates.append(max(1, int(overhead / per_call_saving) + 1))
    return sorted(estimates)[len(estimates) // 2] if estimates else 10**18


def freeze_policy(train: Sequence[Mapping[str, Any]], final_contexts: Sequence[Mapping[str, Any]],
                  manifest: Mapping[str, Any], *, feature: str = "model_parameter_count",
                  aa_noise_floor_log: float = 0.0) -> dict[str, Any]:
    """Fit only on development/calibration receipts, then freeze all final decisions."""
    minimum_effect = max(0.04879016416943205, aa_noise_floor_log)
    stump_feature, stump_threshold, positive_above = fit_stump(train, feature, minimum_effect)
    horizon_threshold = _fit_horizon_threshold(train)
    break_even_threshold = _fit_break_even_threshold(train)
    stump = {"feature": stump_feature, "threshold": stump_threshold,
             "positive_above": positive_above}
    decisions = {}
    for context in final_contexts:
        x = context.get(stump_feature.split(".")[-1])
        supervised = x is not None and stump_threshold is not None and ((float(x) >= stump_threshold) == positive_above)
        decisions[str(context["context_id"])] = {
            "simple_horizon": int(context.get("planned_horizon", 0)) >= horizon_threshold,
            "simple_break_even": int(context.get("planned_horizon", 0)) >= break_even_threshold,
            "supervised_small": bool(supervised),
            "original_acre": _acre(manifest, context),
            "regime_aware_acre": _acre(manifest, context, True),
        }
    payload = {
        "schema_version": 1,
        "training_context_ids": sorted(str(row["context_id"]) for row in train),
        "supervised_stump": stump,
        "minimum_horizon": horizon_threshold,
        "break_even_horizon": break_even_threshold,
        "aa_noise_floor_log": aa_noise_floor_log,
        "minimum_effect_log": minimum_effect,
        "final_context_ids": sorted(str(row["context_id"]) for row in final_contexts),
        "final_decisions": decisions,
        "acre_status": "frozen_rules" if manifest.get("acre_rules") else "abstain_no_certified_support_rule",
    }
    payload["final_decision_digest"] = digest(decisions)
    payload["policy_digest"] = digest(payload)
    return payload


def replay(records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any],
           frozen_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Score saved outcomes only after a pre-execution policy is supplied.

    When policy is omitted this is a training-side diagnostic and returns no
    final scores. A final scorer must pass the frozen policy and decision map.
    """
    records = aggregate_sessions(records)
    train = [row for row in records if row["context"]["split"] in {"development", "gate_calibration"}]
    evaluation = [row for row in records if row["context"]["split"] == "final_untouched"]
    if frozen_policy is None:
        feature, threshold, above = fit_stump(train)
        return {"final_opened": False, "training_pair_count": len(train),
                "supervised_stump": {"feature": feature, "threshold": threshold,
                                     "positive_above": above}}
    if not frozen_policy.get("policy_digest") or not frozen_policy.get("final_decision_digest"):
        raise ValueError("final scoring requires frozen policy and pre-execution decision digests")
    decision_map = frozen_policy.get("final_decisions", {})
    policy_body = dict(frozen_policy)
    policy_body.pop("policy_digest", None)
    if digest(decision_map) != frozen_policy["final_decision_digest"] or digest(policy_body) != frozen_policy["policy_digest"]:
        raise ValueError("frozen policy or decision digest mismatch")
    if set(decision_map) != {row["context_id"] for row in evaluation}:
        raise ValueError("frozen final decisions do not exactly cover the final contexts")
    feature, threshold, above = (frozen_policy["supervised_stump"]["feature"],
                                 frozen_policy["supervised_stump"]["threshold"],
                                 frozen_policy["supervised_stump"]["positive_above"])
    minimum_effect = float(frozen_policy["minimum_effect_log"])
    per_policy = {}
    for policy_name in POLICIES:
        rows = []
        for row in evaluation:
            context = row["context"]
            benefit = _benefit_if_valid(row)
            log_effect = (math.log(float(row["arms"]["eager"]["total_job_s"])
                                   / float(row["arms"]["compile"]["total_job_s"]))
                          if benefit is not None else None)
            if policy_name == "never": deploy = False
            elif policy_name == "always": deploy = True
            elif policy_name in {"simple_horizon", "simple_break_even", "supervised_small",
                                 "original_acre", "regime_aware_acre"}:
                deploy = bool(decision_map[context["context_id"]][policy_name])
            else: deploy = benefit is not None and benefit > 0  # scoring-only oracle; never deployable
            arms = row.get("arms", {})
            invalid_candidate = arms.get("compile", {}).get("status") != "valid" or (
                arms.get("eager", {}).get("status") == "valid"
                and not row.get("correctness", {}).get("passed", False)
            )
            rows.append({"context_id": context["context_id"], "deploy": bool(deploy), "benefit_s": benefit,
                         "log_effect": log_effect,
                         "realized_saving_s": benefit if deploy and benefit is not None else (None if deploy else 0.0),
                         "harm": bool(deploy and log_effect is not None and log_effect < -minimum_effect)
                                 or bool(deploy and invalid_candidate),
                         "regret_s": None if benefit is None else max(0.0, benefit) - (benefit if deploy else 0.0),
                         "outcome_valid": benefit is not None,
                         "candidate_status": arms.get("compile", {}).get("status")})
        n = len(rows)
        valid_rows = [r for r in rows if r["outcome_valid"]]
        deployed = [r for r in rows if r["deploy"]]
        evaluation_contexts = {item["context_id"]: item["context"] for item in evaluation}
        grouped_values: dict[str, list[float]] = {}
        for row in valid_rows:
            lineage = str(evaluation_contexts[row["context_id"]]["lineage_id"])
            grouped_values.setdefault(lineage, []).append(float(row["realized_saving_s"] or 0.0))
        lineage_values = [sum(values) / len(values) for values in grouped_values.values()]
        rng = random.Random(int(manifest.get("seed", 20261018)))
        bootstrap = []
        if len(lineage_values) >= 2:
            for _ in range(10000):
                sample = [lineage_values[rng.randrange(len(lineage_values))] for _ in lineage_values]
                bootstrap.append(sum(sample) / len(sample))
            bootstrap.sort()
        per_policy[policy_name] = {
            "n": n, "outcome_valid_n": len(valid_rows), "coverage": len(deployed) / n if n else None,
            "paired_valid_coverage": len(valid_rows) / n if n else None,
            "deployed_count": len(deployed),
            "valid_benefit_count": sum(float(row["log_effect"]) > minimum_effect for row in valid_rows),
            "valid_harm_count": sum(float(row["log_effect"]) < -minimum_effect for row in valid_rows),
            "unresolved_count": n - sum(abs(float(row["log_effect"])) > minimum_effect for row in valid_rows),
            "deployed_harm_count": sum(bool(row["harm"]) for row in deployed),
            "mean_saving_s": (sum(r["realized_saving_s"] or 0.0 for r in valid_rows) / len(valid_rows)) if valid_rows else None,
            "lineage_macro_saving_s": (sum(lineage_values) / len(lineage_values)) if lineage_values else None,
            "lineage_bootstrap_95_ci_s": ([bootstrap[int(0.025 * len(bootstrap))],
                                             bootstrap[min(len(bootstrap) - 1, int(0.975 * len(bootstrap)))]]
                                            if bootstrap else None),
            "bootstrap_lineage_count": len(lineage_values),
            "deployed_harm_rate": sum(r["harm"] for r in deployed) / len(deployed) if deployed else None,
            "mean_regret_s": (sum(r["regret_s"] or 0.0 for r in valid_rows) / len(valid_rows)) if valid_rows else None,
            "rows": rows,
        }
    return {"policy_names": list(POLICIES), "policy_digest": frozen_policy["policy_digest"],
            "final_decision_digest": frozen_policy["final_decision_digest"],
            "supervised_stump": {"feature": feature, "threshold": threshold, "positive_above": above},
            "evaluation_lineages": sorted({r["context"]["lineage_id"] for r in evaluation}), "results": per_policy,
            "oracle_role": "scoring_only"}


def aggregate_sessions(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["context_id"]), []).append(record)
    aggregate = []
    for context_id, sessions in sorted(grouped.items()):
        context = dict(sessions[0]["context"])
        if any(row["context"] != context for row in sessions):
            raise ValueError(f"context metadata differs across sessions: {context_id}")
        expected = len(context.get("sessions", sessions))
        by_session = {str(row.get("session_id", "session-1")): row for row in sessions}
        arms = {}
        for arm in ("eager", "compile"):
            observed = [row.get("arms", {}).get(arm, {}) for row in sessions]
            statuses = [item.get("status", "unknown") for item in observed]
            valid = len(sessions) == expected and len(set(by_session)) == expected and all(s == "valid" for s in statuses)
            arm_status = "valid" if valid else next((s for s in statuses if s not in {"valid", "unknown"}), "unknown")
            values = {}
            if valid:
                for field in ("total_job_s", "setup_s", "compile_wrapper_s", "cold_horizon_execution_s", "warm_median_s"):
                    finite = [float(item[field]) for item in observed if item.get(field) is not None]
                    if len(finite) == len(observed):
                        values[field] = median(finite)
            arms[arm] = {"status": arm_status, **values,
                         "session_statuses": statuses,
                         "parent_wall_s": sum(float(item.get("parent_wall_s", 0.0)) for item in observed)}
        checks = [row.get("correctness", {}).get("passed", False) for row in sessions]
        correctness = {"passed": len(sessions) == expected and all(checks),
                       "sessions_passed": sum(bool(value) for value in checks),
                       "sessions_expected": expected}
        aggregate.append({"context_id": context_id, "context": context, "arms": arms,
                          "correctness": correctness, "sessions": sessions})
    return aggregate


def analyze(receipt_dir: str | Path, manifest: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    records = []
    for path in sorted(Path(receipt_dir).glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    result = replay(records, manifest)
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
