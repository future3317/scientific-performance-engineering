"""Offline paired fixed-Skill evaluation and reporting; never launches a run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


POLICIES = ("global_fixed_skill", "original_acre_fixed_skill", "regime_acre_fixed_skill")
ARMS = ("off", "on")
TERMINAL_STATUSES = {
    "VERIFIED_SUCCESS", "VERIFIED_FAILURE", "AGENT_POLICY_FAILURE",
    "API_CONTRACT_FAILURE", "PROVIDER_FAILURE", "BUDGET_EXHAUSTED",
    "ENVIRONMENT_FAILURE", "INCONCLUSIVE",
}
COST_FIELDS = (
    "provider_calls", "input_tokens", "completion_tokens", "reasoning_tokens",
    "cache_hit_tokens", "cache_miss_tokens", "provider_usd_estimate",
    "wall_seconds", "gpu_seconds", "verification_seconds",
)
PAIRED_INDEX_FIELDS = (
    "task_id", "template_id", "cluster_id", "split", "arm", "trial_id",
    "baseline_trial_id", "skill_sha", "agent_sha", "evidence_origin",
    "run_status", "launched", "env_won", "verified_outcome", "repair",
    "regression", "paired_cell", "pair_complete", "nonlaunch_reason",
    *COST_FIELDS,
    *(f"{policy}_{field}" for policy in POLICIES for field in ("deployed", "outcome")),
)
FORBIDDEN_DECISION_FIELDS = {
    "outcome", "verified_outcome", "success", "environment_won", "reward",
    "won", "label", "repair", "regression",
}


def _contains_outcome_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(FORBIDDEN_DECISION_FIELDS.intersection(str(key).casefold() for key in value)) or any(
            _contains_outcome_field(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_outcome_field(item) for item in value)
    return False


def freeze_acre_route_decisions(
    engine: Any,
    tasks: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    skill_rule_ids: Sequence[str],
    skill_sha: str,
    agent_sha: str,
    study_id: str,
    run_id: str,
    token_budget: int | None = None,
    support_regimes: Mapping[str, Any] | None = None,
    runtime_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Freeze pre-outcome decisions through the canonical ``AcreEngine.route``."""
    if policy not in POLICIES[1:]:
        raise ValueError("only original/regime ACRE policies route through AcreEngine")
    if not skill_rule_ids or not skill_sha or not agent_sha or not study_id or not run_id:
        raise ValueError("study/run and fixed Skill/Agent identities are required")
    decisions = []
    for task in tasks:
        task_id = task.get("task_id")
        context = task.get("context")
        if not task_id or not isinstance(context, Mapping):
            raise ValueError("route decisions require task_id and pre-execution context")
        if _contains_outcome_field(context):
            raise ValueError("pre-execution routing context cannot contain outcomes or labels")
        result = engine.route(
            context,
            token_budget=token_budget,
            support_regimes=support_regimes if policy == "regime_acre_fixed_skill" else None,
            runtime_versions=runtime_versions,
            candidate_ids=skill_rule_ids,
        )
        selected = tuple(result.selected_rule_ids)
        decisions.append({
            "task_id": str(task_id),
            "policy": policy,
            "deployed": any(rule_id in selected for rule_id in skill_rule_ids),
            "selected_rule_ids": list(selected),
            "decision_source": "AcreEngine.route",
            "skill_sha": skill_sha,
            "agent_sha": agent_sha,
        })
    return {
        "record_type": "frozen_policy_decisions",
        "frozen_before_outcome_join": True,
        "study_id": study_id,
        "run_id": run_id,
        "policy": policy,
        "skill_sha": skill_sha,
        "agent_sha": agent_sha,
        "decisions": decisions,
    }


def freeze_policy_decisions(
    original_engine: Any,
    regime_engine: Any,
    tasks: Sequence[Mapping[str, Any]],
    *,
    skill_rule_ids: Sequence[str],
    skill_sha: str,
    agent_sha: str,
    study_id: str,
    run_id: str,
    token_budget: int | None = None,
    support_regimes: Mapping[str, Any] | None = None,
    runtime_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Freeze global, original ACRE, and regime ACRE decisions before outcomes."""
    if not skill_rule_ids:
        raise ValueError("global and ACRE policies require the same fixed Skill RuleSpecs")
    decisions = [{
        "task_id": str(task["task_id"]),
        "policy": "global_fixed_skill",
        "deployed": True,
        "selected_rule_ids": list(skill_rule_ids),
        "decision_source": "global_fixed_skill_policy",
        "skill_sha": skill_sha,
        "agent_sha": agent_sha,
    } for task in tasks]
    for engine, policy in (
        (original_engine, "original_acre_fixed_skill"),
        (regime_engine, "regime_acre_fixed_skill"),
    ):
        routed = freeze_acre_route_decisions(
            engine, tasks, policy=policy, skill_rule_ids=skill_rule_ids,
            skill_sha=skill_sha, agent_sha=agent_sha, study_id=study_id, run_id=run_id,
            token_budget=token_budget,
            support_regimes=support_regimes if policy == "regime_acre_fixed_skill" else None,
            runtime_versions=runtime_versions,
        )
        decisions.extend(routed["decisions"])
    return {
        "record_type": "frozen_policy_decisions",
        "frozen_before_outcome_join": True,
        "study_id": study_id,
        "run_id": run_id,
        "skill_sha": skill_sha,
        "agent_sha": agent_sha,
        "decisions": decisions,
    }


def _validate_plan(plan: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    if plan.get("plan_type") != "fixed_skill_evaluation_plan":
        raise ValueError("plan_type must be fixed_skill_evaluation_plan")
    if plan.get("evidence_kind") not in {"REAL_AGENT", "SYNTHETIC_FIXTURE"}:
        raise ValueError("evidence_kind must be explicit")
    for field in ("study_id", "run_id", "source_commit", "split_manifest_sha256", "skill_sha", "agent_sha", "model_identity", "configuration_sha", "benchmark_revision"):
        if not plan.get(field):
            raise ValueError(f"plan is missing {field}")
    if not isinstance(plan.get("skill_rule_ids"), list) or not plan["skill_rule_ids"]:
        raise ValueError("plan requires the canonical RuleSpec IDs for the fixed Skill")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("plan requires a nonempty frozen task list")
    seen: set[str] = set()
    template_clusters: dict[str, str] = {}
    trials: dict[str, dict[str, str]] = {}
    clean_tasks = []
    for raw in tasks:
        if not isinstance(raw, dict):
            raise ValueError("planned task entries must be objects")
        task_id, template_id, cluster_id, split = (raw.get(k) for k in ("task_id", "template_id", "cluster_id", "split"))
        if not all(isinstance(value, str) and value for value in (task_id, template_id, cluster_id, split)):
            raise ValueError("each planned task requires task/template/cluster/split identities")
        if type(raw.get("seed")) is not int:
            raise ValueError("each planned task requires its frozen integer environment seed")
        if task_id in seen:
            raise ValueError(f"duplicate planned task_id: {task_id}")
        seen.add(task_id)
        if template_id in template_clusters and template_clusters[template_id] != cluster_id:
            raise ValueError("one template cannot be counted as multiple independent clusters")
        template_clusters[template_id] = cluster_id
        planned = raw.get("trial_ids")
        if not isinstance(planned, dict) or set(planned) != set(ARMS):
            raise ValueError("each task must freeze off/on trial_ids")
        if any(not isinstance(planned[arm], str) or not planned[arm] for arm in ARMS):
            raise ValueError("planned trial IDs must be nonempty strings")
        if planned["off"] == planned["on"] or any(value in trials for value in planned.values()):
            raise ValueError("planned trial IDs must be unique")
        trials[planned["off"]] = {"task_id": task_id, "arm": "off"}
        trials[planned["on"]] = {"task_id": task_id, "arm": "on"}
        clean_tasks.append({**raw, "task_id": task_id, "template_id": template_id, "cluster_id": cluster_id, "split": split})
    policies = plan.get("policies")
    if policies != list(POLICIES):
        raise ValueError(f"policies must be exactly {list(POLICIES)}")
    if type(plan.get("candidate_selection_count", 1)) is not int or plan.get("candidate_selection_count", 1) < 1:
        raise ValueError("candidate_selection_count must be positive")
    if type(plan.get("minimum_independent_clusters_for_interval")) is not int or plan["minimum_independent_clusters_for_interval"] < 2:
        raise ValueError("minimum independent cluster count must be frozen in the plan")
    return clean_tasks, trials


def _validate_decisions(
    plan: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]], record: Mapping[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    if record.get("record_type") != "frozen_policy_decisions" or record.get("frozen_before_outcome_join") is not True:
        raise ValueError("policy decisions must be explicitly frozen before outcome join")
    if record.get("study_id") != plan["study_id"] or record.get("run_id") != plan["run_id"]:
        raise ValueError("frozen decisions belong to a different study/run")
    if record.get("skill_sha") != plan["skill_sha"] or record.get("agent_sha") != plan["agent_sha"]:
        raise ValueError("decisions do not use the plan's fixed Skill and Agent")
    rows = record.get("decisions")
    if not isinstance(rows, list):
        raise ValueError("frozen decisions must be a list")
    expected = {(str(task["task_id"]), policy) for task in tasks for policy in POLICIES}
    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("decision rows must be objects")
        if _contains_outcome_field(row):
            raise ValueError("policy decisions cannot contain outcomes or labels")
        task_id, policy = row.get("task_id"), row.get("policy")
        key = (task_id, policy)
        if key not in expected or key in decisions:
            raise ValueError("decision task/policy is unexpected or duplicated")
        if type(row.get("deployed")) is not bool:
            raise ValueError("every frozen decision must explicitly set deployed true/false")
        if row.get("skill_sha") != plan["skill_sha"] or row.get("agent_sha") != plan["agent_sha"]:
            raise ValueError("all policies must use the same frozen Skill and Agent")
        if policy != "global_fixed_skill" and row.get("decision_source") != "AcreEngine.route":
            raise ValueError("operational ACRE decisions must come from AcreEngine.route")
        if policy != "global_fixed_skill":
            selected = row.get("selected_rule_ids")
            if not isinstance(selected, list) or any(not isinstance(item, str) for item in selected):
                raise ValueError("ACRE decisions must preserve selected canonical RuleSpec IDs")
            expected_deployment = any(item in plan["skill_rule_ids"] for item in selected)
            if row["deployed"] != expected_deployment:
                raise ValueError("ACRE deployment flag differs from AcreEngine.route selection")
        decisions[key] = row
    if set(decisions) != expected:
        raise ValueError("one frozen decision is required for every planned task and policy")
    if any(not decisions[(str(task["task_id"]), "global_fixed_skill")]["deployed"] for task in tasks):
        raise ValueError("global fixed-Skill policy must deploy on every planned task")
    return decisions


def receipt_outcome(row: Mapping[str, Any]) -> bool | None:
    status = row.get("status")
    success, won = row.get("success"), row.get("environment_won")
    done = row.get("environment_done", row.get("done"))
    if status == "VERIFIED_SUCCESS":
        if success is not True or won is not True or done is not True:
            raise ValueError("verified success requires success, environment_won and done true")
        return True
    if status == "VERIFIED_FAILURE":
        if success is not False or won is not False or type(done) is not bool:
            raise ValueError("verified failure requires explicit false success/won and a known done state")
        return False
    if status == "AGENT_POLICY_FAILURE":
        if (success is not False or won is not False or done is not False
                or row.get("termination_reason") != "agent_policy_failure"):
            raise ValueError("agent policy failure requires explicit false success/won and environment_done false")
        return False
    if success is not None:
        raise ValueError("unverified/operational failures must keep success unknown")
    if won is not None and type(won) is not bool:
        raise ValueError("environment_won must be true, false, or null")
    if done is not None and type(done) is not bool:
        raise ValueError("done must be true, false, or null")
    return None


def _request_cost(row: Mapping[str, Any]) -> dict[str, int | float | None]:
    """Use per-request trajectory usage; cumulative provider_budget is ignored."""
    trajectory = row.get("trajectory")
    requests = row.get("request_usage")
    trajectory_usage = requests is None and isinstance(trajectory, list)
    if requests is None and trajectory_usage:
        requests = trajectory
    if not isinstance(requests, list):
        requests = None

    def total(field: str, aliases: tuple[str, ...] = ()) -> int | float | None:
        if requests is None or (not requests and field != "provider_calls"):
            return None
        if field == "provider_calls":
            if not trajectory_usage:
                return len(requests)
            calls = []
            for request in requests:
                if type(request.get("provider_request_made")) is bool:
                    calls.append(request["provider_request_made"])
                elif (any(type(request.get(key)) is int for key in ("input_tokens", "output_tokens", "completion_tokens"))
                      or "raw_output" in request or "provider_raw" in request
                      or request.get("failure_stage") == "provider"):
                    calls.append(True)
                else:
                    return None
            return sum(calls)
        values = []
        for request in requests:
            value = next((request[key] for key in (field, *aliases) if key in request), None)
            if field == "provider_usd_estimate" and isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    return None
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                return None
            values.append(value)
        return sum(values)

    return {
        "provider_calls": total("provider_calls"),
        "input_tokens": total("input_tokens"),
        "completion_tokens": total("completion_tokens", ("output_tokens",)),
        "reasoning_tokens": total("reasoning_tokens"),
        "cache_hit_tokens": total("cache_hit_tokens"),
        "cache_miss_tokens": total("cache_miss_tokens"),
        "provider_usd_estimate": total("provider_usd_estimate", ("estimated_provider_usd",)),
        "wall_seconds": row.get("wall_seconds"),
        "gpu_seconds": row.get("gpu_seconds"),
        "verification_seconds": row.get("verification_seconds"),
    }


def _sum_costs(costs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for field in COST_FIELDS:
        values = [cost.get(field) for cost in costs]
        known = bool(values) and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values)
        result[field] = {"value": sum(values) if known else None,
                         "status": "MEASURED" if known else "UNKNOWN"}
    return result


def _paired_counts(pairs: Sequence[tuple[bool, bool] | None]) -> dict[str, int]:
    counts = {"repair": 0, "regression": 0, "both_correct": 0, "both_wrong": 0, "missing": 0}
    for pair in pairs:
        if pair is None:
            counts["missing"] += 1
        elif pair == (False, True):
            counts["repair"] += 1
        elif pair == (True, False):
            counts["regression"] += 1
        elif pair == (True, True):
            counts["both_correct"] += 1
        else:
            counts["both_wrong"] += 1
    return counts


def _delta_summary(differences: Sequence[int], planned: int) -> dict[str, Any]:
    missing = planned - len(differences)
    return {
        "complete_case": {"value": mean(differences) if differences else None,
                          "n": len(differences), "status": "MEASURED" if differences else "UNDEFINED_NO_COMPLETE_PAIRS"},
        "planned_itt_bounds": {
            "lower": (sum(differences) - missing) / planned if planned else None,
            "upper": (sum(differences) + missing) / planned if planned else None,
            "n_planned": planned, "n_missing": missing,
            "status": "BOUNDED_WITH_MISSING" if missing else "COMPLETE",
        },
    }


def _binary_rate(values: Sequence[bool | None], planned: int) -> dict[str, Any]:
    observed = [value for value in values if type(value) is bool]
    successes = sum(value is True for value in observed)
    missing = planned - len(observed)
    return {
        "complete_case": {"value": successes / len(observed) if observed else None,
                          "n": len(observed), "status": "MEASURED" if observed else "UNDEFINED_NO_OBSERVED_OUTCOMES"},
        "planned_bounds": {"lower": successes / planned if planned else None,
                           "upper": (successes + missing) / planned if planned else None,
                           "n_planned": planned, "n_missing": missing},
    }


def _cluster_interval(
    differences: Sequence[int], cluster_ids: Sequence[str], *,
    selection_count: int, minimum_clusters: int
) -> dict[str, Any]:
    if selection_count != 1:
        return {"status": "NOT_REPORTED_SELECTED_CANDIDATE_MULTIPLICITY", "value": None}
    values: dict[str, list[int]] = {}
    for difference, cluster in zip(differences, cluster_ids):
        values.setdefault(cluster, []).append(difference)
    if len(values) < minimum_clusters:
        return {"status": "NOT_REPORTED_TOO_FEW_INDEPENDENT_SEMANTIC_CLUSTERS", "value": None,
                "independent_clusters": len(values), "minimum_clusters": minimum_clusters}
    # Keep the task-weighted complete-case estimand used by the paired delta.
    # Cluster-level score sums account for within-cluster dependence without
    # silently changing the estimand to an equally weighted cluster mean.
    estimate = mean(differences)
    n_tasks = len(differences)
    cluster_scores = [sum(value - estimate for value in items) for items in values.values()]
    se = math.sqrt(
        len(values) / (len(values) - 1)
        * sum(score * score for score in cluster_scores)
        / (n_tasks * n_tasks)
    )
    return {"status": "EXPLORATORY_SEMANTIC_CLUSTER_NORMAL_95", "value": estimate,
            "lower_95": estimate - 1.96 * se, "upper_95": estimate + 1.96 * se,
            "independent_clusters": len(values), "complete_case_tasks": n_tasks,
            "estimand": "task-weighted complete-case paired delta; semantic-cluster robust normal interval",
            "note": "Contrasts reuse paired outcomes and are not independent across policies."}


def evaluate_fixed_skill(
    plan: Mapping[str, Any],
    frozen_decisions: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    shared_cost_items: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Join explicit frozen decisions to typed receipts and summarize one fixed Skill."""
    tasks, planned_trials = _validate_plan(plan)
    decisions = _validate_decisions(plan, tasks, frozen_decisions)
    expected_origin = plan["evidence_kind"]
    expected_type = "real_agent_episode_receipt" if expected_origin == "REAL_AGENT" else "fixture_episode_receipt"
    by_trial: dict[str, Mapping[str, Any]] = {}
    cost_by_trial: dict[str, Mapping[str, Any]] = {}
    response_models: set[str] = set()
    allowed_status = TERMINAL_STATUSES | {"NOT_LAUNCHED", "STARTED", "PLANNED", "RESERVED", "ELIGIBLE"}
    for row in receipts:
        if row.get("evidence_origin") != expected_origin or row.get("schema_version") not in {2, "2"}:
            raise ValueError("receipt type/origin does not match the plan; fixture evidence cannot be called real")
        if row.get("receipt_type") not in {None, expected_type}:
            raise ValueError("receipt_type conflicts with its evidence_origin")
        if any(row.get(field) != plan[field] for field in ("study_id", "run_id", "source_commit", "split_manifest_sha256")):
            raise ValueError("receipt study/run/source commit identity differs from the frozen plan")
        trial_id = row.get("trial_id")
        if trial_id not in planned_trials or trial_id in by_trial:
            raise ValueError("receipt trial_id is unplanned or duplicated")
        expected = planned_trials[trial_id]
        if row.get("task_id") != expected["task_id"] or row.get("arm") != expected["arm"]:
            raise ValueError("receipt task/arm differs from frozen trial assignment")
        task = next(item for item in tasks if item["task_id"] == expected["task_id"])
        if any(row.get(field) != task[field] for field in ("template_id", "cluster_id", "split")):
            raise ValueError("receipt template/cluster/split identity differs from frozen task assignment")
        if row.get("agent_sha", row.get("agent_revision_sha256")) != plan["agent_sha"]:
            raise ValueError("receipt agent/model differs from fixed evaluation")
        request_model = row.get("model_request_name", row.get("model_identity"))
        if request_model != plan["model_identity"]:
            raise ValueError("receipt model request differs from fixed evaluation")
        response_model = row.get("model_response_identity", row.get("model_fingerprint"))
        planned_response_model = plan.get("model_response_identity")
        if planned_response_model is not None and response_model != planned_response_model:
            raise ValueError("receipt response model identity differs from fixed evaluation")
        if response_model is not None:
            response_models.add(str(response_model))
        if row.get("configuration_sha", row.get("prompt_template_sha256")) != plan["configuration_sha"]:
            raise ValueError("receipt prompt/configuration differs from fixed evaluation")
        if row.get("benchmark_revision") not in {None, plan["benchmark_revision"]}:
            raise ValueError("receipt benchmark revision differs from fixed evaluation")
        if row.get("seed") != task["seed"]:
            raise ValueError("receipt seed differs from frozen task assignment")
        skill = row.get("skill_sha", row.get("skill_version_sha256"))
        if (expected["arm"] == "off" and skill is not None) or (expected["arm"] == "on" and skill != plan["skill_sha"]):
            raise ValueError("receipt skill identity does not match its off/on assignment")
        status = row.get("status")
        if status not in allowed_status:
            raise ValueError("receipt status is not a recognized lifecycle/outcome enum")
        if status in TERMINAL_STATUSES:
            receipt_outcome(row)
        elif row.get("success") is not None:
            raise ValueError("nonterminal and not-launched receipts cannot carry an outcome")
        by_trial[trial_id] = row
        cost_by_trial[trial_id] = _request_cost(row)
    if len(response_models) > 1:
        raise ValueError("provider response model identity changed within the frozen evaluation")

    outcomes: dict[tuple[str, str], bool | None] = {}
    index_rows: list[dict[str, Any]] = []
    content_pairs: list[tuple[bool, bool] | None] = []
    content_diffs: list[int] = []
    content_clusters: list[str] = []
    gate_data: dict[str, dict[str, Any]] = {}
    task_count = len(tasks)
    for policy in POLICIES[1:]:
        gate_data[policy] = {"pairs": [], "delta_base": [], "delta_global": [], "rows": []}

    for task in tasks:
        task_id = str(task["task_id"])
        trial_rows = {arm: by_trial.get(task["trial_ids"][arm]) for arm in ARMS}
        arm_outcomes = {}
        for arm, row in trial_rows.items():
            outcome = receipt_outcome(row) if row and row.get("status") in TERMINAL_STATUSES else None
            outcomes[(task_id, arm)] = outcome
            arm_outcomes[arm] = outcome
        pair = (arm_outcomes["off"], arm_outcomes["on"]) if all(type(arm_outcomes[arm]) is bool for arm in ARMS) else None
        content_pairs.append(pair)
        if pair is not None:
            content_diffs.append(int(pair[1]) - int(pair[0]))
            content_clusters.append(str(task["cluster_id"]))
        idx = {
            "task_id": task_id, "template_id": task["template_id"], "cluster_id": task["cluster_id"],
            "split": task["split"], "skill_sha": plan["skill_sha"], "agent_sha": plan["agent_sha"],
            "off_trial_id": task["trial_ids"]["off"], "on_trial_id": task["trial_ids"]["on"],
            "off_status": trial_rows["off"].get("status") if trial_rows["off"] else "NOT_LAUNCHED",
            "on_status": trial_rows["on"].get("status") if trial_rows["on"] else "NOT_LAUNCHED",
            "off_outcome": arm_outcomes["off"], "on_outcome": arm_outcomes["on"],
            "repair": pair == (False, True) if pair is not None else None,
            "regression": pair == (True, False) if pair is not None else None,
            "pair_complete": pair is not None,
        }
        for policy in POLICIES:
            decision = decisions[(task_id, policy)]
            selected_arm = "on" if decision["deployed"] else "off"
            idx[f"{policy}_deployed"] = decision["deployed"]
            idx[f"{policy}_outcome"] = arm_outcomes[selected_arm]
        paired_cell = (
            "missing" if pair is None else "repair" if pair == (False, True)
            else "regression" if pair == (True, False)
            else "both_correct" if pair == (True, True) else "both_wrong"
        )
        for arm in ARMS:
            receipt = trial_rows[arm]
            status = receipt.get("status") if receipt else "NOT_LAUNCHED"
            launched = status not in {"PLANNED", "ELIGIBLE", "RESERVED", "NOT_LAUNCHED"}
            row = {
                "task_id": task_id, "template_id": task["template_id"],
                "cluster_id": task["cluster_id"], "split": task["split"],
                "arm": arm, "trial_id": task["trial_ids"][arm],
                "baseline_trial_id": task["trial_ids"]["off"],
                "skill_sha": plan["skill_sha"] if arm == "on" else None,
                "agent_sha": plan["agent_sha"],
                "evidence_origin": receipt.get("evidence_origin") if receipt else None,
                "run_status": status, "launched": launched,
                "env_won": receipt.get("environment_won") if receipt else None,
                "verified_outcome": arm_outcomes[arm],
                "repair": pair == (False, True) if pair is not None else None,
                "regression": pair == (True, False) if pair is not None else None,
                "paired_cell": paired_cell, "pair_complete": pair is not None,
                "nonlaunch_reason": (receipt.get("termination_reason", status)
                                     if receipt and not launched else "NOT_LAUNCHED" if not receipt else None),
            }
            usage = cost_by_trial.get(task["trial_ids"][arm], {})
            row.update({field: usage.get(field) for field in COST_FIELDS})
            row.update({key: value for key, value in idx.items() if key not in {
                "task_id", "template_id", "cluster_id", "split", "skill_sha", "agent_sha",
                "off_trial_id", "on_trial_id", "off_status", "on_status", "off_outcome",
                "on_outcome", "repair", "regression", "pair_complete",
            }})
            index_rows.append(row)

        for policy in POLICIES[1:]:
            deployed = decisions[(task_id, policy)]["deployed"]
            gate_outcome = arm_outcomes["on"] if deployed else arm_outcomes["off"]
            global_outcome = arm_outcomes["on"]
            gate_data[policy]["pairs"].append((global_outcome, gate_outcome)
                                                if type(global_outcome) is bool and type(gate_outcome) is bool else None)
            if type(arm_outcomes["off"]) is bool and type(gate_outcome) is bool:
                gate_data[policy]["delta_base"].append((int(gate_outcome) - int(arm_outcomes["off"]), task))
            if type(global_outcome) is bool and type(gate_outcome) is bool:
                gate_data[policy]["delta_global"].append((int(gate_outcome) - int(global_outcome), task))
            gate_data[policy]["rows"].append((deployed, arm_outcomes["off"], arm_outcomes["on"], gate_outcome))

    content_counts = _paired_counts(content_pairs)
    content = {
        "contrast": "fixed skill content on vs off under the same global treatment setup",
        "paired_counts": content_counts,
        "off_success_rate": _binary_rate([outcomes[(str(task["task_id"]), "off")] for task in tasks], task_count),
        "on_success_rate": _binary_rate([outcomes[(str(task["task_id"]), "on")] for task in tasks], task_count),
        "paired_delta_on_minus_off": _delta_summary(content_diffs, task_count),
        "semantic_cluster_interval": _cluster_interval(
            content_diffs, content_clusters,
            selection_count=int(plan.get("candidate_selection_count", 1)),
            minimum_clusters=plan["minimum_independent_clusters_for_interval"],
        ),
    }

    gates = {}
    for policy in POLICIES[1:]:
        data = gate_data[policy]
        pair_counts = _paired_counts(data["pairs"])
        base_diffs = [value for value, _ in data["delta_base"]]
        global_diffs = [value for value, _ in data["delta_global"]]
        base_tasks = [task for _, task in data["delta_base"]]
        global_tasks = [task for _, task in data["delta_global"]]
        rows = data["rows"]
        deployed = sum(item[0] for item in rows)
        observed_deployed = sum(item[0] and type(item[1]) is bool and type(item[2]) is bool for item in rows)
        policy_outcomes = [item[3] for item in rows]
        observed_pairs = sum(type(item[1]) is bool and type(item[2]) is bool for item in rows)
        repair_opportunities = content_counts["repair"]
        retained_repairs = sum(item[0] and item[1] is False and item[2] is True for item in rows)
        avoided_regressions = sum(not item[0] and item[1] is True and item[2] is False for item in rows)
        false_abstentions = sum(not item[0] and item[1] is False and item[2] is True for item in rows)
        harm = sum(item[0] and item[1] is True and item[2] is False for item in rows)
        gates[policy] = {
            "paired_counts_vs_global": pair_counts,
            "coverage": deployed / task_count,
            "n_deployed": deployed,
            "success_rate": _binary_rate(policy_outcomes, task_count),
            "incremental_success_over_base": _delta_summary(base_diffs, task_count),
            "paired_delta_vs_global": _delta_summary(global_diffs, task_count),
            "repairs_retained": retained_repairs,
            "repair_recall": retained_repairs / repair_opportunities if repair_opportunities else None,
            "regressions_avoided": avoided_regressions,
            "false_abstentions": false_abstentions,
            "harm_all": {
                "events": harm,
                "complete_case": harm / observed_pairs if observed_pairs else None,
                "n_complete_pairs": observed_pairs,
                "planned_itt_bounds": {"lower": harm / task_count,
                                        "upper": (harm + task_count - observed_pairs) / task_count},
            },
            "n_deployed_observed": observed_deployed,
            "harm_deployed": harm / observed_deployed if observed_deployed else None,
            "harm_deployed_status": "MEASURED" if observed_deployed else "UNDEFINED_ZERO_OBSERVED_DEPLOYMENT",
            "semantic_cluster_interval_vs_global": _cluster_interval(
                global_diffs, [str(task["cluster_id"]) for task in global_tasks],
                selection_count=int(plan.get("candidate_selection_count", 1)),
                minimum_clusters=plan["minimum_independent_clusters_for_interval"],
            ),
        }

    trial_costs = [cost_by_trial[trial_id] for trial_id in planned_trials if trial_id in cost_by_trial]
    shared_ids: set[str] = set()
    shared_costs = []
    for item in shared_cost_items:
        item_id = item.get("cost_id")
        if not item_id or item_id in shared_ids:
            raise ValueError("shared cost items require unique cost_id values")
        if item_id in planned_trials:
            raise ValueError("shared cost item duplicates a planned trial; count that physical receipt only once")
        shared_ids.add(item_id)
        values = item.get("cost")
        if not isinstance(values, dict):
            raise ValueError("shared cost item requires a cost object")
        shared_costs.append({field: values.get(field) for field in COST_FIELDS})
    task_cost = _sum_costs(trial_costs)
    all_cost = _sum_costs([*trial_costs, *shared_costs])
    terminal_receipts = [row for row in by_trial.values() if row.get("status") in TERMINAL_STATUSES]
    started_trials = [row for row in by_trial.values() if row.get("status") not in {"PLANNED", "ELIGIBLE", "RESERVED", "NOT_LAUNCHED"}]
    completed_task_ids = {
        str(task["task_id"]) for task in tasks
        if all(by_trial.get(task["trial_ids"][arm], {}).get("status") in TERMINAL_STATUSES for arm in ARMS)
    }
    report = {
        "status": "REAL_RECEIPT_ANALYSIS" if expected_origin == "REAL_AGENT" else "SYNTHETIC_FIXTURE_ONLY",
        "claim_scope": "one fixed Skill, one fixed Agent/model/configuration; policies reuse the same task-level off/on outcomes",
        "evidence_kind": expected_origin,
        "identity": {key: plan[key] for key in ("study_id", "run_id", "source_commit", "split_manifest_sha256", "skill_sha", "agent_sha", "model_identity", "configuration_sha", "benchmark_revision") if key in plan},
        "observed_model_response_identity": next(iter(response_models), None),
        "counts": {
            "n_planned": task_count, "n_started": len({row["task_id"] for row in started_trials}),
            "n_completed": len(completed_task_ids),
            "n_paired_valid": sum(pair is not None for pair in content_pairs),
            "n_success_true": sum(value is True for value in outcomes.values()),
            "n_success_false": sum(value is False for value in outcomes.values()),
            "n_success_unknown": sum(value is None for value in outcomes.values()),
            "n_not_launched": sum(1 for trial_id in planned_trials if trial_id not in by_trial or by_trial[trial_id].get("status") in {"PLANNED", "ELIGIBLE", "RESERVED", "NOT_LAUNCHED"}),
            "n_started_trials": len(started_trials), "n_completed_trials": len(terminal_receipts),
        },
        "content_contrast": content,
        "gate_contrasts": gates,
        "cost": {"task_only": task_cost, "all_in": all_cost,
                 "accounting_note": "Each physical trial and shared cost_id is summed once. Provider usage comes from per-request trajectory/request_usage; cumulative provider_budget snapshots are ignored."},
        "uncertainty_note": f"No interval is emitted below the predeclared minimum of {plan['minimum_independent_clusters_for_interval']} independent semantic clusters or for an unadjusted selected-candidate set. Reported intervals retain task weights and cluster dependence; policy intervals reuse outcomes and are correlated, not independent claims.",
        "independent_claims": False,
    }
    return report, index_rows


def dry_run_report(plan: Mapping[str, Any]) -> dict[str, Any]:
    tasks, _ = _validate_plan(plan)
    return {
        "status": "DRY_RUN_NOT_EXECUTED",
        "evidence_kind": plan["evidence_kind"],
        "study_id": plan["study_id"], "run_id": plan["run_id"],
        "n_planned": len(tasks), "n_started": 0, "n_completed": 0,
        "n_paired_valid": 0, "provider_calls": 0,
        "outcomes": None, "claim_scope": "plan only; no receipts joined and no efficacy result",
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Fixed-Skill evaluation", "", f"Status: `{report['status']}`", ""]
    if report["status"] == "DRY_RUN_NOT_EXECUTED":
        lines.extend([f"Planned tasks: {report['n_planned']}", "Provider calls: 0", "", report["claim_scope"], ""])
        return "\n".join(lines)
    counts = report["counts"]
    lines.extend(["## Trial coverage", "", "| Planned | Started | Completed | Paired valid | Success true | Success false | Unknown | Not launched |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|",
                  f"| {counts['n_planned']} | {counts['n_started']} | {counts['n_completed']} | {counts['n_paired_valid']} | {counts['n_success_true']} | {counts['n_success_false']} | {counts['n_success_unknown']} | {counts['n_not_launched']} |",
                  "", "## Skill content", "", f"Paired cells: `{json.dumps(report['content_contrast']['paired_counts'], sort_keys=True)}`",
                  f"On minus off: `{json.dumps(report['content_contrast']['paired_delta_on_minus_off'], sort_keys=True)}`", "",
                  "## Routing policies", ""])
    for policy, result in report["gate_contrasts"].items():
        lines.extend([f"### {policy}", "",
                      f"Coverage: {result['coverage']:.3f}; repair retained: {result['repairs_retained']}; regressions avoided: {result['regressions_avoided']}; false abstentions: {result['false_abstentions']}.",
                      f"Harm/all: `{json.dumps(result['harm_all'], sort_keys=True)}`; harm/deployed: {result['harm_deployed'] if result['harm_deployed'] is not None else 'undefined (zero observed deployments)'}.",
                      f"Paired delta vs global: `{json.dumps(result['paired_delta_vs_global'], sort_keys=True)}`", ""])
    lines.extend(["## Cost", "", "Task-only and all-in cost are in `analysis.json`; shared trial IDs are counted once.", "",
                  report["cost"]["accounting_note"], "", report["uncertainty_note"], ""])
    return "\n".join(lines)


def _write_outputs(output_dir: Path, report: dict[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "REPORT.md").write_text(_markdown(report), encoding="utf-8")
    with (output_dir / "paired_task_index.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIRED_INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--shared-costs", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="render plan-only output (default)")
    mode.add_argument("--evaluate", action="store_true", help="join frozen decisions to supplied receipts")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.evaluate:
        if not args.decisions or not args.receipts:
            parser.error("--evaluate requires --decisions and --receipts")
        decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
        receipt_document = json.loads(args.receipts.read_text(encoding="utf-8"))
        receipts = receipt_document.get("episodes") if isinstance(receipt_document, dict) else receipt_document
        if not isinstance(receipts, list):
            parser.error("--receipts must contain a receipt list or a runner results object with episodes")
        shared_costs = json.loads(args.shared_costs.read_text(encoding="utf-8")) if args.shared_costs else []
        report, rows = evaluate_fixed_skill(plan, decisions, receipts, shared_cost_items=shared_costs)
    else:
        report, rows = dry_run_report(plan), []
    _write_outputs(args.output_dir, report, rows)
    print(json.dumps({"status": report["status"], "output_dir": str(args.output_dir), "provider_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
