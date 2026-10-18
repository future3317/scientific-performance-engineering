from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Mapping, Sequence

from .models import PairedOutcome, PolicyDecision


def _metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    valid = [row for row in rows if bool(row["execution_valid"])]
    deployed = [row for row in valid if bool(row["deploy"])]
    repairs_all = sum(bool(row["repair"]) for row in valid)
    repairs_deployed = sum(bool(row["repair"]) and bool(row["deploy"]) for row in valid)
    regressions_deployed = sum(bool(row["regression"]) and bool(row["deploy"]) for row in valid)
    gains = [float(row["skill_score"]) - float(row["baseline_score"]) for row in valid]
    selected = [float(row["selected_score"]) for row in valid]
    incremental = [float(row["selected_score"]) - float(row["baseline_score"]) for row in valid]
    coverage = len(deployed) / len(valid) if valid else None
    return {
        "n": len(valid),
        "invalid_outcomes": len(rows) - len(valid),
        "deployed": len(deployed),
        "coverage": coverage,
        "repairs_all": repairs_all,
        "repairs_deployed": repairs_deployed,
        "repairs_retained": repairs_deployed,
        "regressions_deployed": regressions_deployed,
        "regression_rate_deployed": regressions_deployed / len(deployed) if deployed else None,
        "fallback_rate": (len(valid) - len(deployed)) / len(valid) if valid else None,
        "repair_recall": repairs_deployed / repairs_all if repairs_all else None,
        "paired_score_gain_mean": sum(gains) / len(gains) if gains else None,
        "policy_utility_mean": sum(selected) / len(selected) if selected else None,
        "incremental_policy_utility": sum(incremental) / len(incremental) if incremental else None,
        "official_score_mean": sum(selected) / len(selected) if selected else None,
    }


def score_policies(
    decisions: Mapping[str, Sequence[PolicyDecision]],
    outcomes: Sequence[PairedOutcome],
    native_scorer: Callable[[PairedOutcome, bool], Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    by_task: dict[str, list[PairedOutcome]] = defaultdict(list)
    for outcome in outcomes:
        by_task[outcome.task_id].append(outcome)
    report: dict[str, dict[str, object]] = {}
    for policy_name, policy_decisions in decisions.items():
        decision_by_task: dict[str, PolicyDecision] = {}
        for decision in policy_decisions:
            if decision.policy_name != policy_name:
                raise ValueError(f"decision policy mismatch for {decision.task_id}")
            if decision.task_id in decision_by_task:
                raise ValueError(f"duplicate decision for {policy_name}/{decision.task_id}")
            decision_by_task[decision.task_id] = decision
        rows: list[dict[str, object]] = []
        for task_id, task_outcomes in by_task.items():
            decision = decision_by_task.get(task_id)
            if decision is None:
                raise ValueError(f"missing decision for {policy_name}/{task_id}")
            for outcome in task_outcomes:
                selected = native_scorer(outcome, decision.deploy)
                rows.append({
                    "task_id": task_id,
                    "repeat_seed": outcome.repeat_seed,
                    "deploy": decision.deploy,
                    "execution_valid": outcome.execution_valid,
                    "baseline_score": outcome.baseline_score,
                    "skill_score": outcome.skill_score,
                    "selected_score": selected["score"],
                    "repair": (not outcome.baseline_success) and outcome.skill_success,
                    "regression": outcome.baseline_success and (not outcome.skill_success),
                })
        report[policy_name] = _metrics(rows)
    return report


def paired_task_bootstrap(rows: Sequence[Mapping[str, object]], reps: int = 10000, seed: int = 0) -> dict[str, object]:
    by_task: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    task_ids = sorted(by_task)
    rng = random.Random(seed)
    coverage: list[float] = []
    regression_rate_deployed: list[float | None] = []
    for _ in range(reps):
        sampled = [row for task_id in (rng.choice(task_ids) for _ in task_ids) for row in by_task[task_id]]
        deployed = [row for row in sampled if bool(row.get("deployed"))]
        coverage.append(sum(bool(row.get("deployed")) for row in sampled) / len(sampled) if sampled else 0.0)
        regression_rate_deployed.append(
            sum(bool(row.get("regression")) for row in deployed) / len(deployed) if deployed else None
        )
    return {
        "cluster_unit": "task_id",
        "replicates": reps,
        "coverage": coverage,
        "regression_rate_deployed": regression_rate_deployed,
    }
