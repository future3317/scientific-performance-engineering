"""Bounded live pilot for evidence-directed semantic Skill selection.

The command has a provider-free --prepare phase and one exact-identity --live
phase. A durable provider intent without a response is terminal UNKNOWN and is
never retried.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from core.acre.resource_budget import ApprovedProviderBudget, ProviderBudgetExceeded, manifest_digest
from core.skills import ModelSkillProposer
from core.skills.semantic import SemanticItem
from experiments.skill_evolution.evaluation import receipt_outcome
from experiments.skill_evolution.real_learning import source_bundle_from_receipts
from experiments.skill_evolution.receipt_journal import DurableProviderJournal
from experiments.skill_evolution.split_manifest import validate_split_membership
from .discriminative_planner import DecisionModel, choose_evolution_action

MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
EXPECTED_IDENTITY = (MODEL, "bc19ba97cbe016479903222bb1578db6")
PRICING = {
    "currency": "USD", "model": MODEL,
    "pricing_source_url": "https://api-docs.deepseek.com/quick_start/pricing",
    "checked_at_utc": "2026-11-01T18:47:42Z", "price_period": "2026-11-01 official peak rates",
    "estimate_quality": "official peak list price; cache counters must reconcile",
    "input_hit_per_million": "0.006", "input_miss_per_million": "0.30",
    "output_per_million": "1.32",
}
LIKELIHOODS = {
    "A_ONLY": (0.55, 0.80, 0.55, 0.55),
    "B_ONLY": (0.55, 0.55, 0.80, 0.55),
    "NEITHER": (0.75, 0.70, 0.70, 0.75),
}
PRIORS = {"A_ONLY": 0.25, "B_ONLY": 0.25, "NEITHER": 0.50}
LOSSES = {
    "retain_a": {"A_ONLY": 0.0, "B_ONLY": 1.0, "NEITHER": 1.0},
    "retain_b": {"A_ONLY": 1.0, "B_ONLY": 0.0, "NEITHER": 1.0},
    "abstain": {"A_ONLY": 0.6, "B_ONLY": 0.6, "NEITHER": 0.0},
}
DEV_ARMS = ("parent_00", "item_a_10", "item_b_01", "LENGTH")
CAL_ARMS = ("NO_SKILL", "LENGTH", "STATIC_CDED", "ADAPTIVE_CDED", "FIXED_ORDER_CDED")
MAX_STEPS = 16
MAX_EPISODE_CALLS = 16
MAX_CALLS = 386
MAX_INPUT = 4096
MAX_OUTPUT = 1024
MAX_USD = "1.00"
MAX_EPISODE_USD = "0.05"
WORST_CASE_USD = (Decimal(MAX_CALLS) *
                  (Decimal(MAX_INPUT) * Decimal(PRICING["input_miss_per_million"])
                   + Decimal(MAX_OUTPUT) * Decimal(PRICING["output_per_million"]))
                  / Decimal(1_000_000))


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _append(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _resolve_tasks(pilot_split: Mapping[str, Any], parent_split: Mapping[str, Any],
                   data_root: Path) -> dict[str, list[dict[str, Any]]]:
    expected_parent = pilot_split["source_split_manifest_sha256"]
    if manifest_digest(parent_split) != expected_parent:
        raise ValueError("parent split manifest differs from the frozen research identity")
    validate_split_membership(parent_split["task_membership"])
    members = {row["task_id"]: row for row in parent_split["task_membership"]}
    groups = {
        "source": pilot_split["proposal_sources"],
        "development": pilot_split["development_screen"],
        "calibration": pilot_split["calibration"],
        "final": pilot_split["evaluation"],
    }
    resolved: dict[str, list[dict[str, Any]]] = {}
    template_ids: set[str] = set()
    cluster_ids: set[str] = set()
    task_ids: set[str] = set()
    for group, selected in groups.items():
        resolved[group] = []
        for item in selected:
            row = members.get(item["task_id"])
            if row is None or row.get("identity_status") != "IDENTIFIED":
                raise ValueError("selected task is absent or has unknown identity")
            if row.get("template_id") != item["template_id"] or row.get("cluster_id") != item["cluster_id"]:
                raise ValueError("selected task cluster/template differs from the parent split")
            if row.get("family") != item["family"] and row.get("task_type") != item["family"]:
                raise ValueError("selected task family differs from the frozen split")
            expected_split = {
                "source": "train_dev", "development": "train_inner",
                "calibration": "train_inner", "final": "train_outer",
            }[group]
            expected_exposure = {
                "train_dev": "new_train_dev", "train_inner": "new_train_inner",
                "train_outer": "new_train_outer",
            }[expected_split]
            if row.get("split") != expected_split or row.get("exposure") != expected_exposure:
                raise ValueError("selected task exposure label differs from the frozen stage")
            if expected_split != "train_dev" and row.get("confirmatory_eligible") is not True:
                raise ValueError("inner/final task is not confirmatory eligible")
            if item["family"] in {r["family"] for r in resolved[group]}:
                raise ValueError("each stage requires distinct native task families")
            if (item["template_id"] in template_ids or item["cluster_id"] in cluster_ids
                    or item["task_id"] in task_ids):
                raise ValueError("selected tasks overlap by task, template, or semantic cluster")
            template_ids.add(item["template_id"])
            cluster_ids.add(item["cluster_id"])
            task_ids.add(item["task_id"])
            trial = item["task_id"].split("/", 1)[1]
            gamefile = data_root / "json_2.1.1" / "train" / item["template_id"] / trial / "game.tw-pddl"
            if not gamefile.is_file() or hashlib.sha256(gamefile.read_bytes()).hexdigest() != item["task_file_sha256"]:
                raise ValueError("selected ALFWorld task file is absent or differs from its frozen digest")
            resolved[group].append({
                **dict(row), **dict(item), "gamefile": str(gamefile.resolve()),
                "seed": pilot_split["selection_seed"], "group": group,
            })
    if len(resolved["source"]) != 2 or len(resolved["development"]) != 2:
        raise ValueError("pilot requires two source and two development clusters")
    if len(resolved["calibration"]) != 2 or len(resolved["final"]) != 2:
        raise ValueError("pilot requires two calibration and two final clusters")
    family_sets = [{item["family"] for item in resolved[group]}
                   for group in ("development", "calibration", "final")]
    if any(left & right for index, left in enumerate(family_sets) for right in family_sets[index + 1:]):
        raise ValueError("native task families must not cross the frozen stage partition")
    if ([item["family"] for item in resolved["source"]]
            != [item["family"] for item in resolved["development"]]):
        raise ValueError("each proposal source must correspond to its frozen development family")
    return resolved


def _make_manifest(tasks: Mapping[str, Sequence[Mapping[str, Any]]], *,
                   controller_commit: str, run_id: str, split_digest: str,
                   pilot_split_digest: str, output: Path, spool: Path) -> dict[str, Any]:
    plan = {name: [dict(row) for row in rows] for name, rows in tasks.items()}
    for name, arms in (("development", DEV_ARMS), ("calibration", CAL_ARMS)):
        for index, task in enumerate(plan[name]):
            task["arm_order"] = list(arms if index == 0 else reversed(arms))
    for index, task in enumerate(plan["final"]):
        task["arm_order"] = (["ADAPTIVE_CDED", "CALIBRATION_STRONGEST_SIMPLE"] if index == 0
                             else ["CALIBRATION_STRONGEST_SIMPLE", "ADAPTIVE_CDED"])
    output_path, spool_path = output.resolve(), spool.resolve()
    if output_path == spool_path or output_path in spool_path.parents or spool_path in output_path.parents:
        raise ValueError("provider spool and study output must be separate directories")
    return {
        "study_id": "SPE-CDED-ITEM-ATTRIBUTION-PILOT-20261102",
        "run_id": run_id, "stage": "skill_evolution", "controller_commit": controller_commit,
        "model": MODEL, "base_url": BASE_URL, "expected_response_identity": list(EXPECTED_IDENTITY),
        "response_identity_binding": "exact_predeclared_identity", "temperature": 0.0,
        "thinking": "enabled", "max_retries": 0, "max_steps": MAX_STEPS,
        "max_calls_per_episode": MAX_EPISODE_CALLS, "max_episode_wall_seconds": 1800,
        "max_usd_per_episode": MAX_EPISODE_USD, "max_usd_per_proposal": MAX_EPISODE_USD,
        "max_input_per_request": MAX_INPUT, "max_completion_per_request": MAX_OUTPUT,
        "max_provider_calls": MAX_CALLS, "max_input_tokens": MAX_CALLS * MAX_INPUT,
        "max_completion_tokens": MAX_CALLS * MAX_OUTPUT, "max_total_usd": MAX_USD,
        "max_wall_seconds": 86400, "max_failed_retries": 0,
        "split_manifest_sha256": split_digest, "pilot_split_sha256": pilot_split_digest,
        "output_dir": str(output_path), "spool_dir": str(spool_path),
        "plan": plan, "arms": list(CAL_ARMS),
        "pricing_sha256": manifest_digest(PRICING),
        "peak_rate_worst_case_usd": str(WORST_CASE_USD),
        "claim_boundary": "descriptive two-family pilot; no powered superiority or ACRE certification",
    }


def _posterior(rows: Sequence[Mapping[str, Any]],
               priors: Mapping[str, float] = PRIORS) -> dict[str, float]:
    grouped: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        outcome = receipt_outcome(row)
        if type(outcome) is not bool:
            raise ValueError("development update requires complete verified binary outcomes")
        grouped[str(row["cluster_id"])][str(row["arm"])] = outcome
    if len(grouped) != 2 or any(set(arms) != set(DEV_ARMS) for arms in grouped.values()):
        raise ValueError("posterior update requires two complete four-arm development clusters")
    log_weights: dict[str, float] = {}
    for hypothesis, probabilities in LIKELIHOODS.items():
        value = math.log(priors[hypothesis])
        for arms in grouped.values():
            for arm, probability in zip(DEV_ARMS, probabilities):
                p = probability if arms[arm] else 1.0 - probability
                value += math.log(p)
        log_weights[hypothesis] = value
    maximum = max(log_weights.values())
    weights = {h: math.exp(value - maximum) for h, value in log_weights.items()}
    total = sum(weights.values())
    return {h: value / total for h, value in weights.items()}


def _fixed_order(rows: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    grouped: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        result = receipt_outcome(row)
        if type(result) is not bool:
            raise ValueError("fixed-order selection requires complete development outcomes")
        grouped[str(row["cluster_id"])][str(row["arm"])] = result
    if len(grouped) != 2:
        raise ValueError("fixed-order selection requires two development clusters")
    for item_arm, action in (("item_a_10", "retain_a"), ("item_b_01", "retain_b")):
        differences = [int(arms[item_arm]) - int(arms["parent_00"]) for arms in grouped.values()]
        if sum(differences) > 0 and all(value >= 0 for value in differences):
            return action, {
                "family_differences": differences,
                "evaluated_order": ["item_a_10", "item_b_01"],
                "selected_item": item_arm,
            }
    return "abstain", {
        "family_differences": None,
        "evaluated_order": ["item_a_10", "item_b_01"],
        "selected_item": None,
    }


def _simple_calibration(rows: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    grouped: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        result = receipt_outcome(row)
        if type(result) is not bool:
            raise ValueError("calibration requires complete verified outcomes")
        grouped[str(row["cluster_id"])][str(row["arm"])] = result
    if len(grouped) != 2 or any(set(arms) != set(CAL_ARMS) for arms in grouped.values()):
        raise ValueError("calibration requires two complete five-arm clusters")
    simple = ("NO_SKILL", "LENGTH", "STATIC_CDED")
    means = {arm: sum(row[arm] for row in grouped.values()) / len(grouped) for arm in simple}
    mean_cost = {
        arm: float(sum((Decimal(_episode_cost(row)["estimated_usd"]) for row in rows
                        if row["arm"] == arm), Decimal(0)) / Decimal(len(grouped)))
        for arm in simple
    }
    baseline = min(simple, key=lambda arm: (-means[arm], mean_cost[arm], simple.index(arm)))
    adaptive_differences = [int(arms["ADAPTIVE_CDED"]) - int(arms[baseline])
                           for arms in grouped.values()]
    eligible = all(value >= 0 for value in adaptive_differences) and sum(adaptive_differences) > 0
    frozen_route = "ADAPTIVE_CDED" if eligible else baseline
    return baseline, {
        "simple_mean_success": means, "simple_mean_estimated_usd": mean_cost,
        "adaptive_family_differences": adaptive_differences,
        "adaptive_gate_passed": eligible, "frozen_route": frozen_route,
    }


def _generic_length_control(target: str) -> str:
    filler = "Be concise. Use only admissible commands. Check the resulting observation. "
    if not target:
        return ""
    return (filler * (len(target) // len(filler) + 1))[:len(target)]


def _length_match(skill: str | None, target: str) -> str | None:
    if skill is None:
        return None
    return skill + _generic_length_control(target[len(skill):])


def _episode_cost(row: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = row.get("trajectory", [])
    return {
        "provider_calls": row.get("provider_calls"),
        "input_tokens": row.get("input_tokens"),
        "completion_tokens": row.get("completion_tokens"),
        "reasoning_tokens": sum(int(step.get("reasoning_tokens") or 0) for step in trajectory),
        "estimated_usd": str(sum((Decimal(str(step["estimated_provider_usd"]))
                                   for step in trajectory if step.get("estimated_provider_usd") is not None),
                                  Decimal(0))),
        "wall_seconds": row.get("wall_seconds"),
        "success": receipt_outcome(row),
    }


def _validate_approval(manifest: Mapping[str, Any], approval: Mapping[str, Any]) -> None:
    """Require the human authorization to match the run's entire frozen envelope."""
    expected = {
        "max_provider_calls": manifest["max_provider_calls"],
        "max_input_tokens": manifest["max_input_tokens"],
        "max_completion_tokens": manifest["max_completion_tokens"],
        "max_total_usd": manifest["max_total_usd"],
        "max_wall_seconds": manifest["max_wall_seconds"],
        "max_failed_retries": manifest["max_failed_retries"],
        "per_episode_max_steps": manifest["max_steps"],
        "per_request_max_completion_tokens": manifest["max_completion_per_request"],
        "max_usd_per_episode": manifest["max_usd_per_episode"],
        "max_calls_per_episode": manifest["max_calls_per_episode"],
        "max_episode_wall_seconds": manifest["max_episode_wall_seconds"],
    }
    if any(approval.get(key) != value for key, value in expected.items()):
        raise ProviderBudgetExceeded("approval_usage_envelope_mismatch")


def _run_live(manifest: Mapping[str, Any], pilot_split: Mapping[str, Any],
              parent_split: Mapping[str, Any], approval: Mapping[str, Any],
              data_root: Path, output: Path) -> dict[str, Any]:
    import inspect
    from experiments.external_real.alfworld_stateful_runner import DeepSeekStatefulClient, run_episode
    from experiments.skill_evolution.native_verifier import make_selected_environment
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if revision != manifest["controller_commit"] or subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("live run requires the exact clean frozen server commit")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("ALFWorld pilot requires the frozen CPU-only provider/runtime contract")
    budget = ApprovedProviderBudget(manifest, approval, PRICING, controller_commit=revision,
                                    stage=manifest["stage"], model=manifest["model"], base_url=manifest["base_url"])
    if manifest.get("pricing_sha256") != manifest_digest(PRICING):
        raise ProviderBudgetExceeded("pricing_snapshot_mismatch")
    budget.ensure_capacity(MAX_CALLS, MAX_CALLS * MAX_INPUT, MAX_CALLS * MAX_OUTPUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_config.json").write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n")
    for name in ("episodes.jsonl", "proposals.jsonl", "provider_journal.jsonl"):
        (output / name).open("x").close()
    journal = DurableProviderJournal(manifest["spool_dir"], run_identity=manifest_digest(manifest))
    journal.restore_budget(budget)
    client = DeepSeekStatefulClient(
        os.environ["DEEPSEEK_API_KEY"], manifest["base_url"], manifest["model"], budget=budget,
        max_completion_tokens=MAX_OUTPUT,
        persist_budget=lambda state: _append(output / "provider_journal.jsonl", state),
    )
    client.max_input_per_request = MAX_INPUT
    client.receipt_journal = journal
    client.response_identity = EXPECTED_IDENTITY
    client.require_response_identity = True
    client.response_ids.update(journal.response_ids)
    runner_source = inspect.getsource(run_episode).encode("utf-8")
    agent_sha = hashlib.sha256(runner_source).hexdigest()
    rows: list[dict[str, Any]] = []
    proposals: dict[str, Any] = {}
    items: dict[str, SemanticItem] = {}
    proposal_costs: list[dict[str, Any]] = []

    def run_one(task: Mapping[str, Any], arm: str, skill: str | None, stage: str) -> dict[str, Any]:
        trial_id = f"{manifest['run_id']}:{stage}:{task['cluster_id']}:{arm}"
        budget.begin_episode(max_usd=MAX_EPISODE_USD, max_calls=MAX_EPISODE_CALLS,
                             max_wall_seconds=manifest["max_episode_wall_seconds"])
        client.set_episode_context(trial_id, arm, None)
        environment = make_selected_environment(task["gamefile"], max_steps=MAX_STEPS, seed=int(task["seed"]))
        try:
            row = run_episode(environment, client, skill=skill, max_steps=MAX_STEPS,
                receipt_metadata={
                    "study_id": manifest["study_id"], "run_id": manifest["run_id"],
                    "trial_id": trial_id, "task_id": task["task_id"], "template_id": task["template_id"],
                    "cluster_id": task["cluster_id"], "task_family": task["family"],
                    "arm": arm, "stage": stage, "split": task["split"],
                    "gamefile_sha256": task["task_file_sha256"], "agent_revision": revision,
                    "benchmark_revision": revision, "agent_revision_sha256": agent_sha,
                    "treatment_sha256": hashlib.sha256(skill.encode("utf-8")).hexdigest() if skill is not None else None,
                    "treatment_characters": len(skill) if skill is not None else 0,
                    "model_request_name": MODEL, "manifest_sha256": manifest_digest(manifest),
                    "split_manifest_sha256": manifest["split_manifest_sha256"],
                    "approval_reference": approval["approval_reference"],
                    "evidence_origin": "REAL_AGENT", "source_task_prior_exposure": task["prior_exposure"],
                }, expected_gamefiles=[task["gamefile"]])
        finally:
            environment.close()
        row["receipt_sha256"] = manifest_digest(row)
        rows.append(row)
        _append(output / "episodes.jsonl", row)
        if receipt_outcome(row) is None:
            raise ProviderBudgetExceeded("terminal_unknown_no_retry_or_replacement")
        return row

    try:
        source_rows = []
        for task in manifest["plan"]["source"]:
            source_rows.append(run_one(task, "NO_SKILL_SOURCE", None, "proposal_source"))
        for index, (task, source_row) in enumerate(zip(manifest["plan"]["source"], source_rows)):
            family = task["family"]
            evidence = source_bundle_from_receipts([source_row], parent_split, source_mode="SUCCESS_ONLY_INDUCTION")
            # ModelSkillProposer's initial path builds a provenance-bearing complete candidate.
            proposer = ModelSkillProposer(client, max_completion_calls=1, max_completion_tokens=MAX_OUTPUT,
                max_skill_characters=1800,
                on_completion=lambda record: _append(output / "proposals.jsonl", record))
            trial_id = f"{manifest['run_id']}:proposal:{family}"
            budget.begin_episode(max_usd=MAX_EPISODE_USD, max_calls=1,
                                 max_wall_seconds=manifest["max_episode_wall_seconds"])
            client.set_episode_context(trial_id, f"PROPOSAL_{index}", None)
            before = {"calls": budget.calls_reserved, "input": budget.input_used,
                      "output": budget.output_used, "usd": budget.usd_used}
            candidate = proposer.propose_initial(
                evidence, skill_id=f"cded_item_{index + 1}",
                subtask="general ALFWorld interaction",
                constraints=(
                    "Propose exactly one concise procedure step as a falsifiable research candidate.",
                    "Use only public observations and actions in the supplied trace.",
                    "Avoid object-instance names, task identifiers, hidden state, or unsupported universal claims.",
                    "This item will be tested on different ALFWorld task families; state its limits clearly.",
                ),
            )
            if len(candidate.procedure) != 1:
                raise ValueError("candidate must contain exactly one procedural step")
            proposal_costs.append({
                "family": family, "provider_calls": budget.calls_reserved - before["calls"],
                "input_tokens": budget.input_used - before["input"],
                "completion_tokens": budget.output_used - before["output"],
                "estimated_usd": str(budget.usd_used - before["usd"]),
            })
            label = "A" if index == 0 else "B"
            proposals[label] = candidate
            items[label] = SemanticItem(
                item_id=f"cded-item-{label.casefold()}", section="procedure", kind="procedure",
                status="recommended", protected=False, semantic_tags=("source-grounded-candidate",),
                evidence=candidate.source_trial_ids, text=candidate.procedure[0],
            )
            _append(output / "proposals.jsonl", {
                "status": "validated_candidate", "family": family, "skill_digest": candidate.digest,
                "source_receipt_sha256": source_row["receipt_sha256"], "semantic_item": items[label].to_dict(),
            })

        skills = {key: value.text for key, value in items.items()}
        dev_target = " " * max(len(skills["A"]), len(skills["B"]))
        dev_skills = {key: value + _generic_length_control(dev_target[len(value):])
                      for key, value in skills.items()}
        dev_length = _generic_length_control(dev_target)
        dev_rows = []
        for task in manifest["plan"]["development"]:
            for arm in task["arm_order"]:
                skill = {"parent_00": None, "item_a_10": dev_skills["A"],
                         "item_b_01": dev_skills["B"], "LENGTH": dev_length}[arm]
                dev_rows.append(run_one(task, arm, skill, "development"))
        posterior = _posterior(dev_rows)
        decision_model = DecisionModel(posterior, LOSSES)
        adaptive_action, risks = choose_evolution_action(decision_model)
        decision_sensitivity = {}
        for neither_prior in (0.25, 0.50, 0.75):
            prior = {"A_ONLY": (1.0 - neither_prior) / 2,
                     "B_ONLY": (1.0 - neither_prior) / 2,
                     "NEITHER": neither_prior}
            alternate = _posterior(dev_rows, prior)
            action, alternate_risks = choose_evolution_action(DecisionModel(alternate, LOSSES))
            decision_sensitivity[str(neither_prior)] = {
                "posterior": alternate, "action": action, "risks": alternate_risks,
            }
        fixed_action, fixed_details = _fixed_order(dev_rows)
        action_text = {"retain_a": skills["A"], "retain_b": skills["B"], "abstain": None}
        adaptive_skill = action_text[adaptive_action]
        fixed_skill = action_text[fixed_action]
        static_skill = skills["A"] + "\n\n" + skills["B"]
        target = static_skill
        length_skill = _generic_length_control(target)
        adaptive_skill = _length_match(adaptive_skill, target)
        fixed_skill = _length_match(fixed_skill, target)
        cal_treatments = {
            "NO_SKILL": None, "LENGTH": length_skill, "STATIC_CDED": static_skill,
            "ADAPTIVE_CDED": adaptive_skill, "FIXED_ORDER_CDED": fixed_skill,
        }
        calibration_rows = []
        for task in manifest["plan"]["calibration"]:
            for arm in task["arm_order"]:
                calibration_rows.append(run_one(task, arm, cal_treatments[arm], "calibration"))
        baseline, calibration = _simple_calibration(calibration_rows)
        frozen = {
            "status": "FINAL_POLICY_FROZEN_BEFORE_FINAL_OUTCOMES",
            "study_id": manifest["study_id"], "run_id": manifest["run_id"],
            "manifest_sha256": manifest_digest(manifest),
            "development_posterior": posterior, "adaptive_action": adaptive_action,
            "decision_sensitivity": decision_sensitivity,
            "fixed_order_action": fixed_action, "calibration": calibration,
            "strongest_simple": baseline,
            "final_arms": ["ADAPTIVE_CDED", "CALIBRATION_STRONGEST_SIMPLE"],
            "final_task_clusters": [task["cluster_id"] for task in manifest["plan"]["final"]],
            "candidate_items": {key: value.to_dict() for key, value in items.items()},
            "treatments": {
                "adaptive_sha256": hashlib.sha256((adaptive_skill or "").encode()).hexdigest(),
                "fixed_order_sha256": hashlib.sha256((fixed_skill or "").encode()).hexdigest(),
                "static_sha256": hashlib.sha256(static_skill.encode()).hexdigest(),
                "length_sha256": hashlib.sha256(length_skill.encode()).hexdigest(),
                "matched_characters": len(target),
            },
        }
        freeze_bytes = (json.dumps(frozen, indent=2, sort_keys=True) + "\n").encode("utf-8")
        freeze_path = output / "DECISION_FREEZE.json"
        with freeze_path.open("xb") as stream:
            stream.write(freeze_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        freeze_sha256 = hashlib.sha256(freeze_bytes).hexdigest()
        final_rows = []
        for task in manifest["plan"]["final"]:
            for arm in task["arm_order"]:
                skill = adaptive_skill if arm == "ADAPTIVE_CDED" else cal_treatments[baseline]
                final_rows.append(run_one(task, arm, skill, "final"))
        final_by_cluster: dict[str, dict[str, bool]] = defaultdict(dict)
        for row in final_rows:
            result = receipt_outcome(row)
            if type(result) is not bool:
                raise ValueError("final comparison requires verified binary outcomes")
            final_by_cluster[str(row["cluster_id"])][str(row["arm"])] = result
        differences = [int(pair["ADAPTIVE_CDED"]) - int(pair["CALIBRATION_STRONGEST_SIMPLE"])
                       for pair in final_by_cluster.values()]
        if any(set(pair) != {"ADAPTIVE_CDED", "CALIBRATION_STRONGEST_SIMPLE"} for pair in final_by_cluster.values()):
            raise ValueError("final pairs are incomplete")
        summary = {
            "status": "COMPLETED_DESCRIPTIVE_PILOT", "results_claimed": True,
            "canonical_promotion": False, "acre_certification": "NOT_EVALUATED",
            "cohort": {key: len(value) for key, value in manifest["plan"].items()},
            "candidate_digests": {key: value.digest for key, value in proposals.items()},
            "decision_freeze_sha256": freeze_sha256,
            "development": {"posterior": posterior, "action": adaptive_action,
                "decision_sensitivity": decision_sensitivity,
                "action_risks": risks, "fixed_order_action": fixed_action,
                "fixed_order_details": fixed_details,
                "outcomes": [{"task_id": row["task_id"], "arm": row["arm"], "success": row["success"]}
                             for row in dev_rows]},
            "calibration": {**calibration, "strongest_simple": baseline,
                "outcomes": [{"task_id": row["task_id"], "arm": row["arm"], "success": row["success"]}
                             for row in calibration_rows]},
            "final": {"families": len(final_by_cluster), "family_differences": differences,
                "mean_paired_success_difference": sum(differences) / len(differences),
                "observed_range_of_family_differences": [min(differences), max(differences)],
                "outcomes": [{"task_id": row["task_id"], "arm": row["arm"], "success": row["success"]}
                             for row in final_rows]},
            "cost_by_stage": {
                stage: {"episodes": sum(row.get("stage") == stage for row in rows),
                        "provider_calls": sum((_episode_cost(row)["provider_calls"] or 0) for row in rows if row.get("stage") == stage),
                        "input_tokens": sum((_episode_cost(row)["input_tokens"] or 0) for row in rows if row.get("stage") == stage),
                        "completion_tokens": sum((_episode_cost(row)["completion_tokens"] or 0) for row in rows if row.get("stage") == stage),
                        "estimated_usd": str(sum((Decimal(_episode_cost(row)["estimated_usd"]) for row in rows
                                                   if row.get("stage") == stage), Decimal(0)))}
                for stage in ("proposal_source", "development", "calibration", "final")
            },
            "proposal_generation_cost": {
                "calls": sum(row["provider_calls"] for row in proposal_costs),
                "input_tokens": sum(row["input_tokens"] for row in proposal_costs),
                "completion_tokens": sum(row["completion_tokens"] for row in proposal_costs),
                "estimated_usd": str(sum((Decimal(row["estimated_usd"]) for row in proposal_costs), Decimal(0))),
                "per_candidate": proposal_costs,
            },
            "provider_budget": budget.as_dict(), "unknown_count": 0,
            "total_provider_usage": {
                "calls_reserved": budget.calls_reserved,
                "input_tokens": budget.input_used,
                "completion_tokens": budget.output_used,
                "estimated_usd": str(budget.usd_used),
                "billing_status": "ESTIMATE_NOT_BILL",
            },
            "claim_boundary": manifest["claim_boundary"],
        }
        (output / "report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary
    except BaseException as exc:
        (output / "report.json").write_text(json.dumps({
            "status": "STOPPED", "stop_reason": type(exc).__name__,
            "completed_episodes": len(rows), "completed_proposals": len(proposals),
            "provider_budget": budget.as_dict(), "results_claimed": False,
            "canonical_promotion": False, "acre_certification": "NOT_EVALUATED",
        }, indent=2, sort_keys=True) + "\n")
        raise
    finally:
        journal.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-split", type=Path, required=True)
    parser.add_argument("--parent-split", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.prepare == args.live:
        parser.error("choose exactly one of --prepare or --live")
    if args.live and args.approval is None:
        parser.error("--live requires the exact prepared manifest and bound approval")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    pilot_split, parent_split = _json(args.pilot_split), _json(args.parent_split)
    tasks = _resolve_tasks(pilot_split, parent_split, args.data_root)
    split_digest = manifest_digest(parent_split)
    pilot_digest = manifest_digest(pilot_split)
    if args.prepare:
        manifest = _make_manifest(tasks, controller_commit=revision, run_id=args.run_id,
            split_digest=split_digest, pilot_split_digest=pilot_digest,
            output=args.output, spool=args.spool)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "PREPARED_NO_PROVIDER_CALLS", "manifest_sha256": manifest_digest(manifest),
            "pilot_split_sha256": pilot_digest, "provider_call_cap": MAX_CALLS,
            "worst_case_usd": str(WORST_CASE_USD), "output": str(args.output)}, indent=2))
        return
    manifest = _json(args.manifest)
    approval = _json(args.approval)
    _validate_approval(manifest, approval)
    if manifest_digest(manifest) != approval.get("approved_manifest_sha256"):
        raise ProviderBudgetExceeded("approval_manifest_digest_mismatch")
    if (manifest.get("pilot_split_sha256") != pilot_digest
            or manifest.get("split_manifest_sha256") != split_digest):
        raise ProviderBudgetExceeded("approved_split_manifest_digest_mismatch")
    if (manifest["controller_commit"] != revision or manifest["run_id"] != args.run_id
            or manifest["output_dir"] != str(args.output.resolve())
            or manifest["spool_dir"] != str(args.spool.resolve())):
        raise ProviderBudgetExceeded("run_identity_mismatch")
    summary = _run_live(manifest, pilot_split, parent_split, approval, args.data_root, args.output)
    print(json.dumps({"status": summary["status"], "provider_budget": summary["provider_budget"]}, indent=2))


if __name__ == "__main__":
    main()
