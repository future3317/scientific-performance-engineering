#!/usr/bin/env python3
"""Run or dry-run a complete SPE-EvoBench formal campaign.

Without ``--agent-command`` this writes only a frozen campaign plan and never
claims benchmark results. With a command, the driver gives the agent a fresh
solution workspace for each task and then invokes the immutable verifier.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.harness import conditions, miniyaml, scoring, verifier
from benchmark.harness.evolution_ledger import EvolutionDecisionLedger
from benchmark.harness.fingerprint import capture_fingerprint
from benchmark.formal import aggregate, attest, budget, schedule
from benchmark.formal.condition_adapter import FormalConditionAdapter
from scripts.render_skill_view import render_skill_view, validate_skill_view_bundle


def _copy_workspace(task_dir: Path, destination: Path) -> None:
    source = task_dir / "workspace"
    if not source.is_dir():
        raise FileNotFoundError(f"workspace missing: {source}")
    shutil.copytree(source, destination, dirs_exist_ok=True)


def materialize_agent_task(task_dir: Path, destination: Path) -> None:
    """Create the public task view used by the external worker.

    Oracle and verifier code are harness-only.  The worker receives the task
    contract, workspace, and public tests, never the task package root.
    """
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("task.yaml", "metadata.json", "scientific_contract.py"):
        source = task_dir / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    for name in ("workspace", "public_tests"):
        source = task_dir / name
        if source.is_dir():
            shutil.copytree(source, destination / name, dirs_exist_ok=True)


def _materialize_read_only_view(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    for entry in destination.rglob("*"):
        if entry.is_file():
            entry.chmod(stat.S_IRUSR | stat.S_IRGRP)
        elif entry.is_dir():
            entry.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)


def _run_agent(command_template: str, env: dict[str, str], cwd: Path, timeout: float) -> dict[str, Any]:
    command = command_template.format(**env)
    try:
        inherited = {key: value for key, value in os.environ.items() if not key.startswith("SPE_") and key not in {"PYTHONPATH", "PYTHONHOME"}}
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            env={**inherited, **env},
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + "timeout",
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def post_task_update(
    *,
    condition: str,
    store: Path,
    task_id: str,
    result: dict[str, Any],
    scored: dict[str, Any],
    core_repo: Path,
    out_dir: Path,
    context_mode: str = "reset",
    ledger: EvolutionDecisionLedger | None = None,
    allow_maintenance: bool = True,
) -> dict[str, Any]:
    """Run the explicit execute -> evidence -> maintenance -> attest transition."""
    condition = condition.upper()
    pre_digest = conditions.store_digest(store)
    evidence_id = f"EXP-{task_id}"
    added_experience_ids: list[str] = []
    promoted_rule_ids: list[str] = []
    if condition in {"C", "C_STRESS", "D"}:
        evidence = {
            "schema_version": 2,
            "id": evidence_id,
            "event_id": evidence_id,
            "task_id": task_id,
            "condition": condition,
            "context_mode": context_mode,
            "context": {"task_id": task_id, "condition": condition, "context_mode": context_mode, "rule_versions": {}},
            "assignment": {"interventions": {task_id: 1}, "propensity": 0.5, "design_id": "formal-task-v2"},
            "evidence_stream": "representative",
            "query_id": evidence_id,
            "outcome_vector": {"task_score": float(scored.get("task_score", 0.0) or 0.0)},
            "artifacts": {},
            "versions": {},
            "source_id": evidence_id,
            "independence_group": task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trust_zone": "local",
            "attacker_controlled_fields": [],
            "intervention": {"action": task_id},
            "verdict": result.get("verdict"),
            "task_score": scored.get("task_score"),
            "scientific_gates": result.get("scientific_gates", {}),
            "measurement": result.get("measurement", {}),
        }
        inbox = store / "experience" / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        evidence_path = inbox / f"{evidence_id}.json"
        if not evidence_path.exists():
            evidence_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            added_experience_ids.append(evidence_id)

    if condition == "D" and allow_maintenance:
        valid, policy_errors = conditions.verify_condition_policy(store)
        if not valid:
            raise ValueError("governed transition failed store policy: " + "; ".join(policy_errors))
        active_ledger = ledger or EvolutionDecisionLedger()
        candidates_dir = store / "evolution" / "candidates"
        candidates = []
        for path in sorted(candidates_dir.glob("*.json")):
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if candidate.get("cases") and candidate.get("status") == "candidate":
                candidates.append(candidate)
        if candidates:
            promoted_rule_ids = _promote_via_replay(store, candidates, core_repo, out_dir, active_ledger)
        from scripts.validate_evolution import audit

        errors = audit(store, schema_root=core_repo)
        if errors:
            raise ValueError("D maintenance transition failed validation: " + "; ".join(errors))
        transition = "governed_maintenance"
    elif condition == "D":
        valid, policy_errors = conditions.verify_condition_policy(store)
        if not valid:
            raise ValueError("governed transition failed store policy: " + "; ".join(policy_errors))
        from scripts.validate_evolution import audit

        errors = audit(store, schema_root=core_repo)
        if errors:
            raise ValueError("D maintenance transition failed validation: " + "; ".join(errors))
        transition = "maintenance_blocked"
    elif condition in {"C", "C_STRESS"}:
        valid, errors = conditions.verify_condition_policy(store)
        if not valid:
            raise ValueError("raw-experience transition failed policy validation: " + "; ".join(errors))
        transition = "raw_experience_capture"
    else:
        transition = "no_op"

    if condition in {"C", "C_STRESS", "D"}:
        conditions.refresh_attestation(store)
    post_digest = conditions.store_digest(store)
    return {
        "status": transition,
        "pre_store_digest": pre_digest,
        "post_store_digest": post_digest,
        "added_experience_ids": added_experience_ids,
        "promoted_rule_ids": promoted_rule_ids,
    }


def _experiment_manifest(
    *,
    repo_root: Path,
    tasks_root: Path,
    skill_digest: str,
    task_digest: str,
    task_ids: list[str],
    item: dict[str, Any],
    model_id: str,
    agent_config: dict[str, Any],
    budgets: budget.Budget,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    task_spec = miniyaml.load(str(tasks_root / item["task_id"] / "task.yaml"))
    lineage = task_spec.get("lineage", {}) if isinstance(task_spec, dict) else {}
    return {
        "schema_version": 1,
        "experiment_id": f"SPE-EvoBench-v1.0-20-{item['stream_id']}-{item['task_id']}",
        "benchmark_revision": attest.benchmark_revision(repo_root),
        "skill_view_digest": skill_digest,
        "task_manifest_digest": task_digest,
        "agent_model_id": model_id,
        "agent_config": agent_config,
        "condition": item["condition"],
        "context_mode": item["context_mode"],
        "task_order": task_ids,
        "family_id": task_spec.get("family_id", task_spec.get("family", "unknown")),
        "anchor_instance_id": task_spec.get("anchor_instance_id", item["task_id"]),
        "family_instance_digest": task_spec.get("family_instance_digest"),
        "lineage_id": lineage.get("mutation_template_id", item["task_id"]),
        "outer_trial_id": item["outer_trial_id"],
        "budgets": budgets.as_dict(),
        "hardware_fingerprint": fingerprint,
        "software_fingerprint": {
            "python_version": fingerprint.get("python_version"),
            "platform": fingerprint.get("platform"),
            "torch_geometric_version": fingerprint.get("torch_geometric_version"),
        },
        "torch_version": fingerprint.get("torch_version"),
        "cuda_version": fingerprint.get("cuda_version"),
    }


def _formal_claim_gate(campaign: dict[str, Any], records: list[dict[str, Any]], report_path: Path) -> bool:
    """Allow a claim only after the frozen schedule and calibration gate pass."""
    if campaign.get("status") != "complete" or not records:
        return False
    if any(record.get("validity") != "valid" for record in records):
        return False
    expected = int(campaign.get("schedule_size", 0))
    if len(records) != expected:
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    calibration = report.get("empirical_calibration", {})
    return calibration.get("calibration_gate") == "passed"


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    tasks_root = Path(args.tasks_root).resolve()
    split_path = Path(args.split).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    task_items = schedule.task_order(split_path)
    task_ids = [task_id for _, task_id in task_items]
    if args.skill_view:
        skill_view = Path(args.skill_view).resolve()
        if not (skill_view / "skill_view_manifest.json").is_file():
            raise ValueError("--skill-view must point to a rendered skill bundle")
        if validate_skill_view_bundle(skill_view):
            raise ValueError("--skill-view failed bundle validation")
    else:
        skill_view = out_dir / "skill-view"
        render_skill_view(args.skill_source, skill_view)
    skill_digest = attest.skill_view_digest(skill_view)
    task_digest = attest.task_manifest_digest(tasks_root, task_ids)
    conditions_list = tuple(item.strip().upper() for item in args.conditions.split(",") if item.strip())
    context_modes = tuple(item.strip() for item in args.context_modes.split(",") if item.strip())
    plan = schedule.build_schedule(
        split_path,
        conditions=conditions_list,
        context_modes=context_modes,
        outer_trials=args.outer_trials,
    )
    budgets = budget.parse_budget(json.loads(args.budgets) if args.budgets else None)
    fingerprint = capture_fingerprint()
    campaign = {
        "schema_version": 1,
        "population_id": "SPE-EvoBench-v1.0-20-pilot",
        "status": "planned" if not args.agent_command else "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_revision": attest.benchmark_revision(repo_root),
        "skill_view_digest": skill_digest,
        "task_manifest_digest": task_digest,
        "conditions": list(conditions_list),
        "context_modes": list(context_modes),
        "outer_trials": args.outer_trials,
        "task_order": task_ids,
        "budgets": budgets.as_dict(),
        "agent_model_id": args.model_id,
        "agent_config": json.loads(args.agent_config),
        "schedule_size": len(plan),
        "results_claimed": False,
    }
    (out_dir / "campaign.json").write_text(json.dumps(campaign, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.agent_command:
        (out_dir / "schedule.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return campaign

    records: list[dict[str, Any]] = []
    stores: dict[str, Path] = {}
    context_paths: dict[str, Path] = {}
    ledgers: dict[str, EvolutionDecisionLedger] = {}
    for item in plan:
        stream_id = str(item["stream_id"])
        task_id = str(item["task_id"])
        task_spec = miniyaml.load(str(tasks_root / task_id / "task.yaml"))
        trial_dir = out_dir / "trials" / stream_id / task_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        manifest = _experiment_manifest(
            repo_root=repo_root,
            tasks_root=tasks_root,
            skill_digest=skill_digest,
            task_digest=task_digest,
            task_ids=task_ids,
            item=item,
            model_id=args.model_id,
            agent_config=json.loads(args.agent_config),
            budgets=budgets,
            fingerprint=fingerprint,
        )
        attest.write_experiment(trial_dir / "experiment.json", manifest)
        state_path = context_paths.setdefault(stream_id, trial_dir.parent / "context.json")
        if item["context_mode"] == "reset" and state_path.exists():
            state_path.unlink()
        store = stores.get(stream_id)
        if store is None:
            store = trial_dir.parent / "condition-store"
            conditions.materialize_condition(item["condition"], skill_view if item["condition"] != "A" else None, store, item["context_mode"])
            stores[stream_id] = store
        solution_dir = trial_dir / "solution"
        _copy_workspace(tasks_root / task_id, solution_dir)
        agent_task_dir = trial_dir / "agent-task"
        materialize_agent_task(tasks_root / task_id, agent_task_dir)
        condition_view = trial_dir / "condition-view"
        _materialize_read_only_view(store, condition_view)
        env = {
            "SPE_TASK_ID": task_id,
            "SPE_TASK_DIR": str(agent_task_dir),
            "SPE_SOLUTION_DIR": str(solution_dir),
            "SPE_CONDITION": str(item["condition"]),
            "SPE_CONTEXT_MODE": str(item["context_mode"]),
            "SPE_CONDITION_STORE": str(condition_view),
            "SPE_CONTEXT_STATE": str(condition_view / "context.json"),
            "SPE_RESULT_PATH": str(trial_dir / "result.json"),
            "SPE_AGENT_USAGE_PATH": str(trial_dir / "agent_usage.json"),
            "SPE_BUDGET_JSON": json.dumps(budgets.as_dict(), sort_keys=True),
            "SPE_OUTER_TRIAL_ID": str(item["outer_trial_id"]),
        }
        agent = _run_agent(args.agent_command, env, solution_dir, budgets.wall_time_s)
        usage_path = trial_dir / "agent_usage.json"
        try:
            usage = json.loads(usage_path.read_text(encoding="utf-8")) if usage_path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            usage = {}
        budget_errors = budgets.validate_usage(usage)
        if agent["returncode"] != 0:
            budget_errors.append(f"agent command failed with return code {agent['returncode']}")
        result = verifier.verify_task(
            tasks_root / task_id,
            solution_dir,
            out_path=trial_dir / "result.json",
            condition=str(item["condition"]),
            context_mode=str(item["context_mode"]),
            seed=int(item["outer_trial_index"]),
        )
        if str(item["condition"]) in {"C", "C_STRESS", "D"}:
            adapter = FormalConditionAdapter(str(item["condition"]), store, token_budget=budgets.tokens)
            result["condition_adapter"] = {
                "kind": "raw_experience_retrieval" if str(item["condition"]) in {"C", "C_STRESS"} else "governed_acre_routing",
                "token_budget": budgets.tokens,
                "proposed_interventions": adapter.propose_interventions({"mechanism": task_spec.get("mechanism", ""), "family_id": task_spec.get("family_id", task_spec.get("family", ""))}),
            }
        result.setdefault("cost", {}).update({
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "tool_calls": usage.get("tool_calls"),
        })
        scored = scoring.score_task(result)
        transition = post_task_update(
            condition=str(item["condition"]),
            store=store,
            task_id=task_id,
            result=result,
            scored=scored,
            core_repo=repo_root,
            out_dir=trial_dir,
            context_mode=str(item["context_mode"]),
            ledger=ledgers.setdefault(stream_id, EvolutionDecisionLedger()),
            allow_maintenance=not budget_errors,
        )
        attestation_ok, attestation_errors = conditions.verify_attestation(store)
        if not attestation_ok:
            budget_errors.extend(f"condition attestation failed: {error}" for error in attestation_errors)
        record = {
            "experiment": manifest,
            "task_id": task_id,
            "family": task_spec.get("family"),
            "family_id": task_spec.get("family_id", task_spec.get("family")),
            "anchor_instance_id": task_spec.get("anchor_instance_id", task_id),
            "family_instance_digest": task_spec.get("family_instance_digest"),
            "lineage_id": task_spec.get("lineage", {}).get("mutation_template_id", task_id),
            "kind": task_spec.get("kind"),
            "condition": item["condition"],
            "context_mode": item["context_mode"],
            "outer_trial_id": item["outer_trial_id"],
            "phase": item["phase"],
            "agent": agent,
            "agent_usage": usage,
            "budget_errors": budget_errors,
            "attestation_ok": attestation_ok,
            "validity": "invalid" if budget_errors else "valid",
            "transition": transition,
            "score": scored,
        }
        (trial_dir / "trial.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        records.append(record)
    campaign["status"] = "complete"
    campaign["results_claimed"] = bool(
        getattr(args, "claim_results", False)
        and _formal_claim_gate(campaign, records, repo_root / "benchmark" / "population_report.json")
    )
    campaign["aggregate"] = aggregate.aggregate_trials(records)
    (out_dir / "campaign.json").write_text(json.dumps(campaign, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--tasks-root", type=Path, default=root / "benchmark" / "tasks")
    parser.add_argument("--split", type=Path, default=root / "benchmark" / "split" / "sequential.yaml")
    parser.add_argument("--skill-source", type=Path, default=root)
    parser.add_argument("--skill-view", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conditions", default="A,B,C,D")
    parser.add_argument("--context-modes", default="reset")
    parser.add_argument("--outer-trials", type=int, default=3)
    parser.add_argument("--model-id", default="unidentified-agent")
    parser.add_argument("--agent-config", default="{}")
    parser.add_argument("--budgets", default=None, help='JSON, e.g. {"tokens":12000,"tool_calls":80,"wall_time_s":900}')
    parser.add_argument("--agent-command", default=None, help="shell template; receives SPE_* environment variables")
    parser.add_argument("--claim-results", action="store_true", help="claim only if the formal calibration gate is passed")
    args = parser.parse_args()
    result = run_campaign(args)
    print(json.dumps({"status": result["status"], "schedule_size": result["schedule_size"], "results_claimed": result["results_claimed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
