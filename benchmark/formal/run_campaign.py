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
from benchmark.harness.evolution import promote_via_replay
from scripts.render_skill_view import render_skill_view, validate_skill_view_bundle
from core.models import identifier_digest, validate_identifier


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
    task = miniyaml.load(str(task_dir / "task.yaml"))
    measurement = task.get("measurement", {}) if isinstance(task, dict) else {}
    correctness = task.get("correctness", {}) if isinstance(task, dict) else {}
    public_task = {
        "schema_version": 1,
        "task_id": task.get("task_id", task_dir.name),
        "title": task.get("title", ""),
        "requires_cuda": bool(task.get("requires_cuda", False)),
        "time_budget_s": int(task.get("time_budget_s", 0)),
        "workspace": dict(task.get("workspace", {})),
        "allowed_files": [str(task.get("workspace", {}).get("entrypoint", "solution.py"))],
        "scientific_gates": list(task.get("scientific_gates", [])),
        "scientific_contract": "scientific_contract.py",
        "measurement_contract": {
            "primary_metric": measurement.get("primary_metric"),
            "higher_is_better": measurement.get("higher_is_better"),
            "warmup_iterations": measurement.get("warmup_iterations"),
            "measured_iterations": measurement.get("measured_iterations"),
            "repetitions": measurement.get("repetitions"),
        },
        "correctness_contract": correctness,
        # Only fields explicitly declared public may drive retrieval/routing.
        "routing_context": dict(task.get("public_context", {})) if isinstance(task.get("public_context", {}), dict) else {},
    }
    (destination / "public_task.json").write_text(
        json.dumps(public_task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    source = task_dir / "scientific_contract.py"
    if source.is_file():
        shutil.copy2(source, destination / "scientific_contract.py")
    for name in ("workspace", "public_tests"):
        source = task_dir / name
        if source.is_dir():
            shutil.copytree(source, destination / name, dirs_exist_ok=True)


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


def _run_isolated_agent(
    command_template: str,
    executor_command: str,
    env: dict[str, str],
    worker_root: Path,
    timeout: float,
) -> dict[str, Any]:
    """Run through an externally supplied namespace/container executor.

    The executor owns the filesystem boundary.  The campaign never exposes the
    condition store, verifier, or benchmark root as worker paths.
    """
    executor = executor_command.format(
        agent_command=shlex.quote(command_template),
        worker_root=shlex.quote(str(worker_root)),
        task_dir=shlex.quote(env["SPE_TASK_DIR"]),
        solution_dir=shlex.quote(env["SPE_SOLUTION_DIR"]),
        retrieved_context=shlex.quote(env["SPE_RETRIEVED_CONTEXT"]),
        skill_view=shlex.quote(env.get("SPE_SKILL_VIEW_DIR", "")),
        executor_receipt=shlex.quote(env["SPE_EXECUTOR_RECEIPT_PATH"]),
    )
    return _run_agent(executor, env, worker_root, timeout)


def _prepare_worker_root(
    trial_dir: Path,
    agent_task_dir: Path,
    solution_dir: Path,
    skill_view: Path | None,
) -> Path:
    """Create the only host directory visible to an external executor."""
    worker_root = trial_dir / "worker"
    if worker_root.exists():
        shutil.rmtree(worker_root)
    worker_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(agent_task_dir, worker_root / "task")
    shutil.copytree(solution_dir, worker_root / "solution")
    if skill_view is not None:
        shutil.copytree(skill_view, worker_root / "skill_view")
    return worker_root


def _read_executor_receipt(path: Path, skill_digest: str | None) -> tuple[dict[str, Any], list[str]]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, ["external executor receipt missing or invalid"]
    if not isinstance(receipt, dict):
        return {}, ["external executor receipt must be an object"]
    errors: list[str] = []
    required = ("mode", "network_mode", "mount_allowlist", "executor_digest", "worker_uid", "usage")
    errors.extend(f"executor receipt missing {key}" for key in required if key not in receipt)
    if receipt.get("mode") != "external_namespace_executor":
        errors.append("executor receipt mode mismatch")
    if receipt.get("network_mode") != "none":
        errors.append("external executor must declare network_mode=none")
    if not isinstance(receipt.get("mount_allowlist"), list) or not receipt.get("mount_allowlist"):
        errors.append("executor receipt mount_allowlist must be non-empty")
    for key in ("executor_digest", "worker_uid"):
        if not isinstance(receipt.get(key), str) or not receipt.get(key):
            errors.append(f"executor receipt {key} must be non-empty")
    if skill_digest is not None and receipt.get("skill_view_digest") != skill_digest:
        errors.append("executor receipt skill_view_digest mismatch")
    usage = receipt.get("usage")
    if not isinstance(usage, dict):
        errors.append("executor receipt usage must be an object")
    return receipt, errors


def _read_agent_extensions(path: Path) -> dict[str, Any]:
    """Read only worker-supplied lesson/proposal extensions before verification.

    The verifier remains authoritative for scores, gates, and verdicts.  A
    worker may contribute a typed proposal, but cannot author evidence,
    replay cases, confidence, or any scored field.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    extensions: dict[str, Any] = {}
    for key in ("lesson", "acre_proposals"):
        if key in payload:
            extensions[key] = payload[key]
    if not isinstance(extensions.get("lesson", {}), dict):
        extensions["lesson"] = {}
    proposals = extensions.get("acre_proposals", [])
    if not isinstance(proposals, list):
        extensions["acre_proposals"] = []
    else:
        clean: list[dict[str, Any]] = []
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            if any(key in proposal for key in ("cases", "confidence", "promotion", "p_min", "delta", "epsilon", "effect", "scientific_gates")):
                continue
            identifier = proposal.get("rule_id") or proposal.get("relation_id") or proposal.get("id")
            if not isinstance(identifier, str):
                continue
            try:
                validate_identifier(identifier, "proposal_id")
            except ValueError:
                continue
            clean.append(dict(proposal))
        extensions["acre_proposals"] = clean
    return extensions


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
    experience_id = f"EXP-{task_id}"
    added_experience_ids: list[str] = []
    added_replay_case_ids: list[str] = []
    maintenance_decisions: list[dict[str, Any]] = []
    promoted_rule_ids: list[str] = []
    if condition in {"C", "C_STRESS", "D"}:
        experience = {
            "schema_version": 1,
            "record_type": "task_experience",
            "id": experience_id,
            "experience_id": experience_id,
            "task_id": task_id,
            "condition": condition,
            "context_mode": context_mode,
            "observation": {
                "verdict": result.get("verdict"),
                "task_score": scored.get("task_score"),
                "scientific_gates": result.get("scientific_gates", {}),
                "measurement": result.get("measurement", {}),
            },
            "lesson": result.get("lesson", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        inbox = store / "experience" / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        validate_identifier(experience_id, "experience_id")
        experience_path = inbox / f"{identifier_digest(experience_id)}.json"
        if not experience_path.exists():
            experience_path.write_text(json.dumps(experience, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            added_experience_ids.append(experience_id)

        if condition == "D":
            # The worker cannot author evidence.  The harness derives paired
            # case records from the verifier-owned outcome below.
            evidence_dir = store / "experience" / "cases"
            evidence_dir.mkdir(parents=True, exist_ok=True)
            case_id = f"CASE-{task_id}"
            validate_identifier(case_id, "case_id")
            case = {
                "schema_version": 1,
                "record_type": "paired_replay_case",
                "case_id": case_id,
                "utility_on": float(scored.get("task_score", 0.0)),
                "utility_off": 0.0,
                "scientific_ok": bool(scored.get("gates_passed", False)),
                "quality_ok": bool(scored.get("gates_passed", False)),
                "source_id": f"verifier-{task_id}",
                "independence_group": f"task-{task_id}",
                "context": {"task_id": task_id, "context_mode": context_mode},
            }
            evidence_path = evidence_dir / f"{identifier_digest(case_id)}.json"
            if not evidence_path.exists():
                evidence_path.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            added_replay_case_ids.append(case_id)

    if condition == "D" and allow_maintenance:
        valid, policy_errors = conditions.verify_condition_policy(store)
        if not valid:
            raise ValueError("governed transition failed store policy: " + "; ".join(policy_errors))
        from dataclasses import asdict
        from core.acre.engine import AcreEngine
        engine = AcreEngine.from_store(store)
        for subject_id in (*engine.rule_states, *engine.relation_states):
            maintenance_decisions.append(asdict(engine.evolve(subject_id)))
        active_ledger = ledger or EvolutionDecisionLedger()
        candidates_dir = store / "evolution" / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        candidates_by_id: dict[str, dict[str, Any]] = {}
        for candidate in result.get("acre_proposals", []):
            if not isinstance(candidate, dict):
                continue
            if candidate.get("relation_id"):
                # Formal task outcomes cannot establish factorial contrasts;
                # relation proposals stay out of the node replay path.
                continue
            identifier = str(candidate.get("relation_id") or candidate.get("rule_id") or candidate.get("id") or "")
            if not identifier:
                continue
            candidate = dict(candidate)
            validate_identifier(identifier, "candidate_id")
            candidate.setdefault("status", "candidate")
            candidate["cases"] = list(added_replay_case_ids)
            path = candidates_dir / f"{identifier_digest(identifier)}.json"
            path.write_text(json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            candidates_by_id[identifier] = candidate
        for path in sorted(candidates_dir.glob("*.json")):
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if candidate.get("cases") and candidate.get("status") == "candidate":
                identifier = str(candidate.get("relation_id") or candidate.get("rule_id") or candidate.get("id") or path.stem)
                candidates_by_id[identifier] = candidate
        candidates = list(candidates_by_id.values())
        for candidate in candidates:
            hydrated_cases: list[dict[str, Any]] = []
            for case_id in candidate.get("cases", []):
                try:
                    case_path = store / "experience" / "cases" / f"{identifier_digest(str(case_id))}.json"
                    case_value = json.loads(case_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
                if isinstance(case_value, dict):
                    hydrated_cases.append(case_value)
            candidate["cases"] = hydrated_cases
        if candidates:
            promoted_rule_ids = promote_via_replay(store, candidates, core_repo, out_dir, active_ledger)
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
        "added_replay_case_ids": added_replay_case_ids,
        "maintenance_decisions": maintenance_decisions,
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
        "worker_isolation": {
            "mode": "external_namespace_executor",
            "network_mode": "none",
            "mount_allowlist": ["task", "solution", "skill_view", "retrieved_context", "result", "executor_receipt"],
            "executor_receipt": None,
        },
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
        public_task = json.loads((agent_task_dir / "public_task.json").read_text(encoding="utf-8"))
        public_routing = public_task.get("routing_context", {}) if isinstance(public_task, dict) else {}
        if not isinstance(public_routing, dict):
            public_routing = {}
        retrieved_context_path = agent_task_dir / "retrieved_context.json"
        if str(item["condition"]) in {"C", "C_STRESS", "D"}:
            adapter = FormalConditionAdapter(str(item["condition"]), store, token_budget=budgets.tokens)
            retrieval_input = {
                "domain": public_routing.get("domain", "scientific-performance"),
                "workload": dict(public_routing.get("workload", {})),
                "hardware": dict(public_routing.get("hardware", {})),
                "software": dict(public_routing.get("software", {})),
                "evidence": dict(public_routing.get("evidence", {})),
                "token_budget": budgets.tokens,
            }
            retrieved_context = adapter.retrieved_context(retrieval_input)
            exposed_context = retrieved_context.get("context", {})
            for key in ("domain", "workload", "hardware", "software", "evidence"):
                if exposed_context.get(key) != retrieval_input.get(key):
                    raise ValueError(f"retrieval context escaped public task context at {key}")
        else:
            retrieved_context = {
                "schema_version": 1,
                "condition": str(item["condition"]),
                "context": {"task_id": task_id, "context_mode": str(item["context_mode"])},
                "proposed_interventions": [],
            }
        retrieved_context_path.write_text(json.dumps(retrieved_context, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        worker_root = _prepare_worker_root(trial_dir, agent_task_dir, solution_dir, None if item["condition"] == "A" else skill_view)
        worker_task_dir = worker_root / "task"
        worker_solution_dir = worker_root / "solution"
        worker_retrieved_context_path = worker_task_dir / "retrieved_context.json"
        worker_retrieved_context_path.write_text(retrieved_context_path.read_text(encoding="utf-8"), encoding="utf-8")
        worker_result_path = worker_root / "worker_result.json"
        receipt_path = worker_root / "executor_receipt.json"
        env = {
            "SPE_TASK_ID": task_id,
            "SPE_TASK_DIR": str(worker_task_dir),
            "SPE_SOLUTION_DIR": str(worker_solution_dir),
            "SPE_CONDITION": str(item["condition"]),
            "SPE_CONTEXT_MODE": str(item["context_mode"]),
            "SPE_RETRIEVED_CONTEXT": str(worker_retrieved_context_path),
            "SPE_RESULT_PATH": str(worker_result_path),
            "SPE_AGENT_USAGE_PATH": str(worker_root / "agent_usage.json"),
            "SPE_EXECUTOR_RECEIPT_PATH": str(receipt_path),
            "SPE_SKILL_VIEW_DIR": str(worker_root / "skill_view") if item["condition"] != "A" else "",
            "SPE_BUDGET_JSON": json.dumps(budgets.as_dict(), sort_keys=True),
            "SPE_OUTER_TRIAL_ID": str(item["outer_trial_id"]),
        }
        if not getattr(args, "executor_command", None):
            raise ValueError("formal agent runs require --executor-command with a namespace/container executor")
        agent = _run_isolated_agent(args.agent_command, args.executor_command, env, worker_root, budgets.wall_time_s)
        agent_extensions = _read_agent_extensions(worker_result_path)
        receipt, receipt_errors = _read_executor_receipt(receipt_path, None if item["condition"] == "A" else skill_digest)
        usage = receipt.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        budget_errors = budgets.validate_usage(usage)
        budget_errors.extend(receipt_errors)
        manifest["worker_isolation"]["executor_receipt"] = {
            key: receipt.get(key)
            for key in ("mode", "network_mode", "mount_allowlist", "executor_digest", "worker_uid", "skill_view_digest")
            if key in receipt
        }
        attest.write_experiment(trial_dir / "experiment.json", manifest)
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
        result.update(agent_extensions)
        result["condition_adapter"] = retrieved_context
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
    parser.add_argument(
        "--executor-command",
        default=None,
        help="namespace/container executor template; receives {agent_command}, {worker_root}, {task_dir}, {solution_dir}, {retrieved_context}, {skill_view}, {executor_receipt}",
    )
    parser.add_argument("--claim-results", action="store_true", help="claim only if the formal calibration gate is passed")
    args = parser.parse_args()
    result = run_campaign(args)
    print(json.dumps({"status": result["status"], "schedule_size": result["schedule_size"], "results_claimed": result["results_claimed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
