"""Approve and run the existing train-only ALFWorld readiness path.

Dry-run is the default. Neither a manifest nor this program grants spending
permission. The operator must supply a separately approved scope and prices.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from core.acre.resource_budget import ApprovedProviderBudget, ProviderBudgetExceeded, manifest_digest
from .readiness import plan_from_split


def viability_gate(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    """Exploratory B gate; no inference about Skill or Agent superiority."""
    eligible = [row for row in receipts if row.get("evidence_origin") == "REAL_AGENT"
                and row.get("cluster_id") and row.get("input_tokens") is not None
                and row.get("completion_tokens") is not None
                and not (row.get("provider_budget") or {}).get("tokens_unknown", True)]
    wins = {row["cluster_id"] for row in eligible
            if row.get("status") == "VERIFIED_SUCCESS" and row.get("environment_won") is True
            and row.get("environment_done") is True}
    failures = {row["cluster_id"] for row in eligible
                if row.get("status") in {"VERIFIED_FAILURE", "AGENT_POLICY_FAILURE"}
                and row.get("environment_won") is False and row.get("success") is False}
    qualified = len(wins) >= 2 and bool(failures - wins)
    return {"status": "B-VIABLE-TRAIN-ONLY" if qualified else
                      "VIABLE_BUT_NOT_QUALIFIED" if wins else "B-INSUFFICIENT",
            "independent_win_clusters": len(wins), "independent_failure_clusters": len(failures),
            "action": "REQUEST_C_APPROVAL" if qualified else "STOP_PAID_SKILL_OPTIMIZATION",
            "claim": "train-side engineering gate, not efficacy",
            "agent_viability": "VERIFIED_TRAIN_VIABLE" if len(wins) >= 2 else "INCONCLUSIVE",
            "skill_source_readiness": ("CONTRASTIVE_READY" if wins and failures - wins else
                                       "SUCCESS_ONLY_READY" if wins else "NONE"),
            "skill_efficacy": "NOT_RUN",
            "provider_authorization": "SEPARATE_STAGE_APPROVAL_REQUIRED"}


def diagnostic_manifest(sample: dict[str, Any], parent: dict[str, Any], extension: dict[str, Any],
                        *, controller_commit: str, run_id: str, output_dir: str,
                        gamefile_digests: dict[str, str], episode_usd_cap: str = "1.25", stage_usd_cap: str = "15") -> dict[str, Any]:
    """Freeze T1's six-family cohort, not a new Agent or benchmark framework."""
    episodes = [dict(row, gamefile_sha256=gamefile_digests[row["task_id"]]) for row in sample["selected"]]
    if len(episodes) != 12:
        raise ValueError("T1 requires twelve preselected independent clusters")
    plan = {"status": "DRY_RUN_NOT_EXECUTED", "provider_calls": 0,
            "config": {"max_steps": 60}, "selection_seed": sample["seed"], "episodes": episodes}
    manifest = readiness_manifest(plan, controller_commit=controller_commit, run_id=run_id,
                                  max_input_per_request=32768, max_completion_per_request=8192,
                                  split_manifest=parent, live_output_dir=output_dir)
    manifest.update(study_id="phase3_baseline_diagnostic", stage="baseline_diagnostic", split="train_dev",
                    split_extension=extension, split_extension_sha256=manifest_digest(extension),
                    sampling_manifest=sample, sampling_manifest_sha256=manifest_digest(sample),
                    max_usd_per_episode=episode_usd_cap, max_total_usd=stage_usd_cap,
                    max_calls_per_episode=60, max_episode_wall_seconds=3600, max_wall_seconds=43200,
                    task_types=[1, 2, 3, 4, 5, 6], agent_condition="BASE_AGENT_V2_TELEMETRY",
                    reason_for_new_identity="Response-ID telemetry/dedup and selected-only environment loading; no prompt change; do not pool with B.",
                    reasoning_effort="NOT_EXPLICITLY_SENT", temperature=0.0,
                    approval_status="APPROVAL_REQUIRED",
                    stop_rules={"all_success": "SUCCESS_ONLY_READY_NO_AUTOMATIC_EXPANSION",
                                "success_and_failure": "CONTRASTIVE_READY_REQUEST_T2_APPROVAL",
                                "all_failure": "STOP_PAID_SKILL_OPTIMIZATION",
                                "unknown_or_integrity_failure": "STOP_INTEGRITY_NO_RETRY",
                                "task_replacement": "NONE; unavailable task stops study; no post-outcome replacement"})
    return manifest


def _validate_diagnostic(manifest: dict[str, Any], plan: dict[str, Any], split: dict[str, Any],
                         approval: dict[str, Any]) -> None:
    from collections import Counter
    from experiments.external_real.alfworld_runner import TASK_TYPE_NAMES
    if manifest.get("split") != "train_dev" or len(plan["episodes"]) != 12:
        raise ProviderBudgetExceeded("unsupported_diagnostic_scope")
    if (manifest.get("max_steps") != 60 or manifest.get("max_calls_per_episode") != 60
            or manifest.get("max_input_per_request") != 32768
            or manifest.get("max_completion_per_request") != 8192
            or manifest.get("reasoning_effort") != "NOT_EXPLICITLY_SENT"):
        raise ProviderBudgetExceeded("diagnostic_frozen_agent_configuration_mismatch")
    counts = Counter(row.get("task_type") for row in plan["episodes"])
    if set(counts) != set(TASK_TYPE_NAMES) or set(counts.values()) != {2}:
        raise ProviderBudgetExceeded("six_families_two_tasks_each_required")
    if len({row.get("cluster_id") for row in plan["episodes"]}) != 12:
        raise ProviderBudgetExceeded("independent_diagnostic_clusters_required")
    extension, sample = manifest.get("split_extension"), manifest.get("sampling_manifest")
    if (not isinstance(extension, dict) or manifest.get("split_extension_sha256") != manifest_digest(extension)
            or not isinstance(sample, dict) or manifest.get("sampling_manifest_sha256") != manifest_digest(sample)
            or sample.get("selected") != [{k: v for k, v in row.items() if k != "gamefile_sha256"} for row in plan["episodes"]]):
        raise ProviderBudgetExceeded("diagnostic_reservation_or_sampling_mismatch")
    for key in ("max_usd_per_episode", "max_calls_per_episode", "max_episode_wall_seconds"):
        if str(manifest.get(key)) != str(approval.get(key)):
            raise ProviderBudgetExceeded(f"diagnostic_episode_approval_mismatch:{key}")
    if str(manifest.get("max_total_usd")) != str(approval.get("max_total_usd")):
        raise ProviderBudgetExceeded("diagnostic_stage_usd_mismatch")
    if manifest.get("max_wall_seconds") != approval.get("max_wall_seconds"):
        raise ProviderBudgetExceeded("diagnostic_stage_wall_mismatch")
    if any(row.get("split") != "train_dev" or row.get("exposure") not in {"new_train_dev", "unexposed"}
           or row.get("identity_status") != "IDENTIFIED" for row in plan["episodes"]):
        raise ProviderBudgetExceeded("diagnostic_task_exposed_or_unknown")


def diagnostic_summary(episodes: list[dict[str, Any]], planned: int) -> dict[str, Any]:
    """Observed failures only; empty content is a turn event, not a failed task."""
    from collections import Counter
    families: dict[str, Any] = {}
    for row in episodes:
        family = row["task_family"]
        entry = families.setdefault(family, {"completed": 0, "successes": 0, "unknown": 0})
        entry["completed"] += 1
        entry["successes"] += row.get("success") is True
        entry["unknown"] += row.get("success") is None
    return {"planned": planned, "started": len(episodes), "completed": len(episodes),
            "not_started": planned - len(episodes), "families": families,
            "terminal_statuses": dict(Counter(row["status"] for row in episodes)),
            "invalid_decision_recovered": sum(row.get("success") is True and any(
                t.get("finish_reason") == "length" and not t.get("selected_action")
                for t in row.get("trajectory", [])) for row in episodes),
            "failure_taxonomy": [{"task_id": row["task_id"], "category": (
                "budget" if row["status"] == "BUDGET_EXHAUSTED" else
                "action_parser" if row["status"] == "API_CONTRACT_FAILURE" else
                "loop" if row["termination_reason"] == "agent_policy_failure" else "UNIDENTIFIED")}
                for row in episodes if row.get("success") is not True],
            "source_readiness": viability_gate(episodes), "next_stage": "SEPARATE_T2_APPROVAL_REQUIRED"}


def run_diagnostic(manifest: dict[str, Any], client: Any, output: Path, *, environment_factory: Any = None) -> dict[str, Any]:
    """Run the frozen T1 order with the existing Agent and selected games only."""
    import hashlib
    from .native_verifier import make_selected_environment
    from experiments.external_real.alfworld_stateful_runner import run_episode
    environment_factory = environment_factory or make_selected_environment
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows: list[dict[str, Any]] = []
    plan = manifest["plan"]
    budget = client.budget
    for task in plan["episodes"]:
        budget.begin_episode(max_usd=manifest["max_usd_per_episode"], max_calls=manifest["max_calls_per_episode"],
                             max_wall_seconds=manifest["max_episode_wall_seconds"])
        try:
            budget.ensure_capacity(60, 60 * client.max_input_per_request, 60 * client.max_completion_tokens)
        except ProviderBudgetExceeded as exc:
            budget.stop(str(exc))
            break
        actual_game_sha = hashlib.sha256(Path(task["gamefile"]).read_bytes()).hexdigest()
        if actual_game_sha != task["gamefile_sha256"]:
            raise ValueError("frozen gamefile bytes changed; no replacement permitted")
        environment = environment_factory(task["gamefile"], max_steps=manifest["max_steps"], seed=plan["selection_seed"])
        try:
            row = run_episode(environment, client, max_steps=manifest["max_steps"],
                              receipt_metadata=client.receipt_metadata,
                              expected_gamefiles=[task["gamefile"]])
        finally:
            environment.close()
        row.update(task_id=task["task_id"], template_id=task["template_id"], cluster_id=task["cluster_id"],
                   parent_split="train_dev", child_split=None, task_family=task["task_type"],
                   gamefile_sha256=actual_game_sha, trial_id=f"{manifest['run_id']}:{len(rows)}",
                   record_kind=("ACTUAL_REAL_ROLLOUT" if row.get("evidence_origin") == "REAL_AGENT" else "SYNTHETIC_FIXTURE"),
                   seed=plan["selection_seed"],
                   pre_registered_manifest_sha256=manifest_digest(manifest),
                   independently_reset_environment=True, actual_new_model_rollout=row.get("evidence_origin") == "REAL_AGENT")
        row["trace_sha256"] = manifest_digest({"trajectory": row["trajectory"]})
        row["verifier_receipt_sha256"] = manifest_digest({k: row.get(k) for k in ["environment_won", "environment_done", "status"]})
        row["receipt_sha256"] = manifest_digest(row)
        rows.append(row)
        result = {"status": "IN_PROGRESS", "episodes": rows, "summary": diagnostic_summary(rows, 12),
                  "provider_budget": budget.as_dict()}
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        if budget.stopped or row["status"] in {"PROVIDER_FAILURE", "ENVIRONMENT_FAILURE", "INCONCLUSIVE", "BUDGET_EXHAUSTED"}:
            break
    result = {"status": "COMPLETED" if len(rows) == 12 else "STOPPED", "episodes": rows,
              "summary": diagnostic_summary(rows, 12), "provider_budget": budget.as_dict()}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "REPORT.md").write_text("# Frozen T1 diagnostic\n\n" + json.dumps(result["summary"], indent=2) + "\n")
    return result


def readiness_manifest(plan: dict[str, Any], *, controller_commit: str,
                       run_id: str, max_input_per_request: int | None = None,
                       max_completion_per_request: int | None = None,
                       split_manifest: dict[str, Any] | None = None,
                       live_output_dir: str | None = None) -> dict[str, Any]:
    steps = plan["config"]["max_steps"]
    episodes = len(plan["episodes"])
    return {"study_id": "skill_evolution_real_v2", "run_id": run_id,
            "stage": "train_viability", "controller_commit": controller_commit,
            "output_dir": live_output_dir,
            "model": "deepseek-flash", "thinking": "enabled", "native_api_tools": False,
            "split": "train_inner", "plan_sha256": manifest_digest(plan),
            "split_manifest": split_manifest,
            "split_manifest_sha256": manifest_digest(split_manifest) if split_manifest else None,
            "plan": plan, "max_steps": steps, "max_calls": episodes * steps,
            "max_input_per_request": max_input_per_request,
            "max_completion_per_request": max_completion_per_request,
            "max_retries": 0, "arms": ["V1_stateful_no_skill"],
            "stop_rules": {"zero_wins": "NO_GO_FOR_SKILL_OPTIMIZATION",
                           "one_win": "VIABLE_BUT_NOT_QUALIFIED",
                           "two_independent_wins_and_failure": "MAY_REQUEST_STAGE_C_APPROVAL"},
            "status": "READY_FOR_APPROVAL", "provider_calls": 0}


def resource_plan(manifest: dict[str, Any], pricing: dict[str, Any] | None = None) -> dict[str, Any]:
    calls = manifest["max_calls"]
    inp, out = manifest.get("max_input_per_request"), manifest.get("max_completion_per_request")
    inputs, outputs = (calls * inp if inp else None), (calls * out if out else None)
    cost = None
    if pricing is not None and inputs is not None and outputs is not None:
        from decimal import Decimal
        input_rate = max(Decimal(str(pricing["input_hit_per_million"])),
                         Decimal(str(pricing["input_miss_per_million"])))
        cost = str((inputs * input_rate + outputs * Decimal(str(pricing["output_per_million"]))) / Decimal(1_000_000))
    return {"status": "READY_FOR_APPROVAL", "provider_calls": 0,
            "manifest_sha256": manifest_digest(manifest), "max_calls": calls,
            "input_allowance": inputs, "completion_allowance": outputs,
            "estimated_worst_usd": cost, "cache_savings_assumed": False,
            "blockers": ["explicit scope and hard-budget approval"] +
                        ([] if inp and out else ["task-grounded finite request caps"])
                        + ([] if manifest.get("split_manifest") else ["template-level split manifest"])
                        + ([] if manifest.get("output_dir") else ["one fixed fresh live output directory"])
                        + ([] if pricing else ["verified execution-time pricing snapshot"])}


def prepare_live(manifest: dict[str, Any], plan: dict[str, Any], approval: dict[str, Any],
                 pricing: dict[str, Any], *, revision: str, base_url: str,
                 output_dir: Path | None = None) -> ApprovedProviderBudget:
    """All guards run before credentials, SDK import, reset or network I/O."""
    if manifest.get("controller_commit") != revision or manifest.get("plan_sha256") != manifest_digest(plan):
        raise ProviderBudgetExceeded("run_revision_or_plan_mismatch")
    if output_dir is None or not manifest.get("output_dir") or output_dir.resolve() != Path(manifest["output_dir"]).resolve():
        raise ProviderBudgetExceeded("run_output_directory_mismatch")
    split = manifest.get("split_manifest")
    if not isinstance(split, dict) or manifest.get("split_manifest_sha256") != manifest_digest(split):
        raise ProviderBudgetExceeded("split_manifest_required_or_mismatch")
    from .split_manifest import validate_split_membership
    membership = split.get("task_membership", [])
    validate_split_membership(membership)
    by_task = {row["task_id"]: row for row in membership}
    for episode in plan["episodes"]:
        row = by_task.get(episode.get("task_id"))
        expected_split = "train_dev" if manifest.get("stage") == "baseline_diagnostic" else "train_inner"
        if row is None or row.get("split") != expected_split or row.get("gamefile") != episode.get("gamefile"):
            raise ProviderBudgetExceeded("readiness_task_outside_frozen_train_inner")
    if manifest.get("stage") not in {"train_viability", "baseline_diagnostic"} or manifest.get("arms") != ["V1_stateful_no_skill"]:
        raise ProviderBudgetExceeded("unsupported_readiness_scope")
    if manifest["stage"] == "baseline_diagnostic":
        _validate_diagnostic(manifest, plan, split, approval)
        if manifest.get("pricing_sha256") != manifest_digest(pricing):
            raise ProviderBudgetExceeded("diagnostic_pricing_snapshot_mismatch")
    if manifest.get("thinking") != "enabled" or manifest.get("max_retries") != 0:
        raise ProviderBudgetExceeded("unsupported_provider_configuration")
    inp, out = manifest.get("max_input_per_request"), manifest.get("max_completion_per_request")
    if type(inp) is not int or inp <= 0 or type(out) is not int or out <= 0:
        raise ProviderBudgetExceeded("request_caps_require_approval")
    steps = manifest["max_steps"]
    if steps != plan["config"]["max_steps"] or steps > approval.get("per_episode_max_steps", 0):
        raise ProviderBudgetExceeded("episode_horizon_mismatch")
    calls = len(plan["episodes"]) * steps
    if calls != manifest["max_calls"]:
        raise ProviderBudgetExceeded("call_plan_mismatch")
    budget = ApprovedProviderBudget(manifest, approval, pricing, controller_commit=revision,
                                    stage=manifest["stage"], model=manifest["model"], base_url=base_url)
    budget.ensure_capacity(calls, calls * inp, calls * out)
    return budget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--max-input-per-request", type=int)
    parser.add_argument("--max-completion-per-request", type=int)
    parser.add_argument("--run-id", default="phase2-readiness")
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--pricing", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--live-output-dir", type=Path, help="freeze one future live directory, distinct from dry-run output")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.live and args.dry_run:
        parser.error("choose live or dry-run, not both")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        plan = manifest["plan"]
    elif args.plan or args.split_manifest:
        split = json.loads(args.split_manifest.read_text()) if args.split_manifest else None
        plan = json.loads(args.plan.read_text()) if args.plan else plan_from_split(split)
        manifest = readiness_manifest(plan, controller_commit=revision, run_id=args.run_id,
                                      max_input_per_request=args.max_input_per_request,
                                      max_completion_per_request=args.max_completion_per_request,
                                      split_manifest=split,
                                      live_output_dir=str(args.live_output_dir.resolve()) if args.live_output_dir else None)
    else:
        parser.error("--manifest or --plan is required")
    pricing = json.loads(args.pricing.read_text()) if args.pricing else None
    if not args.live:
        report = resource_plan(manifest, pricing)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    if args.approval is None or pricing is None:
        raise ProviderBudgetExceeded("APPROVAL_REQUIRED")
    if not args.data_root or not args.output:
        parser.error("live needs --data-root and fresh --output")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("live execution requires a clean committed checkout")
    approval = json.loads(args.approval.read_text())
    base_url = "https://api.deepseek.com"
    budget = prepare_live(manifest, plan, approval, pricing, revision=revision, base_url=base_url,
                          output_dir=args.output)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required at runtime")
    from experiments.external_real.alfworld_stateful_runner import DeepSeekStatefulClient, run_smoke
    def persist(snapshot: dict[str, Any]) -> None:
        with (args.output / "provider_journal.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    client = DeepSeekStatefulClient(api_key, base_url, manifest["model"], budget=budget,
                                    max_completion_tokens=manifest["max_completion_per_request"],
                                    persist_budget=persist)
    client.max_input_per_request = manifest["max_input_per_request"]
    import hashlib
    import inspect
    runner_source = Path(inspect.getfile(DeepSeekStatefulClient)).read_bytes()
    prompt_source = inspect.getsource(DeepSeekStatefulClient.choose_action).encode("utf-8")
    configuration_sha = manifest_digest({
        "agent_source": hashlib.sha256(runner_source).hexdigest(),
        "prompt_source": hashlib.sha256(prompt_source).hexdigest(),
        "model": manifest["model"], "thinking": "enabled", "max_steps": manifest["max_steps"],
        "max_input_per_request": manifest["max_input_per_request"],
        "max_completion_per_request": manifest["max_completion_per_request"],
    })
    client.run_manifest = manifest
    client.receipt_metadata = {"study_id": manifest["study_id"], "run_id": manifest["run_id"],
                               "split": manifest["split"],
                               "arm": "V1_stateful_no_skill", "evidence_origin": "REAL_AGENT",
                               "source_commit": revision,
                               "agent_revision": revision, "benchmark_revision": revision,
                               "agent_revision_sha256": hashlib.sha256(runner_source).hexdigest(),
                               "prompt_template_sha256": hashlib.sha256(prompt_source).hexdigest(),
                               "configuration_sha": configuration_sha,
                               "model_request_name": manifest["model"],
                               "split_manifest_sha256": manifest["split_manifest_sha256"],
                               "approval_reference": approval["approval_reference"]}
    if manifest["stage"] == "baseline_diagnostic":
        client.receipt_metadata.update(reasoning_effort=manifest["reasoning_effort"],
                                       agent_condition=manifest["agent_condition"],
                                       price_snapshot_sha256=manifest_digest(pricing),
                                       approval_sha256=manifest_digest(approval))
        run_diagnostic(manifest, client, args.output)
    else:
        run_smoke(args.data_root, args.output, client, len(plan["episodes"]),
                  manifest["max_steps"], plan["selection_seed"], plan=plan)


if __name__ == "__main__":
    main()
