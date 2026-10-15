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
            "claim": "train-side engineering gate, not efficacy"}


def readiness_manifest(plan: dict[str, Any], *, controller_commit: str,
                       run_id: str, max_input_per_request: int | None = None,
                       max_completion_per_request: int | None = None,
                       split_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    steps = plan["config"]["max_steps"]
    episodes = len(plan["episodes"])
    return {"study_id": "skill_evolution_real_v2", "run_id": run_id,
            "stage": "train_viability", "controller_commit": controller_commit,
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
                        + ([] if pricing else ["verified execution-time pricing snapshot"])}


def prepare_live(manifest: dict[str, Any], plan: dict[str, Any], approval: dict[str, Any],
                 pricing: dict[str, Any], *, revision: str, base_url: str) -> ApprovedProviderBudget:
    """All guards run before credentials, SDK import, reset or network I/O."""
    if manifest.get("controller_commit") != revision or manifest.get("plan_sha256") != manifest_digest(plan):
        raise ProviderBudgetExceeded("run_revision_or_plan_mismatch")
    split = manifest.get("split_manifest")
    if not isinstance(split, dict) or manifest.get("split_manifest_sha256") != manifest_digest(split):
        raise ProviderBudgetExceeded("split_manifest_required_or_mismatch")
    from .split_manifest import validate_split_membership
    membership = split.get("task_membership", [])
    validate_split_membership(membership)
    by_task = {row["task_id"]: row for row in membership}
    for episode in plan["episodes"]:
        row = by_task.get(episode.get("task_id"))
        if row is None or row.get("split") != "train_inner" or row.get("gamefile") != episode.get("gamefile"):
            raise ProviderBudgetExceeded("readiness_task_outside_frozen_train_inner")
    if manifest.get("stage") != "train_viability" or manifest.get("arms") != ["V1_stateful_no_skill"]:
        raise ProviderBudgetExceeded("unsupported_readiness_scope")
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
                                    stage="train_viability", model=manifest["model"], base_url=base_url)
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
                                      split_manifest=split)
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
    budget = prepare_live(manifest, plan, approval, pricing, revision=revision, base_url=base_url)
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
    client.receipt_metadata = {"study_id": manifest["study_id"], "run_id": manifest["run_id"],
                               "split": "train_inner",
                               "arm": "V1_stateful_no_skill", "evidence_origin": "REAL_AGENT",
                               "source_commit": revision,
                               "agent_revision": revision, "benchmark_revision": revision,
                               "agent_revision_sha256": hashlib.sha256(runner_source).hexdigest(),
                               "prompt_template_sha256": hashlib.sha256(prompt_source).hexdigest(),
                               "model_request_name": manifest["model"],
                               "split_manifest_sha256": manifest["split_manifest_sha256"],
                               "approval_reference": approval["approval_reference"]}
    run_smoke(args.data_root, args.output, client, len(plan["episodes"]),
              manifest["max_steps"], plan["selection_seed"], plan=plan)


if __name__ == "__main__":
    main()
