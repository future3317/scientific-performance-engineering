"""Frozen task-start OFF/LENGTH/SKILL attribution on the existing native runner."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import random
import re
from statistics import mean, median
from typing import Any, Mapping, Sequence

from core.acre.resource_budget import ApprovedProviderBudget, ProviderBudgetExceeded, manifest_digest
from core.skills import SkillContract
from .evaluation import _paired_counts, _request_cost, receipt_outcome

ARMS = ("OFF", "LENGTH", "SKILL")
LENGTH_CONTROL = "Be concise. Do not produce unnecessary explanation. Return the next valid action efficiently."
RESOURCE_FIELDS = ("provider_calls", "input_tokens", "completion_tokens", "reasoning_tokens",
                   "final_content_tokens", "total_tokens", "cache_hit_tokens", "cache_miss_tokens",
                   "environment_steps", "invalid_decisions", "replans", "repeated_actions",
                   "provider_usd_estimate", "peak_usd_estimate", "off_peak_usd_estimate", "wall_seconds")


def arm_texts(skill: SkillContract) -> dict[str, str | None]:
    if re.search(r"\b(?:be concise|concise sentence|unnecessary explanation|avoid re.deriving|brief answer)\b", skill.body, re.I):
        raise ValueError("procedure arm contains length-control wording; preregister a different contrast")
    return {"OFF": None, "LENGTH": LENGTH_CONTROL, "SKILL": skill.body}


def arm_fingerprints(skill: SkillContract) -> dict[str, Any]:
    from experiments.external_real.alfworld_stateful_runner import DeepSeekStatefulClient, EpisodeState
    class Capture:
        def complete_text(self, messages):
            return messages
    state = EpisodeState.start("Your task is to: put the target at the destination.", 60)
    result = {}
    for arm, text in arm_texts(skill).items():
        messages = DeepSeekStatefulClient.choose_action(Capture(), state, ["look"], text, "", None)
        result[arm] = {"messages": messages, "prompt_sha256": manifest_digest({"messages": messages}),
                       "treatment_sha256": manifest_digest({"text": text}), "skill_content": arm == "SKILL"}
    return result


def diagnose_failure(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Public trajectory only; no task files, expert state or private reasoning."""
    trace = episode["trajectory"]
    actions = [r["selected_action"] for r in trace if r.get("selected_action")]
    navigation = [a.removeprefix("go to ") for a in actions if a.startswith("go to ")]
    initial_locations = {a.removeprefix("go to ") for a in trace[0].get("admissible_actions", []) if a.startswith("go to ")}
    completed, progression, last_change, longest = set(), [], -1, 0
    for row in trace:
        now = set(row.get("completed_subgoals", []))
        if now - completed:
            last_change = row["step"]
            progression.append({"step": row["step"], "new_public_subgoals": sorted(now - completed)})
        completed |= now
        longest = max(longest, row["step"] - last_change)
    unique_actions = len(set(actions))
    repeated_actions = len(actions) - unique_actions
    revisits = len(navigation) - len(set(navigation))
    return {
        "episode_id": episode["trial_id"], "family": episode.get("task_family"),
        "success": receipt_outcome(episode), "termination": episode["termination_reason"],
        "valid_actions": len(actions), "invalid_actions": sum(bool(r.get("invalid_action")) for r in trace),
        "unique_actions": unique_actions, "repeated_actions": repeated_actions,
        "repeated_action_ratio": repeated_actions / len(actions) if actions else None,
        "navigation_actions": len(navigation), "unique_locations": len(set(navigation)),
        "repeated_location_ratio": revisits / len(navigation) if navigation else None,
        "location_ratio_denominator": "navigation actions; repeats after first visit",
        "initial_visible_locations": len(initial_locations),
        "unvisited_initial_locations": sorted(initial_locations - set(navigation)),
        "progress_changes": len(progression), "longest_no_progress_window": longest,
        "progress_definition": "new public completed_subgoals; not merely observation text changed",
        "goal_predicate_progression": progression,
        "goal_state_distance": None, "goal_state_distance_status": "UNOBSERVABLE_WITHOUT_ORACLE",
        "last_state": {k: episode.get("state", {}).get(k) for k in
                       ("goal", "current_room_or_location", "inventory", "completed_subgoals", "last_observation")},
        "replans": sum(e.get("kind") == "replan" for e in episode.get("progress_events", [])),
        "action_diversity": unique_actions / len(actions) if actions else None,
        "failure_class": "navigation loop" if revisits and not completed else "other / unknown",
        "contributors": ["search coverage incomplete", "official horizon exhausted"],
        "evidence": [{"step": r["step"], "action": r.get("selected_action"), "observation": r.get("observation"),
                      "completed_subgoals": r.get("completed_subgoals", [])} for r in trace],
        "inference_boundary": "No inference about target's actual location or counterfactual success. State-tracking causality remains a hypothesis.",
    }


def episode_metrics(row: Mapping[str, Any], pricing: Mapping[str, Any]) -> dict[str, Any]:
    usage = _request_cost(row)
    trace = row.get("trajectory", [])
    actions = [r["selected_action"] for r in trace if r.get("selected_action")]
    seen_subgoals, stalls, longest_stall, current_stall = set(), 0, 0, 0
    for turn in trace:
        current = set(turn.get("completed_subgoals", []))
        current_stall = 0 if current-seen_subgoals else current_stall+1
        stalls += not bool(current-seen_subgoals)
        seen_subgoals |= current
        longest_stall = max(longest_stall,current_stall)
    result = {**usage, "success": receipt_outcome(row), "termination_reason": row.get("termination_reason"),
              "environment_steps": row.get("env_steps"),
              "invalid_decisions": sum(bool(r.get("invalid_action")) for r in trace),
              "replans": sum(e.get("kind") == "replan" for e in row.get("progress_events", [])),
              "repeated_actions": len(actions) - len(set(actions)),
              "progress_stalls": stalls, "longest_no_subgoal_progress": longest_stall,
              "progress_definition": "no new public completed subgoal; observation novelty is not goal progress",
              "actual_billed_usd": None, "platform_cost_usd": None,
              "skill_injected": row.get("skill_injected"), "skill_loaded": row.get("skill_rendered"),
              "skill_usage_confidence": "UNKNOWN", "trigger_matched": row.get("trigger_matched"),
              "trigger_mismatch": None, "cost_excludes_codex_luna": True}
    n, reasoning = usage["completion_tokens"], usage["reasoning_tokens"]
    result["total_tokens"] = usage["input_tokens"]+n if usage["input_tokens"] is not None and n is not None else None
    result["final_content_tokens"] = n - reasoning if n is not None and reasoning is not None and n >= reasoning else None
    # Official returned total-minus-reasoning, not a tokenizer estimate of the text.
    result["final_content_tokens_definition"] = "completion minus reported reasoning; null if unavailable"
    for name, rates in (("peak_usd_estimate", pricing), ("off_peak_usd_estimate", pricing.get("off_peak", {}))):
        inputs, outputs, hits, misses = [usage[k] for k in ("input_tokens", "completion_tokens", "cache_hit_tokens", "cache_miss_tokens")]
        if inputs is None or outputs is None or not all(k in rates for k in
            ("input_hit_per_million", "input_miss_per_million", "output_per_million")):
            result[name] = None
        else:
            hits = hits if hits is not None and misses is not None and hits + misses == inputs else 0
            result[name] = float((Decimal(inputs-hits)*Decimal(str(rates["input_miss_per_million"]))
                + Decimal(hits)*Decimal(str(rates["input_hit_per_million"]))
                + Decimal(outputs)*Decimal(str(rates["output_per_million"]))) / Decimal(1_000_000))
    return result


def paired_resource_summary(values: Sequence[float], *, seed: int = 83, repeats: int = 4000) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "mean_ci95": None, "median_ci95": None}
    rng = random.Random(seed)
    boot = [rng.choices(list(values), k=len(values)) for _ in range(repeats)]
    def bounds(sample):
        sample = sorted(sample)
        return [sample[math.floor(.025*(repeats-1))], sample[math.ceil(.975*(repeats-1))]]
    return {"n": len(values), "mean": mean(values), "median": median(values),
            "mean_ci95": bounds([mean(v) for v in boot]), "median_ci95": bounds([median(v) for v in boot]),
            "method": "independent-task paired percentile bootstrap; descriptive small-sample interval",
            "seed": seed, "resamples": repeats}


def summarize_attribution(plan: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
                          pricing: Mapping[str, Any], arms: Sequence[str] = ARMS) -> dict[str, Any]:
    by = {}
    planned = {r["cluster_id"]: r for r in plan}
    if len(planned) != len(plan):
        raise ValueError("duplicate planned independent cluster")
    for row in rows:
        key = row["cluster_id"], row["arm"]
        if key in by:
            raise ValueError("duplicate task-arm receipt; shared controls are counted once")
        if key[0] not in planned or key[1] not in arms:
            raise ValueError("receipt outside frozen plan")
        expected = planned[key[0]]
        if any(row.get(k) != expected.get(k) for k in ("task_id", "template_id", "seed")):
            raise ValueError("paired task/seed identity mismatch")
        by[key] = row
    comparisons = {}
    pairs = [("OFF", "LENGTH"), ("OFF", "SKILL"), ("LENGTH", "SKILL")] if tuple(arms) == ARMS else [(arms[0], arms[1])]
    costs = {key: episode_metrics(row, pricing) for key, row in by.items()}
    for left, right in pairs:
        outcomes, deltas = [], {k: [] for k in RESOURCE_FIELDS}
        fractions = {k: [] for k in RESOURCE_FIELDS}
        for cluster in planned:
            a, b = by.get((cluster, left)), by.get((cluster, right))
            av, bv = receipt_outcome(a) if a else None, receipt_outcome(b) if b else None
            outcomes.append((av, bv) if av is not None and bv is not None else None)
            if av is True and bv is True:
                ac, bc = costs[cluster,left], costs[cluster,right]
                for k in deltas:
                    if ac.get(k) is not None and bc.get(k) is not None:
                        deltas[k].append(ac[k]-bc[k])
                        if ac[k] > 0:
                            fractions[k].append((ac[k]-bc[k])/ac[k])
        cells = _paired_counts(outcomes)
        valid = len(planned)-cells["missing"]
        discordant = cells["repair"]+cells["regression"]
        from scipy.stats import binomtest
        comparisons[left+"_vs_"+right] = {
            **cells, "valid_pairs": valid, "net_gain": cells["repair"]-cells["regression"],
            "delta_success": (cells["repair"]-cells["regression"])/valid if valid else None,
            "exact_mcnemar_p": binomtest(cells["repair"], discordant).pvalue if discordant else None,
            "discordant_information": discordant,
            "efficiency_conditioning": "both arms officially successful; positive delta means right arm saves",
            "resource_savings": {k: paired_resource_summary(v) for k, v in deltas.items()},
            "saving_fractions": {k: paired_resource_summary(v) for k, v in fractions.items()},
        }
    return {"planned_independent_clusters": len(planned), "completed_arm_records": len(by),
            "arm_metrics": {arm: [costs[key] for key in costs if key[1] == arm] for arm in arms},
            "comparisons": comparisons, "interaction_effect": None,
            "interaction_status": "NOT_IDENTIFIABLE_WITH_THREE_ARMS", "evaluation_scope": "train_inner attribution, not independent outer"}


def validate_manifest(manifest: Mapping[str, Any], parent: Mapping[str, Any], skill: SkillContract) -> None:
    if manifest.get("stage") != "skill_attribution" or manifest.get("split") != "train_inner":
        raise ValueError("attribution requires approved inner stage")
    if manifest.get("split_manifest_sha256") != manifest_digest(parent):
        raise ValueError("parent split identity mismatch")
    if manifest.get("skill_digest") != skill.digest or manifest.get("skill_body_sha256") != skill.body_digest:
        raise ValueError("frozen skill identity mismatch")
    if manifest.get("arms") != list(ARMS) or manifest.get("treatments") != arm_texts(skill):
        raise ValueError("arm content differs from frozen OFF/LENGTH/SKILL design")
    members = {r["task_id"]: r for r in parent["task_membership"]}
    seen, templates = set(), set()
    excluded = set(manifest["excluded_clusters"])
    for task in manifest["plan"]:
        member = members.get(task["task_id"], {})
        if member.get("split") != "train_inner" or member.get("confirmatory_eligible") is not True:
            raise ValueError("task is not eligible inner metadata")
        if any(task.get(k) != member.get(k) for k in ("task_id", "cluster_id", "template_id", "gamefile")):
            raise ValueError("task identity differs from immutable parent")
        if task["cluster_id"] in seen | excluded or task["template_id"] in templates | set(manifest["excluded_templates"]):
            raise ValueError("task is duplicate, exposed or reserved")
        if sorted(task["arm_order"]) != sorted(ARMS) or task["seed"] != manifest["seed"]:
            raise ValueError("task arms/seed differ")
        seen.add(task["cluster_id"]); templates.add(task["template_id"])
    if manifest.get("thinking") != "enabled" or manifest.get("max_retries") != 0:
        raise ValueError("model configuration changed")


def _append(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True)+"\n"); stream.flush(); os.fsync(stream.fileno())


def run_attribution(manifest, parent, skill, approval, pricing, *, output: Path, revision: str,
                    client_factory=None, environment_factory=None) -> dict[str, Any]:
    """Approved-only; resume completed arms, never restart an interrupted environment."""
    validate_manifest(manifest,parent,skill)
    if manifest["controller_commit"] != revision or str(output) != manifest["output_dir"]:
        raise ValueError("run revision/output identity mismatch")
    budget=ApprovedProviderBudget(manifest,approval,pricing,controller_commit=revision,
        stage="skill_attribution",model=manifest["model"],base_url=manifest["base_url"])
    from experiments.external_real.alfworld_stateful_runner import DeepSeekStatefulClient, run_episode
    from .native_verifier import make_selected_environment
    from .receipt_journal import DurableProviderJournal
    output.mkdir(parents=True,exist_ok=True)
    config=output/"run_config.json"
    if config.exists() and json.loads(config.read_text()) != dict(manifest):
        raise ValueError("resume manifest differs")
    if not config.exists():
        with config.open("x")as f:
            json.dump(dict(manifest),f,sort_keys=True);f.flush();os.fsync(f.fileno())
    events=output/"episodes.jsonl"
    prior=[json.loads(line)for line in events.read_text().splitlines()]if events.exists()else[]
    started={e["trial_id"]for e in prior if e["event"]=="STARTED"}
    completed={e["trial_id"]:e["episode"]for e in prior if e["event"]=="COMPLETED"}
    if started-set(completed):
        raise ProviderBudgetExceeded("interrupted_environment_unknown_no_automatic_restart")
    if any(receipt_outcome(row) is None for row in completed.values()):
        raise ProviderBudgetExceeded("unknown_previous_arm_no_automatic_continuation")
    journal=DurableProviderJournal(Path(manifest["spool_dir"]))
    journal.restore_budget(budget)
    factory=client_factory or DeepSeekStatefulClient
    if client_factory is None and not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY required only after exact live approval")
    client=factory(os.environ.get("DEEPSEEK_API_KEY"),manifest["base_url"],manifest["model"],budget=budget,
        max_completion_tokens=manifest["max_completion_per_request"],
        persist_budget=lambda b:_append(output/"provider_journal.jsonl",b))
    client.max_input_per_request=manifest["max_input_per_request"]
    client.receipt_journal=journal
    make_env=environment_factory or make_selected_environment
    for task in manifest["plan"]:
        pending=[arm for arm in task["arm_order"]if f"{manifest['run_id']}:{task['cluster_id']}:{arm}"not in completed]
        budget.ensure_capacity(len(pending)*manifest["max_steps"],len(pending)*manifest["max_steps"]*manifest["max_input_per_request"],
                               len(pending)*manifest["max_steps"]*manifest["max_completion_per_request"])
        for arm in pending:
            trial=f"{manifest['run_id']}:{task['cluster_id']}:{arm}"
            budget.begin_episode(max_usd=manifest["max_usd_per_episode"],max_calls=manifest["max_calls_per_episode"],max_wall_seconds=manifest["max_episode_wall_seconds"])
            client.set_episode_context(trial,arm,str(skill.version)if arm=="SKILL"else None)
            _append(events,{"event":"STARTED","trial_id":trial})
            env=make_env(task["gamefile"],max_steps=manifest["max_steps"],seed=task["seed"])
            try:
                row=run_episode(env,client,skill=manifest["treatments"][arm],max_steps=manifest["max_steps"],
                    skill_version=str(skill.version)if arm=="SKILL"else None,expected_gamefiles=[task["gamefile"]],
                    receipt_metadata={**{k:task[k]for k in ("task_id","cluster_id","template_id","seed")},
                        "trial_id":trial,"arm":arm,"split":"train_inner","study_id":manifest["study_id"],
                        "run_id":manifest["run_id"],"evidence_origin":"REAL_AGENT","source_commit":revision,
                        "skill_digest":skill.digest if arm=="SKILL"else None})
            finally:
                env.close()
            row["prompt_treatment_present"]=arm!="OFF"
            row["skill_injected"]=row["skill_rendered"]=arm=="SKILL"
            _append(events,{"event":"COMPLETED","trial_id":trial,"episode":row})
            completed[trial]=row
            # Raw episode is durable before any derived summary can fail.
            if receipt_outcome(row)is None:
                return {"status":"STOP_UNKNOWN","completed":len(completed),"provider_budget":budget.as_dict()}
            cluster_rows = {r["arm"]: r for r in completed.values() if r["cluster_id"] == task["cluster_id"]}
            if "OFF" in cluster_rows and receipt_outcome(cluster_rows["OFF"]) is True and any(
                receipt_outcome(cluster_rows[a]) is False for a in ("LENGTH","SKILL") if a in cluster_rows
            ):
                return {"status":"STOP_HARD_REGRESSION","completed":len(completed),"provider_budget":budget.as_dict()}
    report=summarize_attribution(manifest["plan"],list(completed.values()),pricing)
    report.update(status="COMPLETED",provider_budget=budget.as_dict(),outer_status="NOT_RUN")
    temporary=output/"summary.tmp"
    with temporary.open("w")as f:
        json.dump(report,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temporary,output/"report.json")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("manifest","split-manifest","skill","approval","pricing"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--live",action="store_true")
    args=parser.parse_args(argv)
    read=lambda p:json.loads(p.read_text())
    manifest,parent,skill=read(args.manifest),read(args.split_manifest),SkillContract.from_dict(read(args.skill))
    validate_manifest(manifest,parent,skill)
    if not args.live:
        print(json.dumps({"status":"READY_FOR_APPROVAL","provider_calls":0,"manifest_sha256":manifest_digest(manifest)}));return 0
    import subprocess
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise ValueError("live run requires clean source")
    revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    run_attribution(manifest,parent,skill,read(args.approval),read(args.pricing),output=Path(manifest["output_dir"]),revision=revision)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
