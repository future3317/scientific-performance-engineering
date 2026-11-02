"""Batched, single-attempt live LogicBench extension for Soft-ACRE."""

from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
from core.acre.resource_budget import ApprovedProviderBudget, ProviderBudgetExceeded, manifest_digest

STUDY_ID = "EVOSPE_SOFT_ACRE_LOGICBENCH_EXTENSION_20261102_03"
SKILLS = ("logicbench_007", "logicbench_011", "logicbench_017", "logicbench_010")
MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
MAX_COMPLETION = 512
TOTAL_REMAINING_USD = Decimal("0.80")
STAGE_CAPS = {"calibration": Decimal("0.30"), "final": Decimal("0.50")}
PRICING = {
    "currency": "USD", "model": MODEL,
    "pricing_source_url": "https://api-docs.deepseek.com/quick_start/pricing",
    "checked_at_utc": "2026-11-01T18:47:42Z", "price_period": "official peak rates",
    "estimate_quality": "provider-reported token usage at official peak rates; estimate, not invoice",
    "input_hit_per_million": "0.006", "input_miss_per_million": "0.30", "output_per_million": "1.32",
}
INPUT_OVERHEAD_TOKENS = 128
KNOWN_PRIOR_ESTIMATE_USD = Decimal("0.193741176")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@lru_cache(maxsize=1)
def _corpus() -> dict[str, Any]:
    from sragents.corpus import load_corpus_dict
    return load_corpus_dict()


def prompt_for(instance: dict[str, Any]) -> tuple[str, str]:
    from sragents.prompts import build_prompt
    corpus = _corpus()
    skills = [corpus[sid] for sid in instance.get("skill_annotations", []) if sid in corpus]
    return build_prompt(instance, skills=[s["content"] for s in skills])


def input_files(input_dir: Path) -> dict[str, Path]:
    paths = {"protocol": input_dir / "confirmatory_protocol.json", "y0": input_dir / "confirmatory_y0_eval.json"}
    for skill in SKILLS:
        for stage in ("verification", "calibration", "final"):
            paths[f"{skill}_{stage}"] = input_dir / f"confirmatory_y1_{skill}_{stage}_instances.json"
        paths[f"{skill}_verification_eval"] = input_dir / f"confirmatory_y1_{skill}_verification_eval_disabled.json"
    return paths


def prepare(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    if (output_dir / "STUDY_MANIFEST.json").exists():
        raise FileExistsError("study manifest already exists")
    paths = input_files(input_dir)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"frozen study inputs are incomplete: {missing}")
    protocol = read(paths["protocol"])
    if (protocol["model"] != MODEL or protocol["temperature"] != 0.0
            or protocol["max_tokens"] != MAX_COMPLETION or protocol["thinking"] is not False
            or protocol["engine"] != "direct" or protocol["workers"] != 1):
        raise ValueError("model/runtime contract differs from the frozen protocol")
    y0_eval = read(paths["y0"])
    y0_by_id = {row["instance_id"]: bool(row["correct"]) for row in y0_eval["details"]}
    prompt_bounds: dict[str, dict[str, Any]] = {}
    stage_ids: dict[str, list[str]] = {"calibration": [], "final": []}
    verification = {"repairs": 0, "regressions": 0, "n": 0}
    for skill in SKILLS:
        verify = read(paths[f"{skill}_verification"])
        verify_eval = read(paths[f"{skill}_verification_eval"])
        outcomes = {row["instance_id"]: bool(row["correct"]) for row in verify_eval["details"]}
        if set(outcomes) != {row["instance_id"] for row in verify}:
            raise ValueError(f"{skill} verification outcome IDs differ")
        repairs = sum(not y0_by_id[item["instance_id"]] and outcomes[item["instance_id"]] for item in verify)
        regressions = sum(y0_by_id[item["instance_id"]] and not outcomes[item["instance_id"]] for item in verify)
        if not (repairs > regressions and repairs >= 2):
            raise ValueError(f"{skill} fails the frozen verification gate")
        verification["repairs"] += repairs
        verification["regressions"] += regressions
        verification["n"] += len(verify)
        for stage in ("calibration", "final"):
            items = read(paths[f"{skill}_{stage}"])
            if len(items) != (10 if stage == "calibration" else 20):
                raise ValueError(f"{skill} {stage} cohort size differs from protocol")
            stage_local: set[str] = set()
            for item in items:
                identifier = item["instance_id"]
                if item["confirmatory_skill"] != skill or item["confirmatory_stage"] != stage:
                    raise ValueError(f"{skill} {stage} identity fields differ")
                if identifier not in y0_by_id or identifier in stage_local:
                    raise ValueError(f"missing Y0 or duplicate instance within {skill} {stage}: {identifier}")
                system, user = prompt_for(item)
                bound = len(system.encode("utf-8")) + len(user.encode("utf-8")) + INPUT_OVERHEAD_TOKENS
                prompt_bounds[f"{stage}:{identifier}"] = {
                    "skill": skill, "stage": stage, "input_token_bound": bound,
                    "prompt_sha256": hashlib.sha256((system + "\0" + user).encode("utf-8")).hexdigest(),
                }
                stage_ids[stage].append(identifier)
                stage_local.add(identifier)
    calibration_ids, original_final_ids = stage_ids["calibration"], stage_ids["final"]
    if len(set(calibration_ids)) != len(calibration_ids) or len(set(original_final_ids)) != len(original_final_ids):
        raise ValueError("duplicate instance across skills within a study stage")
    excluded_final_ids = sorted(set(calibration_ids) & set(original_final_ids))
    if len(calibration_ids) != 40 or len(original_final_ids) != 80 or len(excluded_final_ids) != 4:
        raise ValueError("frozen cohorts or predeclared cross-stage overlap count changed")
    stage_ids["final"] = [identifier for identifier in original_final_ids if identifier not in set(excluded_final_ids)]
    for identifier in excluded_final_ids:
        prompt_bounds.pop(f"final:{identifier}", None)
    costs = {}
    for stage, ids in stage_ids.items():
        input_tokens = sum(prompt_bounds[f"{stage}:{identifier}"]["input_token_bound"] for identifier in ids)
        output_tokens = len(ids) * MAX_COMPLETION
        costs[stage] = str((Decimal(input_tokens) * Decimal(PRICING["input_miss_per_million"])
                            + Decimal(output_tokens) * Decimal(PRICING["output_per_million"])) / Decimal(1_000_000))
        if Decimal(costs[stage]) > STAGE_CAPS[stage]:
            raise ValueError(f"{stage} worst-case token bound exceeds its USD cap")
    source_root = Path(__file__).resolve().parents[2]
    dirty = subprocess.check_output(["git", "-C", str(source_root), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise ValueError("freeze requires a clean source worktree")
    source_revision = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    sra_root = Path("/home/workspace/lrh/CODE/SR-Agents")
    sra_revision = subprocess.check_output(["git", "-C", str(sra_root), "rev-parse", "HEAD"], text=True).strip()
    value = {
        "study_id": STUDY_ID, "status": "FROZEN_BEFORE_NEW_PROVIDER_CALLS",
        "supersedes_stopped_study_id": "EVOSPE_SOFT_ACRE_LOGICBENCH_EXTENSION_20261102_02",
        "prior_study_record": "one unsuccessful request with zero reported input/output tokens and USD 0 estimated usage; final stage unopened",
        "source_revision": source_revision, "sragents_revision": sra_revision,
        "protocol_id": protocol["experiment"],
        "candidate_selection": "four skills passing the pre-existing verification gate; selection outcomes are development evidence",
        "skills": list(SKILLS), "independent_skill_families": len(SKILLS),
        "model": MODEL, "base_url": BASE_URL, "temperature": 0.0, "thinking": False,
        "engine": "direct", "provider": "oracle", "workers": 1,
        "max_completion_tokens_per_request": MAX_COMPLETION, "max_provider_calls": sum(map(len, stage_ids.values())),
        "calls_by_stage": {key: len(value) for key, value in stage_ids.items()},
        "total_new_provider_budget_usd": str(TOTAL_REMAINING_USD),
        "prior_cded_estimate_usd": str(KNOWN_PRIOR_ESTIMATE_USD),
        "per_stage_worst_case_usd": costs,
        "per_stage_usd_caps": {key: str(value) for key, value in STAGE_CAPS.items()},
        "pricing": PRICING, "input_file_sha256": {key: sha256(path) for key, path in paths.items()},
        "stage_instance_ids": stage_ids, "excluded_final_ids_for_calibration_overlap": excluded_final_ids,
        "prompt_bounds": prompt_bounds,
        "verification_gate": protocol["verification_gate"], "verification_screen": verification,
        "calibration_rule": "fit Soft-ACRE and strongest single-feature route on calibration outcomes; require a selective route to exceed the positive-global-or-abstain default by one paired standard error; freeze final decisions before final Y1 calls",
        "policy_parameters": {"risk_weight": 1.0, "complexity_penalty": 0.002, "max_literals": 3,
                             "switching_margin_se": 1.0, "simple_baseline_max_literals": 1},
        "final_outcomes_used_for_policy": False,
        "failure_rule": "provider/executor error or missing usage is terminal UNKNOWN; stop and do not retry or replace",
        "claim_boundary": "conditional four-family prospective evaluation of candidates selected by the prior verification gate",
    }
    value["manifest_sha256"] = manifest_digest(value)
    atomic_json(output_dir / "STUDY_MANIFEST.json", value)
    return value


def run_stage(input_dir: Path, output_dir: Path, stage: str, env_file: Path) -> dict[str, Any]:
    from dotenv import dotenv_values
    from openai import OpenAI
    manifest = read(output_dir / "STUDY_MANIFEST.json")
    if manifest.get("manifest_sha256") != manifest_digest(manifest):
        raise ValueError("frozen study manifest digest mismatch")
    if stage not in ("calibration", "final"):
        raise ValueError("stage must be calibration or final")
    if stage == "final":
        frozen = read(output_dir / "calibration" / "POLICY_FREEZE.json")
        if frozen.get("final_outcomes_read") is not False:
            raise ValueError("policy freeze indicates final leakage")
        if sum(len(row["final_decisions"]) for row in frozen["per_skill"]) != 80:
            raise ValueError("frozen policy decisions do not cover all final IDs")
    usage_path = output_dir / "API_USAGE.json"
    usage = read(usage_path) if usage_path.exists() else {
        "study_id": STUDY_ID, "prior_cded_estimate_usd": str(KNOWN_PRIOR_ESTIMATE_USD), "stages": {}
    }
    if stage in usage["stages"]:
        raise FileExistsError(f"{stage} stage has already started; never rerun a paid stage")
    prior_new = sum((Decimal(row["estimated_usd"]) for row in usage["stages"].values()), Decimal(0))
    cap = min(STAGE_CAPS[stage], TOTAL_REMAINING_USD - prior_new)
    if cap <= 0 or prior_new + Decimal(manifest["per_stage_worst_case_usd"][stage]) > TOTAL_REMAINING_USD:
        raise ProviderBudgetExceeded("frozen_worst_case_exceeds_remaining_total_budget")
    bounds = manifest["prompt_bounds"]
    instances_by_id: dict[str, dict[str, Any]] = {}
    for skill in SKILLS:
        path = input_dir / f"confirmatory_y1_{skill}_{stage}_instances.json"
        if sha256(path) != manifest["input_file_sha256"][f"{skill}_{stage}"]:
            raise ValueError(f"frozen input changed: {path.name}")
        allowed = set(manifest["stage_instance_ids"][stage])
        for item in read(path):
            if item["instance_id"] not in allowed:
                continue
            if item["instance_id"] in instances_by_id:
                raise ValueError("duplicate instance IDs in stage")
            instances_by_id[item["instance_id"]] = item
    ids = manifest["stage_instance_ids"][stage]
    if set(ids) != set(instances_by_id):
        raise ValueError("stage instance IDs differ from frozen manifest")
    for identifier, item in instances_by_id.items():
        system, user = prompt_for(item)
        if hashlib.sha256((system + "\0" + user).encode("utf-8")).hexdigest() != bounds[f"{stage}:{identifier}"]["prompt_sha256"]:
            raise ValueError(f"prompt changed after freeze: {identifier}")
    stage_manifest = {
        "study_id": STUDY_ID, "run_id": f"{STUDY_ID}_{stage.upper()}",
        "stage": stage, "split_manifest_sha256": manifest["manifest_sha256"],
        "instances": ids, "source_revision": manifest["source_revision"],
    }
    approval = {
        "provider_enabled": True, "execute": True, "approval_status": "APPROVED",
        "controller_commit": manifest["source_revision"], "model": MODEL,
        "study_id": STUDY_ID, "run_id": stage_manifest["run_id"],
        "approved_manifest_sha256": manifest_digest(stage_manifest),
        "approved_split_manifest_sha256": manifest["manifest_sha256"],
        "permitted_stages": [stage], "api_base_allowlist": [BASE_URL],
        "approved_by": "user", "approval_reference": "user-authorized-USD-1.00-cumulative-LogicBench-extension",
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "expiration_utc": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
        "max_provider_calls": len(ids),
        "max_input_tokens": sum(bounds[f"{stage}:{i}"]["input_token_bound"] for i in ids),
        "max_completion_tokens": len(ids) * MAX_COMPLETION,
        "per_episode_max_steps": 1, "per_request_max_completion_tokens": MAX_COMPLETION,
        "max_wall_seconds": 86400, "max_failed_retries": 0,
        "max_total_usd": str(cap), "max_usd_per_episode": str(cap),
    }
    budget = ApprovedProviderBudget(stage_manifest, approval, PRICING,
                                    controller_commit=manifest["source_revision"], stage=stage,
                                    model=MODEL, base_url=BASE_URL)
    env = dotenv_values(env_file)
    api_key = env.get("OPENAI_API_KEY") or env.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("configured server .env has no supported DeepSeek API key")
    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=120.0, max_retries=0)
    raw_dir = output_dir / "raw" / stage
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = {skill: raw_dir / f"confirmatory_y1_{skill}_{stage}.jsonl" for skill in SKILLS}
    if any(path.exists() for path in raw_paths.values()):
        raise FileExistsError(f"{stage} raw receipts already exist; no retries or resume are permitted")
    started, models = time.time(), set()
    try:
        for identifier in ids:
            item = instances_by_id[identifier]
            _, user = prompt_for(item)
            budget.reserve(bounds[f"{stage}:{identifier}"]["input_token_bound"], MAX_COMPLETION)
            try:
                response = client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": user}],
                    temperature=0.0, max_tokens=MAX_COMPLETION,
                    extra_body={"thinking": {"type": "disabled"}},
                )
            except Exception as exc:
                budget.mark_unknown("provider_request_exception")
                append_jsonl(raw_paths[bounds[f"{stage}:{identifier}"]["skill"]], {
                    "instance_id": identifier, "dataset": "logicbench", "method": "oracle_direct",
                    "model": MODEL, "raw_output": "", "error": type(exc).__name__,
                    "provider_status": "UNKNOWN",
                })
                break
            u = response.usage
            input_tokens = getattr(u, "prompt_tokens", None) if u else None
            output_tokens = getattr(u, "completion_tokens", None) if u else None
            if type(input_tokens) is not int or type(output_tokens) is not int:
                budget.mark_unknown("provider_usage_missing")
                append_jsonl(raw_paths[bounds[f"{stage}:{identifier}"]["skill"]], {
                    "instance_id": identifier, "dataset": "logicbench", "method": "oracle_direct",
                    "model": MODEL, "raw_output": response.choices[0].message.content or "",
                    "provider_status": "UNKNOWN", "usage": None,
                })
                break
            details = getattr(u, "prompt_tokens_details", None)
            cache_hit = getattr(details, "cached_tokens", None) if details is not None else None
            if cache_hit is None:
                cache_hit = getattr(u, "prompt_cache_hit_tokens", None)
            cache_miss = input_tokens - cache_hit if type(cache_hit) is int else None
            budget.record_usage(input_tokens, output_tokens, cache_hit_tokens=cache_hit,
                                cache_miss_tokens=cache_miss)
            response_model = str(getattr(response, "model", MODEL))
            models.add(response_model)
            append_jsonl(raw_paths[bounds[f"{stage}:{identifier}"]["skill"]], {
                "instance_id": identifier, "dataset": "logicbench", "method": "oracle_direct",
                "model": MODEL, "raw_output": response.choices[0].message.content or "",
                "provider_status": "RECEIVED", "provider_model": response_model,
                "usage": {"input_tokens": input_tokens, "completion_tokens": output_tokens,
                          "cache_hit_tokens": cache_hit, "cache_miss_tokens": cache_miss},
            })
            if budget.stopped:
                break
    finally:
        client.close()
    snapshot = budget.as_dict()
    status = {
        "study_id": STUDY_ID, "stage": stage,
        "status": "COMPLETE" if not budget.stopped and budget.calls_reserved == len(ids) else "STOPPED_OR_UNKNOWN",
        "provider_models_returned": sorted(models), "provider_budget": snapshot,
        "wall_seconds": time.time() - started,
    }
    usage["stages"][stage] = {
        "status": status["status"], "calls": budget.calls_reserved,
        "estimated_usd": snapshot["estimated_usd_known"],
        "input_tokens": snapshot["input_tokens_known"],
        "completion_tokens": snapshot["completion_tokens_known"],
        "report": f"{stage}/RUN_REPORT.json",
    }
    atomic_json(output_dir / stage / "RUN_REPORT.json", status)
    atomic_json(usage_path, usage)
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "run"), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("calibration", "final"))
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(args.input_dir, args.output_dir)
    else:
        if not args.stage or not args.env_file:
            parser.error("--mode run requires --stage and --env-file")
        result = run_stage(args.input_dir, args.output_dir, args.stage, args.env_file)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
