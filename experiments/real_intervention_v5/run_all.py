"""Freeze once, qualify blind, shakedown, then run development/calibration/final.

Both studies have independent roots. Started identities cannot be replayed;
ACRE infeasibility never gates a separately qualified independent final.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import importlib.metadata
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import append_jsonl, digest, read_json, utc, write_csv, write_json
from .registry import assert_split_integrity, build_registry, TORCHBENCH, OVERLAY

PYTHON = "/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python"
CANONICAL = Path("/home/workspace/lrh/CODE/scientific-performance-engineering")
V4_ROOT = Path("/home/workspace/lrh/RESULTS/evospe-rt-v4/EVOSPE_REAL_INTERVENTION_V4_CHANNELS_LAST_20261020_CONTEXT_PASS_THROUGH_V2")


def preflight():
    if Path.cwd().resolve() != CANONICAL or Path(sys.executable).resolve() != Path(PYTHON).resolve():
        raise RuntimeError("V5 must run from canonical server repository using equivcompiler")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("freeze requires a clean source worktree")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def gpu_inventory():
    import pynvml as n
    n.nvmlInit()
    try:
        rows = []
        for i in range(n.nvmlDeviceGetCount()):
            h = n.nvmlDeviceGetHandleByIndex(i)
            uid = n.nvmlDeviceGetUUID(h)
            name, driver = n.nvmlDeviceGetName(h), n.nvmlSystemGetDriverVersion()
            rows.append({"index": i, "uuid": uid.decode() if isinstance(uid, bytes) else uid,
                         "name": name.decode() if isinstance(name, bytes) else name, "memory_used_mib": n.nvmlDeviceGetMemoryInfo(h).used / 2**20,
                         "utilization_percent": n.nvmlDeviceGetUtilizationRates(h).gpu,
                         "compute_pids": [p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(h)],
                         "driver": driver.decode() if isinstance(driver, bytes) else driver, "timestamp": utc()})
        return rows
    finally:
        n.nvmlShutdown()


def select_gpu(root, wait_s, required_uuid=None):
    deadline = time.monotonic() + wait_s
    while True:
        inventory = gpu_inventory()
        append_jsonl(root / "GPU_RESOURCE_SCANS.jsonl", {"timestamp": utc(), "gpus": inventory})
        idle = [g for g in inventory if (required_uuid is None or g["uuid"] == required_uuid) and not g["compute_pids"] and g["memory_used_mib"] <= 512 and g["utilization_percent"] == 0]
        if idle:
            selected = idle[0]
            write_json(root / "GPU_SELECTION.json", selected)
            return selected
        write_json(root / "PROGRESS.json", {"phase": "WAITING_FOR_IDLE_GPU", "timestamp": utc(), "gpus": inventory})
        if time.monotonic() >= deadline:
            return None
        time.sleep(min(30, max(0, deadline - time.monotonic())))


def environment():
    return {"PYTHONPATH": os.pathsep.join([str(CANONICAL), OVERLAY]), "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1"}


def make_spec(row, config, gpu, root, *, batch, horizon, mode, arm, scope, qualification=None):
    return {**row, "study_id": config["study_id"], "study": config["study"], "mode": mode, "arm": arm,
            "seed": config["seed"], "batch": batch, "horizon": horizon, "gpu_uuid": gpu["uuid"],
            "torchbench_root": TORCHBENCH, "correctness": config["correctness"],
            "timeout_s": config["session_timeout_seconds"], "qualification": qualification,
            "results_root": str(root),
            "source_sha": read_json(root / "FROZEN_MANIFEST.json")["source_sha"] if (root / "FROZEN_MANIFEST.json").exists() else None,
            "cache_path": str(root / "cache" / row["split"] / scope / arm),
            "reference_path": str(root / "references" / f"{row['family_id']}-b{batch}.pt"),
            "final_release_path": str(root / "FINAL_RELEASE.json")}


def execute(specs, directory, config, gpu):
    from .runtime import run_session
    directory.mkdir(parents=True, exist_ok=True)
    terminal = directory / "terminal.json"
    if terminal.exists():
        return read_json(terminal)
    if (directory / "intent.json").exists():
        # Preserve raw partial receipts; add an explicit recovery record only.
        result = {"execution_status": "UNKNOWN", "scientific_status": "UNVERIFIED", "platform_status": "UNCERTAIN",
                  "exposure_status": "UNKNOWN", "analysis_status": "MISSING", "first_failing_gate": "INTERRUPTED_SESSION",
                  "arms": [], "error_code": "STARTED_WITHOUT_TERMINAL_NO_RETRY"}
        write_json(directory / "recovery.json", result)
        return result
    study_root = Path(specs[0]["results_root"]) if specs[0].get("results_root") else None
    if study_root and sum(p.stat().st_size for p in study_root.rglob("*") if p.is_file()) >= config["disk_cap_bytes"]:
        return {"execution_status": "NOT_STARTED", "scientific_status": "UNVERIFIED", "platform_status": "UNCERTAIN", "exposure_status": "UNKNOWN", "analysis_status": "MISSING", "first_failing_gate": "STORAGE_CAP", "error_code": "STORAGE_CAP", "arms": []}
    commands = []
    for i, spec in enumerate(specs):
        cache = Path(spec["cache_path"])
        claim = cache.parent / (cache.name + ".claim.json")
        identity = {"study_id": spec["study_id"], "split": spec["split"], "family_id": spec["family_id"],
                    "batch": spec["batch"], "shape": spec["shape"], "source_version": spec["source_version"],
                    "scope": str(directory), "precision": config["precision"], "gpu_uuid": gpu["uuid"]}
        identity["source_sha"] = spec.get("source_sha")
        if claim.exists() and read_json(claim) != identity:
            raise ValueError("compiler cache belongs to a different experimental scope")
        if not claim.exists():
            write_json(claim, identity)
        spec_path = directory / f"spec-{i}.json"
        write_json(spec_path, spec)
        commands.append([PYTHON, "-u", "-m", "experiments.real_intervention_v5.worker", "--spec", str(spec_path),
                         "--output", "{output}", "--ack-path", "{ack}"])
    return run_session(commands, directory, gpu["uuid"], config["session_timeout_seconds"], environment())


def qualify(registry, config, gpu, root, deadline):
    rows = []
    for row in registry:
        if not row["representative"]:
            rows.append({**row, "eligibility": "ALIAS_NOT_EXTRA_FAMILY", "baseline_valid": False, "candidate_correct": False})
            continue
        checks = []
        for batch in config["batches"]:
            if not row["available"] or time.monotonic() >= deadline:
                checks.append({"batch": batch, "error_code": "SOURCE_NOT_AVAILABLE" if not row["available"] else "RESOURCE_CAP"})
                continue
            directory = root / "qualification" / row["split"] / row["family_id"] / f"b{batch}"
            spec = make_spec(row, config, gpu, root, batch=batch, horizon=1, mode="qualification", arm="REFERENCE",
                             scope=f"qualification-{row['family_id']}-b{batch}")
            repeated_spec = {**spec, "arm": "REFERENCE_REPEAT", "reference_path": str(root / "references" / f"{row['family_id']}-b{batch}-repeat.pt"),
                             "cache_path": spec["cache_path"] + "-repeat"}
            result = execute([spec, repeated_spec], directory, config, gpu)
            payload = result.get("arms", [{}])[0].get("worker_result", {}) if result.get("arms") else {}
            repeated = result["arms"][1].get("worker_result", {}) if len(result.get("arms", [])) == 2 else {}
            identity_repeatable = bool(payload) and bool(repeated) and all(payload.get(k) == repeated.get(k) for k in ("weights_sha256", "inputs_sha256", "reference_sha256"))
            repeat_baseline = repeated.get("baseline", {}).get("passed") is True
            payload["construction_repeatability"] = {"passed": identity_repeatable and repeat_baseline,
                                                       "reference_repeat_passed": repeat_baseline,
                                                       "same_weights_inputs_bf16_output": identity_repeatable}
            if payload.get("candidate", {}).get("passed") is True and not repeated.get("candidate", {}).get("passed"):
                payload["candidate"] = {"passed": False, "reason": "candidate_correctness_failed_independent_reconstruction"}
            if config["study"] == "B" and payload.get("baseline", {}).get("passed") and payload["construction_repeatability"]["passed"]:
                compile_specs = []
                for arm in ("COMPILE_COLD", "CACHE_PREFILL", "COMPILE_PERSISTENT_CACHE"):
                    s = make_spec(row, config, gpu, root, batch=batch, horizon=1, mode="compile_qualification", arm=arm,
                                  scope=f"compile-qualification-{row['family_id']}-b{batch}", qualification=payload)
                    if arm != "COMPILE_COLD":
                        s["cache_path"] = str(root / "cache" / row["split"] / f"compile-qualification-{row['family_id']}-b{batch}" / "persistent")
                    compile_specs.append(s)
                compilation = execute(compile_specs, directory / "compile", config, gpu)
                cp = [a.get("worker_result", {}) for a in compilation.get("arms", [])]
                compile_pass = len(cp) == 3 and all(a.get("scientific_status") == "PASS" for a in cp) and cp[-1].get("cache", {}).get("persistent_hit_proven")
                payload["candidate"] = {"passed": bool(compile_pass), "compile_qualification": cp}
                if compilation["analysis_status"] != "VALID_PAIRED":
                    result["analysis_status"] = compilation["analysis_status"]
            checks.append({"batch": batch, "payload": payload, "session_status": result["analysis_status"]})
        baseline = bool(checks) and all(c.get("payload", {}).get("baseline", {}).get("passed") is True and c.get("payload", {}).get("construction_repeatability", {}).get("passed") is True for c in checks)
        correct = baseline and all(c["payload"].get("candidate", {}).get("passed") is True for c in checks)
        exposure = correct and all(c["session_status"] == "VALID_PAIRED" for c in checks)
        if config["study"] == "A":
            exposure = exposure and all(c["payload"].get("kernel_exposure", {}).get("active") and c["payload"]["kernel_exposure"].get("cuda_kernel_events", 0) > 0 for c in checks)
        qualified = {**row, "baseline_valid": baseline, "candidate_correct": correct, "kernel_exposure_valid": bool(exposure),
                     "eligibility": "ELIGIBLE" if exposure else "QUALIFICATION_FAILED", "checks": checks}
        rows.append(qualified)
        write_json(root / "QUALIFICATION_PROGRESS.json", {"timestamp": utc(), "families_processed": len(rows), "latest": row["family_id"]})
    return rows


def plan_pairs(qualified, config):
    plan = []
    for split in ("development", "calibration", "final"):
        for row in qualified:
            if row["split"] != split or row["eligibility"] != "ELIGIBLE": continue
            for batch in config["batches"]:
                for h in config["horizons"]:
                    for session in range(config["sessions_per_context"]):
                        pair_id = f"{split}-{row['family_id']}-b{batch}-h{h}-s{session}"
                        plan.append({"pair_id": pair_id, "family_id": row["family_id"], "split": split, "batch": batch,
                                     "horizon": h, "session": session, "order": "AB" if int(digest([config["seed"], pair_id])[:8], 16) % 2 == 0 else "BA",
                                     "contrast": "LAYOUT" if config["study"] == "A" else None,
                                     "features": next(c["payload"]["features"] for c in row["checks"] if c["batch"] == batch)})
    return plan


def missing_pair(cell, reason):
    return {**cell, "execution_status": "NOT_STARTED", "scientific_status": "UNVERIFIED", "platform_status": "UNCERTAIN",
            "exposure_status": "UNKNOWN", "analysis_status": "MISSING", "first_failing_gate": reason, "error_code": reason,
            "timestamp": utc(), "s0_s": None, "s1_s": None, "log_benefit": None}


def missing_measurements(cell, config, reason):
    contrasts = ["LAYOUT"] if config["study"] == "A" else ["COMPILE_COLD"] + [f"PERSISTENT_R{r}" for r in config["reuse_counts"]]
    return [{**missing_pair(cell, reason), "contrast": c,
             "pair_id": cell["pair_id"] if config["study"] == "A" else cell["pair_id"] + "-" + c,
             "session_receipt": str(Path("pairs") / cell["split"] / cell["pair_id"] / "terminal.json")}
            for c in contrasts]


def measure(cell, row, config, gpu, root, *, engineering=False):
    scope = ("engineering-" if engineering else "") + cell["pair_id"]
    q = next(c["payload"] for c in row["checks"] if c["batch"] == cell["batch"])
    arms = ["S0", "S1"] if config["study"] == "A" else ["EAGER", "COMPILE_COLD", "CACHE_PREFILL", "COMPILE_PERSISTENT_CACHE"]
    if cell["order"] == "BA":
        arms = list(reversed(arms)) if config["study"] == "A" else ["CACHE_PREFILL", "COMPILE_PERSISTENT_CACHE", "COMPILE_COLD", "EAGER"]
    specs = []
    for arm in arms:
        spec = make_spec(row, config, gpu, root, batch=cell["batch"], horizon=1 if arm == "CACHE_PREFILL" else cell["horizon"],
                         mode="engineering" if engineering else "measurement", arm=arm, scope=scope, qualification=q)
        if engineering:
            # Engineering is excluded from efficacy and may use any prequalified split.
            spec["split"] = "engineering"
        if arm in ("CACHE_PREFILL", "COMPILE_PERSISTENT_CACHE"):
            spec["cache_path"] = str(root / "cache" / spec["split"] / scope / "persistent")
        specs.append(spec)
    directory = root / ("shakedown" if engineering else "pairs") / cell["split"] / cell["pair_id"]
    result = execute(specs, directory, config, gpu)
    payloads = {s["arm"]: a.get("worker_result", {}) for s, a in zip(specs, result.get("arms", []))}
    base = {**cell, "study_id": config["study_id"], "source_sha": specs[0]["source_sha"], **{k: result.get(k) for k in ("execution_status", "scientific_status", "platform_status", "exposure_status", "analysis_status", "first_failing_gate", "error_code", "raw_digest")},
            "timestamp": utc(), "session_receipt": str(directory.relative_to(root) / "terminal.json"),
            "arm_payloads": payloads, "parent_wall_s": result.get("timings", {}).get("session_wall_seconds")}
    if config["study"] == "A":
        variants = [("LAYOUT", "S1", None)]
        reference_arm = "S0"
    else:
        variants = [("COMPILE_COLD", "COMPILE_COLD", None)] + [(f"PERSISTENT_R{r}", "COMPILE_PERSISTENT_CACHE", r) for r in config["reuse_counts"]]
        reference_arm = "EAGER"
    rows = []
    for contrast, candidate_arm, reuse in variants:
        item = {**base, "contrast": contrast, "reuse_count": reuse, "s0_s": None, "s1_s": None, "log_benefit": None}
        if config["study"] == "B":
            item["pair_id"] = cell["pair_id"] + "-" + contrast
        if item["analysis_status"] == "VALID_PAIRED":
            s0 = payloads[reference_arm]["timing"]["total_cold_job_s"]
            s1 = payloads[candidate_arm]["timing"]["total_cold_job_s"]
            if reuse:
                if not payloads[candidate_arm]["cache"]["persistent_hit_proven"]:
                    item.update({"analysis_status": "EXCLUDED_PRESPECIFIED", "first_failing_gate": "PERSISTENT_CACHE_HIT_UNPROVEN"})
                s1 += payloads["CACHE_PREFILL"]["timing"]["total_cold_job_s"] / reuse
            item.update({"s0_s": s0, "s1_s": s1, "log_benefit": math.log(s0 / s1) if item["analysis_status"] == "VALID_PAIRED" else None,
                         "net_saved_s": s0 - s1, "steady_s0_s": payloads[reference_arm]["timing"]["steady_state_per_forward_s"],
                         "steady_s1_s": payloads[candidate_arm]["timing"]["steady_state_per_forward_s"]})
        rows.append(item)
    return rows


def checkpoint(root, phase, pairs):
    write_json(root / "PROGRESS.json", {"timestamp": utc(), "phase": phase, "recorded_pairs": len(pairs),
                                     "analysis_counts": dict(Counter(p["analysis_status"] for p in pairs))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["full"], default="full")
    parser.add_argument("--resume-safe", action="store_true")
    args = parser.parse_args()
    sha = preflight()
    config = read_json(args.study_config)
    root = args.results_root.resolve()
    if not str(root).startswith("/home/workspace/lrh/RESULTS/"):
        raise ValueError("V5 results must have their own server RESULTS identity")
    if root.exists() and not args.resume_safe:
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "RUN_COMPLETE.json").exists():
            if not (root / "BUNDLE_VALIDATION.json").exists() or read_json(root / "BUNDLE_VALIDATION.json").get("status") != "PASS":
                raise RuntimeError("terminal study is missing its validated evidence bundle; do not replay experiments")
            print(read_json(root / "RUN_COMPLETE.json")); return
        run(root, config, sha, args.resume_safe)


def run(root, config, sha, resume):
    from .certificate_feasibility import feasibility
    from .old_studies_readonly_audit import audit_v4
    from .analysis import freeze_policies
    from .report import finish
    for prior_name in ("REGISTRATION_FREEZE.json", "FROZEN_MANIFEST.json"):
        if (root / prior_name).exists():
            prior = read_json(root / prior_name)
            if prior.get("source_sha") != sha or prior.get("config") != config:
                raise RuntimeError("resume source/config identity mismatch; existing study remains untouched")
    if not (root / "RUN_START.json").exists():
        write_json(root / "RUN_START.json", {"timestamp": utc(), "unix": time.time(), "source_sha": sha})
    already_elapsed = time.time() - read_json(root / "RUN_START.json")["unix"]
    start = time.monotonic() - already_elapsed
    deadline = start + config["wall_cap_seconds"]
    registry, versions, tb_sha = build_registry(config["seed"])
    splits = assert_split_integrity(registry)
    write_json(root / "V4_READONLY_AUDIT.json", audit_v4(V4_ROOT))
    write_csv(root / "REGISTERED_FAMILIES.csv", registry)
    write_csv(root / "FAMILY_LINEAGE_MAP.csv", [{"family_id": r["family_id"], "candidate_id": r["candidate_id"], "basis": r["lineage_basis"], "split": r["split"]} for r in registry])
    (root / "registry").mkdir(exist_ok=True)
    with (root / "registry" / "REGISTERED_UNIVERSE.jsonl").open("w") as f:
        import json
        for row in registry: f.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(root / "registry" / "SPLIT_POLICY.json", {"seed": config["seed"], "algorithm": "sort unique lineage by SHA256([seed,lineage]), round-robin dev/cal/final before qualification", "splits": splits})
    write_json(root / "registry" / "PREEXEC_FEATURES.schema.json", {"allowed": ["batch", "horizon", "shape", "parameters", "layers", "conv2d", "source", "family_id"], "prohibited": ["timing", "correctness", "support_label", "candidate_outcome"]})
    cal = sum(s == "calibration" for s in splits.values())
    feasible = feasibility(cal)
    feasible["primary_effect_final_feasible"] = None
    feasible["effect_precision_sensitivity"] = [{"assumed_family_sd": sd, "target_log_effect": effect,
                                               "approximate_groups_80pct_power_two_sided_5pct": math.ceil(((1.96 + .8416) * sd / effect) ** 2)}
                                              for sd in (.05, .1, .2) for effect in (.02, .05, .1)]
    power_design = feasible["effect_precision_sensitivity"]
    registered_final_n = sum(s == "final" for s in splits.values())
    for scenario in power_design:
        shift = math.sqrt(registered_final_n) * scenario["target_log_effect"] / scenario["assumed_family_sd"]
        cdf = lambda z: (1 + math.erf(z / math.sqrt(2))) / 2
        scenario["registered_final_families_optimistic_upper_bound"] = registered_final_n
        scenario["approximate_power_at_registered_final_n_under_assumptions"] = 1 - cdf(1.96 - shift) + cdf(-1.96 - shift)
    feasible["primary_effect_power_status"] = "Sensitivity under stated family-SD assumptions; no assured 80 percent power"
    if config["canonical_p_min"] != feasible["canonical_certificate_lower_bound"]["p_min"] or config["delta_total"] != .05:
        raise ValueError("V5 cannot relabel altered canonical certificate thresholds as original ACRE")
    write_json(root / "CERTIFICATE_FEASIBILITY.json", feasible)
    write_json(root / "REGISTRATION_FREEZE.json", {"source_sha": sha, "registry_sha256": digest(registry), "config": config, "timestamp": utc(), "versions": versions, "torchbench_commit": tb_sha})
    print(f"registered {len(registry)} candidates / {len(splits)} lineages; canonical necessary groups={feasible['required_independent_groups_lower_bound']}; cal={cal}", flush=True)
    if (root / "FROZEN_MANIFEST.json").exists():
        manifest = read_json(root / "FROZEN_MANIFEST.json")
        if manifest["source_sha"] != sha or manifest["config"] != config or manifest["registry_sha256"] != digest(registry):
            raise RuntimeError("resume identity mismatch")
        gpu = manifest["gpu"]
    else:
        gpu = select_gpu(root, config["resource_wait_seconds"], config.get("required_gpu_uuid"))
        manifest = {"study_id": config["study_id"], "source_sha": sha, "config": config, "registry_sha256": digest(registry),
                    "prequalification_certificate_and_power_design": feasible,
                    "registry": registry, "versions": versions, "torchbench_commit": tb_sha, "gpu": gpu,
                    "timestamp": utc(), "python_environment": "equivcompiler/bin/python (absolute canonical path enforced before execution)", "precompute": "B cache prefill measured and amortized; A no precompute",
                    "resource_topology": {"gpu_workers": 1, "simultaneous_seeds": 1, "cpu_threads": 1, "dataloader_workers": 0,
                                          "reservation": "Per-session advisory flock; no lab-wide scheduler found; shared contention detection required",
                                          "cpu_affinity": sorted(os.sched_getaffinity(0)), "cpu_numa_binding": "unbound_observed_shared_host"},
                    "lifecycle": {"startup": "reported separately", "construction": "reported separately", "prepare_and_H_forwards": "primary",
                                  "steady_probe": "secondary only", "evaluation": "numerical only; no task accuracy", "teardown": "parent wall", "failure_retry": "none"}}
        test_receipt = root.parent / "PRE_EXPERIMENT_TESTS.json"
        if test_receipt.exists():
            tests = read_json(test_receipt)
            if tests.get("source_sha") != sha or tests.get("exit_code") != 0:
                raise RuntimeError("pre-experiment test receipt does not attest this source")
            manifest["pre_experiment_tests"] = tests
        write_json(root / "FROZEN_MANIFEST.json", manifest)
    (root / "references").mkdir(exist_ok=True)
    pairs, qualified = [], []
    if gpu is None:
        finish(root, manifest, registry, qualified, pairs, "PLATFORM_BLOCKED_NO_IDLE_GPU", feasible, {"required": 12, "actual": 0}, time.monotonic() - start)
        return
    qualified = qualify(registry, config, gpu, root, deadline)
    write_json(root / "QUALIFICATION_SNAPSHOT.json", qualified)
    write_csv(root / "QUALIFICATION_MATRIX.csv", [{k: v for k, v in r.items() if k != "checks"} for r in qualified])
    by_split = Counter(r["split"] for r in qualified if r["eligibility"] == "ELIGIBLE")
    feasible = feasibility(cal, by_split["calibration"])
    feasible["effect_precision_sensitivity"] = power_design
    feasible["primary_effect_power_status"] = "Independent low-power estimate; no assured 80 percent power; report achieved family interval"
    feasible["primary_effect_final_feasible"] = all(by_split[s] >= config["minimum_pilot_families_per_split"] for s in ("development", "calibration", "final"))
    write_json(root / "CERTIFICATE_FEASIBILITY.json", feasible)
    plan = plan_pairs(qualified, config)
    write_json(root / "PAIR_PLAN.json", plan)
    index = {r["family_id"]: r for r in qualified if r["eligibility"] == "ELIGIBLE"}
    engineering = []
    engineering_pool = [p for p in plan if p["split"] == "development"]
    # Round-robin families so shakedown is not twelve repeats of one model.
    engineering_pool.sort(key=lambda p: (p["session"], p["horizon"], p["batch"], p["family_id"]))
    for cell in engineering_pool[:config["engineering_pairs"]]:
        if time.monotonic() >= deadline: break
        measured = measure(cell, index[cell["family_id"]], config, gpu, root, engineering=True)
        timers_valid = all(a.get("timer_crosscheck", {}).get("passed") is True for a in measured[0].get("arm_payloads", {}).values())
        engineering.append({"pair_id": cell["pair_id"], "valid": all(p["analysis_status"] == "VALID_PAIRED" for p in measured) and timers_valid, "timers_valid": timers_valid, "records": measured})
    shake = {"required": config["engineering_pairs"], "actual": len(engineering), "valid": sum(p["valid"] for p in engineering), "pairs": engineering}
    shake["known_scientific_violations"] = sum(any(r["scientific_status"] == "NUMERICAL_UNSAFE" for r in p["records"]) for p in engineering)
    shake["known_lifecycle_violations"] = sum(any(str(r.get("error_code") or "").startswith("WORKER_") for r in p["records"]) for p in engineering)
    write_json(root / "ENGINEERING_SHAKEDOWN.json", shake)
    if shake["valid"] < config["engineering_minimum_valid"] or shake["known_scientific_violations"] or shake["known_lifecycle_violations"]:
        finish(root, manifest, registry, qualified, [r for p in plan for r in missing_measurements(p, config, "ENGINEERING_SHAKEDOWN_FAILED")], "PLATFORM_UNQUALIFIED_SHAKEDOWN" if shake["actual"] == 12 else "UNDERPOWERED_QUALIFICATION_SHAKEDOWN_INCOMPLETE", feasible, shake, time.monotonic() - start)
        return
    from .fault_injection import run_fault_injection
    faults = run_fault_injection(root, config, gpu, engineering_pool[0], index[engineering_pool[0]["family_id"]])
    write_json(root / "FAULT_INJECTION.json", faults)
    if not faults["passed"]:
        finish(root, manifest, registry, qualified, [r for p in plan for r in missing_measurements(p, config, "FAULT_INJECTION_FAILED")], "PLATFORM_UNQUALIFIED_FAULT_INJECTION", feasible, shake, time.monotonic() - start)
        return
    final_allowed = feasible["primary_effect_final_feasible"]
    for split in ("development", "calibration", "final"):
        if split == "final":
            final_allowed = final_allowed and all(len({p["family_id"] for p in pairs if p["split"] == s and p["analysis_status"] == "VALID_PAIRED"}) >= config["minimum_pilot_families_per_split"] for s in ("development", "calibration"))
            if config["study"] == "A":
                freeze = freeze_policies([p for p in pairs if p["split"] == "development"], [p for p in pairs if p["split"] == "calibration"], plan)
            else:
                contrasts = ["COMPILE_COLD"] + [f"PERSISTENT_R{r}" for r in config["reuse_counts"]]
                freeze = {"by_contrast": {c: freeze_policies([p for p in pairs if p["split"] == "development" and p.get("contrast") == c], [p for p in pairs if p["split"] == "calibration" and p.get("contrast") == c], [{**p, "contrast": c, "pair_id": p["pair_id"] + "-" + c} for p in plan]) for c in contrasts}}
            write_json(root / "POLICY_FREEZE.json", freeze)
            if final_allowed:
                write_json(root / "FINAL_RELEASE.json", {"study_id": config["study_id"], "timestamp": utc(), "policy_sha256": digest(freeze), "manifest_sha256": digest(manifest)})
        for cell in (p for p in plan if p["split"] == split):
            if split == "final" and not final_allowed:
                measured = missing_measurements(cell, config, "MINIMUM_INDEPENDENT_PILOT_FAMILIES_NOT_MET")
            elif time.monotonic() - start >= config["wall_cap_seconds"] or shutil.disk_usage(root).free < 2**30:
                measured = missing_measurements(cell, config, "RESOURCE_CAP")
            else:
                measured = measure(cell, index[cell["family_id"]], config, gpu, root)
            pairs.extend(measured)
            for item in measured: append_jsonl(root / "PAIRS.progress.jsonl", item)
            checkpoint(root, split, pairs)
    status = "FULL_FINAL_COMPLETE" if final_allowed and all(p["execution_status"] == "COMPLETE" for p in pairs) else "UNDERPOWERED_INDEPENDENT_PILOT_CLOSEOUT" if final_allowed else "UNDERPOWERED_FINAL_UNOPENED"
    finish(root, manifest, registry, qualified, pairs, status, feasible, shake, time.monotonic() - start)


if __name__ == "__main__":
    main()
