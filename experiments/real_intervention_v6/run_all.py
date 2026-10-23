"""Single V6 entry: repair validation -> frozen pilot -> policies -> final -> export."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.real_intervention_v5.common import read_json, write_json, digest, append_jsonl, utc
from experiments.real_intervention_v5.run_all import preflight, environment
from experiments.real_intervention_v5.runtime import _terminate_group
from .audit_v5 import audit
from .benchmark_timing_contract import run_microfixture
from .gpu_attestation import idle_device, probe, snapshot
from .execution_adapter import run_arm, idle
from .study import freeze, measure, spec_for, old_root
from .policy_experiment import freeze_policies, _predict_row
from .report import finish


def checkpoint(root, phase, **fields):
    write_json(root / "PROGRESS.json", {"timestamp": utc(), "phase": phase, **fields})
    print(phase, fields, flush=True)


def wait_selected(root, uuid, seconds):
    end = time.monotonic() + seconds
    while True:
        row = snapshot(uuid)
        append_jsonl(root / "RESOURCE_SCANS.jsonl", row)
        if idle(row):
            return
        if time.monotonic() >= end:
            raise RuntimeError("RESOURCE_BLOCKED: selected GPU never became idle within frozen wait")
        checkpoint(root, "WAITING_FOR_SELECTED_IDLE_GPU", gpu_uuid=uuid)
        time.sleep(min(30, max(0, end - time.monotonic())))


def platform_validation(root, gpu, config):
    """Real sentinel, normalized timer and microfixture before cohort freeze."""
    # Every GPU action starts from an observed idle selected device.
    wait_selected(root, gpu["uuid"], config["idle_wait_s"])
    sentinel_path = root / "platform" / "sentinel" / "GPU_SENTINEL.json"
    attestation = read_json(sentinel_path) if sentinel_path.exists() else probe(sentinel_path.parent, gpu["uuid"])
    if attestation["worker_cuda_provenance"] != "VERIFIED":
        raise RuntimeError("CUDA_SENTINEL_PROVENANCE_UNVERIFIED")
    timer_path = root / "platform" / "PREFILL_REPRODUCER.json"
    if config["study"] == "B" and not timer_path.exists():
        old = old_root("B") / "shakedown/development/development-efficientformer-b1-h128-s0"
        old_spec = next(p for p in old.glob("spec-*.json") if read_json(p)["arm"] == "CACHE_PREFILL")
        wait_selected(root, gpu["uuid"], config["idle_wait_s"])
        env = {**os.environ, **environment(), "CUDA_VISIBLE_DEVICES": gpu["uuid"]}
        with (root / "platform" / "timer.stdout.log").open("xb") as out, (root / "platform" / "timer.stderr.log").open("xb") as err:
            process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.real_intervention_v6.timer_reproducer", "--spec", str(old_spec), "--output", str(timer_path), "--gpu-uuid", gpu["uuid"]], env=env, stdout=out, stderr=err, start_new_session=True)
            write_json(root / "platform" / "timer.started.json", {"pid": process.pid, "source_sha": preflight(), "timestamp": utc()})
            try:
                process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                _terminate_group(process, process.pid)
                raise RuntimeError("PREFILL_DIAGNOSTIC_TIMEOUT")
            if process.returncode:
                raise RuntimeError("PREFILL_DIAGNOSTIC_ERROR: inspect timer.stderr.log")
    if config["study"] == "B":
        reproduction = read_json(timer_path)
        groups = reproduction["post_warmup_crosscheck"]
        ratios = [host["per_call_seconds"] / bench["median_seconds_per_call"] for host, bench in zip(groups["host_groups"], groups["benchmark_groups"])]
        write_json(root / "platform" / "TIMER_UNIT_VALIDATION.json", {"ratios": ratios, "passed": all(.5 <= r <= 2 for r in ratios),
            "historical_manual_normalization": "V5 already divided10; cold/warm mutation is tested by per-call counters, not assumed from a400x ratio."})
        if not all(.5 <= r <= 2 for r in ratios):
            raise RuntimeError("NORMALIZED_STEADY_TIMER_STILL_INCONSISTENT")
    micro_path = root / "platform" / "TIMING_MICROFIXTURE.json"
    if not micro_path.exists():
        wait_selected(root, gpu["uuid"], config["idle_wait_s"])
        # Keep the controller CUDA-free: a separate short process owns the fixture.
        process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.real_intervention_v6.validation", "--micro-output", str(micro_path), "--gpu-uuid", gpu["uuid"]], env={**os.environ, **environment(), "CUDA_VISIBLE_DEVICES": gpu["uuid"]}, start_new_session=True)
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            _terminate_group(process, process.pid)
            raise RuntimeError("MICROFIXTURE_TIMEOUT")
        if process.returncode or not read_json(micro_path)["passed"]:
            raise RuntimeError("CPU_CUDA_TIMING_MICROFIXTURE_FAILED")


def engineering_validation(root, gpu, config, roster, plan, deadline):
    """CUDA timeout cleanup and independent-process cache shakedown."""
    pool = sorted([p for p in plan if p["split"] == "development"], key=lambda p: (p["session"], p["horizon"], p["family_id"]))
    fault_path = root / "FAULT_INJECTION.json"
    if not fault_path.exists():
        faults = {}
        cell = pool[0]
        for kind in ("hang_with_descendant", "exit_without_terminal"):
            spec = spec_for(roster[cell["family_id"]], cell, config, gpu, root, "EAGER", f"fault/{kind}", True)
            spec.update({"fault": kind, "timeout_s": 15})
            arm = run_arm(spec, root / "fault" / kind, idle_wait_s=config["idle_wait_s"])
            faults[kind] = {"passed": arm["primary_pair_status"] not in ("VALID", "EXPOSURE_UNCERTAIN") and not arm.get("owned_group_remaining") and arm.get("lifecycle_wall_s", 999) <= 20,
                            "result": arm}
        write_json(fault_path, {"passed": all(f["passed"] for f in faults.values()), "tests": faults,
            "cpu_fault_scope": "process-group descendants, no-retry recovery, identity/UUID/exposure contradictions, output metadata/numerical mismatch, cache atomic publication and units additionally covered by attested pytest"})
    if not read_json(fault_path)["passed"]:
        raise RuntimeError("CUDA_LIFECYCLE_FAULT_INJECTION_FAILED")
    shakedown_path = root / "ENGINEERING_SHAKEDOWN.json"
    if shakedown_path.exists():
        shakedown = read_json(shakedown_path)
    else:
        slots = []
        for cell in pool[:config["engineering_slots"]]:
            checkpoint(root, "ENGINEERING_SHAKEDOWN", slot=len(slots), pair=cell["pair_id"])
            result = measure(root, cell, roster[cell["family_id"]], config, gpu, engineering=True, deadline=deadline)
            eligible = result["timers_valid"] and all(r["analytically_comparable"] for r in result["records"])
            slots.append({"pair_id": cell["pair_id"], "complete_comparable": eligible, "result": result})
        shakedown = {"planned": config["engineering_slots"], "slots": slots, "comparable": sum(s["complete_comparable"] for s in slots),
                     "known_scientific_violations": sum(any(r["primary_pair_status"] == "SCIENTIFIC_UNSAFE" for r in s["result"]["records"]) for s in slots),
                     "exposure_grade": "EXPLORATORY_SHARED_GPU_ONLY", "fault_injection_passed": True}
        write_json(shakedown_path, shakedown)
    if shakedown["comparable"] < config["engineering_minimum_comparable"] or shakedown.get("known_scientific_violations", 0):
        raise RuntimeError("PLATFORM_UNQUALIFIED_SHAKEDOWN: frozen complete-comparable threshold not reached")


def unstarted(cell, config, reason):
    contrasts = [("LAYOUT", 1)] if config["study"] == "A" else [("COLD", 1)] + [(f"PERSISTENT_R{r}", r) for r in config["reported_actual_reuse_counts"]] + [(f"PROJECTED_R{r}", r) for r in config["projected_reuse_counts"]]
    return [{**cell, "context_id": f"{cell['family_id']}-h{cell['horizon']}-{c}", "contrast": c, "reuse_count": r,
             "primary_pair_status": "NOT_STARTED", "analytically_comparable": False, "s0_s": None, "s1_s": None, "log_benefit": None, "error": reason} for c, r in contrasts]


def run(root, config, diagnostics_root=None):
    sha = preflight()
    start_path = root / "RUN_START.json"
    if not start_path.exists():
        write_json(start_path, {"unix": time.time(), "timestamp": utc(), "source_sha": sha, "config": config})
    initial = read_json(start_path)
    if initial["source_sha"] != sha or initial["config"] != config:
        raise RuntimeError("resume refuses a different code/config identity")
    deadline = time.monotonic() + max(0, config["wall_cap_s"] - (time.time() - initial["unix"]))
    def elapsed():
        return time.time() - initial["unix"]
    rows, plan = [], []
    try:
        if not (root / "V6_ROOT_CAUSE_REGISTER.json").exists():
            if diagnostics_root and (diagnostics_root / "V6_ROOT_CAUSE_REGISTER.json").exists():
                write_json(root / "V6_ROOT_CAUSE_REGISTER.json", read_json(diagnostics_root / "V6_ROOT_CAUSE_REGISTER.json"))
            else:
                audit(root)
        if (root / "FROZEN_MANIFEST.json").exists():
            gpu = read_json(root / "FROZEN_MANIFEST.json")["gpu"]
        else:
            resource_end = time.monotonic() + 3600
            while True:
                gpu, inventory = idle_device()
                append_jsonl(root / "RESOURCE_SCANS.jsonl", {"timestamp": utc(), "gpus": inventory})
                if gpu:
                    break
                if time.monotonic() >= resource_end:
                    raise RuntimeError("RESOURCE_BLOCKED: all observed laboratory GPUs occupied throughout one-hour admission window")
                checkpoint(root, "WAITING_FOR_IDLE_GPU", occupied=len(inventory))
                time.sleep(min(30, max(0, resource_end - time.monotonic())))
        checkpoint(root, "PLATFORM_AND_TIMER_VALIDATION", source_sha=sha, gpu_uuid=gpu["uuid"])
        platform_validation(root, gpu, config)
        manifest, roster, plan = freeze(root, config, gpu)
        engineering_validation(root, gpu, config, roster, plan, deadline)
        for split in ("development", "calibration", "final"):
            if split == "final":
                contrasts = sorted({r["contrast"] for r in rows if not r["contrast"].startswith("PROJECTED")})
                freezes, gates = {}, {}
                for contrast in contrasts:
                    dev = [r for r in rows if r["split"] == "development" and r["contrast"] == contrast]
                    cal = [r for r in rows if r["split"] == "calibration" and r["contrast"] == contrast]
                    reuse = next(r["reuse_count"] for r in dev)
                    final_contexts = {}
                    for p in plan:
                        if p["split"] == "final":
                            context = {k: p[k] for k in ("family_id", "horizon", "features")}
                            context.update({"context_id": f"{p['family_id']}-h{p['horizon']}-{contrast}", "reuse_count": reuse})
                            final_contexts[context["context_id"]] = context
                    freeze_record = freeze_policies(dev, cal, list(final_contexts.values()))
                    overhead = {}
                    for name, policy in freeze_record["policies"].items():
                        durations = []
                        for context in final_contexts.values():
                            begin = time.perf_counter()
                            for _ in range(100):
                                _predict_row(context, policy)
                            durations.append((time.perf_counter() - begin) / 100)
                        overhead[name] = sum(durations) / max(1, len(durations))
                    freeze_record["routing_overhead_s"] = overhead
                    freezes[contrast] = freeze_record
                    gates[contrast] = (len({r["family_id"] for r in dev if r["analytically_comparable"]}) >= config["minimum_observed_dev_families"]
                                       and len({r["family_id"] for r in cal if r["analytically_comparable"]}) >= config["minimum_observed_cal_families"])
                write_json(root / "POLICY_FREEZE.json", {"by_contrast": freezes, "timestamp": utc(), "source_sha": sha})
                allowed = bool(gates) and all(gates.values()) and manifest["cohort"]["split_counts"]["final"] >= config["minimum_registered_final_families"]
                if not allowed:
                    rows += [r for cell in plan if cell["split"] == "final" for r in unstarted(cell, config, "PREFINAL_PILOT_GATE_NOT_MET")]
                    return finish(root, "UNDERPOWERED_PREFINAL_GATE", rows, elapsed_s=elapsed(), blocker="Frozen LevelE/P observed development/calibration family minimum not met; canonical certificate was not the gate")
                write_json(root / "FINAL_RELEASE.json", {"study_id": config["study_id"], "manifest_sha256": digest(read_json(root / "FROZEN_MANIFEST.json")),
                    "policy_sha256": digest(read_json(root / "POLICY_FREEZE.json")), "timestamp": utc(), "scope": "EXPLORATORY_INDEPENDENT_PILOT", "original_acre_certified": False})
            for cell in [p for p in plan if p["split"] == split]:
                checkpoint(root, f"MEASURE_{split.upper()}", pair=cell["pair_id"], recorded_rows=len(rows))
                result = measure(root, cell, roster[cell["family_id"]], config, gpu, deadline=deadline)
                rows += result["records"]
                write_json(root / "MEASURED_ROWS_CHECKPOINT.json", rows)
                used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
                if used > config["disk_cap_bytes"]:
                    raise RuntimeError("V6_STORAGE_CAP_REACHED")
        complete = rows and all(r["analytically_comparable"] for r in rows if not r["contrast"].startswith("PROJECTED"))
        return finish(root, "PILOT_COMPLETE" if complete else "PILOT_CLOSED_WITH_MISSING_OUTCOMES", rows, elapsed_s=elapsed())
    except Exception as exc:
        write_json(root / "ERROR_TRACE.json", {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        existing = {r["pair_id"] for r in rows}
        rows += [r for cell in plan if cell["pair_id"] not in existing for r in unstarted(cell, config, str(exc))]
        return finish(root, "ENGINEERING_OR_RESOURCE_NO_GO", rows, elapsed_s=elapsed(), blocker=str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path)
    parser.add_argument("--resume-safe", action="store_true")
    args = parser.parse_args()
    preflight()
    root = args.results_root.resolve()
    if not root.is_relative_to(Path("/home/workspace/lrh/RESULTS")) or "evospe-v6" not in root.name:
        raise ValueError("a new V6 RESULTS root is required")
    if root.exists() and not args.resume_safe:
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "RUN_EXIT.json").exists():
            print(read_json(root / "RUN_EXIT.json"))
            return
        result = run(root, read_json(args.config), args.diagnostics_root)
        print(json.dumps({k: result[k] for k in ("status", "scientific_status", "analytically_comparable_rows", "independent_measured_families")}))
        if result["status"] == "ENGINEERING_OR_RESOURCE_NO_GO":
            raise SystemExit(3)


if __name__ == "__main__":
    main()
