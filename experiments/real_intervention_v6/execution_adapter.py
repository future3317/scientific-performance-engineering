"""Bounded V6 process lifecycle using the existing process-group cleanup."""
from __future__ import annotations

import fcntl
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from experiments.real_intervention_v5.common import read_json, write_json, digest, append_jsonl, utc, file_digest
from experiments.real_intervention_v5.runtime import _terminate_group, _reservation_path
from experiments.real_intervention_v4.process_supervisor import _enable_child_subreaper, _process_group_pids
from experiments.real_intervention_v5.run_all import environment
from .gpu_attestation import classify, proc_identity, snapshot, idle_snapshot as idle
from .benchmark_timing_contract import TimingObservation


def primary_status(row):
    if row["execution_status"] == "NOT_STARTED":
        return "NOT_STARTED"
    if row.get("scientific_correctness") == "FAIL":
        return "SCIENTIFIC_UNSAFE"
    if row["execution_status"] == "UNKNOWN":
        return "UNKNOWN"
    if (row["execution_status"] != "COMPLETE" or row.get("worker_cuda_provenance") != "VERIFIED"
            or row.get("workload_identity") != "MATCH" or row.get("measurement_contract") != "COMPARABLE"
            or row.get("scientific_correctness") != "PASS"):
        return "INFRA_INVALID"
    if row.get("co_tenancy_exposure") == "OBSERVED_FOREIGN":
        return "INFRA_INVALID"
    return "VALID" if row["co_tenancy_exposure"] == "VERIFIED_QUIET" else "EXPOSURE_UNCERTAIN"


def run_arm(spec, folder, *, published_cache=None, idle_wait_s=1800, campaign_deadline=None):
    folder = Path(folder)
    terminal = folder / "terminal.json"
    spec_identity = digest({k: v for k, v in spec.items() if k != "timeout_s"})
    if terminal.exists():
        receipt = read_json(terminal)
        if receipt.get("spec_identity") != spec_identity:
            raise RuntimeError("existing arm code/workload/cache/final identity differs; no replay")
        return receipt
    folder.mkdir(parents=True, exist_ok=True)
    intent = folder / "intent.json"
    empty = {"execution_status": "NOT_STARTED", "scientific_correctness": "UNKNOWN", "workload_identity": "UNKNOWN",
             "worker_cuda_provenance": "UNKNOWN", "co_tenancy_exposure": "UNKNOWN", "measurement_contract": "UNKNOWN"}
    if intent.exists():
        result = {**empty, "execution_status": "UNKNOWN", "primary_pair_status": "UNKNOWN", "error": "STARTED_IDENTITY_NO_RETRY"}
        write_json(folder / "recovery.json", result)
        return result
    lock_path = _reservation_path(spec["gpu_uuid"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        wait_deadline = min(time.monotonic() + idle_wait_s, campaign_deadline) if campaign_deadline is not None else time.monotonic() + idle_wait_s
        while True:
            initial = snapshot(spec["gpu_uuid"])
            append_jsonl(folder / "idle_scans.jsonl", initial)
            if idle(initial):
                break
            if time.monotonic() >= wait_deadline:
                result = {**empty, "error": "IDLE_WAIT_TIMEOUT", "primary_pair_status": "NOT_STARTED", "spec_identity": spec_identity}
                write_json(terminal, result)
                return result
            time.sleep(min(10, max(0, wait_deadline - time.monotonic())))
        write_json(intent, {"spec_sha256": digest(spec), "source_sha": spec["source_sha"], "timestamp": utc(), "gpu_uuid": spec["gpu_uuid"]})
        job_start = time.monotonic()
        deadline = min(job_start + spec["timeout_s"], campaign_deadline) if campaign_deadline is not None else job_start + spec["timeout_s"]
        try:
            if published_cache:
                manifest = read_json(Path(published_cache).with_suffix(".manifest.json"))
                if manifest["compatibility"] != spec["cache_compatibility"]:
                    raise RuntimeError("published cache compatibility mismatch")
                for name, expected in manifest["members"].items():
                    if file_digest(Path(published_cache) / name) != expected:
                        raise RuntimeError("published cache member changed")
                shutil.copytree(published_cache, spec["cache_path"])
            write_json(folder / "spec.json", spec)
            if time.monotonic() >= deadline:
                raise TimeoutError("campaign budget expired during prelaunch preparation")
        except (OSError, ValueError, RuntimeError) as exc:
            result = {**empty, "error": str(exc), "failure_phase": "BEFORE_PROCESS_LAUNCH", "primary_pair_status": "NOT_STARTED", "spec_identity": spec_identity}
            write_json(terminal, result)
            return result
        materialization_s = time.monotonic() - job_start
        env = {**os.environ, **environment(), "CUDA_VISIBLE_DEVICES": spec["gpu_uuid"]}
        _enable_child_subreaper()
        samples, result = [], dict(empty)
        with (folder / "stdout.log").open("xb") as out, (folder / "stderr.log").open("xb") as err:
            process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.real_intervention_v6.worker", "--spec", str(folder / "spec.json")], env=env, stdout=out, stderr=err, start_new_session=True)
            try:
                identity = proc_identity(process.pid)
                write_json(folder / "started.json", {**identity, "source_sha": spec["source_sha"], "timestamp": utc()})
            except BaseException:
                _terminate_group(process, process.pid)
                raise
            done = threading.Event()
            def watchdog():
                if not done.wait(max(0, deadline - time.monotonic()) + 5):
                    _terminate_group(process, process.pid)
                    os._exit(124)  # preserve STARTED without terminal as permanent UNKNOWN
            threading.Thread(target=watchdog, daemon=True).start()
            try:
                ready = None
                while time.monotonic() < deadline:
                    if (folder / "ready.json").exists():
                        ready = read_json(folder / "ready.json")
                        # Fix V5's ordering: read ready first, THEN capture NVML.
                        break
                    if process.poll() is not None:
                        raise RuntimeError("worker exited before ready")
                    time.sleep(.05)
                if ready is None:
                    raise TimeoutError("worker readiness deadline")
                sample = snapshot(spec["gpu_uuid"])
                samples.append(sample)
                attestation = classify(ready, identity, samples, spec["gpu_uuid"], baseline_memory=initial["memory_used_bytes"])
                if attestation["worker_cuda_provenance"] != "VERIFIED":
                    raise RuntimeError(f"CUDA provenance {attestation}")
                if attestation["co_tenancy_exposure"] == "OBSERVED_FOREIGN":
                    raise RuntimeError("foreign GPU process before GO")
                write_json(folder / "go.json", {"timestamp": utc()})
                while not (folder / "result.json").exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("deployment deadline")
                    if process.poll() is not None:
                        raise RuntimeError("worker exited without result")
                    sample = snapshot(spec["gpu_uuid"])
                    samples.append(sample)
                    append_jsonl(folder / "telemetry.jsonl", sample)
                    if classify(ready, identity, [sample], spec["gpu_uuid"])["co_tenancy_exposure"] == "OBSERVED_FOREIGN":
                        raise RuntimeError("foreign GPU process during job")
                    time.sleep(.2)
                job_end = time.monotonic()
                payload = read_json(folder / "result.json")
                samples.append(snapshot(spec["gpu_uuid"]))
                attestation = classify(ready, identity, samples, spec["gpu_uuid"], baseline_memory=initial["memory_used_bytes"])
                if process.poll() is not None:
                    raise RuntimeError("CUDA owner exited before ACK")
                write_json(folder / "ack.json", {"timestamp": utc()})
                process.wait(timeout=max(.1, deadline - time.monotonic()))
                result = {**empty, **payload, **attestation, "parent_wall_s": job_end - job_start,
                          "cache_copy_and_spec_preparation_s": materialization_s,
                          "parent_wall_scope": "instrumented deployment including cache copy, process import/construction, identity checks, H forwards, numerical/steady probes, result publication; excludes ACK hold/teardown",
                          "exit_code": process.returncode, "ready": ready, "parent_identity": identity}
                parent_clock = TimingObservation("CACHE_RELOAD" if spec["arm"] == "RELOAD" else "PREFILL" if spec["arm"] == "PREFILL" else "JOB_TOTAL",
                    "HOST_MONOTONIC", "s", job_end - job_start, 1, 1, 1, "PREFILLED" if spec["arm"] == "RELOAD" else "EMPTY", "BEFORE_AFTER",
                    spec["arm"] in ("COLD", "PREFILL", "RELOAD"), spec["context_id"] + ":instrumented_parent_job")
                result["parent_timing_observation"] = {**asdict(parent_clock), "aggregate_seconds": parent_clock.aggregate_seconds, "per_call_seconds": parent_clock.per_call_seconds}
                timer_ok = not spec["engineering"] or payload.get("timer_crosscheck", {}).get("passed") is True
                result["measurement_contract"] = "COMPARABLE" if timer_ok else "NOT_COMPARABLE"
                if spec["arm"] == "RELOAD" and not payload.get("persistent_hit_proven"):
                    result["measurement_contract"] = "NOT_COMPARABLE"
                if spec["arm"] == "LAYOUT" and not payload.get("layout_evidence", {}).get("active"):
                    result["measurement_contract"] = "NOT_COMPARABLE"
                if process.returncode != 0:
                    result["execution_status"] = "ERROR"
            except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as exc:
                result.update({"execution_status": "TIMEOUT" if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else "ERROR", "error": str(exc)})
                if (folder / "result.json").exists():
                    known = read_json(folder / "result.json")
                    result.update({"scientific_correctness": known.get("scientific_correctness", "UNKNOWN"),
                                   "workload_identity": known.get("workload_identity", "UNKNOWN"), "completed_worker_result_path": "result.json"})
            finally:
                if process.poll() is None or _process_group_pids(process.pid):
                    _terminate_group(process, process.pid)
                done.set()
                result["owned_group_remaining"] = sorted(_process_group_pids(process.pid))
        result["primary_pair_status"] = primary_status(result)
        result["spec_identity"] = spec_identity
        result["final_identity"] = spec.get("final_identity")
        result["lifecycle_wall_s"] = time.monotonic() - job_start
        write_json(folder / "all_samples.json", samples)
        write_json(terminal, result)
        return result


def publish_cache(cache, destination, compatibility):
    cache, destination = Path(cache), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    members = {str(p.relative_to(cache)): file_digest(p) for p in sorted(cache.rglob("*")) if p.is_file() and p.suffix != ".lock"}
    if not members:
        raise RuntimeError("cannot publish an empty compiler cache")
    os.rename(cache, destination)
    # Published tree is never used directly by a worker. Each reload gets its
    # own verified copy; writes remain confined to that arm's private directory.
    write_json(destination.with_suffix(".manifest.json"), {"compatibility": compatibility, "members": members})
    return str(destination)
