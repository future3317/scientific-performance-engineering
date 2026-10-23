"""Bounded V6 process lifecycle using the existing process-group cleanup."""
from __future__ import annotations

import fcntl
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
from .gpu_attestation import classify, proc_identity, snapshot


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


def idle(snapshot_row):
    return (snapshot_row["memory_used_bytes"] <= 512 * 1024**2 and snapshot_row["util_percent"] == 0
            and all(not r.get("rows") and r["status"] == "SUCCESS" for r in snapshot_row["nvml"].values()))


def run_arm(spec, folder, *, published_cache=None, idle_wait_s=1800):
    folder = Path(folder)
    terminal = folder / "terminal.json"
    if terminal.exists():
        return read_json(terminal)
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
        wait_deadline = time.monotonic() + idle_wait_s
        while True:
            initial = snapshot(spec["gpu_uuid"])
            append_jsonl(folder / "idle_scans.jsonl", initial)
            if idle(initial):
                break
            if time.monotonic() >= wait_deadline:
                result = {**empty, "error": "IDLE_WAIT_TIMEOUT", "primary_pair_status": "NOT_STARTED"}
                write_json(terminal, result)
                return result
            time.sleep(min(10, max(0, wait_deadline - time.monotonic())))
        write_json(intent, {"spec_sha256": digest(spec), "source_sha": spec["source_sha"], "timestamp": utc(), "gpu_uuid": spec["gpu_uuid"]})
        job_start = time.monotonic()
        if published_cache:
            manifest = read_json(Path(published_cache).with_suffix(".manifest.json"))
            if manifest["compatibility"] != spec["cache_compatibility"]:
                raise RuntimeError("published cache compatibility mismatch")
            for name, expected in manifest["members"].items():
                if file_digest(Path(published_cache) / name) != expected:
                    raise RuntimeError("published cache member changed")
            shutil.copytree(published_cache, spec["cache_path"])
        write_json(folder / "spec.json", spec)
        env = {**os.environ, **environment(), "CUDA_VISIBLE_DEVICES": spec["gpu_uuid"]}
        _enable_child_subreaper()
        samples, result = [], dict(empty)
        deadline = job_start + spec["timeout_s"]
        with (folder / "stdout.log").open("xb") as out, (folder / "stderr.log").open("xb") as err:
            process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.real_intervention_v6.worker", "--spec", str(folder / "spec.json")], env=env, stdout=out, stderr=err, start_new_session=True)
            identity = proc_identity(process.pid)
            write_json(folder / "started.json", {**identity, "source_sha": spec["source_sha"], "timestamp": utc()})
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
                          "parent_wall_scope": "instrumented deployment including cache copy, process import/construction, identity checks, H forwards, numerical/steady probes, result publication; excludes ACK hold/teardown",
                          "exit_code": process.returncode, "ready": ready, "parent_identity": identity}
                timer_ok = not spec["engineering"] or payload.get("timer_crosscheck", {}).get("passed") is True
                result["measurement_contract"] = "COMPARABLE" if timer_ok else "NOT_COMPARABLE"
                if spec["arm"] == "RELOAD" and not payload.get("persistent_hit_proven"):
                    result["measurement_contract"] = "NOT_COMPARABLE"
                if process.returncode != 0:
                    result["execution_status"] = "ERROR"
            except (RuntimeError, TimeoutError) as exc:
                result.update({"execution_status": "TIMEOUT" if isinstance(exc, TimeoutError) else "ERROR", "error": str(exc)})
            finally:
                if process.poll() is None or _process_group_pids(process.pid):
                    _terminate_group(process, process.pid)
                done.set()
                result["owned_group_remaining"] = sorted(_process_group_pids(process.pid))
        result["primary_pair_status"] = primary_status(result)
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
