from __future__ import annotations

import json
import csv
import io
import os
from pathlib import Path
import resource
import signal
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping

from .arm_journal import complete_arm, start_arm
from .receipts import write_pair


def _gpu_snapshot(gpu_uuid: str | None) -> dict[str, Any] | None:
    if not gpu_uuid:
        return None
    query = subprocess.run(
        ["nvidia-smi", f"--id={gpu_uuid}",
         "--query-gpu=uuid,memory.used,memory.free,utilization.gpu,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True,
    )
    if query.returncode:
        return {"error": query.stderr.strip()}
    rows = list(csv.reader(io.StringIO(query.stdout)))
    if not rows:
        return {"error": "empty_gpu_snapshot"}
    process_query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True,
    )
    processes = []
    if process_query.returncode == 0:
        for values in csv.reader(io.StringIO(process_query.stdout)):
            if len(values) >= 4 and values[0].strip() == gpu_uuid:
                processes.append({"pid": values[1].strip(), "name": values[2].strip(),
                                  "memory_mb": values[3].strip()})
    return {"gpu_uuid": rows[0][0].strip(), "memory_used_mb": rows[0][1].strip(),
            "memory_free_mb": rows[0][2].strip(), "utilization_percent": rows[0][3].strip(),
            "sm_clock_mhz": rows[0][4].strip(), "memory_clock_mhz": rows[0][5].strip(),
            "pstate": rows[0][6].strip(), "power_w": rows[0][7].strip(),
            "temperature_c": rows[0][8].strip(), "compute_processes": processes}


def _run_worker(command: list[str], env: dict[str, str], timeout_seconds: int) -> tuple[int | None, str, str, bool, float]:
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        return process.returncode, stdout, stderr, False, (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        return None, stdout, stderr, True, (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)


def run_pair(
    context: Mapping[str, Any],
    *,
    session_id: str,
    identity: str,
    project_root: str | Path,
    python_executable: str,
    run_root: str | Path,
    device: str,
    order: str,
    horizon: int,
    warmups: int,
    repetitions: int,
    rtol: float,
    atol: float,
    timeout_seconds: int,
    storage_cap_bytes: int,
    arm_timeout_budget: Callable[[], int] | None = None,
    arm_names: tuple[str, str] = ("eager", "compile"),
    storage_scope_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run arms in clean processes and separate compile-cache directories."""
    from .scientific_checks import compare_outputs
    import torch

    if order not in {"eager_first", "compile_first"}:
        raise ValueError("order must be frozen as eager_first or compile_first")
    if len(set(arm_names)) != 2 or any(name not in {"eager", "compile", "eager_control"} for name in arm_names):
        raise ValueError("invalid arm names")
    if horizon < 1 or warmups < 0 or repetitions < 1 or timeout_seconds < 1:
        raise ValueError("invalid horizon, repetitions, warmups, or timeout")
    root = Path(run_root).resolve()
    project = Path(project_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    context_id = str(context["context_id"])
    if not session_id or Path(session_id).name != session_id:
        raise ValueError("invalid session_id")
    pair_key = f"{context_id}.{session_id}"
    completed_pair = root / "pairs" / f"{pair_key}.json"
    if completed_pair.exists():
        saved = json.loads(completed_pair.read_text(encoding="utf-8"))
        if saved.get("identity") != identity:
            raise ValueError("saved pair belongs to a different run identity")
        return saved
    arms: dict[str, dict[str, Any]] = {}
    outputs: dict[str, Any] = {}

    ordered_arms = arm_names if order == "eager_first" else tuple(reversed(arm_names))
    worker_arms = {}
    for arm in ordered_arms:
        worker_arm = "eager" if arm == "eager_control" else arm
        worker_arms[arm] = worker_arm
        allowed_timeout = arm_timeout_budget() if arm_timeout_budget is not None else timeout_seconds
        intent = start_arm(root / "arms", context_id, session_id, arm, identity)
        if intent.get("status") != "started":
            arms[arm] = intent
            if intent.get("status") == "valid":
                output_path = root / "scratch" / f"{pair_key}.{arm}.output.pt"
                if output_path.exists():
                    outputs[arm] = torch.load(output_path, map_location="cpu", weights_only=True)
            continue

        if allowed_timeout <= 0:
            result = {"status": "not_started", "reason": "resource_cap_reached_before_arm"}
            complete_arm(root / "arms", context_id, session_id, arm, identity, result)
            arms[arm] = result
            continue

        scratch = root / "scratch"
        cache = root / "inductor-cache" / context_id / session_id / arm
        scratch.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)
        config_path = scratch / f"{pair_key}.{arm}.config.json"
        result_path = scratch / f"{pair_key}.{arm}.result.json"
        output_path = scratch / f"{pair_key}.{arm}.output.pt"
        started = time.perf_counter()
        config = {"context": dict(context), "arm": worker_arm, "device": device,
                  "horizon": horizon, "warmups": warmups, "repetitions": repetitions,
                  "seed": int(context.get("seed", 0)), "scratch_output": str(output_path),
                  "parent_launch_monotonic": started}
        config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
        env = os.environ.copy()
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
        env["PYTHONPATH"] = str(project) + os.pathsep + env.get("PYTHONPATH", "")
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if device.startswith("cuda"):
            gpu_uuid = context.get("gpu_uuid")
            if not gpu_uuid:
                complete_arm(root / "arms", context_id, session_id, arm, identity,
                             {"status": "protocol_invalid", "reason": "missing_frozen_gpu_uuid"})
                arms[arm] = {"status": "protocol_invalid", "reason": "missing_frozen_gpu_uuid"}
                continue
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_uuid)
        command = [python_executable, "-m", "experiments.real_intervention_v2.worker",
                   str(config_path), str(result_path)]
        gpu_uuid = context.get("gpu_uuid") if device.startswith("cuda") else None
        telemetry_before = _gpu_snapshot(gpu_uuid)
        return_code, stdout, stderr, timed_out, child_cpu_s = _run_worker(
            command, env, min(timeout_seconds, allowed_timeout))
        telemetry_after = _gpu_snapshot(gpu_uuid)
        parent_wall_s = time.perf_counter() - started
        if timed_out:
            result = {"status": "timeout", "reason": "worker_timeout", "parent_wall_s": parent_wall_s,
                      "child_cpu_s": child_cpu_s}
        elif return_code != 0 or not result_path.exists():
            result = {"status": "unknown", "reason": "worker_exit_without_receipt",
                      "return_code": return_code, "stderr_tail": stderr[-4000:], "stdout_tail": stdout[-1000:],
                      "parent_wall_s": parent_wall_s, "child_cpu_s": child_cpu_s}
        else:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["parent_wall_s"] = parent_wall_s
            result["child_cpu_s"] = child_cpu_s
            result["worker_stderr_tail"] = stderr[-4000:]
        if telemetry_before is not None:
            result["gpu_telemetry_before"] = telemetry_before
            result["gpu_telemetry_after"] = telemetry_after
        complete_arm(root / "arms", context_id, session_id, arm, identity, result)
        arms[arm] = result
        if result.get("status") == "valid" and output_path.exists():
            outputs[arm] = torch.load(output_path, map_location="cpu", weights_only=True)

    if set(outputs) == set(arm_names):
        reference_arm, candidate_arm = arm_names
        reference_receipt = arms[reference_arm]
        candidate_receipt = arms[candidate_arm]
        if reference_receipt.get("input_digest") != candidate_receipt.get("input_digest"):
            correctness = {"passed": False, "reason": "input_digest_mismatch"}
        elif reference_receipt.get("model_digest") != candidate_receipt.get("model_digest"):
            correctness = {"passed": False, "reason": "model_state_mismatch"}
        else:
            correctness = compare_outputs(outputs[reference_arm], outputs[candidate_arm], rtol=rtol, atol=atol).__dict__
    else:
        correctness = {"passed": False, "reason": "arm_output_unavailable"}
    for arm in arm_names:
        (root / "scratch" / f"{pair_key}.{arm}.output.pt").unlink(missing_ok=True)
    receipt = {"context_id": context_id, "session_id": session_id, "context": dict(context),
               "identity": identity, "order": order, "horizon": horizon,
               "arm_names": list(arm_names), "worker_arms": worker_arms,
               "arms": arms, "correctness": correctness,
               "classification": "paired_valid" if all(arms.get(arm, {}).get("status") == "valid" for arm in arm_names) and correctness["passed"] else "invalid_or_unknown"}
    write_pair(root / "pairs", receipt, storage_cap=storage_cap_bytes,
               storage_scope=storage_scope_root or root)
    return receipt
