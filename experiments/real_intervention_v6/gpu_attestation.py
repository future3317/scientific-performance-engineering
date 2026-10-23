"""Phased CUDA execution proof, independent of foreign-process visibility."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.real_intervention_v5.common import read_json, write_json
from experiments.real_intervention_v5.runtime import _terminate_group


def device_uuid(value):
    """PyTorch properties omit NVML's GPU- prefix on supported versions."""
    return "GPU-" + str(value).removeprefix("GPU-")


def proc_identity(pid):
    base = Path(f"/proc/{pid}")
    tail = (base / "stat").read_text().rsplit(")", 1)[1].split()
    status = (base / "status").read_text()
    ns = next((line.split()[1:] for line in status.splitlines() if line.startswith("NSpid:")), [str(pid)])
    return {"pid": pid, "ppid": int(tail[1]), "pgid": int(tail[2]), "start_ticks": int(tail[19]),
            "nspid": [int(x) for x in ns], "cgroup": (base / "cgroup").read_text()}


def snapshot(uuid):
    import pynvml as n
    begin = time.monotonic()
    n.nvmlInit()
    try:
        handle = n.nvmlDeviceGetHandleByUUID(uuid)
        results = {}
        for name in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetComputeRunningProcesses_v3"):
            try:
                rows = getattr(n, name)(handle)
                results[name] = {"status": "SUCCESS", "rows": [{"pid": int(p.pid), "memory_bytes": int(p.usedGpuMemory)} for p in rows]}
            except Exception as e:
                results[name] = {"status": type(e).__name__, "response": str(e)}
        memory, util = n.nvmlDeviceGetMemoryInfo(handle), n.nvmlDeviceGetUtilizationRates(handle)
        metrics = {"memory_used_bytes": memory.used, "memory_total_bytes": memory.total, "util_percent": util.gpu}
        for key, call in (("power_mw", lambda: n.nvmlDeviceGetPowerUsage(handle)), ("sm_clock_mhz", lambda: n.nvmlDeviceGetClockInfo(handle, n.NVML_CLOCK_SM)), ("temperature_c", lambda: n.nvmlDeviceGetTemperature(handle, n.NVML_TEMPERATURE_GPU))):
            try:
                metrics[key] = call()
            except Exception as e:
                metrics[key] = type(e).__name__
    finally:
        n.nvmlShutdown()
    cli = subprocess.run(["nvidia-smi", f"--id={uuid}", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
    return {"capture_begin_mono": begin, "capture_end_mono": time.monotonic(), "timestamp_unix": time.time(),
            "gpu_uuid": uuid, "nvml": results, "cli": {"returncode": cli.returncode, "stdout": cli.stdout, "stderr": cli.stderr}, **metrics}


def classify(ready, parent_identity, samples, uuid, *, baseline_memory=None, reservation_verified=False):
    result = {"worker_cuda_provenance": "UNKNOWN", "co_tenancy_exposure": "UNKNOWN"}
    if (ready.get("pid") != parent_identity.get("pid") or ready.get("start_ticks") != parent_identity.get("start_ticks")
            or ready.get("gpu_uuid") != uuid or any(s.get("gpu_uuid") != uuid for s in samples)):
        return {**result, "worker_cuda_provenance": "CONTRADICTED"}
    # A sample begun before ready cannot attest that ready, even if its read finishes later.
    after = [s for s in samples if s["capture_begin_mono"] >= ready["ready_mono"]]
    owned = set(ready.get("nspid", [])) | {ready["pid"]}
    visible = set()
    complete = bool(after)
    for sample in after:
        query = sample["nvml"]["nvmlDeviceGetComputeRunningProcesses"]
        complete &= query["status"] == "SUCCESS" and sample["cli"]["returncode"] == 0
        for channel in sample["nvml"].values():
            if channel.get("status") == "SUCCESS":
                visible.update(p["pid"] for p in channel["rows"])
        for line in sample["cli"]["stdout"].splitlines():
            fields = [x.strip() for x in line.split(",")]
            if len(fields) >= 2 and fields[0] == uuid and fields[1].isdigit():
                visible.add(int(fields[1]))
    foreign = visible - owned
    if foreign:
        result["co_tenancy_exposure"] = "OBSERVED_FOREIGN"
    elif complete and reservation_verified and visible & owned:
        result["co_tenancy_exposure"] = "VERIFIED_QUIET"
    event = ready.get("cuda_event_ms", 0)
    temporal_memory = baseline_memory is not None and any(s["memory_used_bytes"] > baseline_memory + ready.get("allocation_bytes", 0) * .5 for s in after)
    if event > 0 and after and (visible & owned or temporal_memory):
        result["worker_cuda_provenance"] = "VERIFIED"
    return {**result, "foreign_pids": sorted(foreign), "externally_visible_owner": bool(visible & owned),
            "post_ready_samples": len(after), "reservation_verified": reservation_verified}


def idle_device():
    from experiments.real_intervention_v5.run_all import gpu_inventory
    rows = gpu_inventory()
    selected = next((r for r in rows if not r["compute_pids"] and r["memory_used_mib"] <= 512 and r["utilization_percent"] == 0), None)
    return selected, rows


def wait_file(path, process, deadline):
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError(f"child exited {process.returncode}: {path}")
        if time.monotonic() >= deadline:
            raise TimeoutError(str(path))
        time.sleep(.02)
    return read_json(path)


def child(folder, uuid):
    identity = proc_identity(os.getpid())
    write_json(folder / "before.json", {**identity, "phase": "BEFORE_CUDA_INIT", "ready_mono": time.monotonic()})
    limit = time.monotonic() + 90
    def wait(name):
        while not (folder / name).exists():
            if time.monotonic() >= limit:
                raise TimeoutError(name)
            time.sleep(.02)
    wait("init.go")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    actual = device_uuid(props.uuid)
    a = torch.ones((1024, 1024), device="cuda")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    b = a @ a
    end.record()
    torch.cuda.synchronize()
    ready = {**identity, "gpu_uuid": actual, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "cuda_event_ms": start.elapsed_time(end), "allocation_bytes": a.numel() * a.element_size() * 2,
             "ready_mono": time.monotonic(), "phase": "CONTEXT_ALLOCATED", "device_name": props.name}
    write_json(folder / "allocated.json", ready)
    wait("active.go")
    write_json(folder / "active.json", {**ready, "phase": "KERNELS_ACTIVE", "ready_mono": time.monotonic()})
    until = time.monotonic() + 2
    while time.monotonic() < until:
        for _ in range(32):
            torch.mm(a, a, out=b)
        torch.cuda.synchronize()
    write_json(folder / "sync.json", {**ready, "phase": "SYNCHRONIZED", "ready_mono": time.monotonic()})
    wait("exit.go")


def probe(folder, uuid, timeout_s=90):
    deadline = time.monotonic() + timeout_s
    folder.mkdir(parents=True, exist_ok=False)
    before = snapshot(uuid)
    if before["memory_used_bytes"] > 512 * 1024**2 or before["util_percent"] != 0 or any(q.get("rows") for q in before["nvml"].values()):
        raise RuntimeError("selected GPU no longer idle; no sentinel launched")
    if time.monotonic() >= deadline:
        raise TimeoutError("sentinel campaign deadline before launch")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    with (folder / "stdout.log").open("xb") as out, (folder / "stderr.log").open("xb") as err:
        process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.real_intervention_v6.gpu_attestation", "--child", "--root", str(folder), "--uuid", uuid], env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            identity = proc_identity(process.pid)
            wait_file(folder / "before.json", process, deadline)
            samples = [{"phase": "BEFORE_CUDA_INIT", **snapshot(uuid)}]
            write_json(folder / "init.go", {})
            allocated = wait_file(folder / "allocated.json", process, deadline)
            samples.append({"phase": "CONTEXT_ALLOCATED", **snapshot(uuid)})
            write_json(folder / "active.go", {})
            wait_file(folder / "active.json", process, deadline)
            samples.append({"phase": "KERNELS_ACTIVE", **snapshot(uuid)})
            wait_file(folder / "sync.json", process, deadline)
            samples.append({"phase": "SYNCHRONIZED", **snapshot(uuid)})
            decision = classify(allocated, identity, samples, uuid, baseline_memory=before["memory_used_bytes"])
            write_json(folder / "exit.go", {})
            process.wait(timeout=max(.1, deadline - time.monotonic()))
            samples.append({"phase": "CHILD_EXITED", **snapshot(uuid)})
            report = {"parent_observed_identity": identity, "ready": allocated, "before": before, "samples": samples, **decision, "child_exit": process.returncode}
            write_json(folder / "GPU_SENTINEL.json", report)
            return report
        finally:
            if process.poll() is None:
                _terminate_group(process, process.pid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        child(args.root, args.uuid)
    else:
        print(json.dumps(probe(args.root, args.uuid)))
