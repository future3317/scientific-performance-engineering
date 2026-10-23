"""New-process V6 deployment worker; reuse only V5 numerical/model adapters."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import random
import time
import traceback

from experiments.real_intervention_v5.common import read_json, write_json, digest, file_digest
from .gpu_attestation import proc_identity, device_uuid
from .benchmark_timing_contract import TimingObservation, compare_timing


def run(spec):
    if spec.get("fault") == "hang_with_descendant":
        import signal
        import subprocess
        import sys
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        while True:
            time.sleep(1)
    if spec.get("fault") == "exit_without_terminal":
        os._exit(19)
    import numpy as np
    import torch
    from experiments.real_intervention_v5.worker import load_model, map_tree, check
    from experiments.real_intervention_v3.numerical_checks import configure_determinism, nested_digest
    from experiments.real_intervention_v4.worker import _layout_inputs, _layout_evidence
    from torch._dynamo.utils import counters

    configure_determinism(torch)
    random.seed(spec["seed"])
    np.random.seed(spec["seed"])
    torch.manual_seed(spec["seed"])
    torch.cuda.manual_seed_all(spec["seed"])
    cpu_start = time.process_time()
    construction_start = time.perf_counter()
    model, inputs = load_model(spec)
    model.to(memory_format=torch.contiguous_format)
    inputs = map_tree(inputs, lambda t: t.contiguous() if t.ndim == 4 else t)
    torch.cuda.synchronize()
    construction = time.perf_counter() - construction_start
    weights, input_id = nested_digest(model.state_dict()), nested_digest(inputs)
    reference = torch.load(spec["reference_path"], map_location="cpu", weights_only=False)
    for key, value in (("weights_sha256", weights), ("inputs_sha256", input_id), ("reference_sha256", nested_digest(reference))):
        if value != spec["qualification"][key]:
            raise ValueError(f"frozen qualification identity mismatch: {key}")
    features = {"parameters": sum(p.numel() for p in model.parameters()), "layers": len(list(model.modules())),
                "conv2d": sum(isinstance(m, torch.nn.Conv2d) for m in model.modules()), "batch": spec["batch"], "shape": spec["shape"]}
    features["conv_fraction"] = features["conv2d"] / max(1, features["layers"])
    counters.clear()
    prepare_start = time.perf_counter()
    if spec["arm"] == "LAYOUT":
        model.to(memory_format=torch.channels_last)
        inputs = _layout_inputs(inputs)
    elif spec["arm"] in ("COLD", "PREFILL", "RELOAD"):
        model = torch.compile(model, backend="inductor", mode="default", fullgraph=True, dynamic=False)
    torch.cuda.synchronize()
    prepare_s = time.perf_counter() - prepare_start
    def forward():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return model(*inputs)
    # Workloads have identical numeric contexts at every invocation. Warm probes
    # do not run until the declared H-call deployment job has finished.
    start = time.perf_counter()
    first = forward()
    torch.cuda.synchronize()
    first_s = time.perf_counter() - start
    start = time.perf_counter()
    last = first
    for _ in range(spec["horizon"] - 1):
        last = forward()
    torch.cuda.synchronize()
    remaining_s = time.perf_counter() - start
    job_counter = {k: dict(v) for k, v in counters.items()}
    checks = [check(reference, map_tree(v, lambda t: t.detach().cpu()), spec["correctness"]) for v in (first, last)]
    # Explicit warm-up is outside the deployment estimand. Record every call so
    # subsequent compile/guard changes cannot be hidden inside "steady".
    warm = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        forward()
        torch.cuda.synchronize()
        warm.append(time.perf_counter() - start)
    start_event, stop_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    start_event.record()
    for _ in range(10):
        forward()
    stop_event.record()
    torch.cuda.synchronize()
    aggregate = time.perf_counter() - start
    host = TimingObservation("STEADY", "HOST_MONOTONIC", "s", aggregate, 10, 1, 1, "PREFILLED", "BEFORE_AFTER", False, spec["context_id"])
    event = TimingObservation("STEADY", "CUDA_EVENT", "ms", start_event.elapsed_time(stop_event), 10, 1, 1, "PREFILLED", "CUDA_EVENT", False, spec["context_id"])
    timer = None
    if spec["engineering"]:
        import torch.utils.benchmark as benchmark
        measurement = benchmark.Timer(stmt="forward()", globals={"forward": forward}, num_threads=1).blocked_autorange(min_run_time=.1)
        official = TimingObservation("STEADY", "HOST_MONOTONIC", "s", measurement.median * measurement.number_per_run, 1,
                                     len(measurement.raw_times), measurement.number_per_run, "PREFILLED", "BEFORE_AFTER", False, spec["context_id"])
        comparable = compare_timing(host, official)
        ratio = host.per_call_seconds / official.per_call_seconds
        timer = {"status": comparable.status, "ratio": ratio, "passed": comparable.status == "COMPARABLE" and .5 <= ratio <= 2,
                 "raw_times": measurement.raw_times, "number_per_run": measurement.number_per_run, "median_s_per_call": measurement.median}
    cache = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    members = [{"path": str(p.relative_to(cache)), "bytes": p.stat().st_size, "sha256": file_digest(p)} for p in sorted(cache.rglob("*")) if p.is_file() and p.suffix != ".lock"]
    hits = sum(v for k, v in job_counter.get("inductor", {}).items() if "cache_hit" in k)
    # Retain CUDA owners through parent's final sample and ACK.
    globals()["_HOLD"] = model, inputs, last
    return {"scientific_correctness": "PASS" if all(c["passed"] for c in checks) else "FAIL", "correctness": checks,
            "workload_identity": "MATCH", "features": features, "weights_sha256": weights, "inputs_sha256": input_id,
            "reference_sha256": nested_digest(reference), "timing": {"construction_s": construction, "prepare_s": prepare_s, "first_forward_s": first_s,
                "remaining_s": remaining_s, "in_worker_job_s": prepare_s + first_s + remaining_s, "steady_per_call_s": host.per_call_seconds,
                "cuda_event_per_call_s": event.per_call_seconds, "warmup_call_s": warm, "child_cpu_s": time.process_time() - cpu_start},
            "timing_observations": [{**asdict(t), "aggregate_seconds": t.aggregate_seconds, "per_call_seconds": t.per_call_seconds} for t in (host, event)],
            "timer_crosscheck": timer, "compile_counters_job_only": job_counter, "cache_members": members,
            "persistent_hit_proven": spec["arm"] == "RELOAD" and hits > 0,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "layout_evidence": _layout_evidence(model, inputs, torch) if spec["arm"] == "LAYOUT" else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = read_json(args.spec)
    folder = args.spec.parent
    from experiments.real_intervention_v5.run_all import preflight
    if preflight() != spec["source_sha"]:
        raise RuntimeError("worker frozen code mismatch")
    root = Path(spec["results_root"])
    if spec["split"] == "final":
        release = read_json(root / "FINAL_RELEASE.json")
        if release["policy_sha256"] != digest(read_json(root / "POLICY_FREEZE.json")) or release["manifest_sha256"] != digest(read_json(root / "FROZEN_MANIFEST.json")):
            raise RuntimeError("final release identity mismatch")
    for name, subdir in (("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton"), ("CUDA_CACHE_PATH", "cuda")):
        directory = Path(spec["cache_path"]) / subdir
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    os.environ.update({"TORCHINDUCTOR_FX_GRAPH_CACHE": "1", "TORCHINDUCTOR_AUTOGRAD_CACHE": "1", "TORCHINDUCTOR_COMPILE_THREADS": "1"})
    deadline = time.monotonic() + spec["timeout_s"]
    ready_written = False
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.init()
        sentinel = torch.ones((1024, 1024), device="cuda")
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        result = sentinel @ sentinel
        b.record()
        torch.cuda.synchronize()
        ready = {**proc_identity(os.getpid()), "gpu_uuid": device_uuid(torch.cuda.get_device_properties(0).uuid),
                 "cuda_event_ms": a.elapsed_time(b), "allocation_bytes": 8 * 1024**2, "ready_mono": time.monotonic()}
        write_json(folder / "ready.json", ready)
        ready_written = True
        while not (folder / "go.json").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("GO deadline")
            time.sleep(.02)
        payload = run(spec)
        payload["execution_status"] = "COMPLETE"
    except Exception as exc:
        payload = {"execution_status": "ERROR", "scientific_correctness": "UNKNOWN", "error": str(exc), "traceback": traceback.format_exc()}
    write_json(folder / "result.json", {**payload, "pid": os.getpid(), "source_sha": spec["source_sha"]})
    while ready_written and not (folder / "ack.json").exists() and time.monotonic() < deadline:
        time.sleep(.02)


if __name__ == "__main__":
    main()
