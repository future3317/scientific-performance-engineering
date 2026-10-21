"""One CUDA owner, held alive through the parent's last attribution sample."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import random
import time
import traceback

from .common import read_json, utc, write_json, digest, file_digest, require_final_access

_HOLD = None


def map_tree(value, fn):
    import torch
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: map_tree(v, fn) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(map_tree(v, fn) for v in value)
    return value


def load_model(spec):
    from experiments.real_intervention_v3.source_adapters import load_model as old_load, _deny_network
    import torch
    _deny_network()
    import inspect
    if spec["source"] == "timm":
        import timm
        if timm.__version__ != spec["source_version"]:
            raise RuntimeError("timm version differs from registered source")
        if file_digest(inspect.getsourcefile(inspect.unwrap(timm.models.model_entrypoint(spec["model"])))) != spec["source_file_sha256"]:
            raise RuntimeError("timm model implementation differs from registered source")
        model = timm.create_model(spec["model"], pretrained=False, num_classes=1000).cuda().eval()
        x = torch.randn((spec["batch"], *spec["shape"]), generator=torch.Generator().manual_seed(spec["seed"])).cuda()
        return model, (x,)
    if spec["source"] == "torchvision":
        import torchvision.models as tv
        import importlib.metadata
        if importlib.metadata.version("torchvision") != spec["source_version"] or file_digest(inspect.getsourcefile(inspect.unwrap(tv.get_model_builder(spec["model"])))) != spec["source_file_sha256"]:
            raise RuntimeError("torchvision model implementation differs from registered source")
    if spec["source"] == "torchbench":
        source_file = Path(spec["torchbench_root"]) / "torchbenchmark/models" / spec["model"] / "__init__.py"
        if file_digest(source_file) != spec["source_file_sha256"]:
            raise RuntimeError("TorchBench model implementation differs from registered source")
    return old_load({**spec, "torchbench_root": spec["torchbench_root"]})


def check(reference, output, tolerance):
    from experiments.real_intervention_v3.numerical_checks import compare_outputs
    import torch
    result = asdict(compare_outputs(reference, output, rtol=tolerance["rtol"], atol=tolerance["atol"]))
    # Report a scale-aware error as well as elementwise tolerance. It does not
    # relax the elementwise gate near zero or change the output dtype contract.
    sums = [0.0, 0.0]
    def walk(a, b):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape:
            sums[0] += float(((a.double() - b.double()) ** 2).sum())
            sums[1] += float((a.double() ** 2).sum())
        elif isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
            for k in a: walk(a[k], b[k])
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
            for x, y in zip(a, b): walk(x, y)
    walk(reference, output)
    result["relative_l2"] = (sums[0] / max(sums[1], 1e-300)) ** .5
    if tolerance.get("relative_l2_max") is not None and result["relative_l2"] > tolerance["relative_l2_max"]:
        result.update({"passed": False, "reason": "relative_l2_limit_exceeded"})
    return result


def run(spec):
    global _HOLD
    require_final_access(spec)
    import numpy as np
    import torch
    from experiments.real_intervention_v3.numerical_checks import configure_determinism, nested_digest
    from experiments.real_intervention_v4.worker import _layout_inputs, _layout_evidence
    from experiments.real_intervention_v4.runtime_identity import capture_runtime

    configure_determinism(torch)
    random.seed(spec["seed"])
    np.random.seed(spec["seed"])
    torch.manual_seed(spec["seed"])
    torch.cuda.manual_seed_all(spec["seed"])
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    model, inputs = load_model(spec)
    model.to(memory_format=torch.contiguous_format)
    inputs = map_tree(inputs, lambda t: t.contiguous(memory_format=torch.contiguous_format) if t.ndim == 4 else t)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        raise RuntimeError("model constructor changed the frozen CPU inter-op topology")
    torch.cuda.synchronize()
    construction = time.perf_counter() - wall_start
    weights, input_hash = nested_digest(model.state_dict()), nested_digest(inputs)
    runtime = capture_runtime(spec["gpu_uuid"])
    runtime["torch_intraop_threads"] = torch.get_num_threads()
    runtime["torch_interop_threads"] = torch.get_num_interop_threads()
    features = {"parameters": sum(p.numel() for p in model.parameters()),
                "layers": sum(1 for _ in model.modules()),
                "conv2d": sum(isinstance(m, torch.nn.Conv2d) for m in model.modules()),
                "batch": spec["batch"], "shape": spec["shape"]}
    def cpu(x): return map_tree(x, lambda t: t.detach().cpu())
    def forward(m=model, xs=inputs):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return m(*xs)
    if spec["mode"] == "qualification":
        a = cpu(forward())
        aa = cpu(forward())
        baseline = check(a, aa, {"rtol": 0.0, "atol": 0.0})
        with torch.inference_mode():
            fp32 = cpu(model(*inputs))
        fp32 = map_tree(fp32, lambda t: t.float())
        deviation = check(fp32, map_tree(a, lambda t: t.float()), {"rtol": .1, "atol": .01})
        torch.save(a, spec["reference_path"])
        candidate = {"passed": False, "reason": "not_tested"}
        mechanism = None
        if spec["study"] == "A":
            model.to(memory_format=torch.channels_last)
            inputs = _layout_inputs(inputs)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                b = cpu(model(*inputs))
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    model(*inputs)
                    torch.cuda.synchronize()
            candidate = check(a, b, spec["correctness"])
            mechanism = _layout_evidence(model, inputs, torch)
            mechanism["cuda_kernel_events"] = sum("CUDA" in str(e.device_type) for e in profiler.events())
        else:
            candidate = {"passed": True, "reason": "compile_and_cross_process_cache_qualification_pending"}
        _HOLD = model, inputs, a
        return {"status": "valid" if baseline["passed"] and candidate["passed"] else "numeric_mismatch",
                "execution_status": "COMPLETE", "scientific_status": "PASS" if baseline["passed"] and candidate["passed"] else "NUMERICAL_UNSAFE",
                "baseline": baseline, "candidate": candidate, "fp32_deviation_sanity_only": deviation,
                "kernel_exposure": mechanism, "features": features,
                "weights_sha256": weights, "inputs_sha256": input_hash,
                "reference_sha256": nested_digest(a), "runtime_identity": runtime, "gpu_uuid": spec["gpu_uuid"],
                "timings_withheld": True}

    reference = torch.load(spec["reference_path"], map_location="cpu", weights_only=False)
    expected = spec["qualification"]
    for name, value in (("weights_sha256", weights), ("inputs_sha256", input_hash), ("reference_sha256", nested_digest(reference))):
        if value != expected[name]:
            raise ValueError(f"qualification identity mismatch: {name}")
    cache_path = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    cache_before = sorted(str(p.relative_to(cache_path)) for p in cache_path.rglob("*") if p.is_file())
    if spec["arm"] in ("COMPILE_COLD", "CACHE_PREFILL") and cache_before:
        raise ValueError("cold compiler cache was not empty")
    from torch._dynamo.utils import counters
    counters.clear()
    torch.cuda.reset_peak_memory_stats()
    prepare_start = time.perf_counter()
    if spec["arm"] == "S1":
        model.to(memory_format=torch.channels_last)
        inputs = _layout_inputs(inputs)
    elif spec["arm"] in ("COMPILE_COLD", "CACHE_PREFILL", "COMPILE_PERSISTENT_CACHE"):
        model = torch.compile(model, backend="inductor", mode="default", fullgraph=True, dynamic=False)
    torch.cuda.synchronize()
    prepare = time.perf_counter() - prepare_start
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        start = time.perf_counter()
        first = model(*inputs)
        torch.cuda.synchronize()
        first_s = time.perf_counter() - start
        start = time.perf_counter()
        last = first
        for _ in range(spec["horizon"] - 1):
            last = model(*inputs)
        torch.cuda.synchronize()
        remaining_s = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(10):
            steady = model(*inputs)
        torch.cuda.synchronize()
        steady_s = (time.perf_counter() - start) / 10
    checks = [check(reference, cpu(v), spec["correctness"]) for v in (first, last, steady)]
    passed = all(x["passed"] for x in checks)
    compile_counters = {k: dict(v) for k, v in counters.items()}
    cache_after = [{"path": str(p.relative_to(cache_path)), "bytes": p.stat().st_size, "sha256": file_digest(p)} for p in sorted(cache_path.rglob("*")) if p.is_file() and not p.name.endswith(".lock")]
    cache_hits = sum(v for k, v in compile_counters.get("inductor", {}).items() if "cache_hit" in k)
    cache_misses = sum(v for k, v in compile_counters.get("inductor", {}).items() if "cache_miss" in k)
    _HOLD = model, inputs, last
    receipt = {"status": "valid" if passed else "numeric_mismatch", "execution_status": "COMPLETE",
            "scientific_status": "PASS" if passed else "NUMERICAL_UNSAFE", "correctness": checks,
            "gpu_uuid": spec["gpu_uuid"], "runtime_identity": runtime, "arm": spec["arm"],
            "weights_sha256": weights, "inputs_sha256": input_hash, "reference_sha256": nested_digest(reference),
            "timing": {"prepare_s": prepare, "first_forward_s": first_s, "remaining_s": remaining_s,
                       "total_cold_job_s": prepare + first_s + remaining_s, "steady_state_per_forward_s": steady_s,
                       "model_construction_s": construction, "child_cpu_s": time.process_time() - cpu_start,
                       "worker_wall_s": time.perf_counter() - wall_start},
            "features": features, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "compile_counters": compile_counters,
            "cache": {"before_members": cache_before, "after_members": cache_after, "hits": cache_hits,
                      "misses": cache_misses, "observed_graph_cache_hit_fraction": cache_hits / (cache_hits + cache_misses) if cache_hits + cache_misses else None,
                      "persistent_hit_proven": spec["arm"] == "COMPILE_PERSISTENT_CACHE" and cache_hits > 0},
            "layout_evidence": _layout_evidence(model, inputs, torch) if spec["arm"] in ("S0", "S1") else None}
    if spec["mode"] == "engineering":
        import torch.utils.benchmark as benchmark
        def timed_forward():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(*inputs)
        official = benchmark.Timer(stmt="forward()", globals={"forward": timed_forward}, num_threads=1).blocked_autorange(min_run_time=.05)
        ratio = steady_s / official.median
        receipt["timer_crosscheck"] = {"manual_steady_s": steady_s, "torch_utils_benchmark_median_s": official.median,
                                       "ratio": ratio, "passed": .5 <= ratio <= 2.0,
                                       "scope": "Warm forward only, gross async/timer error check; not a speedup acceptance test"}
    if spec["mode"] == "compile_qualification":
        receipt.pop("timing")
        receipt["timings_withheld"] = True
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ack-path", type=Path, required=True)
    args = parser.parse_args()
    spec = read_json(args.spec)
    cache_path = Path(spec["cache_path"])
    cache_path.mkdir(parents=True, exist_ok=True)
    for key, subdir in (("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton"),
                        ("CUDA_CACHE_PATH", "cuda")):
        path = cache_path / subdir
        path.mkdir(exist_ok=True)
        os.environ[key] = str(path)
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    deadline = time.monotonic() + spec["timeout_s"]
    ready_written = False
    try:
        from .run_all import preflight
        if preflight() != spec["source_sha"]:
            raise RuntimeError("worker source differs from frozen study source")
        import torch
        torch.cuda.init()
        sentinel = torch.ones(1, device="cuda")
        torch.cuda.synchronize()
        write_json(args.output.with_suffix(".ready.json"), {"pid": os.getpid(), "pgid": os.getpgid(0), "gpu_uuid": spec["gpu_uuid"], "timestamp": utc()})
        ready_written = True
        while not args.output.with_suffix(".go").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("parent attribution start handshake expired")
            time.sleep(.02)
        result = run(spec)
    except Exception as exc:
        result = {"status": "error", "execution_status": "ERROR", "scientific_status": "UNVERIFIED",
                  "error": str(exc), "error_code": type(exc).__name__, "traceback": traceback.format_exc()}
    result.update({"pid": os.getpid(), "pgid": os.getpgid(0), "timestamp": utc(), "arm_identity": digest(spec)})
    write_json(args.output, result)
    while ready_written and not args.ack_path.exists() and time.monotonic() < deadline:
        time.sleep(.02)


if __name__ == "__main__":
    main()
