"""Read-only V5 EfficientFormer timer reproducer writing to a new output root."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4


def _counter_snapshot(counters) -> dict[str, dict[str, int]]:
    return {
        str(group): {str(key): int(value) for key, value in Counter(values).items()}
        for group, values in counters.items()
    }


def _guard_snapshot(guard_failures) -> list[dict[str, str]]:
    failures = []
    for code, reasons in guard_failures.items():
        failures.extend({
            "filename": str(getattr(code, "co_filename", "")),
            "function": str(getattr(code, "co_name", code)),
            "reason": str(reason),
        } for reason in reasons)
    return failures


def _dynamo_snapshot(counters, guard_failures) -> dict[str, object]:
    counter_values = _counter_snapshot(counters)
    stats = counter_values.get("stats", {})
    return {
        "counters": counter_values,
        "unique_graphs": stats.get("unique_graphs", 0),
        "calls_captured": stats.get("calls_captured", 0),
        "guard_failures": _guard_snapshot(guard_failures),
    }


def _dynamo_delta(before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    before_failures = before["guard_failures"]
    after_failures = after["guard_failures"]
    return {
        "unique_graphs_delta_recompile_indicator": after["unique_graphs"] - before["unique_graphs"],
        "guard_failures_count_delta": len(after_failures) - len(before_failures),
    }


def _attention_cache_snapshot(model) -> list[dict[str, object]]:
    return [
        {"module": name, "training": module.training, "keys": sorted(cache)}
        for name, module in model.named_modules()
        if isinstance((cache := getattr(module, "attention_bias_cache", None)), dict)
    ]


def _host_group(torch, forward, calls: int) -> dict[str, object]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        forward()
    torch.cuda.synchronize()
    aggregate = time.perf_counter() - start
    return {"aggregate_seconds": aggregate, "calls": calls, "per_call_seconds": aggregate / calls}


def _event_group(torch, forward, calls: int) -> dict[str, object]:
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(calls)]
    for start_event, end_event in events:
        start_event.record()
        forward()
        end_event.record()
    torch.cuda.synchronize()
    per_call = [start.elapsed_time(end) / 1000 for start, end in events]
    return {"call_seconds": per_call, "mean_per_call_seconds": sum(per_call) / calls}


def reproduce(
    spec_path: Path,
    output_path: Path,
    gpu_uuid_override: str | None = None,
) -> dict[str, object]:
    from experiments.real_intervention_v5.run_all import preflight

    canonical_commit = preflight()
    with spec_path.open(encoding="utf-8") as stream:
        spec = json.load(stream)
    if int(spec["batch"]) != 1 or "efficientformer" not in str(spec["model"]).lower():
        raise ValueError("spec must identify the registered EfficientFormer batch-1 workload")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    cache_identity = f"timer-reproducer-{uuid4().hex}"
    cache_root = output_path.parent / f"{output_path.stem}.{cache_identity}.cache"
    cache_root.mkdir()
    for variable, directory in (
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("CUDA_CACHE_PATH", "cuda"),
    ):
        path = cache_root / directory
        path.mkdir()
        os.environ[variable] = str(path)
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    original_gpu_uuid = spec["gpu_uuid"]
    actual_gpu_uuid = gpu_uuid_override or original_gpu_uuid
    spec["gpu_uuid"] = actual_gpu_uuid
    os.environ["CUDA_VISIBLE_DEVICES"] = actual_gpu_uuid

    import torch
    import torch.utils.benchmark as benchmark
    import numpy as np
    import random
    from torch._dynamo.utils import counters, guard_failures
    from experiments.real_intervention_v5.worker import load_model
    from experiments.real_intervention_v3.numerical_checks import configure_determinism

    seed = int(spec["seed"])
    configure_determinism(torch)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("V5 deterministic thread contract was not applied")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, inputs = load_model(spec)
    model.to(memory_format=torch.contiguous_format)
    inputs = tuple(
        item.contiguous(memory_format=torch.contiguous_format) if item.ndim == 4 else item
        for item in inputs
    )
    counters.clear()
    guard_failures.clear()
    prepare_start = time.perf_counter()
    compiled = torch.compile(model, backend="inductor", mode="default", fullgraph=True, dynamic=False)
    torch.cuda.synchronize()
    prepare_seconds = time.perf_counter() - prepare_start

    def forward():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return compiled(*inputs)

    torch.cuda.synchronize()
    cold_counters_before = _dynamo_snapshot(counters, guard_failures)
    cold_attention_cache_before = _attention_cache_snapshot(model)
    cold_host_start = time.perf_counter()
    cold_event_start = torch.cuda.Event(enable_timing=True)
    cold_event_end = torch.cuda.Event(enable_timing=True)
    cold_event_start.record()
    cold_output = forward()
    cold_event_end.record()
    torch.cuda.synchronize()
    cold_host_seconds = time.perf_counter() - cold_host_start
    cold_event_seconds = cold_event_start.elapsed_time(cold_event_end) / 1000
    cold_counters_after = _dynamo_snapshot(counters, guard_failures)
    cold_attention_cache_after = _attention_cache_snapshot(model)
    cold_compile_delta = _dynamo_delta(cold_counters_before, cold_counters_after)

    warmup_calls = []
    for call_index in range(10):
        before = _dynamo_snapshot(counters, guard_failures)
        attention_before = _attention_cache_snapshot(model)
        host_start = time.perf_counter()
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
        output = forward()
        event_end.record()
        torch.cuda.synchronize()
        host_seconds = time.perf_counter() - host_start
        event_seconds = event_start.elapsed_time(event_end) / 1000
        after = _dynamo_snapshot(counters, guard_failures)
        warmup_calls.append({
            "call": call_index + 1,
            "host_seconds": host_seconds,
            "cuda_event_seconds": event_seconds,
            "counter_before": before,
            "counter_after": after,
            "compile_guard_delta": _dynamo_delta(before, after),
            "attention_bias_cache_before": attention_before,
            "attention_bias_cache_after": _attention_cache_snapshot(model),
        })

    group_calls = 10
    def benchmark_forward():
        forward()

    host_groups = []
    event_groups = []
    benchmark_groups = []
    for _ in range(3):
        host_groups.append(_host_group(torch, forward, group_calls))
        event_groups.append(_event_group(torch, forward, group_calls))
        measurement = benchmark.Timer(
            stmt="benchmark_forward()",
            globals={"benchmark_forward": benchmark_forward},
            num_threads=1,
        ).blocked_autorange(min_run_time=0.05)
        benchmark_groups.append({
            "raw_times_seconds": list(measurement.raw_times),
            "median_seconds_per_call": measurement.median,
            "number_per_run": measurement.number_per_run,
            "per_call_times_seconds": list(measurement.times),
        })

    result = {
        "status": "COMPLETE",
        "spec_path": str(spec_path),
        "canonical_commit": canonical_commit,
        "model": spec["model"],
        "model_source_version": spec.get("source_version"),
        "model_source_sha256": spec.get("source_file_sha256"),
        "batch": spec["batch"],
        "horizon": 1,
        "precision": "BF16 autocast",
        "compile": {"backend": "inductor", "mode": "default", "fullgraph": True, "dynamic": False},
        "original_spec_gpu_uuid": original_gpu_uuid,
        "actual_gpu_uuid": actual_gpu_uuid,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "seed": seed,
            "device_name": torch.cuda.get_device_name(),
            "device_properties": repr(torch.cuda.get_device_properties(0)),
        },
        "cache_identity": cache_identity,
        "cache_root": str(cache_root),
        "cold_h1": {
            "prepare_host_seconds": prepare_seconds,
            "first_call_host_seconds": cold_host_seconds,
            "first_call_cuda_event_seconds": cold_event_seconds,
            "cuda_event_scope_note": "The cold first-call interval may span host-side graph compilation.",
            "job_total_host_seconds": prepare_seconds + cold_host_seconds,
            "counter_before": cold_counters_before,
            "counter_after": cold_counters_after,
            "compile_guard_delta": cold_compile_delta,
            "attention_bias_cache_before": cold_attention_cache_before,
            "attention_bias_cache_after": cold_attention_cache_after,
        },
        "ten_synchronized_warm_calls": warmup_calls,
        "post_warmup_crosscheck": {
            "calls_per_group": group_calls,
            "host_groups": host_groups,
            "cuda_event_groups": event_groups,
            "benchmark_groups": benchmark_groups,
        },
        "output_shape": list(cold_output.shape) if hasattr(cold_output, "shape") else None,
        "final_warm_output_shape": list(output.shape) if hasattr(output, "shape") else None,
    }
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True, help="read-only V5 worker spec")
    parser.add_argument("--output", type=Path, required=True, help="new-root JSON output path")
    parser.add_argument("--gpu-uuid", help="selected idle GPU UUID; defaults to the UUID in the V5 spec")
    args = parser.parse_args()
    reproduce(args.spec, args.output, args.gpu_uuid)


if __name__ == "__main__":
    main()
