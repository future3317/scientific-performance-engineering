from __future__ import annotations

import argparse
import json
import random
import os
import time
import traceback
from pathlib import Path
from typing import Any


def _sync(torch: Any) -> None:
    torch.cuda.synchronize()


def _strict_compile_setup(torch: Any, receipt: dict[str, Any]):
    import torch._dynamo

    torch._dynamo.config.suppress_errors = False
    receipt["dynamo_suppress_errors"] = bool(torch._dynamo.config.suppress_errors)
    from torch._inductor import metrics

    metrics.reset()
    return metrics


def _cpu_tree(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def _validate_spec(spec: dict[str, Any]) -> None:
    required = {"source", "model", "family", "batch", "shape", "seed", "horizon", "arm", "intervention"}
    missing = sorted(required - spec.keys())
    if missing:
        raise ValueError(f"missing spec fields: {missing}")
    if spec["source"] not in {"torchvision", "torchbench"}:
        raise ValueError(f"unsupported source: {spec['source']!r}")
    if spec["arm"] not in {"S0", "S1"} or spec["intervention"] not in {"compile", "channels_last"}:
        raise ValueError("arm must be S0/S1 and intervention compile/channels_last")
    if int(spec["batch"]) < 1 or int(spec["horizon"]) < 1:
        raise ValueError("batch and horizon must be positive")
    if spec["source"] == "torchbench" and not spec.get("torchbench_root"):
        raise ValueError("torchbench_root is required for TorchBench models")
    if not spec.get("reference_path"):
        raise ValueError("both arms require the frozen admission reference_path")


def run(spec: dict[str, Any]) -> dict[str, Any]:
    _validate_spec(spec)
    return _run(spec)


def _run(spec: dict[str, Any]) -> dict[str, Any]:
    import torch

    from .numerical_checks import compare_outputs, configure_determinism, nested_digest
    from .source_adapters import load_model

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    receipt: dict[str, Any] = {
        "status": "error", "arm": spec["arm"], "intervention": spec["intervention"],
        "source": spec["source"], "model": spec["model"], "family": spec["family"],
        "horizon": int(spec["horizon"]),
        "cache_scope": "parent_managed_durable" if spec["intervention"] == "compile" else "not_applicable",
        "cache_paths": {key: os.environ.get(key) for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH")},
    }
    try:
        configure_determinism(torch)
        torch.manual_seed(int(spec["seed"]))
        random.seed(int(spec["seed"]))
        import numpy as np
        np.random.seed(int(spec["seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(spec["seed"]))

        setup_start = time.perf_counter()
        model, inputs = load_model(spec)
        _sync(torch)
        receipt["setup_module_and_data_s"] = time.perf_counter() - setup_start
        receipt["weights_sha256"] = nested_digest(model.state_dict())
        receipt["inputs_sha256"] = nested_digest(inputs)
        properties = torch.cuda.get_device_properties(0)
        device_uuid = getattr(properties, "uuid", None)
        if device_uuid is None:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            device_uuid = visible if visible and not visible.isdigit() else None
        receipt["runtime_identity"] = {
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "device_index": torch.cuda.current_device(),
            "device_uuid": str(device_uuid) if device_uuid is not None else None,
        }
        torch.cuda.reset_peak_memory_stats()

        inductor_metrics = None
        initialize_start = time.perf_counter()
        if spec["arm"] == "S1":
            if spec["intervention"] == "compile":
                inductor_metrics = _strict_compile_setup(torch, receipt)
                model = torch.compile(model, backend="inductor", mode="default", dynamic=None)
            else:
                model = model.to(memory_format=torch.channels_last)
                inputs = tuple(value.contiguous(memory_format=torch.channels_last) if isinstance(value, torch.Tensor) and value.ndim == 4 else value for value in inputs)
        _sync(torch)
        receipt["intervention_initialize_s"] = time.perf_counter() - initialize_start

        with torch.inference_mode():
            first_start = time.perf_counter()
            output = model(*inputs)
            _sync(torch)
            receipt["first_execution_s"] = time.perf_counter() - first_start
            if inductor_metrics is not None:
                receipt["generated_kernel_count"] = int(inductor_metrics.generated_kernel_count)

            remaining_start = time.perf_counter()
            for _ in range(int(spec["horizon"]) - 1):
                output = model(*inputs)
            _sync(torch)
            receipt["remaining_H_minus_1_s"] = time.perf_counter() - remaining_start

            primary_output = output
            steady_start = time.perf_counter()
            for _ in range(10):
                output = model(*inputs)
                _sync(torch)
            receipt["steady_state_10_calls_s"] = time.perf_counter() - steady_start
        receipt["steady_state_per_step_s"] = receipt["steady_state_10_calls_s"] / 10

        reference = torch.load(spec["reference_path"], map_location="cpu", weights_only=False)
        candidate = _cpu_tree(primary_output)
        rtol = float(spec.get("correctness", {}).get("rtol", 1e-4))
        atol = float(spec.get("correctness", {}).get("atol", 1e-5))
        comparison = compare_outputs(reference, candidate, rtol=rtol, atol=atol)
        receipt["correctness"] = comparison.__dict__
        receipt["reference_sha256"] = nested_digest(reference)
        receipt["output_sha256"] = nested_digest(candidate)
        if not comparison.passed:
            receipt["status"] = "numeric_mismatch"

        receipt["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        receipt["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if receipt["status"] != "numeric_mismatch":
            receipt["status"] = "valid"
    except Exception as exc:
        receipt.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc),
                        "traceback": traceback.format_exc()})
    receipt["cpu_seconds"] = time.process_time() - started_cpu
    receipt["wall_seconds"] = time.perf_counter() - started_wall
    parts = ("intervention_initialize_s", "first_execution_s", "remaining_H_minus_1_s")
    if all(key in receipt for key in parts):
        receipt["total_cold_inclusive_s"] = sum(receipt[key] for key in parts)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise ValueError("spec must contain a JSON object")
        if spec.get("output_path"):
            args.output = spec["output_path"]
        result = run(spec)
    except Exception as exc:
        result = {"status": "error", "error_type": type(exc).__name__, "error": str(exc),
                  "traceback": traceback.format_exc()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
