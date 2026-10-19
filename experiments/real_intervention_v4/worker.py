"""One independent eager S0 or channels-last eager S1 inference process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time
import traceback
from typing import Any

from .outcomes import atomic_json


def _cpu_tree(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    return value


def _layout_inputs(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.contiguous(memory_format=torch.channels_last) if value.ndim == 4 else value
    if isinstance(value, tuple):
        return tuple(_layout_inputs(item) for item in value)
    if isinstance(value, list):
        return [_layout_inputs(item) for item in value]
    if isinstance(value, dict):
        return {key: _layout_inputs(item) for key, item in value.items()}
    return value


def _layout_evidence(model: Any, inputs: Any, torch: Any) -> dict[str, Any]:
    input_tensors = []

    def visit(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            if value.ndim == 4:
                input_tensors.append({
                    "shape": list(value.shape), "stride": list(value.stride()),
                    "channels_last": value.is_contiguous(memory_format=torch.channels_last),
                })
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(inputs)
    convolutions = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    channels_last_weights = [
        module.weight.is_contiguous(memory_format=torch.channels_last)
        for module in convolutions
    ]
    active = bool(input_tensors) and all(item["channels_last"] for item in input_tensors)
    active = active and bool(convolutions) and all(channels_last_weights)
    return {
        "input_4d_tensors": input_tensors,
        "conv2d_module_count": len(convolutions),
        "conv2d_channels_last_weight_count": sum(channels_last_weights),
        "active": active,
    }


def run(spec: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    from experiments.real_intervention_v3.numerical_checks import (
        compare_outputs, configure_determinism, nested_digest,
    )
    from .model_adapter import load_model
    from .qualification import has_pairable_kernel_exposure
    from .runtime_identity import capture_runtime

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    receipt: dict[str, Any] = {
        "status": "ERROR", "arm": spec.get("arm"),
        "intervention_id": spec.get("intervention_id"),
        "source": spec.get("source"), "model": spec.get("model"),
        "family_id": spec.get("family_id"), "batch": spec.get("batch"),
        "horizon": spec.get("horizon"), "split": spec.get("split"),
        "context_id": spec.get("context_id"),
        "timing_boundary": "layout conversion plus H forwards; model/input construction reported separately",
        "cache_paths": {
            key: os.environ.get(key) for key in (
                "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TMPDIR", "XDG_CACHE_HOME",
            )
        },
    }
    try:
        if spec.get("intervention_id") != "CHANNELS_LAST_EAGER_LAYOUT_V1":
            raise ValueError("worker intervention differs from frozen v4 primary")
        if spec.get("arm") not in {"S0", "S1"}:
            raise ValueError("arm must be S0 or S1")
        if spec["arm"] == "S1" and not spec.get("eligible_channels_last"):
            raise ValueError("S1 cannot run outside the prequalified channels-last cohort")
        if spec["arm"] == "S1":
            exposure = spec.get("qualification", {}).get("kernel_exposure") or {}
            if not has_pairable_kernel_exposure(exposure):
                raise ValueError("S1 requires paired layout and S0/S1 CUDA kernel-path qualification evidence")
        horizon = int(spec["horizon"])
        if horizon < 1:
            raise ValueError("horizon must be positive")
        configure_determinism(torch)
        seed = int(spec["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        construction_started = time.perf_counter()
        model, inputs = load_model(spec)
        torch.cuda.synchronize()
        construction_seconds = time.perf_counter() - construction_started
        weights_digest = nested_digest(model.state_dict())
        inputs_digest = nested_digest(inputs)
        if spec.get("expected_weights_sha256") and weights_digest != spec["expected_weights_sha256"]:
            raise ValueError("materialized model weights differ from performance-blind qualification")
        if spec.get("expected_inputs_sha256") and inputs_digest != spec["expected_inputs_sha256"]:
            raise ValueError("materialized inputs differ from frozen pair context")
        runtime = capture_runtime(str(spec["gpu_uuid"]))
        reference = torch.load(spec["reference_path"], map_location="cpu", weights_only=False)
        reference_digest = nested_digest(reference)
        if spec.get("reference_sha256") and reference_digest != spec["reference_sha256"]:
            raise ValueError("correctness reference differs from the frozen qualification output")

        torch.cuda.reset_peak_memory_stats()
        prepare_started = time.perf_counter()
        if spec["arm"] == "S1":
            model.to(memory_format=torch.channels_last)
            inputs = _layout_inputs(inputs)
        torch.cuda.synchronize()
        prepare_seconds = time.perf_counter() - prepare_started
        mechanism = _layout_evidence(model, inputs, torch)
        mechanism["qualification_kernel_exposure"] = (
            spec.get("qualification", {}).get("kernel_exposure")
        )
        if spec["arm"] == "S0":
            mechanism.update({"intervention_id": spec["intervention_id"],
                              "active": False, "role": "REFERENCE_ARM"})
        else:
            mechanism.update({"intervention_id": spec["intervention_id"],
                              "role": "CANDIDATE_ARM"})

        with torch.inference_mode():
            first_started = time.perf_counter()
            first_output = model(*inputs)
            torch.cuda.synchronize()
            first_seconds = time.perf_counter() - first_started
            remaining_started = time.perf_counter()
            last_output = first_output
            for _ in range(horizon - 1):
                last_output = model(*inputs)
            torch.cuda.synchronize()
            remaining_seconds = time.perf_counter() - remaining_started

            steady_started = time.perf_counter()
            for _ in range(10):
                steady_output = model(*inputs)
            torch.cuda.synchronize()
            steady_total = time.perf_counter() - steady_started

        first_cpu, last_cpu = _cpu_tree(first_output), _cpu_tree(last_output)
        first_check = compare_outputs(
            reference, first_cpu, rtol=float(spec["correctness"]["rtol"]),
            atol=float(spec["correctness"]["atol"]),
        )
        last_check = compare_outputs(
            reference, last_cpu, rtol=float(spec["correctness"]["rtol"]),
            atol=float(spec["correctness"]["atol"]),
        )
        steady_check = compare_outputs(
            reference, _cpu_tree(steady_output), rtol=float(spec["correctness"]["rtol"]),
            atol=float(spec["correctness"]["atol"]),
        )
        passed = first_check.passed and last_check.passed and steady_check.passed
        receipt.update({
            "status": "valid" if passed else "numeric_mismatch",
            "arm_execution": "COMPLETE",
            "scientific_correctness": "PASS" if passed else "FAIL",
            "correctness": {
                "passed": passed,
                "first_forward": first_check.__dict__,
                "last_forward": last_check.__dict__,
                "steady_forward": steady_check.__dict__,
            },
            "mechanism_exposure": mechanism,
            "first_failure_stage": (None if mechanism["active"] or spec["arm"] == "S0"
                                     else "S1_MECHANISM_ACTIVATION"),
            "runtime_identity": runtime,
            "runtime_gpu_uuid": runtime["device_uuid"],
            "weights_sha256": weights_digest, "inputs_sha256": inputs_digest,
            "reference_sha256": reference_digest,
            "output_sha256": nested_digest(last_cpu),
            "timing": {
                "prepare_seconds": prepare_seconds,
                "first_forward_seconds": first_seconds,
                "remaining_H_minus_1_seconds": remaining_seconds,
                "cold_seconds": prepare_seconds + first_seconds + remaining_seconds,
                "steady_seconds": steady_total / 10.0,
                "steady_repetitions": 10,
            },
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "model_and_input_construction_seconds": construction_seconds,
            "cache_scope_path": spec.get("cache_scope_path"),
            "cache_state": "EMPTY_ISOLATED_PER_ATTEMPT",
        })
    except Exception as exc:
        receipt.update({
            "status": "error", "scientific_correctness": "UNASSESSED",
            "arm_execution": "FAILED",
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
            "root_cause": f"{type(exc).__name__}:{exc}",
            "infrastructure_classification": (
                "INFRA" if isinstance(exc, (PermissionError, FileNotFoundError)) else None
            ),
            "first_failure_stage": "MODEL_WORKER",
        })
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    receipt["child_cpu_seconds"] = time.process_time() - started_cpu
    receipt["worker_wall_seconds"] = time.perf_counter() - started_wall
    receipt["finished_monotonic"] = time.monotonic()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("worker spec must be a JSON object")
    atomic_json(args.output, run(spec))


if __name__ == "__main__":
    main()
