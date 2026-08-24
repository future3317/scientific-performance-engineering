"""Harness entry point for the CUDA batched-telemetry scalar-sync anchor."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
_METRICS_PER_EVENT = 16
_TRACE_STEPS = 8


def load_solution(path: str | Path, device: str | None = None):
    path = Path(path).resolve()
    module_name = f"scalar_sync_solution_{hashlib.sha1(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load solution module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_fixtures(seed: int, device: str = "cpu") -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    input_dim = 16
    hidden_dim = 16
    output_dim = 1
    n_train = 512
    n_test = 128
    batch_size = 32

    x_train = torch.randn(n_train, input_dim, generator=generator)
    true_w = torch.randn(input_dim, output_dim, generator=generator)
    true_b = torch.randn(output_dim, generator=generator)
    y_train = x_train @ true_w + true_b + 0.1 * torch.randn(n_train, output_dim, generator=generator)
    x_test = torch.randn(n_test, input_dim, generator=generator)
    y_test = x_test @ true_w + true_b + 0.05 * torch.randn(n_test, output_dim, generator=generator)

    return {
        "seed": seed,
        "device": device,
        "requested_device": device,
        "batch_size": batch_size,
        "lr": 0.01,
        "model_dims": [input_dim, hidden_dim, hidden_dim, hidden_dim, output_dim],
        "scalar_syncs_per_step": _METRICS_PER_EVENT,
        "metric_cadence": 1,
        "telemetry_width": _METRICS_PER_EVENT,
        "train_data": (x_train, y_train),
        "test_data": (x_test, y_test),
    }


def _build_model(fixtures: dict[str, Any]) -> nn.Module:
    layers: list[nn.Module] = []
    dims = fixtures["model_dims"]
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.extend((nn.Linear(in_dim, out_dim), nn.ReLU()))
    layers.pop()
    return nn.Sequential(*layers)


def _make_optimizer(model: nn.Module, fixtures: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=float(fixtures["lr"]))


def _run_trace(solution: Any, fixtures: dict[str, Any], steps: int) -> dict[str, Any]:
    torch.manual_seed(int(fixtures["seed"]))
    device = str(fixtures["device"])
    model = solution.build_model(fixtures).to(device)
    optimizer = _make_optimizer(model, fixtures)
    x_all, y_all = (value.to(device) for value in fixtures["train_data"])
    batch_size = int(fixtures["batch_size"])
    loss_trajectory: list[float] = []
    telemetry_trajectory: list[list[float]] = []
    work_units: list[dict[str, Any]] = []

    for step in range(steps):
        start = (step * batch_size) % x_all.shape[0]
        out = solution.train_step(
            model,
            (x_all[start : start + batch_size], y_all[start : start + batch_size]),
            optimizer,
        )
        loss = out.get("loss") if isinstance(out, dict) else None
        telemetry = out.get("telemetry") if isinstance(out, dict) else None
        if not isinstance(loss, torch.Tensor) or not isinstance(telemetry, list):
            raise AssertionError("train_step must return tensor loss and telemetry list")
        loss_trajectory.append(float(loss.detach().cpu()))
        telemetry_trajectory.append([float(value) for value in telemetry])
        work_units.append(dict(out.get("work_units", {})))

    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return {
        "state": state,
        "loss_trajectory": loss_trajectory,
        "telemetry_trajectory": telemetry_trajectory,
        "work_units": work_units,
    }


def _baseline_solution():
    return load_solution(_TASK_DIR / "workspace" / "solution.py")


def _compare_traces(candidate: dict[str, Any], baseline: dict[str, Any], fixtures: dict[str, Any]) -> dict[str, Any]:
    tolerance = {"rtol": 1.0e-5, "atol": 1.0e-6}
    state_match = all(
        name in candidate["state"]
        and torch.allclose(candidate["state"][name], value, **tolerance)
        for name, value in baseline["state"].items()
    ) and set(candidate["state"]) == set(baseline["state"])
    loss_match = len(candidate["loss_trajectory"]) == len(baseline["loss_trajectory"]) and (
        candidate["loss_trajectory"] == baseline["loss_trajectory"]
        or all(
            abs(left - right) <= tolerance["atol"] + tolerance["rtol"] * abs(right)
            for left, right in zip(candidate["loss_trajectory"], baseline["loss_trajectory"])
        )
    )
    telemetry_match = len(candidate["telemetry_trajectory"]) == len(baseline["telemetry_trajectory"]) and all(
        len(left) == len(right)
        and all(abs(a - b) <= tolerance["atol"] + tolerance["rtol"] * abs(b) for a, b in zip(left, right))
        for left, right in zip(candidate["telemetry_trajectory"], baseline["telemetry_trajectory"])
    )
    cadence_match = all(len(event) == int(fixtures["telemetry_width"]) for event in candidate["telemetry_trajectory"])
    work_units_match = candidate["work_units"] == baseline["work_units"]
    return {
        "final_model_equivalence": state_match,
        "loss_trajectory_equivalence": loss_match,
        "telemetry_vector_equivalence": telemetry_match,
        "cadence_preserved": cadence_match,
        "work_units_preserved": work_units_match,
        "details": {
            "steps": len(candidate["loss_trajectory"]),
            "telemetry_width": int(fixtures["telemetry_width"]),
            "baseline_work_units": baseline["work_units"],
            "candidate_work_units": candidate["work_units"],
        },
    }


def run_correctness(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    baseline = _run_trace(_baseline_solution(), copy.deepcopy(fixtures), _TRACE_STEPS)
    candidate = _run_trace(solution, copy.deepcopy(fixtures), _TRACE_STEPS)
    checks = _compare_traces(candidate, baseline, fixtures)
    return {"passed": all(checks[name] for name in checks if name != "details"), "details": checks}


def run_scientific_gates(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    result = run_correctness(solution, fixtures)
    checks = result["details"]
    return {
        name: (bool(checks[name]), checks["details"])
        for name in (
            "final_model_equivalence",
            "loss_trajectory_equivalence",
            "telemetry_vector_equivalence",
            "cadence_preserved",
            "work_units_preserved",
        )
    }


def _profile_host_reads(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    device = str(fixtures["device"])
    model = solution.build_model(fixtures).to(device)
    optimizer = _make_optimizer(model, fixtures)
    x, y = (value.to(device) for value in fixtures["train_data"])
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda") and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize()
    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False) as profile:
        solution.train_step(model, (x[: fixtures["batch_size"]], y[: fixtures["batch_size"]]), optimizer)
    item_calls = sum(event.count for event in profile.key_averages() if event.key == "aten::item")
    local_scalar_calls = sum(
        event.count for event in profile.key_averages() if event.key == "aten::_local_scalar_dense"
    )
    # Count explicit scalar reads, not implementation-internal scalarization
    # inside a batched tensor-to-host conversion.
    host_reads = int(item_calls)
    return {
        "aggregated_scalar_syncs": host_reads,
        "host_scalar_reads_per_logging_event": host_reads,
        "local_scalar_dense_events": int(local_scalar_calls),
        "telemetry_vector_width": _METRICS_PER_EVENT,
        "logging_events": 1,
    }


def run_activation_evidence(solution: Any, baseline_solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_metrics": _profile_host_reads(solution, fixtures),
        "baseline_metrics": _profile_host_reads(baseline_solution, fixtures),
    }


def run_performance(solution: Any, fixtures: dict[str, Any], warmup: int, iterations: int, device: str = "cuda") -> dict[str, Any]:
    model = solution.build_model(fixtures).to(device)
    optimizer = _make_optimizer(model, fixtures)
    x, y = (value.to(device) for value in fixtures["train_data"])
    fixed_batch = (x[: fixtures["batch_size"]], y[: fixtures["batch_size"]])
    for _ in range(warmup):
        solution.train_step(model, fixed_batch, optimizer)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        solution.train_step(model, fixed_batch, optimizer)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - started) * 1000.0
    return {
        "value": total_ms / iterations,
        "work_units": {
            "forward_calls": iterations,
            "backward_calls": iterations,
            "optimizer_steps": iterations,
            "logging_events": iterations,
            "telemetry_values": iterations * _METRICS_PER_EVENT,
        },
        "output_checksums": {},
        "timing": {"total_ms": total_ms, "per_step_ms": total_ms / iterations},
    }
