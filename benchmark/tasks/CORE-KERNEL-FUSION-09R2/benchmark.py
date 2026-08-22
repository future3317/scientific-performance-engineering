#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import importlib.util
import statistics
import time
from pathlib import Path
import torch
from benchmark.harness.api import validate_solution_api

_TASK_DIR = Path(__file__).resolve().parent


def _load(path, prefix):
    spec = importlib.util.spec_from_file_location(f"{prefix}_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}", path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


_checks = _load(_TASK_DIR / "hidden_verifier" / "checks.py", "kernel09_checks")
_science = _load(_TASK_DIR / "scientific_contract.py", "kernel09_science")


def load_solution(path, device=None):
    path = Path(path); module = _load(path / "solution.py" if path.is_dir() else path, "kernel09_solution")
    violations = validate_solution_api(module, "kernel_module_v1")
    if violations: raise RuntimeError("API violations: " + "; ".join(violations))
    return module


def make_fixtures(seed, device="cpu"):
    generator = torch.Generator().manual_seed(seed)
    graph_size, logical_steps, dynamic_shape_rate = 320, 192, 0.2
    return {
        "shape": (graph_size, 128), "graph_size": graph_size, "logical_steps": logical_steps, "dynamic_shape_rate": dynamic_shape_rate,
        "compile_profile": {"backend": "inductor", "baseline_backend": "eager", "mode": "reduce-overhead", "dynamic": True},
        "a1": 0.5 + torch.rand(1, generator=generator).item() * 1.5, "b1": -1.0 + torch.rand(1, generator=generator).item() * 2.0,
        "a2": 0.3 + torch.rand(1, generator=generator).item() * 1.7, "b2": -0.5 + torch.rand(1, generator=generator).item() * 1.0,
        "clamp_min": -2.0, "clamp_max": 2.0, "device": device, "seed": seed,
    }


def workload_shapes(fixtures):
    base_rows, width = fixtures["shape"]
    alternate_rows = base_rows + max(1, round(base_rows * fixtures["dynamic_shape_rate"]))
    switch_period = max(1, round(1.0 / fixtures["dynamic_shape_rate"]))
    return [(alternate_rows, width) if index % switch_period == 0 else (base_rows, width) for index in range(fixtures["logical_steps"])]


def _draw_inputs(fixtures, shape=None, seed_offset=33331):
    generator = torch.Generator(device=fixtures["device"]).manual_seed(fixtures["seed"] + seed_offset)
    shape = shape or fixtures["shape"]
    return torch.randn(shape, generator=generator, device=fixtures["device"]), torch.randn(shape, generator=generator, device=fixtures["device"])


def run_correctness(solution, fixtures):
    x, residual = _draw_inputs(fixtures)
    context = solution.init(fixtures)
    return _checks.check_output(lambda **kwargs: solution.forward(context, **kwargs), x, residual, fixtures, 1e-4, 1e-5)


def run_scientific_gates(solution, fixtures):
    x, residual = _draw_inputs(fixtures); context = solution.init(fixtures); output = solution.forward(context, x, residual)
    return {"finite_output": _science.finite_output(output), "output_moment_match": _science.output_moment_match(output, x, residual, fixtures)}


def run_performance(solution, fixtures, warmup=5, iterations=25, device="cpu"):
    context = solution.init(fixtures); shapes = workload_shapes(fixtures); generator = torch.Generator(device=fixtures["device"]).manual_seed(fixtures["seed"] + 9001)
    inputs = [(torch.randn(shape, generator=generator, device=fixtures["device"]), torch.randn(shape, generator=generator, device=fixtures["device"])) for shape in shapes]
    for index in range(warmup): solution.forward(context, *inputs[index % len(inputs)])
    times, output = [], None
    for index in range(fixtures["logical_steps"]):
        start = time.perf_counter(); output = solution.forward(context, *inputs[index]); times.append((time.perf_counter() - start) * 1000.0)
    value = statistics.median(times)
    return {"value": value, "work_units": {"forward_calls": fixtures["logical_steps"], "logical_steps": fixtures["logical_steps"], "graph_size": fixtures["graph_size"]}, "output_checksums": {"final": _checks.checksum_tensor(output)}, "timing": {"times_ms": times, "median_ms": value}}



