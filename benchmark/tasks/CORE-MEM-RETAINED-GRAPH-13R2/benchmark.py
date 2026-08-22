"""Harness entry for the real retained-graph segmented workload."""
from __future__ import annotations

import hashlib
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn
from benchmark.harness.api import validate_solution_api

_TASK_DIR = Path(__file__).resolve().parent


def _load(path, prefix):
    spec = importlib.util.spec_from_file_location(f"{prefix}_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


_checks = _load(_TASK_DIR / "hidden_verifier" / "checks.py", "mem13_checks")
_science = _load(_TASK_DIR / "scientific_contract.py", "mem13_science")


class _FixtureModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, segment_count):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.segments = nn.ModuleList(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(segment_count))
        self.output = nn.Linear(hidden_dim, 1)


def load_solution(path, device=None):
    path = Path(path)
    module = _load(path / "solution.py" if path.is_dir() else path, "mem13_solution")
    violations = validate_solution_api(module, "train_loop_v1")
    if violations:
        raise RuntimeError("API violations: " + "; ".join(violations))
    return module


def make_fixtures(seed, device="cpu"):
    generator = torch.Generator().manual_seed(seed)
    memory_pressure, segment_count, recompute_ratio = 0.7, 4, 0.2
    hidden_dim = 64 + int(128 * memory_pressure)
    model = _FixtureModel(16, hidden_dim, segment_count)
    for parameter in model.parameters():
        parameter.data.normal_(0.0, 0.02, generator=generator)
    return {
        "device": device, "model_config": {"in_dim": 16, "hidden_dim": hidden_dim, "segment_count": segment_count, "checkpointed_segments": max(1, round(segment_count * recompute_ratio))},
        "optimizer_config": {"lr": 0.001}, "memory_pressure": memory_pressure,
        "segment_count": segment_count, "recompute_ratio": recompute_ratio,
        "checkpointed_segments": max(1, round(segment_count * recompute_ratio)),
        "batch_sizes": [16, 24, 32, 40],
        "inputs": torch.randn(512, 16, generator=generator),
        "targets": torch.randn(512, generator=generator), "init_state": model.state_dict(),
        "eval_inputs": torch.randn(32, 16, generator=generator),
    }


def run_correctness(solution, fixtures):
    return _checks.check_training_correctness(solution, fixtures, 1e-4, 1e-5)


def run_scientific_gates(solution, fixtures):
    return {"finite_loss": _science.finite_loss_gate(solution, fixtures)}


def run_performance(solution, fixtures, warmup=5, iterations=24, device="cpu"):
    model = solution.build_model(fixtures)
    optimizer = torch.optim.SGD(model.parameters(), lr=fixtures["optimizer_config"]["lr"])
    def batch_at(i):
        size = 16
        offset = (i * size) % (fixtures["inputs"].shape[0] - size)
        return fixtures["inputs"][offset:offset + size], fixtures["targets"][offset:offset + size]
    for i in range(warmup):
        solution.train_step(model, batch_at(i), optimizer)
    times, last = [], None
    for i in range(iterations):
        start = time.perf_counter(); last = solution.train_step(model, batch_at(i + warmup), optimizer)
        times.append((time.perf_counter() - start) * 1000.0)
    value = statistics.median(times)
    return {"value": value, "work_units": {"forward": iterations, "backward": iterations, "optimizer": iterations}, "output_checksums": {"loss": _checks.checksum_tensor(last["loss"].reshape(1))}, "timing": {"step_times_ms": times, "median_ms": value}}


