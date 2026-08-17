"""Shared harness for the three compile-family calibration anchors.

The anchors deliberately share measurement plumbing but not workload semantics:
recompile measures graph-break plus cold shape specialization, dynamic measures
shape specialization with a tensor-only graph, and tiny measures whether a
short-lived workload should be compiled at all.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.util
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from benchmark.harness.api import validate_solution_api


def _import_module_by_path(path: str | Path, prefix: str) -> Any:
    path = Path(path)
    name = f"{prefix}_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass
    return module


class TinyResMLP(torch.nn.Module):
    """Reference fixture architecture; task solutions may use the same API."""

    def __init__(self, in_dim: int, hidden_dim: int, num_blocks: int = 4):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.blocks = torch.nn.ModuleList(
            torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_blocks)
        )
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.relu(self.fc1(x))
        for lin in self.blocks:
            out = torch.nn.functional.relu(lin(h))
            h = h + out
        return self.fc2(h)


def _dynamo_diagnostics() -> dict[str, Any]:
    try:
        import torch._dynamo.utils as dynamo_utils

        counters = {
            str(group): {str(key): int(value) for key, value in values.items()}
            for group, values in dynamo_utils.counters.items()
            if values
        }
        compile_times = dynamo_utils.compile_times()
    except Exception as exc:  # pragma: no cover - version-specific torch internals
        return {"available": False, "error": repr(exc)}
    return {"available": True, "counters": counters, "compile_times": str(compile_times)}


def _reset_dynamo_diagnostics() -> None:
    try:
        import torch._dynamo.utils as dynamo_utils

        dynamo_utils.counters.clear()
    except Exception:
        pass


def _set_compile_threads(profile: dict[str, Any]) -> None:
    # The task contract fixes this value for A/B/C/D; it must not depend on the
    # host's affinity-derived default (which can fan out to dozens of workers).
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(profile["compile_threads"])


def _make_fixtures(seed: int, device: str, profile: dict[str, Any]) -> dict[str, Any]:
    rng = torch.Generator().manual_seed(seed)
    data_config = {"num_samples": int(profile["num_samples"]), "in_dim": int(profile["in_dim"])}
    model_config = {
        "in_dim": data_config["in_dim"],
        "hidden_dim": int(profile["hidden_dim"]),
        "num_blocks": int(profile["num_blocks"]),
    }
    batch_sizes = list(profile["batch_sizes"])
    inputs = torch.randn(
        data_config["num_samples"], data_config["in_dim"], generator=rng, dtype=torch.float32
    )
    teacher = torch.nn.Linear(data_config["in_dim"], 1)
    with torch.no_grad():
        teacher.weight.normal_(0.0, 0.1, generator=rng)
        teacher.bias.normal_(0.0, 0.1, generator=rng)
        targets = teacher(inputs).squeeze(-1)

    init_model = TinyResMLP(
        model_config["in_dim"], model_config["hidden_dim"], model_config["num_blocks"]
    )
    with torch.no_grad():
        for parameter in init_model.parameters():
            parameter.normal_(0.0, 0.01, generator=rng)
    return {
        "device": device,
        "data_config": data_config,
        "model_config": model_config,
        "optimizer_config": {"lr": 0.001},
        "batch_sizes": batch_sizes,
        "inputs": inputs,
        "targets": targets,
        "init_state": init_model.state_dict(),
        "eval_inputs": torch.randn(128, data_config["in_dim"], generator=rng, dtype=torch.float32),
        "compile_profile": dict(profile),
    }


def _batch_at(fixtures: dict[str, Any], index: int) -> tuple[torch.Tensor, torch.Tensor]:
    sizes = fixtures["batch_sizes"]
    size = sizes[index % len(sizes)]
    offset = (index * size) % (fixtures["inputs"].shape[0] - size)
    return fixtures["inputs"][offset : offset + size], fixtures["targets"][offset : offset + size]


def _run_performance(
    solution: Any,
    fixtures: dict[str, Any],
    warmup: int,
    iterations: int,
    device: str,
) -> dict[str, Any]:
    profile = fixtures["compile_profile"]
    _set_compile_threads(profile)
    _reset_dynamo_diagnostics()
    model = solution.build_model(fixtures)
    optimizer = torch.optim.SGD(model.parameters(), lr=fixtures["optimizer_config"]["lr"])
    batch_at = getattr(solution, "_batch_at", None)
    schedule = batch_at or _batch_at
    measured: list[float] = []
    final_loss = None

    def run_step(index: int) -> None:
        nonlocal final_loss
        batch = schedule(fixtures, index)
        final_loss = solution.train_step(model, batch, optimizer)["loss"]

    for index in range(warmup):
        run_step(index)
    for index in range(iterations):
        start = time.perf_counter()
        run_step(index)
        measured.append((time.perf_counter() - start) * 1000.0)

    cycle = max(1, len(fixtures["batch_sizes"]))
    cold_schedule_ms = sum(measured[:cycle])
    steady = measured[cycle:] or measured
    peak_rss_mb = None
    try:
        import resource

        peak_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0)
    except Exception:
        pass
    return {
        # The primary estimand includes the first encounter with every
        # preregistered shape; the post-cycle median remains diagnostic only.
        "value": sum(measured) if profile.get("primary_scope") == "full_schedule" else cold_schedule_ms,
        "work_units": {"forward": iterations, "backward": iterations, "optimizer": iterations},
        "output_checksums": {"final_loss": hashlib.sha256(final_loss.detach().cpu().numpy().tobytes()).hexdigest() if final_loss is not None else None},
        "timing": {
            "metric": "schedule_wall_ms" if profile.get("primary_scope") == "full_schedule" else "cold_shape_schedule_ms",
            "step_times_ms": measured,
            "cold_schedule_ms": cold_schedule_ms,
            "steady_state_median_ms": statistics.median(steady),
            "shape_schedule": list(fixtures["batch_sizes"]),
            "compile_threads": int(profile["compile_threads"]),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "peak_rss_mb": peak_rss_mb,
            "dynamo": _dynamo_diagnostics(),
        },
    }


def configure(task_dir: str | Path, profile: dict[str, Any]) -> dict[str, Any]:
    task_dir = Path(task_dir)
    checks = _import_module_by_path(task_dir / "hidden_verifier" / "checks.py", "spe_compile_checks")
    science = _import_module_by_path(task_dir / "scientific_contract.py", "spe_compile_science")
    cache_roots: dict[str, tempfile.TemporaryDirectory[str]] = {}

    def cleanup_compile_caches() -> None:
        for cache in cache_roots.values():
            cache.cleanup()

    atexit.register(cleanup_compile_caches)

    def load_solution(path: str, device: str | None = None) -> Any:
        _set_compile_threads(profile)
        cache_key = str(Path(path).resolve())
        if cache_key not in cache_roots:
            cache_roots[cache_key] = tempfile.TemporaryDirectory(prefix="spe-compile-cache-")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_roots[cache_key].name
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        module = _import_module_by_path(path, "spe_compile_solution")
        violations = validate_solution_api(module, "train_loop_v1")
        if violations:
            raise RuntimeError("API violations: " + "; ".join(violations))
        return module

    def make_fixtures(seed: int, device: str = "cpu") -> dict[str, Any]:
        return _make_fixtures(seed, device, profile)

    def run_correctness(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
        return checks.check_training_correctness(solution, fixtures, rtol=1.0e-4, atol=1.0e-5, train_steps=5)

    def run_scientific_gates(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
        return science.run_gates(solution, fixtures)

    def run_performance(solution: Any, fixtures: dict[str, Any], warmup: int = 0, iterations: int = 16, device: str = "cpu") -> dict[str, Any]:
        return _run_performance(solution, fixtures, warmup, iterations, device)

    return {
        "load_solution": load_solution,
        "make_fixtures": make_fixtures,
        "run_correctness": run_correctness,
        "run_scientific_gates": run_scientific_gates,
        "run_performance": run_performance,
    }
