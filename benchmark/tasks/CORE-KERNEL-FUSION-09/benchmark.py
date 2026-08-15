#!/usr/bin/env python3
"""Harness-side entry for CORE-KERNEL-FUSION-09 (kernel_module_v1 API).

Exposes the five functions the harness verifier/runner expect:
  load_solution, make_fixtures, run_correctness,
  run_scientific_gates, run_performance.

NOTE: The harness inserts the task directory at the front of ``sys.path`` when
it imports this module by path. Because the task directory contains a file named
``benchmark.py``, that file shadows the top-level ``benchmark`` package, so any
``import benchmark.harness.*`` from this module or its submodules would resolve
to the task file instead of the harness package. We work around this by
removing the task directory from ``sys.path`` while importing harness modules,
then restoring it to import task-local helpers (hidden_verifier/,
scientific_contract.py). This is an in-task workaround for a harness path
shadowing issue; the harness itself is not modified.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import torch

_TASK_DIR = str(Path(__file__).resolve().parent)

# Work around task-dir ``benchmark.py`` shadowing the top-level package.
if _TASK_DIR in sys.path:
    sys.path.remove(_TASK_DIR)
from benchmark.harness import anticheat  # noqa: E402
from benchmark.harness.api import validate_solution_api  # noqa: E402

sys.path.insert(0, _TASK_DIR)
from hidden_verifier import checks  # noqa: E402
from scientific_contract import finite_output, output_moment_match  # noqa: E402


def load_solution(path: str | Path, device: str | None = None) -> Any:
    """Import the workspace solution.py and validate its kernel_module_v1 API."""
    from benchmark.harness import runner

    path = Path(path)
    if path.is_dir():
        path = path / "solution.py"
    module = runner.import_module_by_path(path)
    violations = validate_solution_api(module, "kernel_module_v1")
    if violations:
        raise RuntimeError("API violations: " + "; ".join(violations))
    return module


def make_fixtures(seed: int, device: str = "cpu") -> dict[str, Any]:
    """Deterministic fixture: scalar params + shape, input tensors drawn per call."""
    generator = torch.Generator().manual_seed(seed)
    shape = (1024, 4096)
    fixtures = {
        "shape": shape,
        "a1": 0.5 + torch.rand(1, generator=generator).item() * 1.5,
        "b1": -1.0 + torch.rand(1, generator=generator).item() * 2.0,
        "a2": 0.3 + torch.rand(1, generator=generator).item() * 1.7,
        "b2": -0.5 + torch.rand(1, generator=generator).item() * 1.0,
        "clamp_min": -2.0,
        "clamp_max": 2.0,
        "device": device,
        "seed": seed,
    }
    return fixtures


def _draw_inputs(fixtures: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Fresh random inputs for one correctness/probe trial."""
    device = fixtures["device"]
    shape = fixtures["shape"]
    # Use a separate generator seeded from fixture metadata so inputs vary
    # between trials while remaining deterministic for a given fixture.
    generator = torch.Generator(device=device).manual_seed(fixtures["seed"] + 33331)
    x = torch.randn(shape, generator=generator, dtype=torch.float32, device=device)
    residual = torch.randn(shape, generator=generator, dtype=torch.float32, device=device)
    return x, residual


def run_correctness(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """S2: correctness vs fp64 reference + adversarial result-reuse probes."""
    x, residual = _draw_inputs(fixtures)
    context = solution.init(fixtures)

    def forward_fn(*, x, residual):
        return solution.forward(context, x, residual)

    result = checks.check_output(forward_fn, x, residual, fixtures, rtol=1.0e-4, atol=1.0e-5)
    return result


def run_scientific_gates(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """S3: task-declared scientific gates (finite_output, output_moment_match)."""
    x, residual = _draw_inputs(fixtures)
    context = solution.init(fixtures)
    output = solution.forward(context, x, residual)

    gates = {}
    passed, details = finite_output(output)
    gates["finite_output"] = (passed, details)
    passed, details = output_moment_match(output, x, residual, fixtures)
    gates["output_moment_match"] = (passed, details)
    return gates


def _output_checksum(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous()
    return hashlib.sha256(arr.numpy().tobytes()).hexdigest()


def run_performance(
    solution: Any,
    fixtures: dict[str, Any],
    warmup: int = 5,
    iterations: int = 25,
    device: str = "cpu",
) -> dict[str, Any]:
    """S5: time the forward call over repeated iterations.

    Returns a dict with ``value`` (median step ms), work-unit counters, and
    output checksums so the harness can compare baseline vs candidate.
    """
    x, residual = _draw_inputs(fixtures)
    context = solution.init(fixtures)

    # Warmup
    for _ in range(warmup):
        solution.forward(context, x, residual)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(iterations):
        if device.startswith("cuda"):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            out = solution.forward(context, x, residual)
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))
        else:
            start = time.perf_counter()
            out = solution.forward(context, x, residual)
            times_ms.append((time.perf_counter() - start) * 1000.0)

    median_ms = sorted(times_ms)[len(times_ms) // 2]
    return {
        "value": median_ms,
        "work_units": {"forward_calls": iterations},
        "output_checksums": {"final": _output_checksum(out)},
        "timing": {"times_ms": times_ms},
    }
