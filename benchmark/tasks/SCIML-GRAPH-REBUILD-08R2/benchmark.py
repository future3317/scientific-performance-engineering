"""Harness-side measurement entry for SCIML-GRAPH-REBUILD-08R2."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from benchmark.harness import runner

from hidden_verifier.checks import check_correctness, reference_sample
from scientific_contract import distribution_moment_check, energy_force_consistency, neighbor_consistency


CONFIG = {
    "num_atoms": 24,
    "num_steps": 120,
    "r_cut": 2.5,
    "r0": 2.2,
    "dt": 0.01,
    "box": 8.0,
    "noise_scale": 0.12,
    "neighbor_count": 12,
    "sample_count": 24,
    "geometry_variation": 0.7,
}


def load_solution(path: str | Path, device: str = "cpu") -> Any:
    """Import the workspace solution module."""
    module = runner.import_module_by_path(Path(path))
    return module


def _make_initial_positions(
    config: dict[str, Any],
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Place atoms on a grid with small noise to stay valid."""
    num_atoms = int(config["num_atoms"])
    box = float(config["box"])
    side = max(2, int(round(num_atoms ** (1.0 / 3.0))))
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(2.0, box - 2.0, side),
            torch.linspace(2.0, box - 2.0, side),
            torch.linspace(2.0, box - 2.0, side),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    grid = grid[:num_atoms]

    positions = []
    for _ in range(batch_size):
        noise = torch.randn(num_atoms, 3, generator=generator) * (0.1 * float(config["geometry_variation"]))
        positions.append((grid + noise).clamp(min=0.0, max=box))
    return torch.stack(positions, dim=0).to(device)


def _reference_moments(
    config: dict[str, Any],
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    """Empirical moments of the fresh-graph reference distribution."""
    n_ref = 64
    samples: list[torch.Tensor] = []
    for _ in range(n_ref):
        positions = _make_initial_positions(config, 1, generator, device)
        with torch.no_grad():
            samples.append(reference_sample(positions, config, int(config["num_steps"])))
    flat = torch.cat(samples, dim=0).detach().flatten().double()
    return {"mean": flat.mean().item(), "std": flat.std(unbiased=False).item()}


def make_fixtures(seed: int, device: str = "cpu") -> dict[str, Any]:
    """Deterministic, self-contained fixtures."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    dev = torch.device(device)

    config = dict(CONFIG)
    config["num_atoms"] = int(config["sample_count"])
    init_positions = _make_initial_positions(config, 1, generator, dev)
    reference_moments = _reference_moments(config, dev, generator)

    return {
        "config": config,
        "init_positions": init_positions,
        "initial": init_positions.squeeze(0),
        "neighbor_count": int(config["neighbor_count"]),
        "sample_count": int(config["sample_count"]),
        "geometry_variation": float(config["geometry_variation"]),
        "reference_moments": reference_moments,
        "device": device,
    }


def run_correctness(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """S2 correctness: candidate vs fp64 live recompute + probes."""
    sampler = solution.build_sampler(fixtures)

    def sample_fn(positions: torch.Tensor) -> torch.Tensor:
        sub_fixtures = {**fixtures, "init_positions": positions}
        return solution.sample(sampler, sub_fixtures, int(fixtures["config"]["num_steps"]))

    return check_correctness(sample_fn, fixtures, rtol=1.0, atol=3.0)


def run_scientific_gates(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """S3 scientific gates."""
    samples = solution.sample(solution.build_sampler(fixtures), fixtures, int(fixtures["config"]["num_steps"]))

    passed_efc, details_efc = energy_force_consistency(solution, fixtures)
    passed_nc, details_nc = neighbor_consistency(solution, fixtures)
    passed_moments, details_moments = distribution_moment_check(samples, fixtures["reference_moments"])

    return {
        "energy_force_consistency": (passed_efc, details_efc),
        "neighbor_consistency": (passed_nc, details_nc),
        "distribution_moment_check": (passed_moments, details_moments),
    }


def run_performance(
    solution: Any,
    fixtures: dict[str, Any],
    warmup: int = 0,
    iterations: int = 1,
    device: str = "cpu",
) -> dict[str, Any]:
    """Per-step wall-clock measurement."""
    sampler = solution.build_sampler(fixtures)
    num_steps = int(fixtures["config"]["num_steps"])

    for _ in range(warmup):
        solution.sample(sampler, fixtures, num_steps)

    times_s: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        solution.sample(sampler, fixtures, num_steps)
        times_s.append(time.perf_counter() - start)

    times_s.sort()
    median_s = times_s[len(times_s) // 2] if times_s else 0.0
    per_step_ms = (median_s / num_steps) * 1000.0

    num_atoms = int(fixtures["config"]["num_atoms"])
    work_units = {
        "chains": 1,
        "atoms": num_atoms,
        "steps": num_steps,
    }
    return {
        "value": per_step_ms,
        "work_units": work_units,
        "output_checksums": {},
        "timing": {"median_total_ms": median_s * 1000.0, "per_step_ms": per_step_ms},
    }
