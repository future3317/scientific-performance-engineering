"""Harness-side measurement entry for SCIML-CRYSTAL-DIFFUSION-07R2."""

from __future__ import annotations

import time
import math
from pathlib import Path
from typing import Any

import torch

from benchmark.harness import runner

from hidden_verifier.checks import check_correctness, reference_sample
from scientific_contract import crystal_validity_rate, distribution_moment_check


CONFIG = {
    "num_atoms": 32,
    "num_steps": 100,
    "hidden": 64,
    "time_emb_dim": 16,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "guidance_scale": 2.0,
    "validity_threshold": 0.9,
    "time_budget_s": 600.0,
}


def load_solution(path: str | Path, device: str = "cpu") -> Any:
    """Import the workspace solution module."""
    module = runner.import_module_by_path(Path(path))
    return module


def _make_score_state(config: dict[str, Any], generator: torch.Generator, device: torch.device) -> dict[str, torch.Tensor]:
    """Deterministic small random weights for the score network."""
    num_atoms = int(config["num_atoms"])
    time_emb_dim = int(config["time_emb_dim"])
    hidden = int(config["hidden"])
    in_dim = num_atoms * 3 + 3 + time_emb_dim
    out_dim = num_atoms * 3 + 3
    state = {
        "fc1.weight": (torch.rand(hidden, in_dim, generator=generator) * 0.1 - 0.05).to(device),
        "fc1.bias": torch.zeros(hidden, device=device),
        "fc2.weight": (torch.rand(out_dim, hidden, generator=generator) * 0.1 - 0.05).to(device),
        "fc2.bias": torch.zeros(out_dim, device=device),
    }
    return state


def _make_initial_state(
    config: dict[str, Any],
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate near-valid perturbed cubic crystals (fractional coords + lengths)."""
    num_atoms = int(config["num_atoms"])
    # Simple-cubic grid in fractional space with small noise.
    side = max(2, int(math.ceil(num_atoms ** (1.0 / 3.0))))
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 0.75, side),
            torch.linspace(0.0, 0.75, side),
            torch.linspace(0.0, 0.75, side),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    grid = grid[:num_atoms]

    frac_list = []
    lengths_list = []
    base_length = 10.0
    for _ in range(batch_size):
        noise = (torch.randn(num_atoms, 3, generator=generator) * 0.01).to(device)
        frac = ((grid.to(device) + noise) % 1.0)
        length_noise = (torch.randn(1, generator=generator) * 0.05).to(device)
        lengths = torch.tensor([base_length, base_length, base_length], device=device) + length_noise
        frac_list.append(frac)
        lengths_list.append(lengths)
    frac = torch.stack(frac_list, dim=0)
    lengths = torch.stack(lengths_list, dim=0)
    return frac, lengths


def _reference_moments(
    config: dict[str, Any],
    score_state: dict[str, torch.Tensor],
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    """Empirical moments of the reference sampler output distribution."""
    n_ref = 64
    frac, lengths = _make_initial_state(config, n_ref, generator, device)
    init_state = {"frac": frac, "lengths": lengths}
    with torch.no_grad():
        samples = reference_sample(init_state, score_state, config, int(config["num_steps"]), device)
    flat = samples.detach().flatten().double()
    return {"mean": flat.mean().item(), "std": flat.std(unbiased=False).item()}


def make_fixtures(seed: int, device: str = "cpu") -> dict[str, Any]:
    """Deterministic, self-contained fixtures."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    dev = torch.device(device)

    config = dict(CONFIG)
    score_state = _make_score_state(config, generator, dev)

    # Training/evaluation init states are drawn from the same generator.
    init_frac, init_lengths = _make_initial_state(config, 8, generator, dev)
    eval_frac, eval_lengths = _make_initial_state(config, 8, generator, dev)

    reference_moments = _reference_moments(config, score_state, dev, generator)

    return {
        "config": config,
        "guidance_scale": float(config["guidance_scale"]),
        "score_state": score_state,
        "init_frac": init_frac,
        "init_lengths": init_lengths,
        "eval_frac": eval_frac,
        "eval_lengths": eval_lengths,
        "reference_moments": reference_moments,
        "device": device,
    }


def run_correctness(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """S2 correctness: candidate vs fp64 live recompute + input probes."""
    sampler = solution.build_sampler(fixtures)

    def sample_fn(frac: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        sub_fixtures = {**fixtures, "init_frac": frac, "init_lengths": lengths}
        return solution.sample(sampler, sub_fixtures, int(fixtures["config"]["num_steps"]))

    return check_correctness(sample_fn, fixtures, rtol=1.0e-4, atol=1.0e-4)


def run_scientific_gates(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """S3 scientific gates."""
    sampler = solution.build_sampler(fixtures)
    eval_fixtures = {
        **fixtures,
        "init_frac": fixtures["eval_frac"],
        "init_lengths": fixtures["eval_lengths"],
    }
    samples = solution.sample(sampler, eval_fixtures, int(fixtures["config"]["num_steps"]))

    passed_validity, details_validity = crystal_validity_rate(samples, fixtures["config"])
    passed_moments, details_moments = distribution_moment_check(samples, fixtures["reference_moments"])

    return {
        "crystal_validity": (passed_validity, details_validity),
        "distribution_moment_check": (passed_moments, details_moments),
    }


def run_performance(
    solution: Any,
    fixtures: dict[str, Any],
    warmup: int = 0,
    iterations: int = 1,
    device: str = "cpu",
) -> dict[str, Any]:
    """Time-to-quality measurement over the fixed eval batch."""
    sampler = solution.build_sampler(fixtures)
    eval_fixtures = {
        **fixtures,
        "init_frac": fixtures["eval_frac"],
        "init_lengths": fixtures["eval_lengths"],
    }
    num_steps = int(fixtures["config"]["num_steps"])

    # Warmup
    for _ in range(warmup):
        solution.sample(sampler, eval_fixtures, num_steps)

    # Measured: wall-clock to reach the quality threshold.
    total_time = 0.0
    reached = False
    for _ in range(iterations):
        # The sampler is deterministic; validity is evaluated on the full eval batch.
        start = time.perf_counter()
        samples = solution.sample(sampler, eval_fixtures, num_steps)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        passed_validity, _ = crystal_validity_rate(samples, fixtures["config"])
        if passed_validity:
            reached = True
            break

    value = total_time if reached else float(fixtures["config"]["time_budget_s"])
    batch_size = fixtures["eval_frac"].shape[0]
    num_atoms = int(fixtures["config"]["num_atoms"])
    work_units = {
        "crystals": batch_size,
        "atoms": batch_size * num_atoms,
        "steps": num_steps,
    }
    return {
        "value": value,
        "work_units": work_units,
        "output_checksums": {},
        "timing": {"reached": reached, "iterations_to_reach": 1 if reached else iterations},
    }
