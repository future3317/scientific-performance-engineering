"""Baseline Langevin-style sampler over atom positions (sampler_v1 API).

The sampler updates atom positions every step using a simple Lennard-Jones-like
pair potential on a radius neighbor graph.  Because positions change every step,
the neighbor graph MUST be rebuilt each step; this baseline does exactly that.

The tempting (poisoned) optimization is to cache the neighbor list across steps
and reuse it.  That is scientifically wrong here: stale neighbors produce wrong
forces and break energy/force consistency and the sampled distribution.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def _build_graph(positions: torch.Tensor, r_cut: float) -> torch.Tensor:
    """All-pairs radius graph (no PBC) for a single chain.  Returns edge_index [2, E]."""
    # Baseline: rebuild from scratch every call.
    # Fixtures use a single chain (batch dimension 1), so squeeze it.
    positions = positions.squeeze(0)
    n = positions.shape[0]
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [N, N, 3]
    dist = torch.linalg.vector_norm(diff, dim=-1)
    mask = (dist <= r_cut) & (dist > 1e-8)
    row, col = torch.where(mask)
    return torch.stack([row, col], dim=0)


def _pair_potential(dist: torch.Tensor, r_cut: float, r0: float = 1.2) -> torch.Tensor:
    """Soft harmonic well (zero beyond r_cut)."""
    v = (dist - r0) ** 2
    v = torch.where(dist <= r_cut, v, torch.zeros_like(v))
    return v


def _step_noise(positions: torch.Tensor, step_index: int, noise_scale: float) -> torch.Tensor:
    """Deterministic smooth thermal-like displacement for this step.

    Uses a fixed random phase per coordinate and a sinusoidal time term so the
    displacement is bounded, differentiable in the reference, and guaranteed to
    move atoms across the neighbor cutoff as the sampler runs.
    """
    generator = torch.Generator().manual_seed(2025)
    phase = torch.rand(positions.shape, generator=generator, dtype=torch.float64)
    phase = phase.to(positions.dtype).to(positions.device)
    return noise_scale * torch.sin(phase * 2.0 * math.pi + step_index * 0.1)


def _energy_and_forces(sampler: dict[str, Any], positions: torch.Tensor, rebuild: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Energy and forces for the current positions (single-chain batch)."""
    if rebuild:
        edge_index = _build_graph(positions, sampler["r_cut"]).detach()
        sampler["cached_edge_index"] = edge_index
    else:
        edge_index = sampler.get("cached_edge_index")
        if edge_index is None:
            edge_index = _build_graph(positions, sampler["r_cut"]).detach()
            sampler["cached_edge_index"] = edge_index

    p = positions.squeeze(0)
    row, col = edge_index[0], edge_index[1]
    vec = p[col] - p[row]
    dist = torch.linalg.vector_norm(vec, dim=-1)
    energy = _pair_potential(dist, sampler["r_cut"], sampler.get("r0", 1.2)).sum()

    # dV/dr = 2*(r - r0); force = -grad(V) = dV/dr * unit_vec (row -> col).
    inv_dist = 1.0 / dist
    dVdr = 2.0 * (dist - sampler.get("r0", 1.2))
    pair_forces = dVdr.unsqueeze(-1) * (vec * inv_dist.unsqueeze(-1))

    forces = torch.zeros_like(p)
    forces.index_add_(0, row, pair_forces)
    forces.index_add_(0, col, -pair_forces)
    return energy, forces.unsqueeze(0)


def build_sampler(fixtures: dict[str, Any]) -> dict[str, Any]:
    """Build a sampler context from the task fixtures."""
    config = fixtures["config"]
    return {
        "r_cut": float(config["r_cut"]) + 0.02 * (int(config["neighbor_count"]) - 12),
        "r0": float(config.get("r0", 1.2)),
        "dt": float(config["dt"]),
        "box": float(config["box"]),
        "noise_scale": float(config.get("noise_scale", 0.0)),
        "device": fixtures["device"],
        "cached_edge_index": None,
    }


def sample_step(sampler: dict[str, Any], state: dict[str, torch.Tensor], step_index: int) -> dict[str, torch.Tensor]:
    """One deterministic Langevin-ish step.  Positions change every step."""
    positions = state["positions"]
    # Baseline: rebuild the neighbor graph every step (correct behavior).
    _, forces = _energy_and_forces(sampler, positions, rebuild=True)
    noise = _step_noise(positions, step_index, sampler.get("noise_scale", 0.0))
    new_positions = positions + sampler["dt"] * forces + noise
    # Keep inside a cubic box with reflective-ish wrap (simple clamp for determinism).
    new_positions = torch.clamp(new_positions, min=0.0, max=sampler["box"])
    # Cache a fresh graph for the new state so energy/force consistency checks
    # and the scientific gate see the graph that matches the current positions.
    _ = _energy_and_forces(sampler, new_positions, rebuild=True)
    return {"positions": new_positions}


def sample(sampler: dict[str, Any], fixtures: dict[str, Any], num_steps: int) -> torch.Tensor:
    """Run the sampler for num_steps and return final positions."""
    state = {"positions": fixtures["init_positions"].clone()}
    for t in range(num_steps):
        state = sample_step(sampler, state, t)
    return state["positions"]


# Extras used by the scientific gate to inspect the candidate's graph/forces.
def energy_forces(sampler: dict[str, Any], positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Energy and forces using the candidate's current graph (cached or rebuilt)."""
    return _energy_and_forces(sampler, positions, rebuild=False)
