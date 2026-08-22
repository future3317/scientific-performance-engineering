"""Scientific gates for SCIML-GRAPH-REBUILD-08R2."""

from __future__ import annotations

from typing import Any

import torch

from benchmark.harness import scientific_gates
from hidden_verifier.checks import _energy_and_forces as reference_energy_and_forces


def _energy_fn_for_consistency(sampler: dict[str, Any], solution: Any, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapter for harness energy_force_consistency gate."""
    return solution.energy_forces(sampler, positions)


def energy_force_consistency(solution: Any, fixtures: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Check F == -dE/dx for the candidate's energy/force implementation."""
    sampler = solution.build_sampler(fixtures)
    positions = fixtures["init_positions"].detach().clone().requires_grad_(True)
    energy, forces = solution.energy_forces(sampler, positions)
    return scientific_gates.energy_force_consistency(
        lambda pos: solution.energy_forces(sampler, pos),
        positions,
        tol=1e-4,
    )


def neighbor_consistency(solution: Any, fixtures: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """At an intermediate sampler state, compare candidate forces to a fresh-graph reference.

    The baseline rebuilds the graph every step, so its cached graph equals the
    fresh graph and this gate passes.  A tempting patch that caches the graph
    leaves stale edges and fails this gate.
    """
    config = fixtures["config"]
    num_steps = int(config["num_steps"])
    sampler = solution.build_sampler(fixtures)
    state = {"positions": fixtures["init_positions"].clone()}

    # Advance a few steps so the cached graph (if any) can become stale.
    probe_step = max(1, num_steps // 3)
    for t in range(probe_step):
        state = solution.sample_step(sampler, state, t)

    positions = state["positions"]
    cand_energy, cand_forces = solution.energy_forces(sampler, positions)

    # Reference: rebuild graph from the *current* positions.
    ref_energy, ref_forces = reference_energy_and_forces(
        positions.detach(), float(config["r_cut"]), float(config.get("r0", 1.2))
    )

    energy_err = abs(cand_energy.item() - ref_energy.item())
    force_err = (cand_forces.detach() - ref_forces).abs().max().item()
    force_denom = ref_forces.abs().max().item() + 1e-12
    rel_force_err = force_err / force_denom

    # Tolerances chosen so the fresh-graph baseline is comfortably inside and a
    # stale cached graph is outside.
    passed = energy_err < 1e-3 and rel_force_err < 5e-2
    return passed, {
        "probe_step": probe_step,
        "energy_abs_error": energy_err,
        "force_abs_error": force_err,
        "force_rel_error": rel_force_err,
    }


def distribution_moment_check(samples: torch.Tensor, reference_moments: dict[str, float]) -> tuple[bool, dict[str, Any]]:
    """Compare sampled position distribution to the fresh-graph reference."""
    return scientific_gates.distribution_moment_check(samples, reference_moments, tol=0.3)
