"""Hidden verifier helpers for SCIML-GRAPH-REBUILD-08.

fp64 live recompute of the fresh-graph Langevin dynamics and result-reuse
probes.  Nothing here is visible to the agent sandbox.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable

import torch


def _build_graph(positions: torch.Tensor, r_cut: float) -> torch.Tensor:
    """All-pairs radius graph (no PBC) for a single chain."""
    positions = positions.squeeze(0)
    n = positions.shape[0]
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
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
    """Deterministic smooth thermal-like displacement for this step (mirrors solution.py)."""
    generator = torch.Generator().manual_seed(2025)
    phase = torch.rand(positions.shape, generator=generator, dtype=torch.float64)
    phase = phase.to(positions.dtype).to(positions.device)
    return noise_scale * torch.sin(phase * 2.0 * math.pi + step_index * 0.1)


def _energy_and_forces(positions: torch.Tensor, r_cut: float, r0: float = 1.2) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference energy/forces with a freshly built graph (fp64)."""
    edge_index = _build_graph(positions, r_cut).detach()
    p = positions.squeeze(0)
    row, col = edge_index[0], edge_index[1]
    vec = p[col] - p[row]
    dist = torch.linalg.vector_norm(vec, dim=-1)
    energy = _pair_potential(dist, r_cut, r0).sum()

    inv_dist = 1.0 / dist
    dVdr = 2.0 * (dist - r0)
    pair_forces = dVdr.unsqueeze(-1) * (vec * inv_dist.unsqueeze(-1))

    forces = torch.zeros_like(p)
    forces.index_add_(0, row, pair_forces)
    forces.index_add_(0, col, -pair_forces)
    return energy, forces.unsqueeze(0)


def reference_sample(positions: torch.Tensor, config: dict[str, Any], num_steps: int) -> torch.Tensor:
    """fp64 live recompute using a fresh radius graph every step."""
    r_cut = float(config["r_cut"])
    r0 = float(config.get("r0", 1.2))
    dt = float(config["dt"])
    box = float(config["box"])
    noise_scale = float(config.get("noise_scale", 0.0))
    pos = positions.double()
    for t in range(num_steps):
        _, forces = _energy_and_forces(pos, r_cut, r0)
        noise = _step_noise(pos, t, noise_scale).double()
        pos = torch.clamp(pos + dt * forces + noise, min=0.0, max=box)
    return pos.float()


def _allclose(candidate: torch.Tensor, reference: torch.Tensor, rtol: float, atol: float) -> tuple[bool, float]:
    err = (candidate.double() - reference.double()).abs().max().item()
    ok = bool(torch.isfinite(candidate).all() and torch.allclose(candidate.double(), reference.double(), rtol=rtol, atol=atol))
    return ok, err


def checksum_tensor(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous()
    return hashlib.sha256(arr.numpy().tobytes()).hexdigest()


def _random_probe_inputs(base_state: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Deterministic random valid positions (no overlapping-at-origin singularities)."""
    generator = torch.Generator().manual_seed(314159)
    device = base_state["positions"].device
    box = float(config["box"])
    positions = torch.rand(base_state["positions"].shape, generator=generator, dtype=base_state["positions"].dtype).to(device)
    return {"positions": positions * (box - 0.4) + 0.2}


def _mutated_inputs(base_state: dict[str, torch.Tensor], seed: int = 0, scale: float = 0.01) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    device = base_state["positions"].device
    noise = torch.randn(base_state["positions"].shape, generator=generator, dtype=base_state["positions"].dtype).to(device)
    return {"positions": base_state["positions"] + scale * noise}


def check_correctness(
    sample_fn: Callable[..., torch.Tensor],
    fixtures: dict[str, Any],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Correctness on one fresh draw + zero/mutated input probes."""
    init_state = {"positions": fixtures["init_positions"]}
    num_steps = int(fixtures["config"]["num_steps"])

    candidate = sample_fn(**init_state)
    reference = reference_sample(init_state["positions"], fixtures["config"], num_steps)
    ok, err = _allclose(candidate, reference, rtol, atol)
    base_checksum = checksum_tensor(candidate)

    probes: dict[str, Any] = {}
    probes_passed = True
    for name, inputs in (
        ("random_input", _random_probe_inputs(init_state, fixtures["config"])),
        ("mutated_input", _mutated_inputs(init_state)),
    ):
        probe_out = sample_fn(**inputs)
        probe_ref = reference_sample(inputs["positions"], fixtures["config"], num_steps)
        p_ok, p_err = _allclose(probe_out, probe_ref, rtol, atol)
        reused = checksum_tensor(probe_out) == base_checksum
        probes[name] = {"passed": p_ok and not reused, "max_abs_error": p_err, "reused_result": reused}
        probes_passed = probes_passed and probes[name]["passed"]

    details = {
        "fresh_input": {"passed": ok, "max_abs_error": err},
        "probes": probes,
        "output_checksum": base_checksum,
    }
    return {"passed": ok and probes_passed, "details": details}
