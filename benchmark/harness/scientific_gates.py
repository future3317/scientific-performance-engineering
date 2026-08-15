#!/usr/bin/env python3
"""Shared scientific gate library (BENCHMARK_DESIGN.md sections 3.2, 6 S3).

Torch-based gates; each returns ``(passed: bool, details: dict)``. Gates are
deterministic programs, not prompts: a fast answer that violates one scores 0
(principle P2). The rank-3 rotation math is adapted from
``assets/materials_gnn_checks.py``.
"""

from __future__ import annotations

from typing import Any, Callable

import torch


def _split_energy_forces(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept (energy, forces) tuple or {'energy': E, 'forces': F} dict."""
    if isinstance(output, dict):
        return output["energy"], output["forces"]
    if isinstance(output, (tuple, list)) and len(output) == 2:
        return output[0], output[1]
    raise TypeError("energy_fn must return (energy, forces) or {'energy', 'forces'}")


def energy_force_consistency(
    energy_fn: Callable[[torch.Tensor], Any],
    positions: torch.Tensor,
    tol: float = 1e-4,
) -> tuple[bool, dict[str, Any]]:
    """Check F == -dE/dx via autograd on the given positions.

    *energy_fn* receives a differentiable positions tensor and returns
    (energy, forces) — forces being the model's own head. The gate recomputes
    forces as ``-grad(E, positions)`` and compares.
    """
    positions = positions.detach().clone().requires_grad_(True)
    energy, forces = _split_energy_forces(energy_fn(positions))
    grad_outputs = torch.ones_like(energy)
    expected = -torch.autograd.grad(energy, positions, grad_outputs=grad_outputs)[0]
    forces = forces.detach()
    max_abs_err = (forces - expected).abs().max().item()
    denom = expected.abs().max().item()
    rel_err = max_abs_err / (denom + 1e-12)
    passed = bool(torch.isfinite(forces).all()) and max_abs_err <= tol
    return passed, {
        "max_abs_error": max_abs_err,
        "max_rel_error": rel_err,
        "tol": tol,
        "energy_finite": bool(torch.isfinite(energy).all()),
        "forces_finite": bool(torch.isfinite(forces).all()),
    }


def gradient_agreement(
    candidate_fn: Callable[[torch.Tensor], torch.Tensor],
    reference_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    tol: float = 1e-4,
) -> tuple[bool, dict[str, Any]]:
    """Check d(candidate)/dx agrees with d(reference)/dx on the same inputs."""
    candidate_inputs = inputs.detach().clone().requires_grad_(True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    candidate_out = candidate_fn(candidate_inputs)
    reference_out = reference_fn(reference_inputs)
    if candidate_out.shape != reference_out.shape:
        return False, {
            "reason": "output shape mismatch",
            "candidate_shape": list(candidate_out.shape),
            "reference_shape": list(reference_out.shape),
        }
    weights = torch.ones_like(reference_out)
    candidate_grad = torch.autograd.grad(candidate_out, candidate_inputs, grad_outputs=weights)[0]
    reference_grad = torch.autograd.grad(reference_out, reference_inputs, grad_outputs=weights)[0]
    out_err = (candidate_out.detach() - reference_out).abs().max().item()
    grad_err = (candidate_grad - reference_grad).abs().max().item()
    passed = out_err <= tol and grad_err <= tol
    return passed, {
        "max_output_abs_error": out_err,
        "max_gradient_abs_error": grad_err,
        "tol": tol,
    }


def rotate_rank_three_cartesian(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply a 3x3 rotation to the final three Cartesian dims (from assets/materials_gnn_checks.py)."""
    if tensor.shape[-3:] != (3, 3, 3) or rotation.shape != (3, 3):
        raise ValueError("expected tensor[..., 3, 3, 3] and rotation[3, 3]")
    return torch.einsum("ia,jb,kc,...abc->...ijk", rotation, rotation, rotation, tensor)


def _random_rotation(generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    """Uniform random proper rotation via QR of a Gaussian matrix."""
    matrix = torch.randn(3, 3, generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(r))
    q = q * signs
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def equivariance_rank3(
    pred_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    tol: float = 1e-4,
    seed: int = 0,
) -> tuple[bool, dict[str, Any]]:
    """Rank-3 equivariance: f(Rx) == R^3 f(x) for a seeded random rotation.

    *inputs* has trailing Cartesian dimension 3 (e.g. positions); *pred_fn* maps
    it to a ``[..., 3, 3, 3]`` tensor. The relative error
    ``||f(Rx) - R^3 f(x)|| / (||R^3 f(x)|| + eps)`` must not exceed *tol*.
    """
    dtype = inputs.dtype if inputs.is_floating_point() else torch.float64
    generator = torch.Generator().manual_seed(seed)
    rotation = _random_rotation(generator, dtype).to(inputs.device)
    x = inputs.to(dtype)
    rotated_inputs = torch.einsum("ij,...j->...i", rotation, x)
    prediction = pred_fn(x)
    rotated_prediction = pred_fn(rotated_inputs)
    expected = rotate_rank_three_cartesian(prediction, rotation)
    if rotated_prediction.shape != expected.shape:
        return False, {
            "reason": "prediction shape mismatch after rotation",
            "expected_shape": list(expected.shape),
            "observed_shape": list(rotated_prediction.shape),
        }
    rel_error = (
        torch.linalg.vector_norm(rotated_prediction - expected)
        / (torch.linalg.vector_norm(expected) + 1e-12)
    ).item()
    return rel_error <= tol, {
        "relative_equivariance_error": rel_error,
        "tol": tol,
        "rotation_seed": seed,
    }


def crystal_validity(
    positions: torch.Tensor,
    lattice: torch.Tensor,
    min_dist: float = 0.5,
    lattice_range: tuple[float, float] = (0.1, 100.0),
) -> tuple[bool, dict[str, Any]]:
    """Minimal structural-validity gate (no pymatgen): distances and lattice ranges.

    - positions: [..., N, 3] Cartesian coordinates (batched allowed)
    - lattice:   [..., 3, 3] lattice matrices (rows are lattice vectors)
    Fails on non-finite values, any interatomic distance below *min_dist*, or
    lattice-vector lengths outside *lattice_range*.
    """
    details: dict[str, Any] = {"min_dist": min_dist, "lattice_range": list(lattice_range)}
    if not torch.isfinite(positions).all() or not torch.isfinite(lattice).all():
        return False, {**details, "reason": "non-finite positions or lattice"}
    if positions.shape[-1] != 3:
        return False, {**details, "reason": "positions must have trailing dimension 3"}
    if positions.shape[-2] < 2:
        return False, {**details, "reason": "at least two atoms required"}

    diff = positions.unsqueeze(-2) - positions.unsqueeze(-3)  # [..., N, N, 3]
    dists = torch.linalg.vector_norm(diff, dim=-1)
    n = positions.shape[-2]
    eye = torch.eye(n, dtype=torch.bool, device=positions.device)
    dists = dists.masked_fill(eye, float("inf"))
    min_observed = dists.min().item()

    lengths = torch.linalg.vector_norm(lattice, dim=-1)  # [..., 3]
    lattice_ok = bool(((lengths >= lattice_range[0]) & (lengths <= lattice_range[1])).all())
    passed = min_observed >= min_dist and lattice_ok
    return passed, {
        **details,
        "min_interatomic_distance": min_observed,
        "lattice_vector_lengths": lengths.flatten().tolist(),
        "lattice_in_range": lattice_ok,
    }


def distribution_moment_check(
    samples: torch.Tensor,
    reference_moments: dict[str, float],
    tol: float = 0.1,
) -> tuple[bool, dict[str, Any]]:
    """Check sample distribution moments against a reference (relative tolerance).

    Supported moment keys: ``mean``, ``std``. *samples* is flattened before
    computing moments. A moment is compared as ``abs(observed - reference) <=
    tol * max(abs(reference), 1e-12)`` so it degrades gracefully near zero.
    """
    flat = samples.detach().flatten().double()
    if not torch.isfinite(flat).all():
        return False, {"reason": "non-finite samples"}
    observed = {
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item() if flat.numel() > 1 else 0.0,
    }
    per_moment: dict[str, dict[str, float]] = {}
    passed = True
    for key, reference in reference_moments.items():
        if key not in observed:
            return False, {"reason": f"unsupported moment {key!r}"}
        error = abs(observed[key] - float(reference))
        bound = tol * max(abs(float(reference)), 1e-12)
        per_moment[key] = {"observed": observed[key], "reference": float(reference), "error": error, "bound": bound}
        passed = passed and error <= bound
    return passed, {"moments": per_moment, "tol": tol}
