"""Scientific gates for SCIML-CRYSTAL-DIFFUSION-07R2.

Exposes gate functions consumed by benchmark.py.  The harness uses
`harness/scientific_gates.py` for the core structural and moment checks.
"""

from __future__ import annotations

from typing import Any

import torch

from benchmark.harness import scientific_gates


def _to_cartesian(frac: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Orthorhombic fractional -> Cartesian."""
    # frac: [B, N, 3], lengths: [B, 3]
    return frac * lengths.unsqueeze(-2)


def _build_lattice(lengths: torch.Tensor) -> torch.Tensor:
    """Orthorhombic lattice matrices [B, 3, 3]."""
    b = lengths.shape[0]
    eye = torch.eye(3, dtype=lengths.dtype, device=lengths.device).unsqueeze(0).expand(b, -1, -1)
    return eye * lengths.unsqueeze(-1)


def crystal_validity_rate(samples: torch.Tensor, config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Check that ≥90% of the generated crystals are structurally valid."""
    num_atoms = int(config["num_atoms"])
    frac = samples[..., : num_atoms * 3].reshape(-1, num_atoms, 3)
    lengths = samples[..., num_atoms * 3 :]
    cart = _to_cartesian(frac, lengths)
    lattice = _build_lattice(lengths)

    per_crystal: list[bool] = []
    min_dists: list[float] = []
    for i in range(cart.shape[0]):
        passed, details = scientific_gates.crystal_validity(
            cart[i : i + 1], lattice[i : i + 1], min_dist=0.5, lattice_range=(0.1, 100.0)
        )
        per_crystal.append(bool(passed))
        min_dists.append(float(details["min_interatomic_distance"]))

    rate = sum(per_crystal) / len(per_crystal) if per_crystal else 0.0
    threshold = float(config.get("validity_threshold", 0.9))
    passed = rate >= threshold
    return passed, {
        "validity_rate": rate,
        "threshold": threshold,
        "min_interatomic_distance": min(min_dists) if min_dists else None,
    }


def distribution_moment_check(samples: torch.Tensor, reference_moments: dict[str, float]) -> tuple[bool, dict[str, Any]]:
    """Compare sample moments to the seeded reference distribution."""
    return scientific_gates.distribution_moment_check(samples, reference_moments, tol=0.1)
