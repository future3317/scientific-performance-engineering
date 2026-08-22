#!/usr/bin/env python3
"""Public smoke test for SCIML-EQUIV-RECOMPUTE-06R2 (agent-visible).

Builds a tiny trajectory, checks rank-3 equivariance and translation
invariance, and prints a rough timing. Contains no hidden-verifier logic.
"""

import sys
import time
from pathlib import Path

import torch

WORKSPACE = Path(__file__).resolve().parents[1] / "workspace"
sys.path.insert(0, str(WORKSPACE))

import solution  # noqa: E402


def toy_trajectory():
    gen = torch.Generator().manual_seed(4321)
    n, a, r_cut = 10, 5.5, 2.4
    positions = torch.rand(n, 3, generator=gen) * a
    trajectory = []
    for step in range(4):
        diff = positions[None, :, :] - positions[:, None, :]
        dist = torch.linalg.vector_norm(diff, dim=-1)
        mask = (dist < r_cut) & (dist > 1e-8)
        pairs = mask.nonzero(as_tuple=False)
        edge_index = pairs.t().contiguous().long()
        cell_offsets = torch.zeros(edge_index.shape[1], 3, dtype=torch.long)
        trajectory.append({
            "positions": positions,
            "x0": torch.randn(n, 4, generator=gen),
            "edge_index": edge_index,
            "cell_offsets": cell_offsets,
            "cell_a": a,
            "step_index": step,
        })
        positions = torch.remainder(positions + torch.randn(n, 3, generator=gen) * 0.05, a)
    return trajectory


def main():
    config = {
        "in_feat": 4,
        "hidden": 16,
        "num_channels": 8,
        "num_rbf": 8,
        "gamma": 4.0,
        "rbf_min": 0.4,
        "rbf_max": 2.4,
        "r_cut": 2.4,
    }
    trajectory = toy_trajectory()
    model = solution.Rank3EquivariantHead(config)

    tensors, work_units = solution.eval_trajectory(model, trajectory)
    assert tensors.shape == (len(trajectory), 3, 3, 3), tensors.shape
    assert torch.isfinite(tensors).all()

    # Translation invariance: edge vectors unchanged by a global shift.
    graph = trajectory[0]
    shifted = {k: v for k, v in graph.items() if k != "positions"}
    t0 = solution.energy_fn(model, graph["positions"], **shifted)
    t1 = solution.energy_fn(model, graph["positions"] + torch.tensor([0.3, -0.2, 0.1]), **shifted)
    print(f"smoke: translation delta = {(t0 - t1).abs().max().item():.3e}")
    assert (t0 - t1).abs().max() < 1e-5

    # Rank-3 equivariance: rotate input -> output rotates as R^3 tensor.
    rotation = torch.linalg.qr(torch.randn(3, 3))[0]
    if torch.det(rotation) < 0:
        rotation[:, 0] = -rotation[:, 0]
    rotated_pos = torch.einsum("ij,nj->ni", rotation, graph["positions"])
    t_rot = solution.energy_fn(model, rotated_pos, **shifted)
    expected = torch.einsum("ia,jb,kc,abc->ijk", rotation, rotation, rotation, t0)
    err = (t_rot - expected).abs().max().item()
    print(f"smoke: equivariance max delta = {err:.3e}")
    assert err < 1e-3

    for _ in range(3):
        solution.eval_trajectory(model, trajectory)
    t0 = time.perf_counter()
    reps = 10
    for _ in range(reps):
        solution.eval_trajectory(model, trajectory)
    print(f"smoke: {(time.perf_counter() - t0) / reps * 1e3:.3f} ms per eval_trajectory ({work_units})")
    print("smoke test OK")


if __name__ == "__main__":
    main()
