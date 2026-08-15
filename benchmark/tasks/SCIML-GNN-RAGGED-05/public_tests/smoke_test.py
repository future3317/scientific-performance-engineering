#!/usr/bin/env python3
"""Public smoke test for SCIML-GNN-RAGGED-05 (agent-visible).

Builds two tiny synthetic graphs, runs eval_batch, checks F == -dE/dx to loose
tolerance, and prints a rough timing. Contains no hidden-verifier logic and no
gold data. Run from the workspace directory:

    python ../public_tests/smoke_test.py
"""

import sys
import time
from pathlib import Path

import torch

WORKSPACE = Path(__file__).resolve().parents[1] / "workspace"
sys.path.insert(0, str(WORKSPACE))

import solution  # noqa: E402


def toy_graphs():
    gen = torch.Generator().manual_seed(1234)
    graphs = []
    for n, a in ((10, 5.6), (14, 6.0)):
        positions = torch.rand(n, 3, generator=gen) * (a - 1.5)
        # simple all-pairs-within-cutoff graph, no PBC (smoke level only)
        diff = positions[None, :, :] - positions[:, None, :]
        dist = torch.linalg.vector_norm(diff, dim=-1)
        mask = (dist < 2.4) & (dist > 1e-8)
        pairs = mask.nonzero(as_tuple=False)
        edge_index = pairs.t().contiguous().long()
        cell_offsets = torch.zeros(edge_index.shape[1], 3, dtype=torch.long)
        graphs.append(
            {
                "positions": positions,
                "x0": torch.randn(n, 4, generator=gen),
                "edge_index": edge_index,
                "cell_offsets": cell_offsets,
                "cell_a": a,
            }
        )
    return graphs


def main():
    config = {
        "in_feat": 4,
        "hidden": 16,
        "num_layers": 2,
        "num_rbf": 8,
        "gamma": 4.0,
        "rbf_min": 0.4,
        "rbf_max": 2.4,
        "r_cut": 2.4,
    }
    graphs = toy_graphs()
    model = solution.InvariantEnergyModel(config)  # random init is fine for a smoke test

    energies, forces, work_units = solution.eval_batch(model, graphs)
    assert energies.shape == (len(graphs),), energies.shape
    total_atoms = sum(g["positions"].shape[0] for g in graphs)
    assert forces.shape == (total_atoms, 3), forces.shape
    assert torch.isfinite(energies).all() and torch.isfinite(forces).all()

    # F == -dE/dx, checked per graph against solution.energy_fn
    start = 0
    max_err = 0.0
    for gi, g in enumerate(graphs):
        n = g["positions"].shape[0]
        pos = g["positions"].detach().clone().requires_grad_(True)
        e = solution.energy_fn(
            model, pos, x0=g["x0"], edge_index=g["edge_index"],
            cell_offsets=g["cell_offsets"], cell_a=g["cell_a"],
        )
        expected = -torch.autograd.grad(e, pos)[0]
        max_err = max(max_err, (forces[start : start + n] - expected).abs().max().item())
        start += n
    print(f"smoke: energies={energies.tolist()}")
    print(f"smoke: max |F - (-dE/dx)| = {max_err:.3e}")
    assert max_err < 1e-4, "forces do not match -dE/dx"

    for _ in range(3):
        solution.eval_batch(model, graphs)
    t0 = time.perf_counter()
    reps = 10
    for _ in range(reps):
        solution.eval_batch(model, graphs)
    print(f"smoke: {(time.perf_counter() - t0) / reps * 1e3:.3f} ms per eval_batch ({work_units})")
    print("smoke test OK")


if __name__ == "__main__":
    main()
