#!/usr/bin/env python3
"""Public smoke test for SCIML-GRAPH-REBUILD-08R2 (agent-visible).

Run from the workspace directory:

    python ../public_tests/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

WORKSPACE = Path(__file__).resolve().parents[1] / "workspace"
sys.path.insert(0, str(WORKSPACE))

import solution  # noqa: E402


def toy_fixtures():
    device = torch.device("cpu")
    config = {"num_atoms": 8, "num_steps": 10, "r_cut": 2.0, "dt": 0.001, "box": 5.0}
    gen = torch.Generator().manual_seed(1234)
    positions = torch.rand(1, config["num_atoms"], 3, generator=gen) * (config["box"] - 0.5) + 0.25
    return {"config": config, "init_positions": positions.to(device), "device": "cpu"}


def main():
    fixtures = toy_fixtures()
    sampler = solution.build_sampler(fixtures)
    final = solution.sample(sampler, fixtures, int(fixtures["config"]["num_steps"]))
    assert final.shape == fixtures["init_positions"].shape, final.shape
    assert torch.isfinite(final).all()

    # Self-consistency: same input -> same output.
    final2 = solution.sample(sampler, fixtures, int(fixtures["config"]["num_steps"]))
    assert torch.allclose(final, final2, atol=1e-6)

    # Energy/forces are finite.
    energy, forces = solution.energy_forces(sampler, fixtures["init_positions"])
    assert torch.isfinite(energy) and torch.isfinite(forces).all()

    print(f"smoke: final positions shape {list(final.shape)}, energy {energy.item():.4f}")
    print("smoke test OK")


if __name__ == "__main__":
    main()
