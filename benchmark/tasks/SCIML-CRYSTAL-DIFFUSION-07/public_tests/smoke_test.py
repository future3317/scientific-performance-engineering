#!/usr/bin/env python3
"""Public smoke test for SCIML-CRYSTAL-DIFFUSION-07 (agent-visible).

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
    config = {
        "num_atoms": 4,
        "num_steps": 5,
        "hidden": 16,
        "time_emb_dim": 8,
        "beta_start": 1e-4,
        "beta_end": 0.02,
    }
    gen = torch.Generator().manual_seed(42)
    in_dim = config["num_atoms"] * 3 + 3 + config["time_emb_dim"]
    out_dim = config["num_atoms"] * 3 + 3
    score_state = {
        "fc1.weight": (torch.rand(config["hidden"], in_dim, generator=gen) * 0.1 - 0.05).to(device),
        "fc1.bias": torch.zeros(config["hidden"], device=device),
        "fc2.weight": (torch.rand(out_dim, config["hidden"], generator=gen) * 0.1 - 0.05).to(device),
        "fc2.bias": torch.zeros(out_dim, device=device),
    }
    init_frac = torch.rand(2, config["num_atoms"], 3, generator=gen).to(device)
    init_lengths = torch.full((2, 3), 5.0, device=device)
    return {
        "config": config,
        "score_state": score_state,
        "init_frac": init_frac,
        "init_lengths": init_lengths,
        "device": "cpu",
    }


def main():
    fixtures = toy_fixtures()
    sampler = solution.build_sampler(fixtures)
    samples = solution.sample(sampler, fixtures, int(fixtures["config"]["num_steps"]))

    batch_size = fixtures["init_frac"].shape[0]
    num_atoms = fixtures["config"]["num_atoms"]
    assert samples.shape == (batch_size, num_atoms * 3 + 3), samples.shape
    assert torch.isfinite(samples).all()

    # Loose self-consistency: running twice with the same input gives the same output.
    samples2 = solution.sample(sampler, fixtures, int(fixtures["config"]["num_steps"]))
    assert torch.allclose(samples, samples2, atol=1e-6)

    print(f"smoke: samples shape {list(samples.shape)}, min {samples.min():.3f}, max {samples.max():.3f}")
    print("smoke test OK")


if __name__ == "__main__":
    main()
