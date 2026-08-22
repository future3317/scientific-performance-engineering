#!/usr/bin/env python3
"""Public smoke test for CORE-H2D-PIPELINE-03R2 (agent-visible).

Builds a tiny fixture, exercises the train_loop_v1 API, and checks that
losses are finite. Contains no hidden-verifier logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

WORKSPACE = Path(__file__).resolve().parents[1] / "workspace"
sys.path.insert(0, str(WORKSPACE))

import solution  # noqa: E402


def tiny_fixtures():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator().manual_seed(1234)
    inputs = torch.randn(256, 128, generator=gen)
    targets = torch.randn(256, generator=gen)
    init_model = torch.nn.Sequential(
        torch.nn.Linear(128, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    )
    return {
        "device": device,
        "inputs": inputs,
        "targets": targets,
        "model_config": {"in_dim": 128, "hidden_dim": 16},
        "optimizer_config": {"lr": 0.01},
        "init_state": init_model.state_dict(),
        "data_config": {"batch_size": 64},
    }


def main():
    fixtures = tiny_fixtures()
    model = solution.build_model(fixtures)
    optimizer = torch.optim.SGD(model.parameters(), lr=fixtures["optimizer_config"]["lr"])
    dataset = torch.utils.data.TensorDataset(fixtures["inputs"], fixtures["targets"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=fixtures["data_config"]["batch_size"])
    losses = []
    work_units = {"forward": 0, "backward": 0, "optimizer": 0}
    for batch in loader:
        out = solution.train_step(model, batch, optimizer)
        assert "loss" in out and "work_units" in out
        losses.append(out["loss"])
        for k in work_units:
            work_units[k] += out["work_units"][k]
    loss_tensor = torch.stack(losses)
    assert torch.isfinite(loss_tensor).all(), "non-finite losses"
    assert work_units["forward"] == work_units["backward"] == work_units["optimizer"]
    print(f"smoke: final_loss={loss_tensor[-1].item():.4f} work_units={work_units}")
    print("smoke test OK")


if __name__ == "__main__":
    main()
