#!/usr/bin/env python3
"""Public smoke test for CORE-MEM-RETAINED-GRAPH-13 (agent-visible).

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
    inputs = torch.randn(256, 16, generator=gen)
    teacher = torch.nn.Linear(16, 1)
    with torch.no_grad():
        teacher.weight.normal_(0.0, 0.1, generator=gen)
        teacher.bias.normal_(0.0, 0.1, generator=gen)
        targets = teacher(inputs).squeeze(-1)
    init_model = solution.TinyResMLP(16, 16, num_blocks=4)
    with torch.no_grad():
        init_model.fc1.weight.normal_(0.0, 0.01, generator=gen)
        init_model.fc1.bias.normal_(0.0, 0.01, generator=gen)
        for lin in init_model.blocks:
            lin.weight.normal_(0.0, 0.01, generator=gen)
            lin.bias.normal_(0.0, 0.01, generator=gen)
        init_model.fc2.weight.normal_(0.0, 0.01, generator=gen)
        init_model.fc2.bias.normal_(0.0, 0.01, generator=gen)
    return {
        "device": device,
        "inputs": inputs,
        "targets": targets,
        "model_config": {"in_dim": 16, "hidden_dim": 16},
        "optimizer_config": {"lr": 0.0001},
        "init_state": init_model.state_dict(),
        "batch_sizes": [32, 48, 64],
    }


def main():
    fixtures = tiny_fixtures()
    model = solution.build_model(fixtures)
    optimizer = torch.optim.SGD(model.parameters(), lr=fixtures["optimizer_config"]["lr"])
    losses = []
    work_units = {"forward": 0, "backward": 0, "optimizer": 0}
    for i in range(8):
        batch = (
            fixtures["inputs"][i * 32 : (i + 1) * 32],
            fixtures["targets"][i * 32 : (i + 1) * 32],
        )
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
