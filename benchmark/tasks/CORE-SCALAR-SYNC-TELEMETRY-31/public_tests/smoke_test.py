#!/usr/bin/env python3
"""Public smoke test for the CUDA telemetry-sync task package."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workspace"))
import solution  # noqa: E402


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fixtures = {
        "device": device,
        "batch_size": 32,
        "lr": 0.01,
        "model_dims": [16, 16, 16, 16, 1],
        "train_data": (torch.randn(64, 16), torch.randn(64, 1)),
    }
    model = solution.build_model(fixtures).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=fixtures["lr"])
    x, y = (value.to(device) for value in fixtures["train_data"])
    result = solution.train_step(model, (x[:32], y[:32]), optimizer)
    assert len(result["telemetry"]) == 16
    assert result["work_units"] == {"forward_calls": 1, "backward_calls": 1, "optimizer_steps": 1}
    print("smoke_test: telemetry vector width=16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
