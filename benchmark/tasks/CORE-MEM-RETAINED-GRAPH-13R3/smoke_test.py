#!/usr/bin/env python3
"""Public smoke test for the 13R3 retained-history API."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(__file__).resolve().parent
    repo_root = root.parents[2]
    sys.path.insert(0, str(repo_root))
    benchmark = _load("benchmark_13r3_smoke", root / "benchmark.py")
    solution = _load("solution_13r3_smoke", root / "workspace" / "solution.py")
    fixtures = benchmark.make_fixtures(1234)
    model = solution.build_model(fixtures)
    optimizer = torch.optim.SGD(model.parameters(), lr=fixtures["optimizer_config"]["lr"])
    losses = []
    for i in range(3):
        size = fixtures["batch_sizes"][i]
        batch = (fixtures["inputs"][i * size:(i + 1) * size], fixtures["targets"][i * size:(i + 1) * size])
        out = solution.train_step(model, batch, optimizer)
        assert "loss" in out and "work_units" in out
        losses.append(out["loss"])
    loss_tensor = torch.stack(losses)
    assert torch.isfinite(loss_tensor).all(), "non-finite losses"
    assert len(model.retained_history) == fixtures["segment_count"] * 3
    print(f"smoke: final_loss={loss_tensor[-1].item():.4f}")
    print("smoke test OK")


if __name__ == "__main__":
    main()
