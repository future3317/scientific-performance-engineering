#!/usr/bin/env python3
"""Harness-side entry for CORE-REPEATED-BACKBONE-02."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn

from benchmark.harness import runner


class _ReferenceMultiHeadModel(nn.Module):
    def __init__(self, step_dim: int, fixed_dim: int, emb_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.embed = nn.Sequential(
            nn.Linear(fixed_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.backbone = nn.Sequential(
            nn.Linear(step_dim + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_heads)])

    def project_fixed(self, fixed_input: torch.Tensor) -> torch.Tensor:
        return self.embed(fixed_input).mean(dim=0)

    def forward_all(self, step_input: torch.Tensor, fixed_input: torch.Tensor) -> list[torch.Tensor]:
        context = self.project_fixed(fixed_input)
        ctx = context.unsqueeze(0).expand(step_input.size(0), -1)
        x = torch.cat([step_input, ctx], dim=-1)
        h = self.backbone(x)
        return [head(h) for head in self.heads]


def _build_reference_model(fixtures: dict) -> nn.Module:
    return _ReferenceMultiHeadModel(
        fixtures["step_dim"],
        fixtures["fixed_dim"],
        fixtures["emb_dim"],
        fixtures["hidden_dim"],
        fixtures["num_heads"],
    )


def _reference_run_training(fixtures: dict, steps: int, device: str) -> dict:
    """fp64 live-recomputed reference (no repeated-compute caching)."""
    torch.manual_seed(fixtures["seed"])
    model = _build_reference_model(fixtures).to(torch.float64).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=fixtures["lr"])

    step_data = fixtures["step_data"].to(torch.float64).to(device)
    fixed_data = fixtures["fixed_data"].to(torch.float64).to(device)
    targets = [t.to(torch.float64).to(device) for t in fixtures["targets"]]
    n = step_data.size(0)
    batch_size = fixtures["batch_size"]
    regime = fixtures.get("regime", "fixed")

    for step in range(steps):
        sidx = (step * batch_size) % n
        step_batch = step_data[sidx : sidx + batch_size]
        step_targets = [t[sidx : sidx + batch_size] for t in targets]
        fixed = fixed_data if regime == "fixed" else fixtures["fixed_data_changing"][step % fixtures["fixed_data_changing"].size(0)].to(torch.float64).to(device)
        preds = model.forward_all(step_batch, fixed)
        loss = sum(nn.functional.mse_loss(p, t) for p, t in zip(preds, step_targets))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        x_test = fixtures["test_step_data"].to(torch.float64).to(device)
        fixed_test = fixtures["test_fixed_data"].to(torch.float64).to(device)
        preds = model.forward_all(x_test, fixed_test)
        y_test = [t.to(torch.float64).to(device) for t in fixtures["test_targets"]]
        test_loss = sum(nn.functional.mse_loss(p, t) for p, t in zip(preds, y_test))
    return {"test_loss": test_loss.item(), "model": model}


def load_solution(path: str | Path):
    return runner.import_module_by_path(Path(path))


def make_fixtures(seed: int, device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    n_train_step = 512
    n_train_fixed = 128
    n_test_step = 128
    n_test_fixed = 64
    n_changing = 20
    step_dim = 32
    fixed_dim = 64
    emb_dim = 16
    hidden_dim = 64
    num_heads = 4
    batch_size = 64

    step_data = torch.randn(n_train_step, step_dim)
    fixed_data = torch.randn(n_train_fixed, fixed_dim)
    fixed_data_changing = torch.randn(n_changing, n_train_fixed, fixed_dim)

    # Targets are linear-ish functions of step + embedded fixed context.
    true_w_step = torch.randn(step_dim, 1)
    true_w_head = [torch.randn(hidden_dim, 1) for _ in range(num_heads)]
    true_b_head = [torch.randn(1) for _ in range(num_heads)]
    # Use a simple reference projection for target generation.
    ref_embed = nn.Linear(fixed_dim, emb_dim)
    ref_backbone = nn.Sequential(nn.Linear(step_dim + emb_dim, hidden_dim), nn.ReLU())
    with torch.no_grad():
        context = ref_embed(fixed_data).mean(dim=0)
        ctx_train = context.unsqueeze(0).expand(n_train_step, -1)
        h_train = ref_backbone(torch.cat([step_data, ctx_train], dim=-1))
        targets = [h_train @ true_w_head[k] + true_b_head[k] + 0.1 * torch.randn(n_train_step, 1) for k in range(num_heads)]

        test_step_data = torch.randn(n_test_step, step_dim)
        test_fixed_data = torch.randn(n_test_fixed, fixed_dim)
        context_test = ref_embed(test_fixed_data).mean(dim=0)
        ctx_test = context_test.unsqueeze(0).expand(n_test_step, -1)
        h_test = ref_backbone(torch.cat([test_step_data, ctx_test], dim=-1))
        test_targets = [h_test @ true_w_head[k] + true_b_head[k] + 0.05 * torch.randn(n_test_step, 1) for k in range(num_heads)]

    return {
        "seed": seed,
        "device": device,
        "regime": "fixed",
        "batch_size": batch_size,
        "lr": 0.001,
        "step_dim": step_dim,
        "fixed_dim": fixed_dim,
        "emb_dim": emb_dim,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "step_data": step_data,
        "fixed_data": fixed_data,
        "fixed_data_changing": fixed_data_changing,
        "targets": targets,
        "test_step_data": test_step_data,
        "test_fixed_data": test_fixed_data,
        "test_targets": test_targets,
    }


def _run_candidate(solution, fixtures: dict, steps: int, regime: str) -> dict:
    device = fixtures["device"]
    torch.manual_seed(fixtures["seed"])
    model = solution.build_model(fixtures).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=fixtures["lr"])

    step_data = fixtures["step_data"].to(device)
    fixed_data = fixtures["fixed_data"].to(device)
    targets = [t.to(device) for t in fixtures["targets"]]
    n = step_data.size(0)
    batch_size = fixtures["batch_size"]

    for step in range(steps):
        sidx = (step * batch_size) % n
        step_batch = step_data[sidx : sidx + batch_size]
        step_targets = [t[sidx : sidx + batch_size] for t in targets]
        if regime == "fixed":
            fixed = fixed_data
        else:
            idx = step % fixtures["fixed_data_changing"].size(0)
            fixed = fixtures["fixed_data_changing"][idx].to(device)
        batch = {"step": step_batch, "fixed": fixed, "targets": step_targets}
        solution.train_step(model, batch, optimizer)

    model.eval()
    with torch.no_grad():
        x_test = fixtures["test_step_data"].to(device)
        fixed_test = fixtures["test_fixed_data"].to(device)
        context = model.project_fixed(fixed_test)
        preds = [model.forward_head(x_test, context, head_idx) for head_idx in range(fixtures["num_heads"])]
        y_test = [t.to(device) for t in fixtures["test_targets"]]
        test_loss = sum(nn.functional.mse_loss(p, t) for p, t in zip(preds, y_test))
    return {"test_loss": test_loss.item(), "model": model}


def run_correctness(solution, fixtures: dict) -> dict:
    device = fixtures["device"]
    steps = 5

    # Fixed regime.
    fixtures_fixed = {**fixtures, "regime": "fixed"}
    ref_fixed = _reference_run_training(fixtures_fixed, steps, device)
    cand_fixed = _run_candidate(solution, fixtures_fixed, steps, "fixed")

    # Changing regime (semantic trap for unconditional cache).
    fixtures_changing = {**fixtures, "regime": "changing"}
    ref_changing = _reference_run_training(fixtures_changing, steps, device)
    cand_changing = _run_candidate(solution, fixtures_changing, steps, "changing")

    fixed_pass = bool(
        torch.allclose(torch.tensor(cand_fixed["test_loss"]), torch.tensor(ref_fixed["test_loss"]), rtol=1e-2, atol=1e-3)
    )

    # Changing-regime semantic trap: evaluate both models on the LAST changing
    # fixed input used during training. A solution that caches the first fixed
    # input (the tempting pattern) will return a stale context here and its
    # predictions will diverge from the reference, while baseline/oracle
    # recompute the context and stay matched.
    last_changing_fixed = fixtures["fixed_data_changing"][(steps - 1) % fixtures["fixed_data_changing"].size(0)].to(device)
    ref_model = ref_changing["model"]
    cand_model = cand_changing["model"]
    x_test = fixtures["test_step_data"].to(device)
    with torch.no_grad():
        ref_preds = torch.stack(ref_model.forward_all(x_test.to(torch.float64), last_changing_fixed.to(torch.float64)))
        cand_ctx = cand_model.project_fixed(last_changing_fixed)
        cand_preds = torch.stack([cand_model.forward_head(x_test, cand_ctx, k) for k in range(fixtures["num_heads"])])
    changing_pass = bool(torch.allclose(cand_preds, ref_preds.to(cand_preds.dtype), rtol=1e-2, atol=1e-3))

    passed = bool(fixed_pass and changing_pass)

    return {
        "passed": passed,
        "details": {
            "candidate_fixed_test_loss": cand_fixed["test_loss"],
            "reference_fixed_test_loss": ref_fixed["test_loss"],
            "candidate_changing_test_loss": cand_changing["test_loss"],
            "reference_changing_test_loss": ref_changing["test_loss"],
            "changing_regime_prediction_close": changing_pass,
            "output_checksum": cand_fixed["test_loss"],
        },
    }


def run_scientific_gates(solution, fixtures: dict) -> dict:
    return {}


def run_performance(solution, fixtures: dict, warmup: int, iterations: int, device: str = "cpu") -> dict:
    torch.manual_seed(fixtures["seed"])
    model = solution.build_model(fixtures).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=fixtures["lr"])

    step_data = fixtures["step_data"].to(device)
    fixed_data = fixtures["fixed_data"].to(device)
    targets = [t.to(device) for t in fixtures["targets"]]
    n = step_data.size(0)
    batch_size = fixtures["batch_size"]

    # Fixed-input regime for performance: cache is valid.
    fixed_batch = fixed_data

    def one_step(step: int) -> None:
        sidx = (step * batch_size) % n
        step_batch = step_data[sidx : sidx + batch_size]
        step_targets = [t[sidx : sidx + batch_size] for t in targets]
        batch = {"step": step_batch, "fixed": fixed_batch, "targets": step_targets}
        solution.train_step(model, batch, optimizer)

    for step in range(warmup):
        one_step(step)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wall_start = time.perf_counter()
    for step in range(iterations):
        one_step(warmup + step)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - wall_start) * 1000.0
    per_step_ms = total_ms / iterations

    return {
        "value": per_step_ms,
        "work_units": {"forward_calls": iterations * fixtures["num_heads"], "backward_calls": iterations, "optimizer_steps": iterations},
        "output_checksums": {},
        "timing": {"total_ms": total_ms, "per_step_ms": per_step_ms},
    }
