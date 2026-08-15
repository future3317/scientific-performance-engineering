"""Hidden verifier logic for CORE-H2D-PIPELINE-03 (harness-only).

fp64 live-recomputed reference for the training loop, plus output checksums.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset


def checksum_tensor(tensor: torch.Tensor) -> str:
    """SHA-256 of a tensor's bytes (anti-caching evidence)."""
    arr = tensor.detach().cpu().contiguous()
    return hashlib.sha256(arr.numpy().tobytes()).hexdigest()


def _make_fp64_model(fixtures: dict[str, Any]) -> torch.nn.Module:
    config = fixtures["model_config"]
    model = torch.nn.Sequential(
        torch.nn.Linear(config["in_dim"], config["hidden_dim"], dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(config["hidden_dim"], 1, dtype=torch.float64),
    )
    # Convert init state to float64 while preserving names.
    state = {}
    for k, v in fixtures["init_state"].items():
        state[k] = v.double()
    model.load_state_dict(state)
    model.to(fixtures["device"])
    return model


def _run_training_reference(fixtures: dict[str, Any], steps: int) -> torch.nn.Module:
    """Run the same training loop in fp64 on the same data order."""
    model = _make_fp64_model(fixtures)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=fixtures["optimizer_config"]["lr"]
    )
    dataset = TensorDataset(
        fixtures["inputs"].double(), fixtures["targets"].double()
    )
    loader = DataLoader(
        dataset,
        batch_size=fixtures["data_config"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    it = iter(loader)
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        inputs, targets = batch
        inputs = inputs.to(fixtures["device"])
        targets = targets.to(fixtures["device"])
        optimizer.zero_grad()
        preds = model(inputs).squeeze(-1)
        loss = torch.nn.functional.mse_loss(preds, targets)
        loss.backward()
        optimizer.step()
    return model


def _eval_model(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(inputs.to(next(model.parameters()).device)).squeeze(-1)


def check_training_correctness(
    solution: Any,
    fixtures: dict[str, Any],
    rtol: float,
    atol: float,
    train_steps: int = 5,
) -> dict[str, Any]:
    """Compare candidate training result to a live fp64 reference."""
    # Candidate run: use the solution's own train_step on the same data order.
    model = solution.build_model(fixtures)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=fixtures["optimizer_config"]["lr"]
    )
    dataset = TensorDataset(fixtures["inputs"], fixtures["targets"])
    loader = DataLoader(
        dataset,
        batch_size=fixtures["data_config"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    it = iter(loader)
    for _ in range(train_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        solution.train_step(model, batch, optimizer)

    candidate_outputs = _eval_model(model, fixtures["eval_inputs"])
    reference_model = _run_training_reference(fixtures, train_steps)
    reference_outputs = _eval_model(reference_model, fixtures["eval_inputs"].double())

    err = (candidate_outputs.double() - reference_outputs).abs().max().item()
    passed = bool(
        torch.isfinite(candidate_outputs).all()
        and torch.allclose(candidate_outputs.double(), reference_outputs, rtol=rtol, atol=atol)
    )
    return {
        "passed": passed,
        "details": {
            "max_abs_error": err,
            "output_checksum": checksum_tensor(candidate_outputs),
        },
    }
