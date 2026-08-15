"""Hidden verifier logic for CORE-DATALOADER-FANOUT-16 (harness-only).

fp64 live-recomputed reference for the training loop, plus output checksums.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def checksum_tensor(tensor: torch.Tensor) -> str:
    """SHA-256 of a tensor's bytes (anti-caching evidence)."""
    arr = tensor.detach().cpu().contiguous()
    return hashlib.sha256(arr.numpy().tobytes()).hexdigest()


class _ReferenceResMLP(nn.Module):
    """Reference architecture matching the workspace TinyResMLP (no graph break)."""

    def __init__(self, in_dim: int, hidden_dim: int, num_blocks: int = 4, dtype=torch.float32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, dtype=dtype)
        self.blocks = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype) for _ in range(num_blocks)
        )
        self.fc2 = nn.Linear(hidden_dim, 1, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        for lin in self.blocks:
            out = F.relu(lin(h))
            h = h + out
        return self.fc2(h)


def _make_fp64_model(fixtures: dict[str, Any]) -> torch.nn.Module:
    config = fixtures["model_config"]
    model = _ReferenceResMLP(config["in_dim"], config["hidden_dim"], num_blocks=4, dtype=torch.float64)
    state = {}
    for k, v in fixtures["init_state"].items():
        state[k] = v.double()
    model.load_state_dict(state)
    model.to(fixtures["device"])
    return model


def _batch_at(fixtures: dict[str, Any], index: int) -> tuple[torch.Tensor, torch.Tensor]:
    size = fixtures["batch_sizes"][0]  # correctness uses a fixed shape for stability
    offset = (index * size) % (fixtures["inputs"].shape[0] - size)
    return (
        fixtures["inputs"][offset : offset + size],
        fixtures["targets"][offset : offset + size],
    )


def _run_training_reference(fixtures: dict[str, Any], steps: int) -> torch.nn.Module:
    """Run the same training loop in fp64 on a fixed batch size."""
    model = _make_fp64_model(fixtures)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=fixtures["optimizer_config"]["lr"]
    )
    device = fixtures["device"]
    for i in range(steps):
        inputs, targets = _batch_at(fixtures, i)
        inputs = inputs.double().to(device)
        targets = targets.double().to(device)
        optimizer.zero_grad()
        preds = model(inputs).squeeze(-1)
        loss = F.mse_loss(preds, targets)
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
    model = solution.build_model(fixtures)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=fixtures["optimizer_config"]["lr"]
    )
    for i in range(train_steps):
        batch = _batch_at(fixtures, i)
        solution.train_step(model, batch, optimizer)

    candidate_outputs = _eval_model(model, fixtures["eval_inputs"])
    reference_model = _run_training_reference(fixtures, train_steps)
    reference_outputs = _eval_model(reference_model, fixtures["eval_inputs"].double())

    err = (candidate_outputs.double() - reference_outputs).abs().max().item()
    passed = bool(
        torch.isfinite(candidate_outputs).all()
        and torch.allclose(
            candidate_outputs.double(), reference_outputs, rtol=rtol, atol=atol
        )
    )
    return {
        "passed": passed,
        "details": {
            "max_abs_error": err,
            "output_checksum": checksum_tensor(candidate_outputs),
        },
    }
