"""Hidden correctness checks for the batched VJP task."""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch import nn


def checksum_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous()
    return hashlib.sha256(array.numpy().tobytes()).hexdigest()


class _ReferenceModel(nn.Module):
    def __init__(self, input_dim: int, output_count: int, output_mask: torch.Tensor, dtype=torch.float32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64, dtype=dtype),
            nn.Tanh(),
            nn.Linear(64, output_count, dtype=dtype),
        )
        self.register_buffer("vjp_output_mask", output_mask.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _jacobian_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    y = model(x)
    rows = []
    for index in torch.nonzero(model.vjp_output_mask, as_tuple=False).flatten().tolist():
        rows.append(torch.autograd.grad(y[:, index].mean(), x, retain_graph=True, create_graph=True)[0])
    return torch.stack(rows)


def _make_reference(fixtures: dict[str, Any]) -> nn.Module:
    model = _ReferenceModel(
        fixtures["input_dim"],
        fixtures["output_count"],
        fixtures["vjp_output_mask"],
        dtype=torch.float64,
    )
    state = {key: value.double() if value.is_floating_point() else value for key, value in fixtures["init_state"].items()}
    model.load_state_dict(state)
    return model.to(fixtures["device"])


def check_vjp_training(
    solution: Any,
    fixtures: dict[str, Any],
    rtol: float = 1e-4,
    atol: float = 1e-5,
    train_steps: int = 5,
) -> dict[str, Any]:
    candidate = solution.build_model(fixtures)
    candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=fixtures["lr"])
    reference = _make_reference(fixtures)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=fixtures["lr"])
    batch = fixtures["batch"]
    for _ in range(train_steps):
        solution.train_step(candidate, batch, candidate_optimizer)
        x = batch[0].double().to(fixtures["device"])
        reference_optimizer.zero_grad()
        y = reference(x)
        jacobian = _jacobian_features(reference, x)
        loss = y.square().mean() + 0.1 * jacobian.square().mean()
        loss.backward()
        reference_optimizer.step()

    generator = torch.Generator().manual_seed(991)
    eval_inputs = torch.randn(16, fixtures["input_dim"], generator=generator).to(fixtures["device"])
    with torch.no_grad():
        candidate_output = candidate(eval_inputs).double()
        reference_output = reference(eval_inputs.double())
    error = float((candidate_output - reference_output).abs().max())
    return {
        "passed": bool(torch.isfinite(candidate_output).all() and torch.allclose(candidate_output, reference_output, rtol=rtol, atol=atol)),
        "details": {"max_abs_error": error, "output_checksum": checksum_tensor(candidate_output)},
    }


def check_training_correctness(solution: Any, fixtures: dict[str, Any], rtol: float, atol: float) -> dict[str, Any]:
    return check_vjp_training(solution, fixtures, rtol, atol)
