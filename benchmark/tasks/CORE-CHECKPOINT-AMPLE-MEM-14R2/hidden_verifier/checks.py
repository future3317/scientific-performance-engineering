from __future__ import annotations
import hashlib
import torch
from torch import nn


def checksum_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _reference_model(fixtures):
    config = fixtures["model_config"]
    model = nn.Module()
    model.input = nn.Linear(config["in_dim"], config["hidden_dim"])
    model.segments = nn.ModuleList(nn.Sequential(nn.Linear(config["hidden_dim"], config["hidden_dim"]), nn.ReLU(), nn.Linear(config["hidden_dim"], config["hidden_dim"]), nn.ReLU()) for _ in range(fixtures["segment_count"]))
    model.output = nn.Linear(config["hidden_dim"], 1)
    model.load_state_dict(fixtures["init_state"])
    return model.to(fixtures["device"])


def _reference_step(model, x, target, optimizer):
    optimizer.zero_grad(); h = torch.relu(model.input(x))
    for segment in model.segments: h = h + segment(h)
    loss = torch.nn.functional.mse_loss(model.output(h).squeeze(-1), target); loss.backward(); optimizer.step()


def check_training_correctness(solution, fixtures, rtol, atol):
    candidate = solution.build_model(fixtures)
    optimizer = torch.optim.SGD(candidate.parameters(), lr=fixtures["optimizer_config"]["lr"])
    reference = _reference_model(fixtures)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=fixtures["optimizer_config"]["lr"])
    for i in range(3):
        size = fixtures["batch_sizes"][i % len(fixtures["batch_sizes"])]
        offset = (i * size) % (fixtures["inputs"].shape[0] - size)
        x = fixtures["inputs"][offset:offset + size].to(fixtures["device"])
        target = fixtures["targets"][offset:offset + size].to(fixtures["device"])
        solution.train_step(candidate, (x, target), optimizer); _reference_step(reference, x, target, ref_optimizer)
    with torch.no_grad():
        x = fixtures["eval_inputs"].to(fixtures["device"])
        candidate_output = candidate(x).squeeze(-1); h = torch.relu(reference.input(x))
        for segment in reference.segments: h = h + segment(h)
        reference_output = reference.output(h).squeeze(-1)
    error = float((candidate_output.double() - reference_output.double()).abs().max())
    return {"passed": bool(torch.allclose(candidate_output.double(), reference_output.double(), rtol=rtol, atol=atol)), "details": {"max_abs_error": error, "output_checksum": checksum_tensor(candidate_output)}}



