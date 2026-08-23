from __future__ import annotations

import math
from typing import Any

import torch


class _ScoreNet(torch.nn.Module):
    def __init__(self, num_atoms: int, time_emb_dim: int, hidden: int) -> None:
        super().__init__()
        self.time_emb_dim = time_emb_dim
        in_dim = num_atoms * 3 + 3 + time_emb_dim
        out_dim = num_atoms * 3 + 3
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, out_dim)

    def forward(self, frac, lengths, t_emb):
        batch_size, num_atoms, _ = frac.shape
        x = torch.cat([frac.reshape(batch_size, num_atoms * 3), lengths, t_emb], dim=-1)
        out = self.fc2(torch.tanh(self.fc1(x)))
        return out[..., : num_atoms * 3].reshape(batch_size, num_atoms, 3), out[..., num_atoms * 3 :]


def _build_time_embedding(t: int, dim: int, device: torch.device | str) -> torch.Tensor:
    half = dim // 2
    components = []
    for i in range(half):
        freq = 1.0 / (10000.0 ** (i / half))
        t_tensor = torch.tensor(t, dtype=torch.float64, device=device)
        components.append(torch.sin(t_tensor * freq))
        components.append(torch.cos(t_tensor * freq))
    if dim % 2:
        components.append(torch.sin(torch.tensor(t, dtype=torch.float64, device=device)))
    return torch.stack(components, dim=0)


def build_sampler(fixtures: dict[str, Any]) -> dict[str, Any]:
    config = fixtures["config"]
    device = fixtures["device"]
    num_steps = int(config["num_steps"])
    score = _ScoreNet(int(config["num_atoms"]), int(config["time_emb_dim"]), int(config["hidden"]))
    score.load_state_dict(fixtures["score_state"])
    score = score.double()
    score.to(device)
    score.eval()

    betas = [config["beta_start"] + (config["beta_end"] - config["beta_start"]) * t / num_steps for t in range(num_steps + 1)]
    alphas = [1.0 - beta for beta in betas]
    alpha_bars = []
    product = 1.0
    for alpha in alphas:
        product *= alpha
        alpha_bars.append(product)

    return {
        "score": score,
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "time_emb": torch.stack(
            [_build_time_embedding(t, int(config["time_emb_dim"]), device) for t in range(num_steps + 1)]
        ),
        "num_steps": num_steps,
        "guidance_scale": float(config["guidance_scale"]),
        "device": device,
    }


def sample_step(sampler: dict[str, Any], state: dict[str, torch.Tensor], step_index: int):
    beta_t = sampler["betas"][step_index]
    alpha_t = sampler["alphas"][step_index]
    alpha_bar_t = sampler["alpha_bars"][step_index]
    frac = state["frac"]
    lengths = state["lengths"]
    t_emb = sampler["time_emb"][step_index].unsqueeze(0).expand(frac.shape[0], -1)
    with torch.no_grad():
        eps_frac, eps_lengths = sampler["score"](frac, lengths, t_emb)
    eps_frac = sampler["guidance_scale"] * eps_frac
    eps_lengths = sampler["guidance_scale"] * eps_lengths
    coef = beta_t / math.sqrt(1.0 - alpha_bar_t)
    frac = (frac - coef * eps_frac) / math.sqrt(alpha_t)
    lengths = (lengths - coef * eps_lengths) / math.sqrt(alpha_t)
    return {
        "frac": frac - frac.floor(),
        "lengths": torch.clamp(lengths, min=0.1, max=20.0),
    }


def sample(sampler: dict[str, Any], fixtures: dict[str, Any], num_steps: int) -> torch.Tensor:
    state = {"frac": fixtures["init_frac"].double().clone(), "lengths": fixtures["init_lengths"].double().clone()}
    for step_index in range(num_steps, 0, -1):
        state = sample_step(sampler, state, step_index)
    return torch.cat([state["frac"].reshape(state["frac"].shape[0], -1), state["lengths"]], dim=-1)
