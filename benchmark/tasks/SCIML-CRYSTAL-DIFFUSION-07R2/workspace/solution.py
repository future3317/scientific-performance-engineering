"""Baseline DDPM-style crystal sampler (sampler_v1 API).

Correct but inefficient.  The sampler denoises fractional coordinates and
orthorhombic lattice lengths for tiny synthetic crystals.  The baseline
injections are:

* per-step scalar synchronization (schedule scalars read with `.item()`),
* launch fragmentation (time embedding rebuilt from many tiny ops each step),
* one-crystal-at-a-time Python loop over the evaluation batch.

The oracle fix keeps the same score network and update equations but
pre-computes the schedule on device, vectorises the time embedding, and batches
the sampling loop.
"""

from __future__ import annotations

import math
from typing import Any

import torch


class _ScoreNet(torch.nn.Module):
    """Tiny MLP score network: predicts noise from [frac, lengths, t_emb]."""

    def __init__(self, num_atoms: int, time_emb_dim: int, hidden: int) -> None:
        super().__init__()
        self.time_emb_dim = time_emb_dim
        in_dim = num_atoms * 3 + 3 + time_emb_dim
        out_dim = num_atoms * 3 + 3
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, out_dim)

    def forward(self, frac: torch.Tensor, lengths: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, _ = frac.shape
        x = torch.cat([frac.reshape(b, n * 3), lengths, t_emb], dim=-1)
        h = torch.tanh(self.fc1(x))
        out = self.fc2(h)
        noise_frac = out[..., : n * 3].reshape(b, n, 3)
        noise_lengths = out[..., n * 3 :]
        return noise_frac, noise_lengths


def _build_time_embedding_baseline(t: int, dim: int, device: torch.device | str) -> torch.Tensor:
    """Sinusoidal time embedding rebuilt with many tiny ops every step."""
    # Baseline: build each frequency component separately and concatenate.
    components: list[torch.Tensor] = []
    half = dim // 2
    for i in range(half):
        freq = 1.0 / (10000.0 ** (i / half))
        t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
        components.append(torch.sin(t_tensor * freq))
        components.append(torch.cos(t_tensor * freq))
    if dim % 2:
        components.append(torch.sin(torch.tensor(t, dtype=torch.float32, device=device)))
    return torch.stack(components, dim=0)


def build_sampler(fixtures: dict[str, Any]) -> dict[str, Any]:
    """Build a sampler context from the task fixtures."""
    config = fixtures["config"]
    device = fixtures["device"]
    num_steps = int(config["num_steps"])
    score = _ScoreNet(
        num_atoms=int(config["num_atoms"]),
        time_emb_dim=int(config["time_emb_dim"]),
        hidden=int(config["hidden"]),
    )
    score.load_state_dict(fixtures["score_state"])
    score.to(device)
    score.eval()

    # Noise schedule as plain Python lists (baseline keeps scalars on CPU).
    betas = [config["beta_start"] + (config["beta_end"] - config["beta_start"]) * t / num_steps for t in range(num_steps + 1)]
    alphas = [1.0 - b for b in betas]
    alpha_bars = []
    prod = 1.0
    for a in alphas:
        prod *= a
        alpha_bars.append(prod)

    return {
        "score": score,
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "num_steps": num_steps,
        "time_emb_dim": int(config["time_emb_dim"]),
        "guidance_scale": float(config["guidance_scale"]),
        "device": device,
    }


def sample_step(sampler: dict[str, Any], state: dict[str, torch.Tensor], step_index: int) -> dict[str, torch.Tensor]:
    """One DDPM denoising step.  `step_index` is the current timestep t (1..T)."""
    t = step_index
    # Scalar sync: pull schedule values back to host as Python floats.
    beta_t = sampler["betas"][t].item() if isinstance(sampler["betas"][t], torch.Tensor) else sampler["betas"][t]
    alpha_t = sampler["alphas"][t].item() if isinstance(sampler["alphas"][t], torch.Tensor) else sampler["alphas"][t]
    alpha_bar_t = sampler["alpha_bars"][t].item() if isinstance(sampler["alpha_bars"][t], torch.Tensor) else sampler["alpha_bars"][t]

    frac = state["frac"]
    lengths = state["lengths"]
    device = frac.device

    # Launch fragmentation: rebuild the time embedding from many tiny ops.
    t_emb = _build_time_embedding_baseline(t, sampler["time_emb_dim"], device)
    # t_emb is [dim]; score expects [B, dim].
    batch_size = frac.shape[0]
    t_emb = t_emb.unsqueeze(0).expand(batch_size, -1)

    with torch.no_grad():
        eps_frac, eps_lengths = sampler["score"](frac, lengths, t_emb)
        eps_frac = sampler["guidance_scale"] * eps_frac
        eps_lengths = sampler["guidance_scale"] * eps_lengths

    # DDPM posterior mean (deterministic sigma = 0 for reproducibility).
    coef = beta_t / math.sqrt(1.0 - alpha_bar_t)
    new_frac = (frac - coef * eps_frac) / math.sqrt(alpha_t)
    new_lengths = (lengths - coef * eps_lengths) / math.sqrt(alpha_t)

    # Wrap fractional coordinates into [0, 1).
    new_frac = new_frac - new_frac.floor()
    new_lengths = torch.clamp(new_lengths, min=0.1, max=20.0)

    return {"frac": new_frac, "lengths": new_lengths}


def sample(sampler: dict[str, Any], fixtures: dict[str, Any], num_steps: int) -> torch.Tensor:
    """Sample crystals one-at-a-time (baseline inefficiency #3)."""
    state = {
        "frac": fixtures["init_frac"].clone(),
        "lengths": fixtures["init_lengths"].clone(),
    }
    batch_size = state["frac"].shape[0]

    # Baseline: loop over the batch dimension outside the step function.
    per_crystal_states: list[dict[str, torch.Tensor]] = []
    for i in range(batch_size):
        per_crystal_states.append({
            "frac": state["frac"][i : i + 1].clone(),
            "lengths": state["lengths"][i : i + 1].clone(),
        })

    for t in range(num_steps, 0, -1):
        next_states: list[dict[str, torch.Tensor]] = []
        for single in per_crystal_states:
            next_states.append(sample_step(sampler, single, t))
        per_crystal_states = next_states

    frac_out = torch.cat([s["frac"] for s in per_crystal_states], dim=0)
    lengths_out = torch.cat([s["lengths"] for s in per_crystal_states], dim=0)
    # Return concatenated fractional coordinates and lattice lengths so the
    # reference and gates can reconstruct Cartesian coordinates.
    return torch.cat([frac_out.reshape(batch_size, -1), lengths_out], dim=-1)
