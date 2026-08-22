"""Hidden verifier helpers for SCIML-CRYSTAL-DIFFUSION-07R2.

fp64 live recompute of the baseline sampling equations and result-reuse probes.
Nothing here is visible to the agent sandbox.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable

import torch


class _ScoreNet(torch.nn.Module):
    """Architecture must match workspace/solution.py exactly."""

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


def _load_score_double(score_state: dict[str, torch.Tensor], config: dict[str, Any], device: torch.device | str) -> _ScoreNet:
    model = _ScoreNet(
        num_atoms=int(config["num_atoms"]),
        time_emb_dim=int(config["time_emb_dim"]),
        hidden=int(config["hidden"]),
    )
    model = model.double()
    model.load_state_dict({k: v.double() for k, v in score_state.items()})
    model.to(device)
    model.eval()
    return model


def _schedule(config: dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
    num_steps = int(config["num_steps"])
    beta_start = float(config["beta_start"])
    beta_end = float(config["beta_end"])
    betas = [beta_start + (beta_end - beta_start) * t / num_steps for t in range(num_steps + 1)]
    alphas = [1.0 - b for b in betas]
    alpha_bars = []
    prod = 1.0
    for a in alphas:
        prod *= a
        alpha_bars.append(prod)
    return betas, alphas, alpha_bars


def _time_embedding_double(t: int, dim: int, device: torch.device | str) -> torch.Tensor:
    """Baseline time-embedding math in fp64 (mirrors solution.py)."""
    half = dim // 2
    comps: list[torch.Tensor] = []
    for i in range(half):
        freq = 1.0 / (10000.0 ** (i / half))
        comps.append(torch.sin(torch.tensor(t, dtype=torch.float64, device=device) * freq))
        comps.append(torch.cos(torch.tensor(t, dtype=torch.float64, device=device) * freq))
    if dim % 2:
        comps.append(torch.sin(torch.tensor(t, dtype=torch.float64, device=device)))
    return torch.stack(comps, dim=0)


def reference_sample(
    init_state: dict[str, torch.Tensor],
    score_state: dict[str, torch.Tensor],
    config: dict[str, Any],
    num_steps: int,
    device: torch.device | str,
) -> torch.Tensor:
    """fp64 live recompute of the baseline DDPM sampling loop."""
    betas, alphas, alpha_bars = _schedule(config)
    model = _load_score_double(score_state, config, device)
    dim = int(config["time_emb_dim"])

    frac = init_state["frac"].double()
    lengths = init_state["lengths"].double()
    batch_size = frac.shape[0]

    for t in range(num_steps, 0, -1):
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        t_emb = _time_embedding_double(t, dim, device).unsqueeze(0).expand(batch_size, -1)
        with torch.no_grad():
            eps_frac, eps_lengths = model(frac, lengths, t_emb)
        guidance_scale = float(config.get("guidance_scale", 1.0))
        eps_frac = guidance_scale * eps_frac
        eps_lengths = guidance_scale * eps_lengths
        coef = beta_t / math.sqrt(1.0 - alpha_bar_t)
        frac = (frac - coef * eps_frac) / math.sqrt(alpha_t)
        lengths = (lengths - coef * eps_lengths) / math.sqrt(alpha_t)
        frac = frac - frac.floor()
        lengths = torch.clamp(lengths, min=0.1, max=20.0)

    n = frac.shape[1]
    return torch.cat([frac.reshape(batch_size, n * 3), lengths], dim=-1)


def _allclose(candidate: torch.Tensor, reference: torch.Tensor, rtol: float, atol: float) -> tuple[bool, float]:
    err = (candidate.double() - reference).abs().max().item()
    ok = bool(torch.isfinite(candidate).all() and torch.allclose(candidate.double(), reference, rtol=rtol, atol=atol))
    return ok, err


def checksum_tensor(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous()
    return hashlib.sha256(arr.numpy().tobytes()).hexdigest()


def _probe_inputs(base_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {"frac": torch.zeros_like(base_state["frac"]), "lengths": torch.zeros_like(base_state["lengths"])}


def _mutated_inputs(base_state: dict[str, torch.Tensor], seed: int = 0, scale: float = 0.01) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    device = base_state["frac"].device
    noise_frac = torch.randn(base_state["frac"].shape, generator=generator, dtype=base_state["frac"].dtype).to(device)
    noise_lengths = torch.randn(base_state["lengths"].shape, generator=generator, dtype=base_state["lengths"].dtype).to(device)
    return {
        "frac": base_state["frac"] + scale * noise_frac,
        "lengths": base_state["lengths"] + scale * noise_lengths,
    }


def check_correctness(
    sample_fn: Callable[..., torch.Tensor],
    fixtures: dict[str, Any],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Correctness on one fresh draw + zero/mutated input probes."""
    init_state = {"frac": fixtures["init_frac"], "lengths": fixtures["init_lengths"]}
    num_steps = int(fixtures["config"]["num_steps"])
    device = fixtures["device"]

    candidate = sample_fn(**init_state)
    reference = reference_sample(init_state, fixtures["score_state"], fixtures["config"], num_steps, device)
    ok, err = _allclose(candidate, reference, rtol, atol)
    base_checksum = checksum_tensor(candidate)

    probes: dict[str, Any] = {}
    probes_passed = True
    for name, inputs in (
        ("zero_input", _probe_inputs(init_state)),
        ("mutated_input", _mutated_inputs(init_state)),
    ):
        probe_out = sample_fn(**inputs)
        probe_ref = reference_sample(inputs, fixtures["score_state"], fixtures["config"], num_steps, device)
        p_ok, p_err = _allclose(probe_out, probe_ref, rtol, atol)
        reused = checksum_tensor(probe_out) == base_checksum
        probes[name] = {"passed": p_ok and not reused, "max_abs_error": p_err, "reused_result": reused}
        probes_passed = probes_passed and probes[name]["passed"]

    details = {
        "fresh_input": {"passed": ok, "max_abs_error": err},
        "probes": probes,
        "output_checksum": base_checksum,
    }
    return {"passed": ok and probes_passed, "details": details}
