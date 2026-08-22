"""Hidden verifier logic for CORE-KERNEL-FUSION-09R2 (harness-only).

fp64 live-recomputed reference (never stored goldens), adversarial
result-reuse probes (harness anticheat helpers, with an in-task CUDA-device
workaround), and output checksums.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from benchmark.harness import anticheat

_MUTATION_SCALE = 0.01
_MUTATION_SEED = 0


def reference_fp64(x: torch.Tensor, residual: torch.Tensor, params: dict[str, float]) -> torch.Tensor:
    """fp64 live recompute of the pointwise chain (TorchBench-style reference)."""
    z = x.double() * params["a1"] + params["b1"]
    h = z * torch.sigmoid(z)
    y = h + residual.double()
    yc = torch.clamp(y, params["clamp_min"], params["clamp_max"])
    return yc * params["a2"] + params["b2"]


def params_of(fixtures: dict[str, Any]) -> dict[str, float]:
    return {
        "a1": float(fixtures["a1"]),
        "b1": float(fixtures["b1"]),
        "a2": float(fixtures["a2"]),
        "b2": float(fixtures["b2"]),
        "clamp_min": float(fixtures["clamp_min"]),
        "clamp_max": float(fixtures["clamp_max"]),
    }


def checksum_tensor(tensor: torch.Tensor) -> str:
    """SHA-256 of the output bytes (anti-caching evidence recorded in results)."""
    arr = tensor.detach().cpu().contiguous()
    return hashlib.sha256(arr.numpy().tobytes()).hexdigest()


def _allclose(candidate: torch.Tensor, reference: torch.Tensor, rtol: float, atol: float) -> tuple[bool, float]:
    err = (candidate.double() - reference).abs().max().item()
    ok = bool(
        torch.isfinite(candidate).all()
        and torch.allclose(candidate.double(), reference, rtol=rtol, atol=atol)
    )
    return ok, err


def _zero_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Adversarial zero-input probe."""
    return {key: torch.zeros_like(value) for key, value in inputs.items()}


def _mutate_inputs(inputs: dict[str, torch.Tensor], seed: int, scale: float) -> dict[str, torch.Tensor]:
    """Adversarial mutated-input probe with device-aware noise.

    NOTE: This mirrors ``anticheat.mutated_input_probe`` but creates the noise
    tensor on the same device as the input, avoiding a CUDA/CPU device mismatch
    in the harness helper.
    """
    device = next(iter(inputs.values())).device
    generator = torch.Generator(device=device).manual_seed(seed)
    mutated: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            noise = torch.randn(value.shape, generator=generator, dtype=value.dtype, device=value.device)
            mutated[key] = value + scale * noise
        else:
            mutated[key] = value
    return mutated


def _run_probe(
    forward_fn,
    probe_inputs: dict[str, torch.Tensor],
    params: dict[str, float],
    base_checksum: str,
    rtol: float,
    atol: float,
    use_helper: str,
) -> tuple[bool, dict[str, Any]]:
    """Run one adversarial probe and check the output against a live fp64
    recompute on the probe inputs."""
    if use_helper == "zero":
        probed = _zero_inputs(probe_inputs)
        out = forward_fn(**probed)
        reference = reference_fp64(probed["x"], probed["residual"], params)
    else:
        probed = _mutate_inputs(probe_inputs, seed=_MUTATION_SEED, scale=_MUTATION_SCALE)
        out = forward_fn(**probed)
        reference = reference_fp64(probed["x"], probed["residual"], params)
    ok, err = _allclose(out, reference, rtol, atol)
    reused = checksum_tensor(out) == base_checksum
    return ok and not reused, {"max_abs_error": err, "reused_result": reused}


def check_output(
    forward_fn,
    x: torch.Tensor,
    residual: torch.Tensor,
    fixtures: dict[str, Any],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Correctness on one fresh input draw + adversarial result-reuse probes.

    *forward_fn* is called as ``forward_fn(x=..., residual=...)``. A candidate
    that replays a cached/stale result fails: the reference is recomputed live
    on the zeroed/mutated probe inputs, and identical output checksums across
    probes are flagged as reuse.
    """
    params = params_of(fixtures)
    original_inputs = {"x": x, "residual": residual}

    candidate = forward_fn(**original_inputs)
    reference = reference_fp64(x, residual, params)
    ok, err = _allclose(candidate, reference, rtol, atol)
    base_checksum = checksum_tensor(candidate)

    probes: dict[str, Any] = {}
    probes_passed = True
    for name, helper in (
        ("zero_input", "zero"),
        ("mutated_input", "mutated"),
    ):
        p_ok, p_details = _run_probe(forward_fn, original_inputs, params, base_checksum, rtol, atol, helper)
        probes[name] = {"passed": p_ok, **p_details}
        probes_passed = probes_passed and p_ok

    details = {
        "fresh_input": {"passed": ok, "max_abs_error": err},
        "probes": probes,
        "output_checksum": base_checksum,
    }
    return {"passed": ok and probes_passed, "details": details}


