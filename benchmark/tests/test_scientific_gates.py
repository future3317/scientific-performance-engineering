#!/usr/bin/env python3
"""Standalone assert-script tests for harness/scientific_gates.py (tiny tensors)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from benchmark.harness import scientific_gates as gates


def main() -> None:
    torch.manual_seed(0)
    dtype = torch.float64

    # --- energy_force_consistency ---------------------------------------------
    # E(x) = sum(x^2); true forces = -2x.
    positions = torch.randn(5, 3, dtype=dtype)

    def good_energy_fn(pos):
        energy = (pos ** 2).sum()
        return {"energy": energy, "forces": -2.0 * pos}

    def bad_energy_fn(pos):
        energy = (pos ** 2).sum()
        return {"energy": energy, "forces": 2.0 * pos}  # sign flipped

    passed, details = gates.energy_force_consistency(good_energy_fn, positions, tol=1e-8)
    assert passed, details
    assert details["max_abs_error"] < 1e-8
    passed, details = gates.energy_force_consistency(bad_energy_fn, positions, tol=1e-8)
    assert not passed and details["max_abs_error"] > 1.0

    # --- gradient_agreement -----------------------------------------------------
    inputs = torch.randn(7, dtype=dtype)
    ref = lambda x: (x ** 3).sum() + x.sum()
    passed, details = gates.gradient_agreement(ref, ref, inputs, tol=1e-8)
    assert passed, details
    wrong = lambda x: (x ** 2).sum()
    passed, details = gates.gradient_agreement(wrong, ref, inputs, tol=1e-8)
    assert not passed and details["max_gradient_abs_error"] > 0.1

    # --- equivariance_rank3 -------------------------------------------------------
    x = torch.randn(4, 3, dtype=dtype)

    def equivariant_fn(pos):
        return torch.einsum("...i,...j,...k->...ijk", pos, pos, pos)

    def non_equivariant_fn(pos):
        out = torch.einsum("...i,...j,...k->...ijk", pos, pos, pos)
        return out + 0.5  # constant shift breaks equivariance

    passed, details = gates.equivariance_rank3(equivariant_fn, x, tol=1e-8)
    assert passed, details
    assert details["relative_equivariance_error"] < 1e-8
    passed, details = gates.equivariance_rank3(non_equivariant_fn, x, tol=1e-8)
    assert not passed and details["relative_equivariance_error"] > 1e-3

    # --- crystal_validity -----------------------------------------------------------
    good_pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 2.0]], dtype=dtype)
    lattice = torch.eye(3, dtype=dtype) * 10.0
    passed, details = gates.crystal_validity(good_pos, lattice, min_dist=0.5)
    assert passed, details
    assert abs(details["min_interatomic_distance"] - 2.0) < 1e-8

    close_pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=dtype)
    passed, details = gates.crystal_validity(close_pos, lattice, min_dist=0.5)
    assert not passed and details["min_interatomic_distance"] < 0.5

    bad_lattice = torch.eye(3, dtype=dtype) * 0.01
    passed, details = gates.crystal_validity(good_pos, bad_lattice, min_dist=0.5)
    assert not passed and details["lattice_in_range"] is False

    nan_pos = good_pos.clone()
    nan_pos[0, 0] = float("nan")
    passed, details = gates.crystal_validity(nan_pos, lattice, min_dist=0.5)
    assert not passed and "non-finite" in details["reason"]

    # --- distribution_moment_check ------------------------------------------------------
    samples = torch.linspace(-2.0, 2.0, 101, dtype=dtype)
    mean = samples.mean().item()
    std = samples.std(unbiased=False).item()
    passed, details = gates.distribution_moment_check(samples, {"mean": mean, "std": std}, tol=0.05)
    assert passed, details
    passed, details = gates.distribution_moment_check(samples, {"mean": mean + 0.5, "std": std}, tol=0.05)
    assert not passed and details["moments"]["mean"]["error"] > 0.05

    print("test_scientific_gates: OK")


if __name__ == "__main__":
    main()
