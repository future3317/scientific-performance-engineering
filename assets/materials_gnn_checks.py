#!/usr/bin/env python3
"""Small, copyable checks for work-normalized and rank-three tensor metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class WorkWindow:
    """Totals observed over one already-timed benchmark window."""

    seconds: float
    crystals: int
    atoms: int
    edges: int


def work_rates(window: WorkWindow) -> dict[str, float]:
    """Return crystal, atom, and edge throughput for one timing window."""
    if window.seconds <= 0:
        raise ValueError("window.seconds must be positive")
    return {
        "crystals_per_second": window.crystals / window.seconds,
        "atoms_per_second": window.atoms / window.seconds,
        "edges_per_second": window.edges / window.seconds,
    }


def aggregate_work_rates(windows: Iterable[WorkWindow]) -> dict[str, float]:
    """Return aggregate throughput; do not average per-window rates."""
    windows = tuple(windows)
    if not windows:
        raise ValueError("at least one timing window is required")
    return work_rates(
        WorkWindow(
            seconds=sum(window.seconds for window in windows),
            crystals=sum(window.crystals for window in windows),
            atoms=sum(window.atoms for window in windows),
            edges=sum(window.edges for window in windows),
        )
    )


def rotate_rank_three_cartesian(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply a 3x3 rotation to the final three Cartesian dimensions."""
    if tensor.shape[-3:] != (3, 3, 3) or rotation.shape != (3, 3):
        raise ValueError("expected tensor[..., 3, 3, 3] and rotation[3, 3]")
    return torch.einsum("ia,jb,kc,...abc->...ijk", rotation, rotation, rotation, tensor)


def rank_three_equivariance_error(
    prediction: torch.Tensor,
    rotated_prediction: torch.Tensor,
    rotation: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return ||f(Rx)-R^3f(x)|| / (||R^3f(x)|| + epsilon)."""
    expected = rotate_rank_three_cartesian(prediction, rotation)
    if rotated_prediction.shape != expected.shape:
        raise ValueError("prediction and rotated_prediction must have the same shape")
    return torch.linalg.vector_norm(rotated_prediction - expected) / (
        torch.linalg.vector_norm(expected) + epsilon
    )


def self_test() -> None:
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    tensor = torch.arange(27, dtype=torch.float64).reshape(3, 3, 3)
    rotated = rotate_rank_three_cartesian(tensor, rotation)
    assert torch.allclose(
        rank_three_equivariance_error(tensor, rotated, rotation),
        torch.zeros((), dtype=tensor.dtype),
    )
    assert aggregate_work_rates([WorkWindow(2.0, 4, 20, 80), WorkWindow(3.0, 6, 30, 120)]) == {
        "crystals_per_second": 2.0,
        "atoms_per_second": 10.0,
        "edges_per_second": 40.0,
    }
    print("materials_gnn_checks: self-test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    if parser.parse_args().self_test:
        self_test()
    else:
        parser.print_help()
