from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TensorComparison:
    passed: bool
    tensor_count: int
    max_abs_error: float
    max_rel_error: float
    reason: str | None = None


def compare_outputs(reference: Any, candidate: Any, *, rtol: float, atol: float) -> TensorComparison:
    """Compare nested model outputs without flattening away structure."""
    import torch

    errors: list[tuple[float, float]] = []

    def visit(left: Any, right: Any, path: str) -> str | None:
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                return f"tensor_type_mismatch:{path}"
            if left.shape != right.shape or left.dtype != right.dtype:
                return f"tensor_metadata_mismatch:{path}"
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                return f"non_finite_output:{path}"
            delta = (left.detach().to(torch.float64) - right.detach().to(torch.float64)).abs()
            scale = left.detach().to(torch.float64).abs().clamp_min(torch.finfo(torch.float64).tiny)
            errors.append((float(delta.max().item()) if delta.numel() else 0.0,
                           float((delta / scale).max().item()) if delta.numel() else 0.0))
            if not torch.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False):
                return f"numerical_mismatch:{path}"
            return None
        if type(left) is not type(right):
            return f"output_type_mismatch:{path}"
        if isinstance(left, dict):
            if left.keys() != right.keys():
                return f"output_keys_mismatch:{path}"
            for key in left:
                reason = visit(left[key], right[key], f"{path}.{key}")
                if reason:
                    return reason
            return None
        if isinstance(left, (tuple, list)):
            if len(left) != len(right):
                return f"output_length_mismatch:{path}"
            for index, (a, b) in enumerate(zip(left, right)):
                reason = visit(a, b, f"{path}[{index}]")
                if reason:
                    return reason
            return None
        if left != right:
            return f"scalar_output_mismatch:{path}"
        return None

    reason = visit(reference, candidate, "output")
    return TensorComparison(
        passed=reason is None,
        tensor_count=len(errors),
        max_abs_error=max((item[0] for item in errors), default=0.0),
        max_rel_error=max((item[1] for item in errors), default=0.0),
        reason=reason,
    )
