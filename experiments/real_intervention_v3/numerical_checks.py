from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class NumericalComparison:
    passed: bool
    tensor_count: int
    max_abs_error: float
    max_rel_error: float
    reason: str | None = None


def nested_digest(value: Any) -> str:
    """Hash nested tensors by logical values and structure, independent of device."""
    import torch

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(repr((tuple(tensor.shape), str(tensor.dtype))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda key: repr(key)):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"\0")
            for child in item:
                visit(child)
        else:
            digest.update(type(item).__qualname__.encode() + b"\0" + repr(item).encode() + b"\0")

    visit(value)
    return digest.hexdigest()


def compare_outputs(reference: Any, candidate: Any, *, rtol: float, atol: float) -> NumericalComparison:
    """Check nested output structure, tensor metadata, finiteness and tolerance."""
    import torch

    errors: list[tuple[float, float]] = []

    def visit(left: Any, right: Any, path: str) -> str | None:
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                return f"tensor_type_mismatch:{path}"
            if left.shape != right.shape or left.dtype != right.dtype:
                return f"tensor_metadata_mismatch:{path}"
            if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
                return f"non_finite_output:{path}"
            delta = (left.detach().to(torch.float64) - right.detach().to(torch.float64)).abs()
            scale = left.detach().to(torch.float64).abs().clamp_min(torch.finfo(torch.float64).tiny)
            errors.append((float(delta.max()) if delta.numel() else 0.0,
                           float((delta / scale).max()) if delta.numel() else 0.0))
            return None if torch.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False) else f"numerical_mismatch:{path}"
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
        return None if left == right else f"scalar_output_mismatch:{path}"

    reason = visit(reference, candidate, "output")
    return NumericalComparison(reason is None, len(errors), max((e[0] for e in errors), default=0.0),
                               max((e[1] for e in errors), default=0.0), reason)


def comparison_dict(reference: Any, candidate: Any, *, rtol: float, atol: float) -> dict[str, Any]:
    return asdict(compare_outputs(reference, candidate, rtol=rtol, atol=atol))
