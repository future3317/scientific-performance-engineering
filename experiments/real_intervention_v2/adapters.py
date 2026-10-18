from __future__ import annotations

from importlib import metadata
from typing import Any, Mapping


TORCHVISION_MODELS = ("resnet18", "mobilenet_v3_small", "squeezenet1_0")


def census() -> dict[str, Any]:
    """Package metadata only; does not import torch, torchvision, or models."""
    versions = {}
    for name in ("torch", "torchvision", "torchbenchmark", "torch-benchmark"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "packages": versions,
        "torchvision_fallback_models": list(TORCHVISION_MODELS) if versions["torchvision"] else [],
        "torchbench_available": versions["torchbenchmark"] is not None,
        "download_policy": "network disabled in execution child; pretrained weights disabled",
    }


def load_model(row: Mapping[str, Any], device: str):
    """Load one preinstalled model lazily. Call only from an approved run."""
    import torch

    name = str(row["model"])
    if row["adapter"] == "torchbench":
        from importlib import import_module

        module = import_module(f"torchbenchmark.models.{name}")
        benchmark = module.Model(test="eval", device=device, batch_size=int(row.get("batch_size", 1)))
        network, inputs = benchmark.get_module()
        network = network.to(device).eval()
        inputs = _move(inputs, device)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        return network, inputs
    if row["adapter"] == "torchvision":
        import torchvision.models as tv_models

        if name not in TORCHVISION_MODELS:
            raise ValueError(f"unsupported torchvision fallback model: {name}")
        network = tv_models.get_model(name, weights=None).to(device).eval()
        generator = torch.Generator(device="cpu").manual_seed(int(row.get("seed", 0)))
        shape = row.get("input_shapes", row.get("input_shape"))
        if shape and isinstance(shape[0], list):
            shape = shape[0]
        inputs = torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(device)
        return network, (inputs,)
    raise ValueError(f"unknown adapter: {row['adapter']}")


def _move(value: Any, device: str):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value
