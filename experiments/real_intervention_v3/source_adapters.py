from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Mapping


_TV_FAMILY_PREFIXES = {
    "alexnet": "alexnet", "convnext": "convnext", "densenet": "densenet",
    "efficientnet": "efficientnet", "googlenet": "inception", "inception_v3": "inception",
    "maxvit": "maxvit", "mnasnet": "mnasnet", "mobilenet_v2": "mobilenet",
    "mobilenet_v3": "mobilenet", "regnet": "regnet", "resnet": "resnet",
    "resnext": "resnet", "wide_resnet": "resnet", "shufflenet": "shufflenet",
    "squeezenet": "squeezenet", "swin": "swin", "vgg": "vgg", "vit": "vit",
}
_TORCHBENCH_FAMILY_PREFIXES = {
    "resnet": "resnet", "mobilenet": "mobilenet", "efficientnet": "efficientnet",
    "densenet": "densenet", "vgg": "vgg", "alexnet": "alexnet", "squeezenet": "squeezenet",
    "shufflenet": "shufflenet", "mnasnet": "mnasnet", "convnext": "convnext",
    "vit": "vit", "swin": "swin", "maxvit": "maxvit", "inception": "inception",
    "googlenet": "inception", "resnext": "resnet", "wide_resnet": "resnet",
}


def torchvision_models() -> list[str]:
    """Return the installed torchvision model names in its official registry."""
    from torchvision.models import list_models

    return sorted(list_models())


def _torchvision_family(name: str) -> str | None:
    for prefix, family in _TV_FAMILY_PREFIXES.items():
        if name.startswith(prefix):
            return family
    return None


def _torchvision_input_shape(spec: Mapping[str, Any]) -> tuple[int, int, int, int]:
    name = str(spec["model"])
    height, width = int(spec["shape"][-2]), int(spec["shape"][-1])
    if name.startswith("inception_v3"):
        height = width = 299
    return int(spec["batch"]), 3, height, width


def _torchbench_entries(root: str | Path | None) -> list[dict[str, Any]]:
    if root is None:
        return []
    checkout = Path(root).resolve()
    package_root = checkout / "torchbenchmark"
    models_root = package_root / "models"
    if not models_root.is_dir():
        raise FileNotFoundError(f"TorchBench models directory not found: {models_root}")

    sys.path.insert(0, str(checkout))
    entries = []
    for model_dir in sorted(path for path in models_root.iterdir() if path.is_dir() and (path / "__init__.py").exists()):
        model = model_dir.name
        try:
            module = importlib.import_module(f"torchbenchmark.models.{model}")
            metadata = getattr(module, "metadata", {})
        except Exception as exc:
            entries.append({"source": "torchbench", "model": model, "family": f"torchbench:{model}",
                            "availability": "metadata_import_error", "reason": f"{type(exc).__name__}: {exc}"})
            continue
        deterministic = metadata.get("eval_deterministic") if isinstance(metadata, Mapping) else None
        entries.append({"source": "torchbench", "model": model, "family": f"torchbench:{model}",
                        "availability": "available", "eval_deterministic": deterministic,
                        "correctness_status": "deterministic_candidate" if deterministic is True else "stochastic_review_required"})
    return entries


def model_universe(torchbench_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Build a fixed source-derived universe; no timing or outcome is consulted."""
    result: list[dict[str, Any]] = []
    for model in torchvision_models():
        family = _torchvision_family(model)
        result.append({"source": "torchvision", "model": model, "family": family,
                       "availability": "available" if family else "family_unmapped",
                       "input_kind": "REAL_MODEL_MATERIALIZED_INPUT"})

    # The configured checkout is the pinned TorchBench source of record. Its own
    # model names remain distinct unless a known torchvision family is explicit.
    known_by_name = {row["model"]: row["family"] for row in result if row["family"]}
    for row in _torchbench_entries(torchbench_root):
        if row["model"] in known_by_name:
            row["family"] = known_by_name[row["model"]]
        else:
            row["family"] = next((family for prefix, family in _TORCHBENCH_FAMILY_PREFIXES.items()
                                  if row["model"].startswith(prefix)), None)
        if row["family"] is None and row["availability"] == "available":
            row["availability"] = "family_unmapped"
        row["input_kind"] = "TORCHBENCH_MINIATURE_INPUT"
        result.append(row)
    return result


def load_model(spec: Mapping[str, Any]):
    """Load an eval model and reconstruct the requested fixed inputs."""
    import torch

    source, name = spec["source"], str(spec["model"])
    torch.set_num_threads(1)
    torch.manual_seed(int(spec["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(spec["seed"]))
    if source == "torchvision":
        import torchvision.models as models

        model = models.get_model(name, weights=None).cuda().eval()
        inputs = torch.randn(_torchvision_input_shape(spec), generator=torch.Generator().manual_seed(int(spec["seed"])), dtype=torch.float32).cuda()
        return model, (inputs,)
    if source == "torchbench":
        root = Path(str(spec["torchbench_root"])).resolve()
        import sys

        sys.path.insert(0, str(root))
        module = importlib.import_module(f"torchbenchmark.models.{name}")
        benchmark = module.Model(test="eval", device="cuda", batch_size=int(spec["batch"]))
        model, inputs = benchmark.get_module()
        model = model.cuda().eval()
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        return model, tuple(_to_device(value) for value in inputs)
    raise ValueError(f"unsupported source: {source!r}")


def _to_device(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, tuple):
        return tuple(_to_device(item) for item in value)
    if isinstance(value, list):
        return [_to_device(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item) for key, item in value.items()}
    return value
