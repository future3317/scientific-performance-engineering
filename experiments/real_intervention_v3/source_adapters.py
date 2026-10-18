from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Mapping


TORCHVISION_REPRESENTATIVES = {
    "resnet": "resnet18",
    "densenet": "densenet121",
    "mobilenet": "mobilenet_v3_small",
    "efficientnet": "efficientnet_b0",
    "convnext": "convnext_tiny",
    "regnet": "regnet_y_400mf",
    "shufflenet": "shufflenet_v2_x0_5",
    "mnasnet": "mnasnet0_5",
    "squeezenet": "squeezenet1_0",
    "vgg": "vgg11",
    "alexnet": "alexnet",
    "vit": "vit_b_16",
    "swin": "swin_t",
    "maxvit": "maxvit_t",
    "inception": "inception_v3",
}
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
_TORCHBENCH_NAMED_FAMILIES = {
    "pytorch_unet": "unet", "dlrm": "dlrm", "super_slomo": "unet",
    "pytorch_cyclegan_and_pix2pix": "cyclegan_generator", "cyclegan": "cyclegan_generator",
    "maml": "maml", "fastnlp_bert": "bert",
}
_TORCHBENCH_CANDIDATES = (
    "pytorch_unet", "dlrm", "Super_SloMo", "pytorch_CycleGAN_and_pix2pix", "maml", "fastNLP_Bert",
)
DEFAULT_TORCHBENCH_ROOT = Path("/home/workspace/lrh/DATA/evospe-rt-v2/torchbench")


def torchvision_models() -> list[str]:
    """Return every model name registered by the installed torchvision."""
    from torchvision.models import list_models

    return sorted(list_models())


def _torchvision_family(name: str) -> str | None:
    return next((family for prefix, family in _TV_FAMILY_PREFIXES.items() if name.startswith(prefix)), None)


def _torchvision_input_shape(spec: Mapping[str, Any]) -> tuple[int, int, int, int]:
    name = str(spec["model"])
    height, width = int(spec["shape"][-2]), int(spec["shape"][-1])
    if name.startswith("inception_v3"):
        height = width = 299
    return int(spec["batch"]), 3, height, width


def _torchbench_family(name: str) -> str | None:
    normalized = name.lower().replace("-", "_")
    if normalized in _TORCHBENCH_NAMED_FAMILIES:
        return _TORCHBENCH_NAMED_FAMILIES[normalized]
    return next((family for prefix, family in _TORCHBENCH_FAMILY_PREFIXES.items()
                 if normalized.startswith(prefix)), None)


def _torchbench_entries(root: str | Path | None) -> list[dict[str, Any]]:
    """Read model metadata without importing model code or running setup hooks."""
    import yaml

    checkout = Path(root or DEFAULT_TORCHBENCH_ROOT).resolve()
    models_root = checkout / "torchbenchmark" / "models"
    existing = {path.name: path for path in models_root.iterdir() if path.is_dir()} if models_root.is_dir() else {}
    candidates = {name.lower(): name for name in _TORCHBENCH_CANDIDATES}
    existing_by_lower = {name.lower(): name for name in existing}
    names = set(existing) | {existing_by_lower.get(normalized, original)
                             for normalized, original in candidates.items()}
    rows: list[dict[str, Any]] = []
    for name in sorted(names):
        directory = existing.get(name)
        if directory is None:
            directory = next((path for key, path in existing.items() if key.lower() == name.lower()), None)
        metadata_path = directory / "metadata.yaml" if directory else None
        is_named_candidate = name.lower() in candidates
        if metadata_path is None or not metadata_path.is_file():
            if is_named_candidate:
                rows.append({"source": "torchbench", "model": name, "family": _torchbench_family(name),
                             "availability": "error", "metadata_path": str(metadata_path) if metadata_path else None,
                             "error": "model directory missing" if directory is None else "metadata.yaml missing",
                             "correctness_status": "determinism_unknown"})
            continue
        try:
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            rows.append({"source": "torchbench", "model": name, "family": _torchbench_family(name),
                         "availability": "error", "metadata_path": str(metadata_path),
                         "error": f"metadata parse error: {type(exc).__name__}: {exc}", "correctness_status": "determinism_unknown"})
            continue
        if not isinstance(metadata, Mapping):
            metadata = {}
        deterministic = metadata.get("eval_deterministic")
        if not isinstance(deterministic, bool):
            deterministic = None
        rows.append({"source": "torchbench", "model": name, "family": _torchbench_family(name),
                     "availability": "available", "metadata_supported": True, "metadata_path": str(metadata_path),
                     "eval_deterministic": deterministic,
                     "eligible_for_deterministic_primary": deterministic is True,
                     "correctness_status": "deterministic_candidate" if deterministic is True else
                                           "stochastic_unqualified" if deterministic is False else "determinism_unknown",
                     "input_kind": "TORCHBENCH_MINIATURE_INPUT"})
    return rows


def model_universe(torchbench_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Return frozen torchvision representatives, its full census, and metadata-backed TorchBench rows."""
    registered = set(torchvision_models())
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for family, model in TORCHVISION_REPRESENTATIVES.items():
        available = model in registered
        result.append({"source": "torchvision", "model": model, "family": family,
                       "availability": "available" if available else "not_registered",
                       "is_representative": True, "input_kind": "REAL_MODEL_MATERIALIZED_INPUT"})
        seen.add(model)
    for model in sorted(registered - seen):
        family = _torchvision_family(model)
        result.append({"source": "torchvision", "model": model, "family": family,
                       "availability": "available",
                       "family_status": "mapped" if family else "family_unmapped",
                       "is_representative": False, "input_kind": "REAL_MODEL_MATERIALIZED_INPUT"})
    for row in _torchbench_entries(torchbench_root):
        family = _torchbench_family(str(row["model"]))
        row["family"] = family
        if family is None and row.get("availability") == "available":
            row["family_status"] = "family_unmapped"
        result.append(row)
    return result


def load_model(spec: Mapping[str, Any]):
    """Load an eval model and reconstruct the requested fixed inputs."""
    _deny_network()
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
        sys.path.insert(0, str(root))
        module = importlib.import_module(f"torchbenchmark.models.{name}")
        benchmark = module.Model(test="eval", device="cuda", batch_size=int(spec["batch"]))
        model, inputs = benchmark.get_module()
        model = model.cuda().eval()
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        return model, tuple(_to_device(value) for value in inputs)
    raise ValueError(f"unsupported source: {source!r}")


def _deny_network() -> None:
    """Disable remote fetches in this short lived worker before importing model code."""
    import socket
    import urllib.request

    def blocked(*args, **kwargs):
        raise RuntimeError("network access is disabled in real intervention workers")

    urllib.request.urlopen = blocked
    socket.getaddrinfo = blocked
    socket.create_connection = blocked
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked


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
