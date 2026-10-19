"""Load the frozen model sources used by the v4 primary study."""

from __future__ import annotations

import importlib.metadata
from typing import Any, Mapping


def load_model(spec: Mapping[str, Any]):
    from experiments.real_intervention_v3.source_adapters import (
        _deny_network,
        load_model as load_v3_model,
    )

    _deny_network()
    if spec.get("source") != "timm":
        return load_v3_model(spec)

    import torch

    source_identity = spec.get("source_identity")
    revisions = (source_identity.get("source_revisions")
                 if isinstance(source_identity, Mapping) else None)
    expected_version = revisions.get("timm_version") if isinstance(revisions, Mapping) else None
    installed_version = importlib.metadata.version("timm")
    if not expected_version or installed_version != str(expected_version):
        raise RuntimeError(
            f"timm source version differs from the frozen source identity: "
            f"expected {expected_version!r}, observed {installed_version!r}"
        )
    if spec.get("pretrained") is not False:
        raise ValueError("v4 timm candidates must use pretrained=False")

    import timm

    model = timm.create_model(
        str(spec["model"]), pretrained=False,
        num_classes=int(spec.get("num_classes", 1000)),
    ).cuda().eval()
    shape = tuple(int(item) for item in spec["shape"])
    generator = torch.Generator(device="cpu").manual_seed(int(spec["seed"]))
    inputs = torch.randn(
        (int(spec["batch"]), *shape), generator=generator,
        dtype=torch.float32,
    ).cuda()
    return model, (inputs,)
