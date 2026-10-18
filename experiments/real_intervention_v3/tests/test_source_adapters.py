from experiments.real_intervention_v3 import source_adapters


def test_model_universe_uses_registered_models_and_known_shared_families(monkeypatch):
    monkeypatch.setattr(source_adapters, "torchvision_models", lambda: [
        "resnet18", "resnext50_32x4d", "inception_v3", "googlenet", "vit_b_16", "swin_t",
    ])
    monkeypatch.setattr(source_adapters, "_torchbench_entries", lambda root: [
        {"source": "torchbench", "model": "resnet50", "availability": "available", "eval_deterministic": True},
        {"source": "torchbench", "model": "unknown_arch", "availability": "available", "eval_deterministic": False},
    ])

    rows = source_adapters.model_universe("pinned-checkout")
    families = {(row["source"], row["model"]): row["family"] for row in rows}
    assert families[("torchvision", "resnext50_32x4d")] == "resnet"
    assert families[("torchvision", "inception_v3")] == families[("torchvision", "googlenet")] == "inception"
    assert families[("torchvision", "vit_b_16")] == "vit"
    assert families[("torchvision", "swin_t")] == "swin"
    assert families[("torchbench", "resnet50")] == "resnet"
    assert families[("torchbench", "unknown_arch")] is None
    assert next(row for row in rows if row["model"] == "unknown_arch")["availability"] == "family_unmapped"


def test_torchvision_input_sizes_keep_only_inception_at_299():
    base = {"batch": 1, "shape": [3, 224, 224]}
    assert source_adapters._torchvision_input_shape({**base, "model": "vit_b_16"}) == (1, 3, 224, 224)
    assert source_adapters._torchvision_input_shape({**base, "model": "maxvit_t"}) == (1, 3, 224, 224)
    assert source_adapters._torchvision_input_shape({**base, "model": "googlenet"}) == (1, 3, 224, 224)
    assert source_adapters._torchvision_input_shape({**base, "model": "inception_v3"}) == (1, 3, 299, 299)
