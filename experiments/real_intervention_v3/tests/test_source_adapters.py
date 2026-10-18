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
    unknown = next(row for row in rows if row["source"] == "torchbench" and row["model"] == "unknown_arch")
    assert unknown["availability"] == "available"
    assert unknown["family_status"] == "family_unmapped"


def test_torchvision_input_sizes_keep_only_inception_at_299():
    base = {"batch": 1, "shape": [3, 224, 224]}
    assert source_adapters._torchvision_input_shape({**base, "model": "vit_b_16"}) == (1, 3, 224, 224)
    assert source_adapters._torchvision_input_shape({**base, "model": "maxvit_t"}) == (1, 3, 224, 224)
    assert source_adapters._torchvision_input_shape({**base, "model": "googlenet"}) == (1, 3, 224, 224)
    assert source_adapters._torchvision_input_shape({**base, "model": "inception_v3"}) == (1, 3, 299, 299)


def test_fixed_torchvision_representatives_include_fifteen_families_and_mnasnet0_5():
    assert len(source_adapters.TORCHVISION_REPRESENTATIVES) == 15
    assert source_adapters.TORCHVISION_REPRESENTATIVES["mnasnet"] == "mnasnet0_5"


def test_torchbench_universe_reads_metadata_yaml_without_importing_models(tmp_path):
    models = tmp_path / "torchbenchmark" / "models"
    names = {
        "pytorch_unet": "unet", "dlrm": "dlrm", "Super_SloMo": "unet",
        "pytorch_CycleGAN_and_pix2pix": "cyclegan_generator", "maml": "maml", "fastNLP_Bert": "bert",
    }
    for name in names:
        directory = models / name
        directory.mkdir(parents=True)
        (directory / "__init__.py").write_text("raise RuntimeError('model imported')\n", encoding="utf-8")
        (directory / "metadata.yaml").write_text("eval_deterministic: true\n", encoding="utf-8")
    unknown = models / "unknown_arch"
    unknown.mkdir()
    (unknown / "__init__.py").write_text("raise RuntimeError('model imported')\n", encoding="utf-8")
    (unknown / "metadata.yaml").write_text("eval_deterministic: null\nmodel_family: invented\n", encoding="utf-8")

    rows = source_adapters._torchbench_entries(tmp_path)
    indexed = {row["model"]: row for row in rows}
    for name, family in names.items():
        assert indexed[name]["availability"] == "available"
        assert indexed[name]["family"] == family
        assert indexed[name]["eligible_for_deterministic_primary"] is True
    assert indexed["unknown_arch"]["family"] is None
    assert indexed["unknown_arch"]["eligible_for_deterministic_primary"] is False
    assert indexed["unknown_arch"]["correctness_status"] == "determinism_unknown"


def test_unet_family_is_not_split_by_workload_name():
    assert source_adapters._torchbench_family("pytorch_unet") == "unet"
    assert source_adapters._torchbench_family("Super_SloMo") == "unet"
