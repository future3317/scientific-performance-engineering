import pytest

from experiments.real_intervention_v2.prepare_study import _initialize_run_root, _ordered_models


def test_ordered_models_keep_seeded_lineage_split_order_and_group_variants():
    eligible = ["hf_Bert", "resnet18", "BERT_pytorch", "alexnet"]
    lineages = ["resnet18", "bert", "alexnet"]

    assert _ordered_models(eligible, lineages) == [
        "resnet18", "BERT_pytorch", "hf_Bert", "alexnet"
    ]


def test_run_root_reuses_admission_cache_but_never_overwrites_frozen_artifacts(tmp_path):
    run_root = tmp_path / "study"
    cache_file = run_root / "model-cache" / "weights.bin"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"cached")

    _initialize_run_root(run_root)
    assert cache_file.read_bytes() == b"cached"

    (run_root / "RUN_MANIFEST.json").write_text("{}")
    with pytest.raises(FileExistsError, match="study artifacts already exist"):
        _initialize_run_root(run_root)
