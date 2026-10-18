import pytest

from experiments.real_intervention_v2.prepare_study import (
    BATCH_SIZES,
    ISOLATED_MODEL_DEPENDENCIES,
    MODEL_CANDIDATES,
    _determinism_metadata,
    _initialize_run_root,
    _ordered_models,
)


def test_deterministic_cohort_is_the_frozen_pinned_metadata_set():
    assert MODEL_CANDIDATES == (
        "dlrm", "fastNLP_Bert", "maml", "pytorch_CycleGAN_and_pix2pix",
        "pytorch_unet", "Super_SloMo",
    )
    assert BATCH_SIZES == (1,)
    assert ISOLATED_MODEL_DEPENDENCIES == {
        "onnx": "1.17.0",
        "protobuf": "5.29.5",
        "dominate": "2.9.1",
    }


def test_candidate_metadata_must_declare_deterministic_eval(tmp_path):
    for model in MODEL_CANDIDATES:
        folder = tmp_path / "torchbenchmark" / "models" / model
        folder.mkdir(parents=True)
        (folder / "metadata.yaml").write_text("eval_deterministic: true\n")

    assert _determinism_metadata(tmp_path) == {model: "true" for model in MODEL_CANDIDATES}


def test_candidate_metadata_rejects_nondeterministic_eval(tmp_path):
    for model in MODEL_CANDIDATES:
        folder = tmp_path / "torchbenchmark" / "models" / model
        folder.mkdir(parents=True)
        (folder / "metadata.yaml").write_text(
            "eval_deterministic: false\n" if model == MODEL_CANDIDATES[0]
            else "eval_deterministic: true\n"
        )

    with pytest.raises(ValueError, match="eval_deterministic=true"):
        _determinism_metadata(tmp_path)


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
