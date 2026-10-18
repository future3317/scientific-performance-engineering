from experiments.real_intervention_v2.prepare_study import _ordered_models


def test_ordered_models_keep_seeded_lineage_split_order_and_group_variants():
    eligible = ["hf_Bert", "resnet18", "BERT_pytorch", "alexnet"]
    lineages = ["resnet18", "bert", "alexnet"]

    assert _ordered_models(eligible, lineages) == [
        "resnet18", "BERT_pytorch", "hf_Bert", "alexnet"
    ]
