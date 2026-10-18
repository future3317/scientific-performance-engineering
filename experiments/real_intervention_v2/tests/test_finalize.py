import json

from experiments.real_intervention_v2.finalize import _aa_diagnostics


def test_aa_diagnostics_reports_correctness_stop_by_model(tmp_path):
    pair_dir = tmp_path / "aa" / "pairs"
    pair_dir.mkdir(parents=True)
    manifest = {
        "aa_control_context_ids": ["ctx-a", "ctx-b"],
        "contexts": [
            {"context_id": "ctx-a", "model": "model-a", "lineage_id": "lineage-a"},
            {"context_id": "ctx-b", "model": "model-b", "lineage_id": "lineage-b"},
        ],
    }
    rows = [
        {"context_id": "ctx-a", "correctness": {"passed": True}, "arms": {"eager": {"status": "valid"}}},
        {"context_id": "ctx-b", "correctness": {"passed": False, "reason": "numerical_mismatch:output"},
         "arms": {"eager": {"status": "valid"}}},
    ]
    for index, row in enumerate(rows):
        (pair_dir / f"{index}.json").write_text(json.dumps(row))

    summary = _aa_diagnostics(tmp_path, manifest)

    assert summary["scheduled_contexts"] == 2
    assert summary["completed_contexts"] == 2
    assert summary["correctness_passed"] == 1
    assert summary["failure_reasons"] == {"passed": 1, "numerical_mismatch:output": 1}
    assert [row["model"] for row in summary["per_context"]] == ["model-a", "model-b"]
