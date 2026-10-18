from experiments.external_real.upstream_provenance import audit_provenance


def test_missing_released_derivatives_authorizes_resampling(tmp_path):
    report = audit_provenance(tmp_path)

    assert report["frozen_train_derived_artifact_usable"] is False
    assert report["prospective_audit_authorized"] is False
    assert report["decision"] == "enter_u8_resampling"


def test_present_derivative_requires_provenance_review(tmp_path):
    path = tmp_path / "logs" / "sampling"
    path.mkdir(parents=True)
    (path / "train.jsonl").write_text("{}\n", encoding="utf-8")

    report = audit_provenance(tmp_path)

    assert report["artifact_inventory"]["sampled_trajectories"] == ["logs/sampling/train.jsonl"]
    assert report["frozen_train_derived_artifact_usable"] is False
    assert report["prospective_audit_authorized"] is False
    assert report["outcome_label_leakage_check"] == "requires_manual_artifact_review"
