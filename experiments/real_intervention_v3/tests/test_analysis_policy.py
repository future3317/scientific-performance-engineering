from __future__ import annotations

import hashlib
import json
import tarfile

from experiments.real_intervention_v3.analyze_real import analyze
from experiments.real_intervention_v3.policy_bench import evaluate_policies, freeze_policies
from experiments.real_intervention_v3.report import write_report


def pair(pair_id: str, family: str, *, split: str = "development", horizon: int = 1,
         status: str = "VALID_PAIR", s0: str = "valid", s1: str = "valid",
         baseline: float | None = 10.0, candidate: float | None = 8.0) -> dict:
    receipt0 = {"status": s0, "total_cold_inclusive_s": baseline, "steady_state_per_step_s": 0.5,
                "child_cpu_s": 2.0, "parent_wall_s": 3.0}
    receipt1 = {"status": s1, "total_cold_inclusive_s": candidate, "steady_state_per_step_s": 0.4,
                "child_cpu_s": 2.5, "parent_wall_s": 4.0}
    return {"pair_id": pair_id, "family": family, "model": family + "_model", "source": "torchvision",
            "split": split, "horizon": horizon, "session": 0, "order": ["S0", "S1"],
            "status": status, "S0": receipt0, "S1": receipt1,
            "features": {"horizon": horizon, "batch": 1, "parameter_count": 1000}}


def test_analyze_counts_platform_failure_separately_and_marks_single_family_ci_noninferential():
    rows = [pair("p1", "resnet", baseline=10, candidate=5),
            pair("p2", "resnet", status="PLATFORM_LINK_OR_INSTALL", s1="error", candidate=None)]
    result = analyze(rows)
    assert result["failure_denominator"]["platform_invalid_pairs"] == 1
    assert result["failure_denominator"]["candidate_failure_pairs"] == 0
    assert result["source_strata"]["torchvision"]["input_kind"] == "REAL_MODEL_MATERIALIZED_INPUT"
    interval = result["overall_effects"]["family_bootstrap_95_ci"]
    assert interval["status"] == "INSUFFICIENT_INDEPENDENT_FAMILIES"
    assert interval["inferential"] is False
    assert interval["lower"] is interval["upper"] is None


def test_policy_counts_candidate_failure_and_preserves_missed_benefit_and_undefined_zero_deploy_harm():
    gain = pair("gain", "family-a", baseline=12, candidate=8)
    failed = pair("failed", "family-b", status="INTERVENTION_MODEL_UNSUPPORTED", s1="error", candidate=None)
    frozen = {"development": {"horizon_threshold": 1000,
                               "simple_supervised": {"feature": None, "threshold": None, "positive_above": True}},
              "acre": {"configured": False, "status": "ACRE_NOT_FITTED"}}
    results = evaluate_policies([gain, failed], frozen)
    never = next(row for row in results if row["policy"] == "NEVER")
    always = next(row for row in results if row["policy"] == "ALWAYS")
    assert never["candidate_failure_count"] == 1
    assert never["missed_benefit_seconds"] == 4
    assert never["conditional_harm_rate"] is None
    assert never["total_net_seconds_vs_never"] == 0
    assert always["candidate_failure_count"] == 1
    assert always["unsafe_deployment_count"] == 1
    assert always["total_net_seconds_vs_never"] is None
    threshold = next(row for row in results if row["policy"] == "HORIZON_ONLY")
    assert threshold["family_bootstrap_95_ci_seconds"]["families"] == 2
    assert threshold["family_bootstrap_95_ci_seconds"]["lower"] == 0
    assert threshold["family_bootstrap_95_ci_seconds"]["upper"] == 0
    assert threshold["family_bootstrap_95_ci_seconds"]["estimand"] == "family-equal mean net seconds per pair"


def test_missing_deployed_outcome_makes_harm_rates_unknown_with_observed_bounds():
    observed_harm = pair("harm", "family-a", baseline=8, candidate=10)
    unknown = pair("unknown", "family-b", status="PLATFORM_LINK_OR_INSTALL", s1="error", candidate=None)
    frozen = {"development": {"horizon_threshold": 1000,
                               "simple_supervised": {"feature": None, "threshold": None, "positive_above": True}},
              "acre": {"configured": False, "status": "ACRE_NOT_FITTED"}}
    always = next(row for row in evaluate_policies([observed_harm, unknown], frozen) if row["policy"] == "ALWAYS")
    assert always["harm_deployed_count"] == 1
    assert always["unknown_deployed_count"] == 1
    assert always["conditional_harm_rate"] is None
    assert always["conditional_harm_rate_lower"] == 0.5
    assert always["conditional_harm_rate_upper"] == 1
    assert always["harm_unconditional_rate"] is None
    assert always["harm_unconditional_rate_lower"] == 0.5
    assert always["harm_unconditional_rate_upper"] == 1
    assert always["total_net_seconds_vs_never"] is None
    assert always["family_bootstrap_95_ci_seconds"] is None


def test_pair_status_alone_classifies_candidate_harm():
    from experiments.real_intervention_v3.policy_bench import _candidate_failure

    model_failure = pair("unsupported", "family-u", status="INTERVENTION_MODEL_UNSUPPORTED", s1="error", candidate=None)
    mismatch = pair("mismatch", "family-m", status="INTERVENTION_NUMERIC_MISMATCH", s1="numeric_mismatch", candidate=None)
    platform = pair("platform", "family-p", status="PLATFORM_LINK_OR_INSTALL", s1="error", candidate=None)
    baseline = pair("baseline", "family-b", status="BASELINE_INVALID", s0="error", s1="error", baseline=None, candidate=None)
    unresolved = pair("unknown", "family-n", status="SESSION_NOISE_UNRESOLVED", s0="error", s1="error", baseline=None, candidate=None)
    assert [_candidate_failure(row) for row in (model_failure, mismatch, platform, baseline, unresolved)] == [True, True, False, False, False]
    frozen = {"development": {"horizon_threshold": 1000,
                               "simple_supervised": {"feature": None, "threshold": None, "positive_above": True}},
              "acre": {"configured": False, "status": "ACRE_NOT_FITTED"}}
    always = next(row for row in evaluate_policies([model_failure, mismatch, platform, baseline, unresolved], frozen)
                  if row["policy"] == "ALWAYS")
    assert always["candidate_failure_count"] == 2
    assert always["unsafe_deployment_count"] == 2


def test_freeze_uses_only_development_and_records_release_and_acre_not_fitted(tmp_path):
    dev = [pair("d1", "dev-a", horizon=1, baseline=10, candidate=12),
           pair("d2", "dev-b", horizon=16, baseline=10, candidate=12)]
    cal = [pair(f"c{i}", f"cal-{i}", split="calibration") for i in range(5)]
    (tmp_path / "FROZEN_MANIFEST.json").write_text(json.dumps({"family_split": {f"f{i}": "development" for i in range(17)},
                                                                "manifest_digest": "manifest-id"}), encoding="utf-8")
    frozen = freeze_policies(dev, cal, tmp_path)
    assert frozen["status"] == "UNDERPOWERED_NOT_FINAL"
    assert frozen["release_qualification"]["valid_calibration_families"] == 5
    assert frozen["development"]["horizon_threshold"] > 16
    assert frozen["acre"]["status"] == "ACRE_NOT_FITTED"
    assert "acre_store" not in frozen["acre"] or frozen["acre"]["store"] is None
    assert set(frozen["preexecution_feature_schema"]) == {"horizon", "batch", "parameter_count", "input_height", "input_width",
        "input_channels", "input_numel", "sequence_length", "hidden_size", "num_heads", "depth", "width", "num_parameters", "num_nodes",
        "num_edges", "num_features", "num_classes"}


def test_freeze_rejects_final_rows_in_fit_inputs(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="held-out"):
        freeze_policies([pair("f", "family", split="final")], [], tmp_path)


def test_report_bundle_repeat_has_unique_members_hash_closure_and_claim_limits(tmp_path):
    (tmp_path / "raw" / "logs").mkdir(parents=True)
    (tmp_path / "raw" / "logs" / "LOADER_TRACEBACKS.txt").write_text("full traceback\n", encoding="utf-8")
    (tmp_path / "raw" / "canary.json").write_text('{"status":"valid"}\n', encoding="utf-8")
    (tmp_path / "SOURCE_PATCH.diff").write_text("diff --git a/source b/source\n", encoding="utf-8")
    (tmp_path / "caches").mkdir()
    (tmp_path / "caches" / "not_evidence.json").write_text("{}", encoding="utf-8")
    metrics = analyze([pair("p", "family")])
    extra = {"status": "UNDERPOWERED_NOT_FINAL", "selected_intervention": "compile", "canary_passed": 3,
             "identity": {"source_commit": "abc", "config_digest": "cfg", "python": "/python", "gpu_uuid": "GPU-test"},
             "release": {"qualified": False, "status": "UNDERPOWERED_NOT_FINAL"}, "final_opened": False,
             "resources": {"child_cpu_seconds": 4.5}, "claim_scope": "system performance only"}
    write_report(tmp_path, metrics, [pair("p", "family")], [], extra)
    write_report(tmp_path, metrics, [pair("p", "family")], [], extra)
    with tarfile.open(tmp_path / "EVIDENCE_BUNDLE.tar.gz", "r:gz") as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
        assert len(names) == len(set(names))
        index = json.load(archive.extractfile("EVIDENCE_BUNDLE_INDEX.json"))
        members = index["members"]
        assert "raw/logs/LOADER_TRACEBACKS.txt" in members
        assert "raw/canary.json" in members
        assert "SOURCE_PATCH.diff" in members
        assert not any("caches/" in name for name in members)
        for name, expected in members.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected
    matrix = (tmp_path / "PAPER_CLAIM_MATRIX.md").read_text(encoding="utf-8")
    assert "No dataset accuracy" in matrix
    assert "Exploratory only" in matrix
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "REAL_MODEL_MATERIALIZED_INPUT" in report
    assert "TORCHBENCH_MINIATURE_INPUT" in report
