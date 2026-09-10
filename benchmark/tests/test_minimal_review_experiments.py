from __future__ import annotations

from scripts.run_minimal_review_experiments import aggregate_relation_budget_curve, build_risk_coverage_summary


def test_risk_coverage_reaggregation_uses_confusion_counts_without_thresholds():
    raw = {
        "policies": ["acre_full"],
        "evidence_prefixes": ["all"],
        "family_results": [{"prefixes": {"all": {"policies": {"acre_full": {
            "tp": 2, "fp": 1, "fn": 1, "tn": 2,
            "unsafe_generalization_rate": 1 / 6,
            "abstention_rate": 3 / 6,
            "recall": 2 / 3,
            "f1": 2 / 3,
        }}}}}],
    }
    result = build_risk_coverage_summary(raw)
    row = result["summary"]["acre_full"]["all"]["family_cluster_mean"]
    assert row["coverage"] == 0.5
    assert row["unsafe_risk"] == 1 / 3
    assert result["summary"]["acre_full"]["all"]["new_threshold"] is False


def test_relation_budget_curve_reuses_each_persisted_block():
    raw = {
        "block_schedule": [8, 16],
        "surface_results": [
            {"hidden_relation": "synergy", "estimates": [{"blocks": 8, "decision": "unresolved"}, {"blocks": 16, "decision": "synergy"}]},
            {"hidden_relation": "antagonism", "estimates": [{"blocks": 8, "decision": "unresolved"}, {"blocks": 16, "decision": "antagonism"}]},
        ],
    }
    result = aggregate_relation_budget_curve(raw)
    assert result["factorial_rerun"] is False
    assert result["curves"][0]["coverage"] == 0.0
    assert result["curves"][1]["coverage"] == 1.0
    assert result["curves"][1]["resolved_accuracy"] == 1.0
