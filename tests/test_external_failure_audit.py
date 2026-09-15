from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_external_failure import audit_hosted_campaign, audit_skillgen_suite


def test_skillgen_audit_reports_required_benchmark_counts(tmp_path: Path) -> None:
    source = tmp_path / "suite"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps({"benchmarks": ["scienceworld", "mind2web", "toolbench", "mcp_bench"]}),
        encoding="utf-8",
    )
    for benchmark in ("scienceworld", "mind2web", "toolbench", "mcp_bench"):
        (source / benchmark).mkdir()
        (source / benchmark / "report.json").write_text(
            json.dumps(
                {
                    "train_evidence": [{"context": {"family": "a"}, "label": "positive"}],
                    "positive_anchors": 1,
                    "counterexamples": 2,
                    "unresolved": 3,
                    "repairs": 4,
                    "regressions": 5,
                    "both_pass": 6,
                    "both_fail": 7,
                    "candidate_predicates": [
                        {"status": "accepted"},
                        {"status": "no_positive_support"},
                    ],
                    "contexts": [{"family": "a"}, {"family": "b"}],
                    "full_acre_coverage": 0.0,
                    "full_acre_reason": "no_positive_and_no_harm_support_group",
                }
            ),
            encoding="utf-8",
        )

    report = audit_skillgen_suite(source)

    assert set(report["benchmarks"]) == {"scienceworld", "mind2web", "toolbench", "mcp_bench"}
    assert report["benchmarks"]["scienceworld"]["train_evidence"] == 1
    assert report["benchmarks"]["scienceworld"]["candidate_predicate_rejections"]["no_positive_support"] == 1
    assert report["benchmarks"]["scienceworld"]["full_acre_zero_reason"] == "no_positive_and_no_harm_support_group"


def test_hosted_audit_conserves_cells_and_preserves_unknowns(tmp_path: Path) -> None:
    source = tmp_path / "hosted.json"
    source.write_text(
        json.dumps(
            [
                {"condition": "A", "status": "measurement_valid", "input_tokens": 10, "turns": 2},
                {"condition": "B", "status": "turn_limit", "input_tokens": 20, "turns": 4},
                {"condition": "C", "status": "contract_fail", "input_tokens": 30, "turns": 3},
                {"condition": "D", "status": "positive_effect", "input_tokens": 40, "turns": 5},
                {"condition": "D", "status": "unseen_status", "input_tokens": 50, "turns": 6},
            ]
        ),
        encoding="utf-8",
    )

    report = audit_hosted_campaign(source)

    assert report["scheduled"] == 5
    assert report["by_condition"]["D"]["scheduled"] == 2
    assert report["by_condition"]["D"]["unknown_status"] == 1
    assert report["by_condition"]["B"]["turn_limit"] == 1
    assert report["input_tokens"]["median"] == 30
    assert report["input_tokens"]["p90"] == 50
    assert report["invariants"]["status_count_equals_scheduled"] is True


def test_hosted_audit_counts_raw_cell_records_without_derived_status(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "cell.json").write_text(
        json.dumps({"task_id": "task-1", "outer_trial_id": "outer-000", "control_implementation": "baseline"}),
        encoding="utf-8",
    )

    report = audit_hosted_campaign(campaign)

    assert report["scheduled"] == 1
    assert report["by_condition"]["unknown"]["unknown_status"] == 1
