from __future__ import annotations

import pytest

from experiments.external_real.run_scienceworld import build_smoke_manifest, evaluate_smoke_gate, parse_action_plan


def test_smoke_manifest_freezes_provider_and_smoke_sizes() -> None:
    manifest = build_smoke_manifest("scienceworld-1.2.3", "deepseek-v4-flash", (0, 1, 2))

    assert manifest.benchmark == "scienceworld"
    assert manifest.metadata["construction_n"] == 20
    assert manifest.metadata["verification_n"] == 10
    assert manifest.metadata["support_n"] == 20
    assert manifest.metadata["test_n"] == 20
    assert manifest.metadata["provider_calls_authorized"] is True


def test_action_plan_parser_accepts_json_and_numbered_public_actions() -> None:
    assert parse_action_plan('{"actions": ["look around", "wait1"]}') == ["look around", "wait1"]
    assert parse_action_plan("1. look around\n2. wait1") == ["look around", "wait1"]


@pytest.mark.parametrize(
    ("report", "reason"),
    [
        ({"environment_validity": 1.0, "decision_digest": None}, "decision digest"),
        ({"environment_validity": 1.0, "decision_digest": "abc", "outcome_before_decision": True}, "before decision"),
    ],
)
def test_smoke_gate_rejects_invalid_reports(report: dict[str, object], reason: str) -> None:
    passed, reasons = evaluate_smoke_gate(report)

    assert passed is False
    assert any(reason in item.lower() for item in reasons)


def test_smoke_gate_accepts_valid_report() -> None:
    passed, reasons = evaluate_smoke_gate({
        "environment_validity": 1.0,
        "decision_digest": "abc",
        "outcome_before_decision": False,
        "information_flow_valid": True,
        "evaluator_valid": True,
        "acre_coverage": 0.2,
    })

    assert passed is True
    assert reasons == []


def test_zero_acre_coverage_is_a_diagnostic_not_an_engineering_failure() -> None:
    passed, reasons = evaluate_smoke_gate({
        "environment_validity": 1.0,
        "decision_digest": "abc",
        "outcome_before_decision": False,
        "information_flow_valid": True,
        "evaluator_valid": True,
        "acre_coverage": 0.0,
    })

    assert passed is True
    assert reasons == []
