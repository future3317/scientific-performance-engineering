from __future__ import annotations

from scripts.audit_lifecycle_controls import run_lifecycle_controls


def test_lifecycle_controls_emit_four_policies_and_action_ledgers() -> None:
    report = run_lifecycle_controls(replicates=2)
    assert set(report["policies"]) == {"raw_reuse", "no_reuse", "frozen_certificate", "full_acre"}
    assert len(report["ledger"]) == 2 * 3 * 4 * 6
    required = {"action", "blocking_clause", "revision_id", "utility", "oracle_utility", "evidence_cost_units"}
    assert required <= set(report["ledger"][0])
    assert "support_decision_regret" not in report["summary"]["full_acre"]


def test_full_acre_revalidates_after_regime_change_without_permanent_abstention() -> None:
    report = run_lifecycle_controls(replicates=3)
    rows = [row for row in report["ledger"] if row["policy"] == "full_acre" and row["template"] == "compiler_drift"]
    assert any(row["action"] == "revalidate" and row["blocking_clause"] == "L" for row in rows)
    assert any(row["action"] == "reuse" and row["phase"] == 4 for row in rows)
