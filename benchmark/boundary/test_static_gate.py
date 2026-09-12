from __future__ import annotations

from benchmark.boundary.policy_eval import Decision
from benchmark.boundary.static_gate import StaticTestGate


def test_static_suite_pass_admits_every_context_with_zero_uncertainty() -> None:
    gate = StaticTestGate.from_calibration_suite([
        {"test_id": "import", "passed": True, "category": "contract"},
        {"test_id": "regression", "passed": True, "category": "correctness"},
    ])
    decision = gate.decide({"workload": {"x": 1}})
    assert decision == Decision("reuse", 1.0, 0.0, "static_suite_pass")
    assert gate.to_dict()["suite_pass"] is True


def test_static_suite_failure_abstains_globally() -> None:
    gate = StaticTestGate.from_calibration_suite([
        {"test_id": "import", "passed": True},
        {"test_id": "regression", "passed": False},
    ])
    assert gate.decide({"workload": {"x": 1}}) == Decision("abstain", 0.0, 0.0, "static_suite_fail")


def test_static_gate_never_reads_hidden_context_or_effect_fields() -> None:
    gate = StaticTestGate.from_calibration_suite([{"test_id": "public", "passed": True}])
    clean = gate.decide({"workload": {"x": 1}})
    contaminated = gate.decide({
        "workload": {"x": 1},
        "tau": 100.0,
        "scientific_gate": False,
        "oracle_label": True,
        "expected_applicable": False,
    })
    assert contaminated == clean

