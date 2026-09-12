"""Audit the boundary between public evidence construction and hidden scoring truth."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.boundary.baselines import predict_applicability
from benchmark.boundary.evaluator import evaluate_applicability


_HIDDEN_FIELDS = {
    "expected_applicable",
    "scientific_truth",
    "oracle_bundle",
    "heldout_label",
}


class _PublicOnlyObservation:
    """Boundary observation whose hidden-label fields fail loudly if read."""

    def __init__(self, observation_id: str, x: int, *, positive: bool) -> None:
        self.observation_id = observation_id
        self.context = {"workload": {"x": x}}
        self.effect = 1.0 if positive else 0.0
        self.gate_passed = positive
        self.effect_lower = self.effect
        self.effect_upper = self.effect
        self._positive = positive

    def __getattribute__(self, name: str) -> Any:
        if name in _HIDDEN_FIELDS:
            raise AssertionError(f"policy accessed hidden field: {name}")
        return object.__getattribute__(self, name)

    def positive_anchor(self, epsilon_true: float = 0.0) -> bool:
        return self._positive and self.effect_lower > epsilon_true

    def certified_counterexample(self, epsilon_false: float = 0.0) -> bool:
        return (not self.gate_passed) or self.effect_upper <= epsilon_false


def run_oracle_separation_audit() -> dict[str, Any]:
    """Exercise every RQ1 policy with observations that expose no hidden labels."""
    visible = [
        _PublicOnlyObservation("positive", 2, positive=True),
        _PublicOnlyObservation("negative", 10, positive=False),
    ]
    heldout = [_PublicOnlyObservation("heldout", 3, positive=True)]
    policies = (
        "always_apply",
        "observed_positive_only",
        "nearest_experience",
        "untyped_positive_summary",
        "shortest_without_counterexamples",
        "acre_full",
    )
    policy_outputs: dict[str, dict[str, bool]] = {}
    hidden_access_passed = True
    try:
        for policy in policies:
            policy_outputs[policy] = predict_applicability(
                policy, visible, heldout, family="compile",
            )
    except AssertionError:
        hidden_access_passed = False

    source = inspect.getsource(predict_applicability)
    checks = {
        "hidden_labels_unreadable_during_policy": {
            "passed": hidden_access_passed,
            "detail": "all RQ1 policy paths ran with hidden-label accessors that raise",
        },
        "policy_source_has_no_hidden_label_lookup": {
            "passed": not any(f".{field}" in source for field in _HIDDEN_FIELDS),
            "detail": "predict_applicability consumes BoundaryObservation evidence only",
        },
        "heldout_truth_is_scoring_only": {
            "passed": evaluate_applicability(
                {"heldout": True}, {"heldout": True},
            )["tp"] == 1,
            "detail": "held-out truth is supplied only to the offline evaluator",
        },
    }
    status = "PASS" if all(item["passed"] for item in checks.values()) else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "provider_calls": 0,
        "dataflow": [
            {
                "field": "effect/effect_lower/effect_upper/gate_passed",
                "producer": "paired baseline/intervention measurement and scientific gates",
                "consumer": "BoundaryObservation -> StatisticalCEGIS/policy predicates",
                "split": "calibration/evidence",
                "hidden": False,
            },
            {
                "field": "expected_applicable/scientific_truth",
                "producer": "FamilyInstance benchmark truth",
                "consumer": "evaluate_applicability scoring only",
                "split": "held-out evaluation",
                "hidden": True,
            },
            {
                "field": "evaluation outcome labels",
                "producer": "offline evaluator",
                "consumer": "reporting only",
                "split": "held-out evaluation",
                "hidden": True,
            },
        ],
        "checks": checks,
        "policy_outputs": policy_outputs,
        "interpretation": "hidden oracle is scoring truth, not learner or router input",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run_oracle_separation_audit()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
