"""Fixed public regression-suite admission policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .policy_eval import Decision


@dataclass(frozen=True)
class StaticTestGate:
    """A context-independent gate built from a fixed public test suite."""

    suite_pass: bool
    test_ids: tuple[str, ...]

    @classmethod
    def from_calibration_suite(cls, suite_records: Sequence[Mapping[str, Any]]) -> "StaticTestGate":
        if not suite_records:
            raise ValueError("static calibration suite must not be empty")
        allowed = {"test_id", "passed", "category", "details"}
        normalized: list[tuple[str, bool]] = []
        for record in suite_records:
            if set(record) - allowed:
                raise ValueError("static suite records must contain public test results only")
            if "test_id" not in record or "passed" not in record:
                raise ValueError("static suite record requires test_id and passed")
            normalized.append((str(record["test_id"]), bool(record["passed"])))
        return cls(
            suite_pass=all(passed for _, passed in normalized),
            test_ids=tuple(test_id for test_id, _ in normalized),
        )

    def decide(self, context: Mapping[str, Any]) -> Decision:
        del context
        return Decision(
            "reuse" if self.suite_pass else "abstain",
            1.0 if self.suite_pass else 0.0,
            0.0,
            "static_suite_pass" if self.suite_pass else "static_suite_fail",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "suite_pass": self.suite_pass,
            "test_ids": list(self.test_ids),
            "uncertainty": 0.0,
        }


__all__ = ["StaticTestGate"]
