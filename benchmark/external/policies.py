"""Policy adapter contracts; benchmark-specific policy logic stays outside scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol


@dataclass(frozen=True)
class PolicyDecision:
    reuse: bool
    score: float | None
    uncertainty: float | None
    reason: str
    certificate_id: str | None = None
    skill_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reuse": self.reuse,
            "score": self.score,
            "uncertainty": self.uncertainty,
            "reason": self.reason,
            "certificate_id": self.certificate_id,
            "skill_version": self.skill_version,
        }


class Policy(Protocol):
    name: str

    def decide(self, context: Mapping[str, Any], evidence: Mapping[str, Any], skill_snapshot: str | None) -> PolicyDecision:
        ...


class CallablePolicy:
    def __init__(self, name: str, function: Callable[[Mapping[str, Any], Mapping[str, Any], str | None], PolicyDecision]) -> None:
        self.name = str(name)
        self._function = function

    def decide(self, context: Mapping[str, Any], evidence: Mapping[str, Any], skill_snapshot: str | None) -> PolicyDecision:
        return self._function(context, evidence, skill_snapshot)


def global_reuse_policy() -> CallablePolicy:
    return CallablePolicy(
        "global_skill",
        lambda _context, _evidence, skill: PolicyDecision(True, 1.0, 0.0, "global_reuse", skill_version=skill),
    )


def abstaining_policy(name: str = "abstain") -> CallablePolicy:
    return CallablePolicy(name, lambda _context, _evidence, _skill: PolicyDecision(False, 0.0, 1.0, "abstain"))

