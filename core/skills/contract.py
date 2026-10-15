"""Immutable, observable skill contract represented as an ordinary RuleSpec."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any

from core.models import RevisionRef, RuleSpec, validate_identifier


@dataclass(frozen=True)
class SkillContract:
    skill_id: str
    version: int
    task_family: str
    subtask: str
    intent: str
    applicability: tuple[str, ...]
    procedure: tuple[str, ...]
    procedure_step_ids: tuple[str, ...]
    required_procedure_step_ids: tuple[str, ...]
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    protected_semantics: tuple[str, ...]
    source_trial_ids: tuple[str, ...]
    verification_trial_ids: tuple[str, ...] = ()
    source_split: str = "train"
    parent_digest: str | None = None
    parent_version: int | None = None
    status: str = "candidate"

    def __post_init__(self) -> None:
        validate_identifier(self.skill_id, "skill_id")
        if self.version < 1 or self.source_split != "train" or self.status not in {"candidate", "rejected"}:
            raise ValueError("skill candidates require a positive version, train provenance, and candidate/rejected status")
        if self.parent_digest is not None and (not self.parent_version or self.parent_version >= self.version):
            raise ValueError("child skill revisions require an earlier parent version")
        for name in ("task_family", "subtask", "intent"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if not self.procedure:
            raise ValueError("skill procedure must not be empty")
        if len(self.procedure_step_ids) != len(self.procedure) or len(set(self.procedure_step_ids)) != len(self.procedure_step_ids):
            raise ValueError("procedure step identifiers must be unique and aligned with procedure steps")
        if not set(self.required_procedure_step_ids).issubset(self.procedure_step_ids):
            raise ValueError("required procedure steps must exist in the procedure")
        for trial_id in self.source_trial_ids + self.verification_trial_ids:
            validate_identifier(trial_id, "trial_id")

    @property
    def body(self) -> str:
        return "\n".join((
            f"When to use: {', '.join(self.applicability)}",
            f"Preconditions: {', '.join(self.preconditions)}",
            "Procedure:",
            *(f"{i + 1}. {step}" for i, step in enumerate(self.procedure)),
            "Stop or replan when a precondition is unknown or a required action is unavailable.",
            f"Verify: {', '.join(self.postconditions)}",
        ))

    @property
    def body_digest(self) -> str:
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()

    @property
    def digest(self) -> str:
        # Evidence attachment changes provenance, not the intervention identity.
        payload = asdict(self)
        payload.pop("verification_trial_ids")
        payload.pop("status")
        payload = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SkillContract":
        return cls(**{key: tuple(item) if key in {
            "applicability", "procedure", "preconditions", "postconditions",
            "procedure_step_ids", "required_procedure_step_ids", "protected_semantics",
            "source_trial_ids", "verification_trial_ids",
        } else item for key, item in value.items()})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_rule_spec(self) -> RuleSpec:
        parent = RevisionRef(self.skill_id, self.parent_version, self.parent_digest) if self.parent_version and self.parent_digest else None
        return RuleSpec(
            rule_id=self.skill_id,
            version=self.version,
            parent=parent,
            applicability={"all": [
                {"equals": {"domain": "alfworld"}},
                {"equals": {"workload.task_family": self.task_family}},
            ]},
            intervention={"kind": "skill", "contract": self.to_dict(), "contract_digest": self.digest, "body_digest": self.body_digest},
            expected_mechanism=self.intent,
            evidence_requirements=["independent_train_paired_verification"],
            scientific_invariants=list(self.protected_semantics),
            abstain_conditions={},
            relations={},
            runtime_cost={},
            provenance_policy={
                "source_split": self.source_split,
                "source_trial_ids": list(self.source_trial_ids),
                "verification_trial_ids": list(self.verification_trial_ids),
            },
            severity="P2",
            domain="alfworld",
            text=self.body,
        )


_ENTITY_NUMBER = re.compile(r"\b(?:cabinet|drawer|countertop|table|sinkbasin|stoveburner|fridge|microwave|bed|sofa)\s+\d+\b", re.I)


def validate_skill_text(contract: SkillContract, *, max_characters: int) -> list[str]:
    """Check size and concrete unsafe/source-leak patterns in the rendered skill."""
    body = contract.body + "\n" + json.dumps(contract.to_dict(), ensure_ascii=False, sort_keys=True)
    errors: list[str] = []
    if len(body) > max_characters:
        errors.append("skill_too_long")
    if _ENTITY_NUMBER.search(body):
        errors.append("instance_specific_entity")
    if re.search(r"(?i)(?:api[_ -]?key|password|secret|private key|\.env|gamefile|traj_data|test answer)", body):
        errors.append("secret_or_oracle_reference")
    if re.search(r"(?i)\b(?:ignore safety|ignore previous instructions|exfiltrate|curl\s|wget\s|rm\s+-rf|os\.system|subprocess)\b", body):
        errors.append("unsafe_instruction")
    return errors
