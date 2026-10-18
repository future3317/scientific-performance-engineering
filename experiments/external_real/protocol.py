from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping


FORBIDDEN_POLICY_FIELDS = frozenset(
    {
        "ground_truth",
        "baseline_score",
        "skill_score",
        "treatment_effect",
        "oracle_support",
        "judge_result",
        "post_execution_trace",
    }
)
_PHASES = ("construction", "verification", "synthesis", "calibration")


@dataclass(frozen=True)
class ProtocolManifest:
    benchmark: str
    benchmark_revision: str
    protocol_revision: str
    split_seed: str
    model_id: str
    environment: dict[str, object]
    feature_fields: tuple[str, ...]
    metadata: dict[str, object]

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProtocolManifest":
        required = ("benchmark", "benchmark_revision", "protocol_revision", "split_seed", "model_id", "environment", "feature_fields")
        missing = [field for field in required if field not in payload]
        if missing:
            raise ValueError(f"missing manifest fields: {', '.join(missing)}")
        if not isinstance(payload["environment"], Mapping):
            raise ValueError("environment must be an object")
        fields = payload["feature_fields"]
        if not isinstance(fields, (list, tuple)) or not all(isinstance(field, str) for field in fields):
            raise ValueError("feature_fields must be a sequence of strings")
        return cls(
            benchmark=str(payload["benchmark"]),
            benchmark_revision=str(payload["benchmark_revision"]),
            protocol_revision=str(payload["protocol_revision"]),
            split_seed=str(payload["split_seed"]),
            model_id=str(payload["model_id"]),
            environment=dict(payload["environment"]),
            feature_fields=tuple(fields),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark": self.benchmark,
            "benchmark_revision": self.benchmark_revision,
            "protocol_revision": self.protocol_revision,
            "split_seed": self.split_seed,
            "model_id": self.model_id,
            "environment": self.environment,
            "feature_fields": list(self.feature_fields),
            "metadata": self.metadata,
        }


def assign_split(task_id: str, seed: str, phase_sizes: tuple[int, int, int, int]) -> str:
    if len(phase_sizes) != len(_PHASES) or any(size <= 0 for size in phase_sizes):
        raise ValueError("phase_sizes must contain four positive values")
    digest = hashlib.sha256(f"{seed}|{task_id}".encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") % sum(phase_sizes)
    for phase, size in zip(_PHASES, phase_sizes):
        if position < size:
            return phase
        position -= size
    raise AssertionError("split allocation did not return a phase")


def _keys(value: Mapping[str, object]) -> set[str]:
    keys = set(value)
    for nested in value.values():
        if isinstance(nested, Mapping):
            keys.update(_keys(nested))
    return keys


def assert_information_flow(policy_input: Mapping[str, object]) -> None:
    leaked = sorted(_keys(policy_input) & FORBIDDEN_POLICY_FIELDS)
    if leaked:
        raise ValueError(f"forbidden post-execution policy fields: {', '.join(leaked)}")


def validate_acre_r_request(
    *,
    verified_skill: bool,
    support_evidence_count: int,
    strict_coverage: float,
    alpha: float,
) -> None:
    if not verified_skill:
        raise ValueError("ACRE-R requires a verified skill")
    if support_evidence_count <= 0:
        raise ValueError("ACRE-R requires adequate support evidence")
    if not 0.0 <= strict_coverage < 0.10:
        raise ValueError("ACRE-R requires strict ACRE coverage below 0.10")
    if alpha not in {0.05, 0.10, 0.20}:
        raise ValueError("ACRE-R alpha must be one of 0.05, 0.10, 0.20")
