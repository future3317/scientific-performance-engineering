"""Protocol manifest construction for the public ACRE evidence revision."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .contracts import validate_task_split


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


class PublicTreatmentManifest:
    @staticmethod
    def create(
        *,
        revision: str,
        implementation_commit: str,
        benchmark_revisions: Mapping[str, str],
        task_split: Mapping[str, list[str] | tuple[str, ...]],
        model_config: Mapping[str, Any],
        budgets: Mapping[str, Any],
        seeds: list[int] | tuple[int, ...],
        policies: list[str] | tuple[str, ...],
        status: str = "protocol_frozen",
    ) -> dict[str, Any]:
        validate_task_split(task_split)
        body: dict[str, Any] = {
            "schema_version": 1,
            "revision": str(revision),
            "implementation_commit": str(implementation_commit),
            "benchmark_revisions": dict(sorted((str(k), str(v)) for k, v in benchmark_revisions.items())),
            "task_split": {str(k): [str(item) for item in value] for k, value in sorted(task_split.items())},
            "model_config": dict(model_config),
            "budgets": dict(budgets),
            "seeds": [int(seed) for seed in seeds],
            "policies": [str(policy) for policy in policies],
            "retry_rule": "one_same_seed_infrastructure_invalid_only",
            "results_claimed": False,
            "hosted_efficacy_status": "UNSUPPORTED",
            "status": str(status),
        }
        body["manifest_digest"] = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
        return body

