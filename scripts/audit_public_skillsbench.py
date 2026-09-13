#!/usr/bin/env python3
"""Run the frozen, provider-free SkillsBench external deployment audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.boundary.ltt import LTTConfig, LTTLogisticPolicy
from benchmark.boundary.policy_eval import Decision, evaluate_policy
from benchmark.boundary.rscb import RSCBConfig, RSCBModel


REQUIRED_STRATIFICATION_FIELDS = ("category", "difficulty")
ACTIVE_CONDITIONS = ("No Skills", "With Skills")
DEFAULT_SPLIT_SEED = "skillsbench-v1.1-task-split"


class SkillsBenchFeatureSchema:
    """Frozen one-hot encoding for the preregistered public feature set."""

    FIELDS = ("category", "difficulty", "model", "harness")

    def __init__(self, vocabularies: Mapping[str, Sequence[str]]) -> None:
        self.vocabularies = {
            field: tuple(sorted(set(str(value) for value in values))) + ("<UNK>",)
            for field, values in vocabularies.items()
        }
        self.output_width = sum(len(values) + 1 for values in self.vocabularies.values())

    @classmethod
    def from_synthesis(cls, rows: Sequence[Mapping[str, Any]]) -> "SkillsBenchFeatureSchema":
        return cls({field: [str(row["context"][field]) for row in rows] for field in cls.FIELDS})

    def transform(self, context: Mapping[str, Any]) -> tuple[float, ...]:
        encoded: list[float] = []
        for field in self.FIELDS:
            value = context.get(field)
            category = str(value) if value is not None else "<UNK>"
            vocabulary = self.vocabularies[field]
            if category not in vocabulary:
                category = "<UNK>"
            encoded.extend(1.0 if item == category else 0.0 for item in vocabulary)
            encoded.append(0.0 if value is not None else 1.0)
        return tuple(encoded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "feature_set": list(self.FIELDS),
            "fit_source": "synthesis_tasks_only",
            "vocabularies": {field: list(values) for field, values in self.vocabularies.items()},
            "output_width": self.output_width,
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_id(metadata: Mapping[str, Any]) -> str:
    for key in ("task_id", "title", "name"):
        value = metadata.get(key)
        if value:
            return str(value)
    raise ValueError("task metadata lacks canonical task_id/title/name")


def join_registry_rows(rows: Sequence[Mapping[str, Any]], task_manifest: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Join result rows to one and only one active v1.1 task metadata row."""

    all_metadata: dict[str, Mapping[str, Any]] = {}
    for metadata in task_manifest:
        task_id = _task_id(metadata)
        if task_id in all_metadata:
            raise ValueError(f"duplicate task metadata for {task_id!r}")
        all_metadata[task_id] = metadata

    active: dict[str, Mapping[str, Any]] = {
        task_id: metadata
        for task_id, metadata in all_metadata.items()
        if not bool(metadata.get("excluded", False))
    }
    if not active:
        raise ValueError("task manifest has no active task metadata")
    for task_id, metadata in active.items():
        for field in REQUIRED_STRATIFICATION_FIELDS:
            if metadata.get(field) in (None, ""):
                raise ValueError(f"active task {task_id!r} has null {field} metadata")

    joined: list[dict[str, Any]] = []
    for row in rows:
        task_id = str(row.get("task", row.get("task_id", "")))
        metadata = active.get(task_id)
        if metadata is None:
            raise ValueError(f"missing active task metadata for result task {task_id!r}")
        joined.append({
            **dict(row),
            "task_id": task_id,
            "category": str(metadata["category"]),
            "difficulty": str(metadata["difficulty"]),
            "task_version": metadata.get("version"),
        })
    if len(joined) != len(rows):
        raise AssertionError("left join changed result row count")
    return joined


def build_task_split(rows: Sequence[Mapping[str, Any]], *, split_seed: str = DEFAULT_SPLIT_SEED) -> dict[str, list[dict[str, Any]]]:
    """Allocate whole tasks to 50/25/25 phases by stable within-stratum order."""

    task_metadata: dict[str, tuple[str, str]] = {}
    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row.get("task_id", row.get("task", "")))
        if not task_id:
            raise ValueError("result row lacks task_id")
        stratum = (str(row.get("category", "")), str(row.get("difficulty", "")))
        previous = task_metadata.get(task_id)
        if previous is not None and previous != stratum:
            raise ValueError(f"task {task_id!r} has conflicting split metadata")
        task_metadata[task_id] = stratum
        task_rows[task_id].append({**dict(row), "task_id": task_id})

    by_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)
    for task_id, stratum in task_metadata.items():
        by_stratum[stratum].append(task_id)
    task_phase: dict[str, str] = {}
    for stratum, task_ids in sorted(by_stratum.items()):
        ordered = sorted(
            task_ids,
            key=lambda task_id: hashlib.sha256(
                f"{split_seed}|{stratum[0]}|{stratum[1]}|{task_id}".encode()
            ).hexdigest(),
        )
        for index, task_id in enumerate(ordered):
            task_phase[task_id] = ("synthesis", "synthesis", "calibration", "evaluation")[index % 4]

    result = {"synthesis": [], "calibration": [], "evaluation": []}
    for task_id in sorted(task_rows):
        result[task_phase[task_id]].extend(task_rows[task_id])
    if sum(len(members) for members in result.values()) != len(rows):
        raise AssertionError("task split does not conserve rows")
    return result


def _pair_rows(joined: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in joined:
        condition = str(row.get("condition", ""))
        if condition not in ACTIVE_CONDITIONS:
            raise ValueError(f"unexpected SkillsBench condition {condition!r}")
        key = (str(row["task_id"]), str(row["model"]), str(row["harness"]))
        if condition in grouped[key]:
            raise ValueError(f"duplicate result row for {key!r}, condition {condition!r}")
        grouped[key][condition] = row

    pairs: list[dict[str, Any]] = []
    for key in sorted(grouped):
        members = grouped[key]
        if set(members) != set(ACTIVE_CONDITIONS):
            raise ValueError(f"condition pair incomplete for {key!r}")
        no_skills = members["No Skills"]
        with_skills = members["With Skills"]
        no_score = float(no_skills["score"])
        with_score = float(with_skills["score"])
        if not math.isfinite(no_score) or not math.isfinite(with_score):
            raise ValueError(f"non-finite score in {key!r}")
        pairs.append({
            "id": json.dumps(key, separators=(",", ":")),
            "task_id": key[0],
            "model": key[1],
            "harness": key[2],
            "category": str(no_skills["category"]),
            "difficulty": str(no_skills["difficulty"]),
            "context": {
                "category": str(no_skills["category"]),
                "difficulty": str(no_skills["difficulty"]),
                "model": key[1],
                "harness": key[2],
            },
            "no_skills_score": no_score,
            "with_skills_score": with_score,
            "tau": (with_score - no_score) / 100.0,
        })
    return pairs


class _AlwaysReuse:
    def decide(self, context: Mapping[str, Any]) -> Decision:
        del context
        return Decision("reuse", 1.0, 0.0, "global_reuse")


class _GroupCertificate:
    def __init__(self, rows: Sequence[Mapping[str, Any]], fields: Sequence[str], *, min_count: int, reason: str) -> None:
        self.fields = tuple(fields)
        self.min_count = int(min_count)
        self.reason = reason
        values: dict[tuple[str, ...], list[float]] = defaultdict(list)
        for row in rows:
            values[tuple(str(row["context"][field]) for field in self.fields)].append(float(row["tau"]))
        self.groups = {
            key: {"count": len(tau), "mean_tau": sum(tau) / len(tau)}
            for key, tau in values.items()
        }

    def _pass(self, context: Mapping[str, Any]) -> bool:
        group = self.groups.get(tuple(str(context.get(field)) for field in self.fields))
        return bool(group and group["count"] >= self.min_count and group["mean_tau"] > 0.0)

    def decide(self, context: Mapping[str, Any]) -> Decision:
        passed = self._pass(context)
        return Decision(
            "reuse" if passed else "abstain",
            1.0 if passed else 0.0,
            0.0,
            self.reason if passed else f"{self.reason}_withheld",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": list(self.fields),
            "min_count": self.min_count,
            "groups": {"|".join(key): value for key, value in sorted(self.groups.items())},
        }


class _ConjunctiveCertificate:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.category_gate = _GroupCertificate(
            rows, ("category", "difficulty"), min_count=2, reason="category_difficulty_certificate"
        )
        self.model_gate = _GroupCertificate(
            rows, ("model", "harness"), min_count=2, reason="model_harness_certificate"
        )

    def decide(self, context: Mapping[str, Any]) -> Decision:
        category = self.category_gate.decide(context)
        model = self.model_gate.decide(context)
        passed = category.action == "reuse" and model.action == "reuse"
        return Decision(
            "reuse" if passed else "abstain",
            1.0 if passed else 0.0,
            0.0,
            "certificate_pass" if passed else "certificate_withheld",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "category_difficulty_gate": self.category_gate.to_dict(),
            "model_harness_gate": self.model_gate.to_dict(),
            "policy": "conjunctive_public_context_certificate",
        }


def _metrics_without_decisions(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "decisions"}


def _task_cluster_bootstrap(
    policy: Any,
    evaluation: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
    *,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evaluation:
        by_task[str(row["task_id"])].append(row)
    task_ids = sorted(by_task)
    rng = random.Random(seed)
    values: dict[str, list[float]] = {
        "coverage": [],
        "unsupported_all": [],
        "unsupported_deployed": [],
        "support_decision_regret_mean": [],
    }
    for _ in range(reps):
        sampled = [
            row
            for task_id in (rng.choice(task_ids) for _ in task_ids)
            for row in by_task[task_id]
        ]
        report = evaluate_policy(policy, [], sampled, scores)
        for key in values:
            value = report.get(key)
            if value is not None:
                values[key].append(float(value))
    output = {}
    for key, series in values.items():
        ordered = sorted(series)
        output[key] = {
            "lower_2_5": ordered[max(0, int(0.025 * len(ordered)) - 1)] if ordered else None,
            "upper_97_5": ordered[min(len(ordered) - 1, int(0.975 * len(ordered)))] if ordered else None,
            "bootstrap_reps": len(ordered),
        }
    return output


def _audit_with_manifest(
    rows: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
    *,
    registry_path: Path,
    manifest_path: Path,
    registry_source_commit: str,
    manifest_source_commit: str,
    split_seed: str,
    bootstrap_reps: int,
) -> dict[str, Any]:
    joined = join_registry_rows(rows, manifest)
    split = build_task_split(joined, split_seed=split_seed)
    pairs = _pair_rows(joined)
    task_phase = {
        str(row["task_id"]): phase
        for phase, members in split.items()
        for row in members
    }
    pair_split = {
        phase: [row for row in pairs if task_phase[row["task_id"]] == phase]
        for phase in ("synthesis", "calibration", "evaluation")
    }
    synthesis = pair_split["synthesis"]
    calibration = pair_split["calibration"]
    evaluation = pair_split["evaluation"]
    schema = SkillsBenchFeatureSchema.from_synthesis(synthesis)
    synthesis_labels = [
        {"id": row["id"], "context": row["context"], "label": row["tau"] > 0.0}
        for row in synthesis
    ]
    calibration_labels = [
        {"id": row["id"], "context": row["context"], "unsupported": row["tau"] <= 0.0}
        for row in calibration
    ]
    eval_rows = [
        {"id": row["id"], "task_id": row["task_id"], "context": row["context"]}
        for row in evaluation
    ]
    scores = {row["id"]: {"tau": row["tau"], "scientific_gate": True} for row in evaluation}

    ltt = LTTLogisticPolicy.fit(synthesis_labels, calibration_labels, schema, LTTConfig.default())
    rscb = RSCBModel.fit(synthesis_labels, schema, RSCBConfig.default())
    category_gate = _GroupCertificate(
        calibration, ("category", "difficulty"), min_count=2, reason="category_difficulty_certificate"
    )
    acre_gate = _ConjunctiveCertificate(calibration)
    policies = {
        "global_reuse": _AlwaysReuse(),
        "category_difficulty_gate": category_gate,
        "ltt_logistic": ltt,
        "rscb_lcb": rscb,
        "acre_context_certificate": acre_gate,
    }

    reports: dict[str, Any] = {}
    for index, (name, policy) in enumerate(policies.items()):
        metrics = evaluate_policy(policy, calibration_labels, eval_rows, scores, delta=0.0)
        reports[name] = {
            "metrics": _metrics_without_decisions(metrics),
            "task_cluster_bootstrap": _task_cluster_bootstrap(
                policy, eval_rows, scores, reps=bootstrap_reps, seed=20260913 + index
            ),
            "policy": policy.to_dict() if hasattr(policy, "to_dict") else {"type": name},
        }

    strata: dict[str, list[str]] = defaultdict(list)
    tasks = sorted({str(row["task_id"]) for row in joined})
    for task_id in tasks:
        members = [row for row in joined if str(row["task_id"]) == task_id]
        category = str(members[0]["category"])
        difficulty = str(members[0]["difficulty"])
        strata[f"{category}|{difficulty}"].append(task_id)

    return {
        "schema_version": 2,
        "status": "complete",
        "analysis": "public_external_deployment_audit",
        "claim_boundary": "public external deployment audit on frozen public outcomes; not hosted-agent efficacy",
        "provider_calls": 0,
        "result_registry": {
            "path": str(registry_path),
            "sha256": _sha256(registry_path),
            "source_commit": registry_source_commit,
            "rows": len(rows),
        },
        "task_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "source_commit": manifest_source_commit,
            "active_tasks": len(tasks),
        },
        "join": {
            "result_rows": len(rows),
            "joined_rows": len(joined),
            "pairs": len(pairs),
            "duplicate_active_metadata": False,
            "missing_task_metadata": 0,
            "null_category": 0,
            "null_difficulty": 0,
        },
        "feature_set": ["category", "difficulty", "model", "harness"],
        "task_split": {
            "seed": split_seed,
            "unit": "task",
            "allocation": "within category x difficulty stratum, stable sha256 order assigned synthesis/synthesis/calibration/evaluation",
            "task_counts": {
                phase: len({str(row["task_id"]) for row in members})
                for phase, members in split.items()
            },
            "row_counts": {phase: len(members) for phase, members in split.items()},
            "strata": {key: sorted(value) for key, value in sorted(strata.items())},
        },
        "conditions": list(ACTIVE_CONDITIONS),
        "score_definition": "tau = (With Skills score - No Skills score) / 100; supported iff tau > 0 and scientific gate is observed-valid",
        "policies": reports,
        "synthesis_count": len(synthesis),
        "calibration_count": len(calibration),
        "evaluation_count": len(evaluation),
        "evaluation_task_count": len({row["task_id"] for row in evaluation}),
    }


def audit_public_snapshot(
    path: Path,
    *,
    task_manifest_path: Path | None = None,
    registry_source_commit: str | None = None,
    manifest_source_commit: str | None = None,
    split_seed: str = DEFAULT_SPLIT_SEED,
    bootstrap_reps: int = 400,
) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        return {
            "schema_version": 2,
            "status": "invalid_snapshot",
            "provider_calls": 0,
            "claim_boundary": "not external validity evidence",
        }
    if task_manifest_path is None:
        missing = [
            field for field in REQUIRED_STRATIFICATION_FIELDS
            if any(field not in row for row in rows)
        ]
        return {
            "schema_version": 1,
            "status": "metadata_insufficient",
            "provider_calls": 0,
            "rows": len(rows),
            "missing_stratification_fields": missing or list(REQUIRED_STRATIFICATION_FIELDS),
            "source": str(path),
            "claim_boundary": "not external validity evidence",
            "reason": "no version-pinned task manifest supplied; aggregate rows are not analyzed",
        }
    manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("task manifest must be a list")
    if not registry_source_commit or not manifest_source_commit:
        raise ValueError("result and task-manifest source commits must be supplied for external audit")
    return _audit_with_manifest(
        rows,
        manifest,
        registry_path=path,
        manifest_path=task_manifest_path,
        registry_source_commit=registry_source_commit,
        manifest_source_commit=manifest_source_commit,
        split_seed=split_seed,
        bootstrap_reps=bootstrap_reps,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--registry-source-commit")
    parser.add_argument("--manifest-source-commit")
    parser.add_argument("--split-seed", default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--bootstrap-reps", type=int, default=400)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit_public_snapshot(
        args.snapshot,
        task_manifest_path=args.task_manifest,
        registry_source_commit=args.registry_source_commit,
        manifest_source_commit=args.manifest_source_commit,
        split_seed=args.split_seed,
        bootstrap_reps=args.bootstrap_reps,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "provider_calls": 0, "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
