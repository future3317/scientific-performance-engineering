"""Pinned SkillGen-compatible benchmark views and outcome adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any


SCORING_ONLY_FIELDS = frozenset(
    {
        "key_actions", "gold_actions", "ground_truth", "gold_path_length",
        "target_bnids", "relevant_apis", "dependency_analysis", "judge_score",
        "official_score", "score", "trajectory", "verifier_result",
    }
)


@dataclass(frozen=True)
class BenchmarkView:
    benchmark: str
    dataset_path: str
    expected_commit: str
    instances: tuple[dict[str, Any], ...]


def load_pinned_benchmark(benchmark: str, dataset_path: str | Path, *, expected_commit: str) -> BenchmarkView:
    path = Path(dataset_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    instances = tuple(dict(item) for item in payload.get("instances", []))
    ids = [str(item.get("instance_id")) for item in instances]
    if not instances or any(item == "None" for item in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"benchmark instances must have unique instance_id: {path}")
    return BenchmarkView(str(benchmark), str(path), str(expected_commit), instances)


def build_public_context(instance: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(instance.get("metadata") or {})
    context: dict[str, Any] = {
        "benchmark": metadata.get("benchmark"),
        "task_id": metadata.get("task_id"),
        "task_name": metadata.get("task_name"),
        "fold": metadata.get("fold"),
        "task_description": metadata.get("task_description"),
        "initial_observation": metadata.get("initial_observation"),
        "category": metadata.get("category") or metadata.get("domain"),
        "difficulty": metadata.get("difficulty"),
        "model_harness_identity": metadata.get("model_harness_identity"),
    }
    return {key: value for key, value in context.items() if value is not None}


def validate_paired_instance_ids(baseline_ids: list[str] | tuple[str, ...], skill_ids: list[str] | tuple[str, ...]) -> None:
    if list(baseline_ids) != list(skill_ids) or len(set(baseline_ids)) != len(baseline_ids):
        raise ValueError("paired instance ids must match exactly and be unique")


def paired_cache_key(instance_id: str, skill_digest: str, model: str, seed: int) -> str:
    body = f"{instance_id}|{skill_digest}|{model}|{int(seed)}".encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def grade_output(skillgen_root: str | Path, benchmark: str, output: str, instance: dict[str, Any]) -> dict[str, Any]:
    root = Path(skillgen_root)
    module_path = root / "benchmarks" / f"{benchmark}_grader.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"SkillGen grader not found: {module_path}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module(f"benchmarks.{benchmark}_grader")
    function = getattr(module, f"evaluate_{benchmark}_output", None)
    if function is None:
        raise AttributeError(f"grader lacks evaluate_{benchmark}_output")
    return dict(function(output, instance.get("metadata") or {}))
