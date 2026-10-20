from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


SKILLSBENCH_V11_COMMIT = "b63b7b2850226b6aa4fb5929a8c1ac7bc4d9a6af"
BENCHFLOW_RANGE = ">=0.6.3,<0.7"
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]+", re.IGNORECASE)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_task_frontmatter(task_path: Path) -> dict[str, Any]:
    text = task_path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"task has no YAML frontmatter: {task_path}")
    _, frontmatter, _ = text.split("---", 2)
    data = yaml.safe_load(frontmatter)
    if not isinstance(data, dict) or not isinstance(data.get("metadata"), dict):
        raise ValueError(f"task metadata is not an object: {task_path}")
    return dict(data["metadata"])


def _tokens(text: str) -> Counter[str]:
    return Counter(TOKEN_RE.findall(text.lower()))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    dot = sum(left[key] * right[key] for key in keys)
    norm_left = math.sqrt(sum(value * value for value in left.values()))
    norm_right = math.sqrt(sum(value * value for value in right.values()))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


def _field_list(metadata: Mapping[str, Any], key: str) -> list[str]:
    value = metadata.get(key, [])
    if isinstance(value, list):
        return sorted(str(item) for item in value)
    return [str(value)] if value else []


def _task_record(task_dir: Path) -> dict[str, Any]:
    metadata = parse_task_frontmatter(task_dir / "task.md")
    prompt = (task_dir / "task.md").read_text(encoding="utf-8")
    return {
        "task_id": task_dir.name,
        "task_path": str(task_dir),
        "category": str(metadata.get("category", "")),
        "difficulty": str(metadata.get("difficulty", "")),
        "task_type": _field_list(metadata, "task_type"),
        "modality": _field_list(metadata, "modality"),
        "interface": _field_list(metadata, "interface"),
        "skill_type": _field_list(metadata, "skill_type"),
        "tags": _field_list(metadata, "tags"),
        "prompt": prompt,
    }


def enumerate_tasks(benchmark_root: Path) -> list[dict[str, Any]]:
    tasks = [_task_record(path) for path in sorted((benchmark_root / "tasks").iterdir()) if (path / "task.md").is_file()]
    if len(tasks) != 87:
        raise ValueError(f"SkillsBench v1.1 expected 87 active tasks, found {len(tasks)}")
    return tasks


def enumerate_skills(benchmark_root: Path, tasks: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    task_by_id = {str(task["task_id"]): task for task in tasks}
    rows: list[dict[str, Any]] = []
    for skill_path in sorted((benchmark_root / "tasks").glob("*/environment/skills/*/SKILL.md")):
        source_task = skill_path.parents[3].name
        if source_task not in task_by_id:
            raise ValueError(f"skill source task is not active: {source_task}")
        text = skill_path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        _, frontmatter, body = text.split("---", 2)
        header = yaml.safe_load(frontmatter) or {}
        source = task_by_id[source_task]
        description = str(header.get("description", ""))
        row = {
            "skill_hash": sha256_text(text),
            "skill_path": str(skill_path),
            "skill_name": str(header.get("name", skill_path.parent.name)),
            "description": description,
            "body": body.strip(),
            "source_task_id": source_task,
            "source_category": source["category"],
            "source_difficulty": source["difficulty"],
            "source_task_type": source["task_type"],
            "source_modality": source["modality"],
            "source_interface": source["interface"],
            "source_skill_type": source["skill_type"],
            "source_tags": source["tags"],
        }
        rows.append(row)
    return rows


def _related(skill: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    if task["task_id"] == skill["source_task_id"]:
        return False
    same_category = bool(skill["source_category"]) and task["category"] == skill["source_category"]
    same_interface = bool(set(skill["source_interface"]) & set(task["interface"]))
    same_type = bool(set(skill["source_task_type"]) & set(task["task_type"]))
    return same_category or (same_interface and same_type)


def _structurally_general(skill: Mapping[str, Any]) -> bool:
    source = str(skill["source_task_id"]).lower()
    description = str(skill["description"]).lower()
    body = str(skill["body"]).lower()
    source_tokens = {token for token in TOKEN_RE.findall(source) if len(token) > 3}
    return bool(description) and not any(token in description for token in source_tokens) and not any(
        token in body[:1200] for token in source_tokens
    )


def select_skill_cohort(tasks: list[dict[str, Any]], skills: list[dict[str, Any]], *, requested_m: int = 8, k: int = 15) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_text = {str(task["task_id"]): _tokens(" ".join([str(task["prompt"]), str(task["category"]), *task["tags"]])) for task in tasks}
    candidates: list[dict[str, Any]] = []
    for skill in skills:
        related = [task for task in tasks if _related(skill, task)]
        if len(related) < k or not _structurally_general(skill):
            continue
        skill_tokens = _tokens(" ".join([str(skill["description"]), str(skill["body"]), *skill["source_tags"]]))
        scored = sorted(
            related,
            key=lambda task: (-_cosine(skill_tokens, task_text[str(task["task_id"])]), str(task["task_id"])),
        )
        candidates.append({
            **{key: skill[key] for key in skill if key not in {"body"}},
            "related_candidate_count": len(related),
            "candidate_tasks": [str(task["task_id"]) for task in scored[:k]],
            "structural_generality_score": len(related),
        })
    candidates.sort(key=lambda row: (-int(row["structural_generality_score"]), str(row["skill_hash"])))
    m = requested_m if len(candidates) >= requested_m else 6 if len(candidates) >= 6 else len(candidates)
    selected = candidates[:m]
    if len(selected) < 3:
        raise ValueError(f"fewer than three structurally eligible skills: {len(selected)}")
    target_rows = [
        {"skill_hash": row["skill_hash"], "source_task_id": row["source_task_id"], "target_task_id": task_id}
        for row in selected
        for task_id in row["candidate_tasks"]
    ]
    return selected, target_rows


def materialize_no_skill_task(task_dir: Path, destination: Path) -> None:
    shutil.copytree(task_dir, destination, ignore=shutil.ignore_patterns("skills"))
    leaked = list(destination.rglob("SKILL.md")) + list((destination / "environment" / "skills").rglob("*") if (destination / "environment" / "skills").exists() else [])
    if leaked:
        raise AssertionError(f"no-skill materialization leaked skill files: {leaked}")


def materialize_native_skill_task(task_dir: Path, destination: Path) -> None:
    shutil.copytree(task_dir, destination)


def materialize_transferred_skill_task(
    task_dir: Path,
    source_skill_dir: Path,
    skill_name: str,
    destination: Path,
) -> None:
    if not skill_name or skill_name in {".", ".."} or "/" in skill_name or "\\" in skill_name:
        raise ValueError(f"skill_name must be one path component: {skill_name!r}")
    shutil.copytree(task_dir, destination, ignore=shutil.ignore_patterns("skills"))
    target_skill_dir = destination / "environment" / "skills" / skill_name
    target_skill_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_skill_dir / "SKILL.md", target_skill_dir / "SKILL.md")
    skill_files = list((destination / "environment" / "skills").rglob("SKILL.md"))
    if len(skill_files) != 1:
        raise AssertionError(f"transferred materialization must contain one skill: {skill_files}")


def materialize_sb1_smoke(benchmark_root: Path, smoke_manifest: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(smoke_manifest.read_text(encoding="utf-8"))
    if manifest.get("conditions") != ["oracle", "no_skill", "native_skill"]:
        raise ValueError("SB1 smoke manifest must use oracle/no_skill/native_skill conditions")
    task_ids = [str(row["task_id"]) for row in manifest.get("tasks", [])]
    output_dir.mkdir(parents=True, exist_ok=False)
    for task_id in task_ids:
        task_dir = benchmark_root / "tasks" / task_id
        if not task_dir.is_dir():
            raise FileNotFoundError(f"SB1 task is not active: {task_id}")
        materialize_no_skill_task(task_dir, output_dir / "no_skill" / task_id)
        materialize_native_skill_task(task_dir, output_dir / "native_skill" / task_id)
    receipt = {
        "schema_version": 1,
        "benchmark_release": manifest.get("benchmark_release"),
        "tasks": task_ids,
        "conditions": manifest["conditions"],
        "provider_calls": 0,
        "no_skill_root": str(output_dir / "no_skill"),
        "native_skill_root": str(output_dir / "native_skill"),
    }
    (output_dir / "sb1_materialization_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def write_sb0_sb2_artifacts(benchmark_root: Path, output_dir: Path, *, archive_sha256: str | None = None) -> dict[str, Any]:
    tasks = enumerate_tasks(benchmark_root)
    skills = enumerate_skills(benchmark_root, tasks)
    selected, target_rows = select_skill_cohort(tasks, skills)
    smoke_tasks = [task for task in tasks if task["difficulty"] in {"easy", "medium"}][:3]
    if len(smoke_tasks) != 3:
        raise ValueError("SB1 requires three easy/medium smoke tasks")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "skillsbench_protocol_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "benchmark": "SkillsBench",
        "benchmark_release": "v1.1",
        "benchmark_commit": SKILLSBENCH_V11_COMMIT,
        "benchflow_version_range": BENCHFLOW_RANGE,
        "provider": "deepseek-only",
        "model_selection_rule": "current server DeepSeek model, frozen before SB1 provider call",
        "pair_unit": "skill_hash,target_task_id,model_id,harness_id",
        "outcome_blind": True,
        "acre_started": False,
        "final_test_started": False,
        "archive_sha256": archive_sha256,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "skill_cohort_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "selection_is_outcome_blind": True,
        "requested_skill_count": 8,
        "selected_skill_count": len(selected),
        "candidate_target_count_per_skill": 15,
        "selection_rule": "structural generality plus deterministic lexical similarity; no agent outcomes",
        "skills": selected,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "candidate_context_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "selection_is_outcome_blind": True,
        "pair_unit": "skill_hash,target_task_id",
        "pairs": target_rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "sb1_smoke_task_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "benchmark_release": "v1.1",
        "benchmark_commit": SKILLSBENCH_V11_COMMIT,
        "tasks": [{key: value for key, value in task.items() if key not in {"prompt", "task_path"}} for task in smoke_tasks],
        "conditions": ["oracle", "no_skill", "native_skill"],
        "trials_per_condition": 1,
        "provider_calls": "pending",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"tasks": tasks, "skills": skills, "selected": selected, "target_rows": target_rows, "smoke_tasks": smoke_tasks}
