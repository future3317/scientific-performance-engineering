from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_RUN_RE = re.compile(
    r"^(?P<model>.+?)_temp(?P<temperature>[^_]+)_step(?P<max_steps>\d+)"
    r"_demoac_skill_newprompt_sampling(?P<sampling_count>\d+)"
    r"_topac(?P<top_ac>\d+)_topk(?P<top_s>\d+)_fold(?P<fold>\d+)\.jsonl$"
)


def _mean(values: Sequence[float | int]) -> float:
    return sum(values) / len(values)


def _median(values: Sequence[float | int]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def _float(value: object, field: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [0, 1], got {result}")
    return result


def progress_curve(row: Mapping[str, object]) -> list[float]:
    """Reconstruct the upstream reward curve, including the initial zero."""
    num_step = int(row["num_step"])
    if num_step < 0:
        raise ValueError("num_step must be non-negative")
    events = row.get("progress", [])
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise ValueError("progress must be a sequence of [step, reward] pairs")
    by_step: dict[int, float] = {}
    for event in events:
        if not isinstance(event, Sequence) or len(event) != 2:
            raise ValueError("progress events must contain exactly two values")
        step = int(event[0])
        if step < 0 or step >= num_step or step in by_step:
            raise ValueError(f"invalid or duplicate progress step: {step}")
        by_step[step] = _float(event[1], "progress reward")
    curve = [0.0]
    current = 0.0
    for step in range(num_step):
        if step in by_step:
            if by_step[step] < current:
                raise ValueError("progress rewards must be non-decreasing")
            current = by_step[step]
        curve.append(current)
    return curve


def aupc_from_progress(row: Mapping[str, object]) -> float:
    curve = progress_curve(row)
    return sum(curve) / len(curve)


def _distribution(values: Sequence[float | int]) -> dict[str, object]:
    if not values:
        return {"n": 0, "values": []}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "values": ordered,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": _mean(ordered),
        "median": _median(ordered),
    }


def summarize_rows(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot summarize an empty release log")
    task_ids = sorted({str(row["task_uid"]) for row in materialized})
    step_counts = [int(row["num_step"]) for row in materialized]
    grounding = [len(row.get("grounding", [])) / steps if steps else 0.0 for row, steps in zip(materialized, step_counts)]
    progress = [_float(row.get("progress_rate", 0.0), "progress_rate") for row in materialized]
    aupc = [aupc_from_progress(row) for row in materialized]
    total_steps = sum(step_counts)
    total_grounding = sum(len(row.get("grounding", [])) for row in materialized)
    families = Counter(str(row["task_name"]) for row in materialized)
    return {
        "n": len(materialized),
        "task_ids": task_ids,
        "task_classification": "public_reference_burned",
        "grounding_rate": _mean(grounding),
        "grounding_rate_step_weighted": total_grounding / total_steps if total_steps else None,
        "progress_rate": _mean(progress),
        "success_rate": sum(value >= 1.0 for value in progress) / len(progress),
        "aupc": _mean(aupc),
        "step_count_distribution": _distribution(step_counts),
        "task_family_distribution": dict(sorted(families.items())),
        "metric_semantics": {
            "grounding_rate": "macro mean of valid-action indicators per logged action",
            "progress_rate": "macro mean of final upstream progress reward",
            "success_rate": "fraction with final progress_rate == 1.0",
            "aupc": "mean reconstructed progress over initial zero plus each logged action step",
        },
    }


def _read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_metadata(path: Path) -> dict[str, object]:
    match = _RUN_RE.match(path.name)
    if not match:
        return {"filename": path.name, "config_parse": "unparsed"}
    metadata = match.groupdict()
    metadata.update({"filename": path.name, "config_parse": "parsed", "fold": int(metadata["fold"])})
    for key in ("max_steps", "sampling_count", "top_ac", "top_s"):
        metadata[key] = int(metadata[key])
    return metadata


def _is_zero_temperature(value: object) -> bool:
    try:
        return float(str(value)) == 0.0
    except ValueError:
        return False


def build_release_artifacts(
    source_root: Path,
    output_dir: Path,
    *,
    source_commit: str,
    source_url: str,
) -> dict[str, object]:
    log_dir = source_root / "logs" / "inference_skillgen" / "sc"
    files = sorted(log_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no released ScienceWorld logs under {log_dir}")
    runs: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    for path in files:
        rows = _read_rows(path)
        metadata = _run_metadata(path)
        metadata.update({"relative_path": str(path.relative_to(source_root)), "row_count": len(rows)})
        metadata["summary"] = summarize_rows(rows)
        runs.append(metadata)
        all_rows.extend(rows)
    canonical = [
        run for run in runs
        if run.get("model") == "Qwen2.5-7B-Instruct"
        and _is_zero_temperature(run.get("temperature"))
        and run.get("sampling_count") == 6
        and run.get("top_ac") == 1
        and run.get("top_s") == 1
        and run.get("max_steps") == 20
        and run.get("fold") in {0, 1, 2, 3}
    ]
    canonical_folds = sorted(run["fold"] for run in canonical)
    if canonical_folds != [0, 1, 2, 3]:
        raise ValueError(f"expected exactly canonical Qwen folds 0-3, found {canonical_folds}")
    canonical_rows = []
    for run in canonical:
        canonical_rows.extend(_read_rows(source_root / run["relative_path"]))
    inventory = {
        "schema_version": 1,
        "source": {"repository": source_url, "commit": source_commit, "dataset": "sc"},
        "log_root": "logs/inference_skillgen/sc",
        "run_count": len(runs),
        "runs": runs,
        "canonical_run_selection": {
            "model": "Qwen2.5-7B-Instruct",
            "temperature": "0.0",
            "max_steps": 20,
            "sampling_count": 6,
            "top_ac": 1,
            "top_s": 1,
            "folds": [0, 1, 2, 3],
        },
        "provider_calls": 0,
        "prospective_use": "forbidden_public_reference_burned",
    }
    summary = {
        "schema_version": 1,
        "source": inventory["source"],
        "all_released_sc_logs": summarize_rows(all_rows),
        "canonical_qwen_20_step_sampling6": summarize_rows(canonical_rows),
        "canonical_fold_summaries": [run["summary"] for run in sorted(canonical, key=lambda item: item["fold"])],
        "metric_implementation": "experiments.external_real.upstream_reference",
        "provider_calls": 0,
        "prospective_use": "forbidden_public_reference_burned",
    }
    task_ids = sorted({str(row["task_uid"]) for row in all_rows})
    task_artifact = {
        "schema_version": 1,
        "source": inventory["source"],
        "classification": "public_reference_burned",
        "prospective_eligible": False,
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "reason": "released outcome logs were used for reference metric recomputation",
    }
    protocol = {
        "schema_version": 1,
        "stage": "U6_upstream_released_reference_audit",
        "source": inventory["source"],
        "canonical_config": inventory["canonical_run_selection"],
        "metric_semantics": summary["canonical_qwen_20_step_sampling6"]["metric_semantics"],
        "outcome_status": "public_reference_burned",
        "provider_calls": 0,
        "acre_started": False,
        "final_test_started": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "released_run_inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "released_metric_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "released_task_ids.json").write_text(json.dumps(task_artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "released_protocol_manifest.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = "\n".join([
        "# Upstream released ScienceWorld reference audit",
        "",
        f"- Source: `{source_url}@{source_commit}`",
        f"- Released ScienceWorld log files inventoried: **{len(runs)}**",
        f"- Canonical Qwen 20-step / sampling-6 / top_ac-1 / top_s-1 folds: **{len(canonical)}**",
        f"- Unique released task IDs: **{len(task_ids)}**",
        "- Provider calls for this audit: **0**",
        "- All released task IDs and outcomes are classified `public_reference / burned`.",
        "- They are not eligible for prospective train/dev or final-test evaluation.",
        "",
        "The metric implementation reconstructs the upstream progress curve from the persisted `progress` events, includes the initial zero state, and reports macro grounding rate, macro final progress, success as final progress equal to one, AUPC, step-count distribution, and task-family counts.",
        "",
        "The release contains inference outcome logs but no committed `logs/sampling`, `logs/skill_extraction`, action-contribution graph, embedding, or retrieval artifact that can be loaded as a frozen train-derived intervention. U7 provenance audit therefore precedes any new native audit.",
        "",
    ])
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return {"inventory": inventory, "summary": summary, "task_artifact": task_artifact, "protocol": protocol}
