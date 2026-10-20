import json
from pathlib import Path

import pytest

from experiments.external_real.upstream_reference import (
    build_release_artifacts,
    aupc_from_progress,
    summarize_rows,
)


def test_aupc_reconstructs_progress_curve_with_initial_zero():
    row = {"num_step": 3, "progress": [[0, 0.5], [2, 1.0]]}

    assert aupc_from_progress(row) == 0.5


def test_summary_marks_released_tasks_as_burned_and_reports_metrics():
    rows = [
        {
            "task_uid": "task_1",
            "task_name": "family-a",
            "num_step": 2,
            "grounding": [0],
            "grounding_rate": 0.5,
            "progress": [[0, 1.0]],
            "progress_rate": 1.0,
        },
        {
            "task_uid": "task_2",
            "task_name": "family-b",
            "num_step": 2,
            "grounding": [],
            "grounding_rate": 0.0,
            "progress": [],
            "progress_rate": 0.0,
        },
    ]

    summary = summarize_rows(rows)

    assert summary["n"] == 2
    assert summary["grounding_rate"] == 0.25
    assert summary["progress_rate"] == 0.5
    assert summary["success_rate"] == 0.5
    assert summary["aupc"] == 1 / 3
    assert summary["task_ids"] == ["task_1", "task_2"]
    assert summary["task_classification"] == "public_reference_burned"


def _write_release_logs(root: Path, configs: list[tuple[str, int]]) -> None:
    log_dir = root / "logs" / "inference_skillgen" / "sc"
    log_dir.mkdir(parents=True)
    row = {
        "task_uid": "task",
        "task_name": "family",
        "num_step": 1,
        "grounding": [],
        "progress": [],
        "progress_rate": 0.0,
    }
    for temperature, fold in configs:
        path = log_dir / (
            f"Qwen2.5-7B-Instruct_temp{temperature}_step20_demoac_skill_newprompt_sampling6_topac1_topk1_fold{fold}.jsonl"
        )
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_release_reference_requires_zero_temperature_and_exact_fold_set(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_release_logs(source, [("0.0", 0), ("0.0", 1), ("0.0", 2), ("0.0", 3), ("1.0", 0)])

    result = build_release_artifacts(source, tmp_path / "out", source_commit="fixture", source_url="fixture")

    assert result["summary"]["canonical_qwen_20_step_sampling6"]["n"] == 4


def test_release_reference_rejects_duplicate_or_missing_canonical_fold(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_release_logs(source, [("0.0", 0), ("0.0", 1), ("0.0", 2), ("0.00", 2)])

    with pytest.raises(ValueError, match=r"exactly canonical Qwen folds 0-3"):
        build_release_artifacts(source, tmp_path / "out", source_commit="fixture", source_url="fixture")


def test_release_reference_preserves_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_release_logs(source, [("0.0", fold) for fold in range(4)])
    output = tmp_path / "out"
    output.mkdir()
    frozen = output / "released_protocol_manifest.json"
    frozen.write_text("frozen\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_release_artifacts(source, output, source_commit="fixture", source_url="fixture")

    assert frozen.read_text(encoding="utf-8") == "frozen\n"
