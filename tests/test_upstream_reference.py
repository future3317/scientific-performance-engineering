from experiments.external_real.upstream_reference import (
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
