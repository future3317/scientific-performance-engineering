from __future__ import annotations

from scripts.run_skillgen_external_calibration import _public_context, _summarize_policy


def test_public_context_excludes_scoring_only_fields() -> None:
    instance = {
        "metadata": {
            "benchmark": "scienceworld",
            "task_id": "1",
            "task_name": "task",
            "variation_idx": 2,
            "fold": "test",
            "task_description": "public",
            "initial_observation": "scene",
            "key_actions": ["hidden"],
            "gold_actions": ["hidden"],
            "ground_truth": "hidden",
            "gold_path_length": 3,
        }
    }
    context = _public_context(instance)
    assert context["task_name"] == "task"
    assert "key_actions" not in context
    assert "gold_actions" not in context
    assert "ground_truth" not in context


def test_policy_summary_uses_deployed_denominator() -> None:
    rows = [
        {"baseline_score": 0.5, "skill_score": 0.7, "policies": {"p": {"apply": True}}},
        {"baseline_score": 0.8, "skill_score": 0.4, "policies": {"p": {"apply": True}}},
        {"baseline_score": 0.8, "skill_score": 0.8, "policies": {"p": {"apply": False}}},
    ]
    summary = _summarize_policy("p", rows)
    assert summary["deployed"] == 2
    assert summary["coverage"] == 2 / 3
    assert summary["unsupported_deployed"] == 1
    assert summary["unsupported_deployed_rate"] == 0.5
