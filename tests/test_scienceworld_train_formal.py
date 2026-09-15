from __future__ import annotations

from scripts.run_scienceworld_train_formal import generation_prompt, summarize_candidate


def test_generation_prompt_contains_public_construction_view_only() -> None:
    prompt = generation_prompt([
        {"task_id": "scienceworld:boil:1", "task_name": "boil", "task_description": "Change state.", "initial_observation": "A room."}
    ], seed=2)

    assert "scienceworld:boil:1" in prompt
    assert "gold" not in prompt.lower()
    assert "reference" not in prompt.lower()
    assert "seed=2" in prompt


def test_candidate_summary_preserves_admission_and_failure_reasons() -> None:
    summary = summarize_candidate(
        generation_seed=1,
        skill_hash="abc",
        verification={"outcome": "WITHHOLD_SKILL", "failure_reasons": ["skill_failed"]},
    )

    assert summary == {
        "generation_seed": 1,
        "skill_hash": "abc",
        "outcome": "WITHHOLD_SKILL",
        "failure_reasons": ["skill_failed"],
    }
