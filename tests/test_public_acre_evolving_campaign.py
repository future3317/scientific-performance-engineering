from __future__ import annotations

from benchmark.external.evolving_campaign import decision_for_condition, promote_candidate, run_sequential_campaign
from benchmark.external.skill_versions import SkillVersionStore


def test_conditions_have_distinct_pre_outcome_decisions(tmp_path) -> None:
    store = SkillVersionStore(tmp_path / "skills.jsonl")
    context = {"benchmark": "scienceworld", "task_id": "t1"}
    assert decision_for_condition("A", context, [], None).reuse is False
    assert decision_for_condition("C", context, ["prior plan"], None).reuse is True
    assert decision_for_condition("D", context, ["prior plan"], None).reuse is False
    version = store.create(parent_id=None, task_index=0, content="certified plan", evidence_ids=("ev-1",), certificate={"status": "promoted"}, provenance={})
    assert decision_for_condition("D", context, ["prior plan"], version).reuse is True


def test_candidate_promotion_requires_nonnegative_prior_effects(tmp_path) -> None:
    store = SkillVersionStore(tmp_path / "skills.jsonl")
    assert promote_candidate(store, task_index=0, content="v0", evidence_ids=(), prior_effects=[], current_effect=0.2, minimum_effect=0.1) is not None
    assert promote_candidate(store, task_index=1, content="v1", evidence_ids=("ev-2",), prior_effects=[0.2], current_effect=0.2, minimum_effect=0.1) is not None
    assert promote_candidate(store, task_index=2, content="v2", evidence_ids=("ev-3",), prior_effects=[0.2, -0.1], current_effect=0.2, minimum_effect=0.1) is None


def test_sequential_runner_serializes_decision_before_evolution_acquisition(tmp_path) -> None:
    calls = []

    def agent(prompt, task, condition):
        calls.append((task["instance_id"], condition))
        if condition == "D_evolution_acquisition":
            return "good", {"total_tokens": 2}
        return ("base" if condition == "A" else "good"), {"total_tokens": 1}

    def score(output, _task):
        return 1.0 if output == "good" else 0.0

    artifact = run_sequential_campaign(
        [{"instance_id": "t0", "input": "task", "metadata": {}}],
        condition="D",
        run_id="run-1",
        out_dir=tmp_path,
        agent_call=agent,
        score_call=score,
        minimum_effect=0.1,
    )
    assert artifact["skill_versions"]
    assert artifact["decision_ledger"][0]["event"] == "decision"
    assert artifact["decision_ledger"][1]["event"] == "outcome"
    assert calls[-1][1] == "D_evolution_acquisition"
