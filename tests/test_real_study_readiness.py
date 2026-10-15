import json
import sys

import pytest

from core.acre.resource_budget import ProviderBudgetExceeded
from experiments.skill_evolution.real_study import main, prepare_live, readiness_manifest, resource_plan


def plan():
    return {"config": {"max_steps": 60}, "selection_seed": 17,
            "episodes": [{"episode_id": name} for name in ("clean", "heat", "cool")]}


def test_readiness_has_no_default_paid_cap_or_assumed_cache_benefit():
    m = readiness_manifest(plan(), controller_commit="sha", run_id="b1")
    r = resource_plan(m)
    assert r["max_calls"] == 180
    assert r["input_allowance"] is None and r["completion_allowance"] is None
    assert r["estimated_worst_usd"] is None
    assert r["provider_calls"] == 0 and not r["cache_savings_assumed"]


def test_default_dryrun_does_not_import_sdk_or_environment(monkeypatch, tmp_path, capsys):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan()))
    monkeypatch.setattr(sys, "argv", ["real_study", "--plan", str(path)])
    monkeypatch.setattr("experiments.skill_evolution.real_study.subprocess.check_output", lambda *a, **kw: "sha\n")
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.setitem(sys.modules, "alfworld", None)
    main()
    assert json.loads(capsys.readouterr().out)["provider_calls"] == 0


def test_live_missing_approval_stops_before_credentials_sdk_or_reset(monkeypatch, tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan()))
    monkeypatch.setattr(sys, "argv", ["real_study", "--plan", str(path), "--live"])
    monkeypatch.setattr("experiments.skill_evolution.real_study.subprocess.check_output", lambda *a, **kw: "sha\n")
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ProviderBudgetExceeded, match="APPROVAL_REQUIRED"):
        main()


def test_modified_split_is_rejected_before_provider_initialization():
    p = plan()
    m = readiness_manifest(p, controller_commit="sha", run_id="b1")
    changed = {**p, "selection_seed": 18}
    with pytest.raises(ProviderBudgetExceeded, match="plan_mismatch"):
        prepare_live(m, changed, {}, {}, revision="sha", base_url="https://api.deepseek.com")


@pytest.mark.parametrize("won,done,expected", [(True, True, True), (False, True, False),
                                              (None, True, None), (True, False, None)])
def test_production_runner_uses_official_won_and_done_not_reward(won, done, expected):
    from experiments.external_real.alfworld_stateful_runner import run_episode

    class Env:
        def reset(self):
            return ["Your task is to: look."], {"admissible_commands": [["look"]]}

        def step(self, actions):
            assert actions == ["look"]
            return ["Task complete according to a message."], [1.0], [done], {
                "won": [won], "admissible_commands": [["look"]]}

    class Client:
        def choose_action(self, *args):
            return {"raw_output": "Action: look"}

    result = run_episode(Env(), Client(), max_steps=1)
    assert result["success"] is expected
    assert result["environment_won"] is won
    assert result["environment_done"] is done


def test_progress_avoids_false_loop_and_inventory_sources_are_observation_only():
    from experiments.external_real.alfworld_stateful_runner import EpisodeState, ExecutionGuard
    state = EpisodeState.start("Your task is to: put a clean apple in fridge.", 60)
    guard = ExecutionGuard(repeat_limit=2)
    assert guard.observe_result(state, "old", "same", "look") is None
    state.observe("You pick up the apple from the countertop.", "take apple from countertop")
    assert state.fact_sources["inventory"]["step"] == 1
    assert guard.observe_result(state, "old", "same", "look") is None
    state.observe("Inventory: nothing.")
    assert state.inventory == ()
    assert state.fact_sources["inventory"]["previous_contradicted"]


def test_viability_requires_two_semantic_clusters_and_independent_failure():
    from experiments.skill_evolution.real_study import viability_gate
    def receipt(cluster, win):
        return {"evidence_origin": "REAL_AGENT", "cluster_id": cluster, "input_tokens": 10,
                "completion_tokens": 3, "provider_budget": {"tokens_unknown": False},
                "status": "VERIFIED_SUCCESS" if win else "VERIFIED_FAILURE", "success": win,
                "environment_won": win, "environment_done": True}
    same_cluster = [receipt("a", True), receipt("a", True), receipt("b", False)]
    assert viability_gate(same_cluster)["status"] == "VIABLE_BUT_NOT_QUALIFIED"
    independent = [receipt("a", True), receipt("b", True), receipt("c", False)]
    assert viability_gate(independent)["status"] == "B-VIABLE-TRAIN-ONLY"
    independent[0]["evidence_origin"] = "SYNTHETIC_FIXTURE"
    assert viability_gate(independent)["status"] == "VIABLE_BUT_NOT_QUALIFIED"
