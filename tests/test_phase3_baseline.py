"""Execution-contract fixtures only: no provider or native environment."""
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
import sys

import pytest

from core.acre.resource_budget import ApprovedProviderBudget, ProviderBudgetExceeded, manifest_digest
from experiments.skill_evolution.real_study import (
    _validate_diagnostic, diagnostic_manifest, diagnostic_summary, main, viability_gate,
)


def fixture_manifest():
    from experiments.external_real.alfworld_runner import TASK_TYPE_NAMES
    rows = [{"task_id": f"{family}/{i}", "template_id": f"{family}-{i}", "cluster_id": f"{family}-{i}",
             "task_type": family, "identity_status": "IDENTIFIED", "split": "train_dev",
             "exposure": "new_train_dev"} for family in TASK_TYPE_NAMES for i in range(2)]
    return diagnostic_manifest({"selected": rows, "seed": 31}, {"fixture": True}, {}, controller_commit="fixture",
                               gamefile_digests={row["task_id"]: "game-digest" for row in rows},
                               run_id="fixture", output_dir="/fixture")


def approved_budget():
    manifest = fixture_manifest()
    approval = {"provider_enabled": True, "study_id": manifest["study_id"], "run_id": "fixture",
                "controller_commit": "fixture", "model": "deepseek-flash", "approved_manifest_sha256": manifest_digest(manifest),
                "approved_split_manifest_sha256": manifest["split_manifest_sha256"], "approved_by": "SYNTHETIC_FIXTURE",
                "approval_reference": "SYNTHETIC_FIXTURE", "approved_at_utc": datetime.now(timezone.utc).isoformat(),
                "expiration_utc": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
                "permitted_stages": ["baseline_diagnostic"], "api_base_allowlist": ["https://api.deepseek.com"],
                "max_provider_calls": 720, "max_input_tokens": 23592960, "max_completion_tokens": 5898240,
                "per_episode_max_steps": 60, "per_request_max_completion_tokens": 8192,
                "max_total_usd": "15", "max_failed_retries": 0, "max_wall_seconds": 43200,
                "max_usd_per_episode": "1.25", "max_calls_per_episode": 60, "max_episode_wall_seconds": 3600}
    pricing = {"model": "deepseek-flash", "currency": "USD", "pricing_source_url": "https://fixture.test",
               "checked_at_utc": "fixture", "price_period": "fixture", "estimate_quality": "SYNTHETIC_FIXTURE",
               "input_hit_per_million": "0.006", "input_miss_per_million": "0.3", "output_per_million": "1.2"}
    budget = ApprovedProviderBudget(manifest, approval, pricing, controller_commit="fixture",
                                    stage="baseline_diagnostic", model="deepseek-flash", base_url="https://api.deepseek.com")
    return manifest, approval, budget


def test_t1_six_families_plan_has_720_ceiling_not_180():
    m, a, b = approved_budget()
    assert m["max_calls"] == 720
    _validate_diagnostic(m, m["plan"], {}, a)
    b.ensure_capacity(720, 720 * 32768, 720 * 8192)
    assert b.calls_reserved == 0


def test_episode_usd_wall_and_call_caps_share_same_ledger(monkeypatch):
    _, _, b = approved_budget()
    b.begin_episode(max_usd="0.001", max_calls=1, max_wall_seconds=10)
    with pytest.raises(ProviderBudgetExceeded, match="episode_usd_cap"):
        b.reserve(32768, 8192)
    assert b.calls_reserved == 0
    b.reserve(100, 10)
    b.record_usage(100, 10)
    with pytest.raises(ProviderBudgetExceeded, match="episode_call_cap"):
        b.reserve(100, 10)
    b.begin_episode(max_usd="1.25", max_calls=60, max_wall_seconds=10)
    start = b.episode_limits["started"]
    monkeypatch.setattr("core.acre.resource_budget.time.monotonic", lambda: start + 11)
    assert b.remaining_wall_seconds < 0
    with pytest.raises(ProviderBudgetExceeded, match="episode_wall_cap"):
        b.reserve(100, 10)


def test_inconsistent_cache_retains_known_charge_but_stops():
    _, _, b = approved_budget()
    b.reserve(100, 10)
    b.record_usage(100, 10, cache_hit_tokens=20, cache_miss_tokens=20)
    assert b.stopped and b.stop_reason == "cache_telemetry_inconsistent"
    assert b.usd_used > 0


@pytest.mark.parametrize("field,value", [("execute", False), ("approval_status", "APPROVAL_REQUIRED")])
def test_pending_approval_cannot_be_enabled_by_one_flag(field, value):
    manifest, approval, b = approved_budget()
    approval[field] = value
    with pytest.raises(ProviderBudgetExceeded, match="APPROVAL_REQUIRED"):
        ApprovedProviderBudget(manifest, approval, b.pricing, controller_commit="fixture",
                               stage="baseline_diagnostic", model="deepseek-flash", base_url="https://api.deepseek.com")


def test_t1_live_without_approval_never_imports_sdk_or_environment(monkeypatch, tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(fixture_manifest()))
    monkeypatch.setattr(sys, "argv", ["real_study", "--manifest", str(path), "--live"])
    monkeypatch.setattr("experiments.skill_evolution.real_study.subprocess.check_output", lambda *a, **k: "fixture\n")
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.setitem(sys.modules, "alfworld", None)
    with pytest.raises(ProviderBudgetExceeded, match="APPROVAL_REQUIRED"):
        main()


def test_success_only_does_not_invent_episode_failure_from_truncation():
    rows = [{"evidence_origin": "REAL_AGENT", "cluster_id": str(i), "input_tokens": 1, "completion_tokens": 2,
             "provider_budget": {"tokens_unknown": False}, "status": "VERIFIED_SUCCESS", "success": True,
             "environment_won": True, "environment_done": True, "task_family": "heat",
             "task_id": str(i), "trajectory": [{"finish_reason": "length", "selected_action": None}]}
            for i in range(3)]
    gate = viability_gate(rows)
    assert gate["agent_viability"] == "VERIFIED_TRAIN_VIABLE"
    assert gate["skill_source_readiness"] == "SUCCESS_ONLY_READY"
    assert gate["independent_failure_clusters"] == 0
    assert diagnostic_summary(rows, 12)["not_started"] == 9


def test_selected_game_factory_does_not_scan_population(monkeypatch):
    from experiments.skill_evolution.native_verifier import make_selected_environment
    import types
    registered = []
    env = SimpleNamespace(seed=lambda value: None)
    tw = types.ModuleType("textworld")
    gym = types.ModuleType("textworld.gym")
    tw.EnvInfos = lambda **kw: kw
    gym.register_games = lambda games, infos, **kw: registered.append((games, infos, kw)) or "selected"
    gym.make = lambda identity: env
    tw.gym = gym
    wrappers = types.ModuleType("alfworld.agents.environment.alfred_tw_env")
    wrappers.AlfredDemangler = lambda **kw: "demangler"
    wrappers.AlfredInfos = "infos"
    monkeypatch.setitem(sys.modules, "textworld", tw)
    monkeypatch.setitem(sys.modules, "textworld.gym", gym)
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment.alfred_tw_env", wrappers)
    game = "/data/json_2.1.1/train/template/trial_one/game.tw-pddl"
    assert make_selected_environment(game, max_steps=60, seed=31) is env
    assert registered[0][0] == [game]
    assert registered[0][1]["extras"] == ["gamefile"]
    assert registered[0][2]["max_episode_steps"] == 60


def test_production_t1_batch_resets_selected_games_in_frozen_order_without_provider(tmp_path):
    from experiments.skill_evolution.real_study import run_diagnostic
    import hashlib
    m, _, b = approved_budget()
    selected, reset = [], []
    for i, task in enumerate(m["plan"]["episodes"]):
        game = tmp_path / f"game{i}.tw-pddl"
        game.write_bytes(b"SYNTHETIC_FIXTURE")
        task["gamefile"] = str(game)
        task["gamefile_sha256"] = hashlib.sha256(game.read_bytes()).hexdigest()
        selected.append(str(game))

    class Env:
        def __init__(self, game):
            self.game = game

        def reset(self):
            reset.append(self.game)
            return ["Your task is to: look."], {"extra.gamefile": [self.game], "admissible_commands": [["look"]]}

        def step(self, actions):
            assert actions == ["look"]
            return ["fixture completion"], [1], [True], {"won": [True], "admissible_commands": [["look"]]}

        def close(self):
            pass

    class Client:
        budget = b
        max_input_per_request = 32768
        max_completion_tokens = 8192
        receipt_metadata = {"evidence_origin": "SYNTHETIC_FIXTURE", "split": "train_dev"}

        def choose_action(self, *args):
            b.reserve(100, 10)
            b.record_usage(100, 10)
            return {"raw_output": "Action: look", "input_tokens": 100, "output_tokens": 10, "finish_reason": "stop"}

    result = run_diagnostic(m, Client(), tmp_path / "outputs", environment_factory=lambda game, **kw: Env(game))
    assert reset == selected
    assert result["summary"]["not_started"] == 0
    assert b.calls_reserved == 12
    assert all(row["record_kind"] == "SYNTHETIC_FIXTURE" and not row["actual_new_model_rollout"]
               for row in result["episodes"])
    assert result["summary"]["source_readiness"]["independent_win_clusters"] == 0
