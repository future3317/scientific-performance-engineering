from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.acre.resource_budget import ProviderBudgetExceeded, manifest_digest
from core.skills import EvidenceBundle
from experiments.skill_evolution.real_learning import (
    PHASE4_SELECTION_POLICY, PHASE4_T2_MODE, freeze_pair_orders, freeze_triplet_orders,
    select_inner_candidate, validate_learning_inputs,
)


def learning_fixture():
    membership = []
    receipts = []
    for index, (outcome, status) in enumerate(((True, "VERIFIED_SUCCESS"),
                                                (True, "VERIFIED_SUCCESS"),
                                                (False, "AGENT_POLICY_FAILURE"))):
        task_id = f"source-template-{index}/trial_{index}"
        row = {
            "trial_id": f"source-run:{index}",
            "task_id": task_id,
            "template_id": f"source-template-{index}",
            "cluster_id": f"source-cluster-{index}",
            "split": "train_dev",
            "status": status,
            "termination_reason": "success" if outcome else "agent_policy_failure",
            "environment_won": outcome,
            "environment_done": outcome,
            "success": outcome,
            "input_tokens": 17,
            "completion_tokens": 13,
            "provider_budget": {"tokens_unknown": False},
            "evidence_origin": "REAL_AGENT",
            "task_family": "clean_place",
            "agent_revision_sha256": "a" * 64,
            "configuration_sha": "e" * 64, "model_request_name": "deepseek-flash",
        }
        row["receipt_sha256"] = manifest_digest(row)
        receipts.append(row)
        membership.append({
            "task_id": task_id, "template_id": row["template_id"], "cluster_id": row["cluster_id"],
            "split": "train_dev", "task_type": "pick_clean_then_place_in_recep",
            "identity_status": "IDENTIFIED", "confirmatory_eligible": False,
        })

    panels = {"panel_a_pairs": [], "panel_b_pairs": []}
    for panel, count in (("A", 6), ("B", 6)):
        target = panels["panel_a_pairs" if panel == "A" else "panel_b_pairs"]
        for index in range(count):
            suffix = f"{panel.lower()}-{index}"
            task_id = f"{suffix}-template/trial_1"
            gamefile = f"/data/json_2.1.1/train/{task_id}/game.tw-pddl"
            row = {"task_id": task_id, "template_id": f"{suffix}-template", "gamefile": gamefile,
                   "cluster_id": f"{suffix}-cluster", "split": "train_inner",
                   "identity_status": "IDENTIFIED", "confirmatory_eligible": True}
            membership.append(row)
            target.append({
                "pair_id": f"pair-{suffix}", "task_id": task_id, "template_id": row["template_id"],
                "cluster_id": row["cluster_id"], "split": "train_inner", "task_type_id": 3,
                "gamefile": gamefile, "seed": 19,
            })
    panels["panel_a_pairs"] = freeze_pair_orders(panels["panel_a_pairs"], 19)
    panels["panel_b_pairs"] = freeze_triplet_orders(panels["panel_b_pairs"], 23)

    split_manifest = {"task_membership": membership}
    plan = {**panels, "pair_order_seed": 19, "panel_b_arm_order_seed": 23}
    manifest = {
        "stage": "skill_evolution", "split": "train_inner", "thinking": "enabled",
        "phase4_mode": PHASE4_T2_MODE, "phase4_mode_version": 1,
        "selection_policy": {"id": PHASE4_SELECTION_POLICY, "version": 1},
        "source_mode": "CONTRASTIVE_REPAIR",
        "max_retries": 0, "split_manifest_sha256": manifest_digest(split_manifest),
        "plan": plan, "plan_sha256": manifest_digest(plan),
        "viability_receipts_sha256": "b" * 64, "source_receipts_sha256": "c" * 64,
        "max_steps": 2, "max_input_per_request": 100, "max_completion_per_request": 100,
        "max_usd_per_episode": "1.25", "max_calls_per_episode": 60,
        "max_episode_wall_seconds": 3600,
        "proposal_completion_calls": 1, "patch_completion_calls": 1, "max_calls": 62, "agent_sha": "a" * 64,
        "configuration_sha": "e" * 64, "model_request_name": "deepseek-flash",
        "source_commit": "controller-commit",
    }
    evidence = EvidenceBundle(
        source_split="train_dev", source_mode="CONTRASTIVE_REPAIR",
        source_task_ids=tuple(row["task_id"] for row in receipts),
        source_template_ids=tuple(row["template_id"] for row in receipts),
        source_cluster_ids=tuple(row["cluster_id"] for row in receipts),
        source_trace_ids=tuple(row["trial_id"] for row in receipts),
        source_trace_hashes=tuple(row["receipt_sha256"] for row in receipts),
        train_split_manifest_sha256=manifest["split_manifest_sha256"],
        task_family="clean_place", agent_hash=manifest["agent_sha"],
        successful_trajectory_summaries=("Verified visible completion.",),
        failed_trajectory_summaries=("Frozen agent reached a policy loop; environment did not report a win.",),
        verified_failure_patterns=("policy loop",),
        redaction_report_sha256="d" * 64,
    )
    viability = []
    for index in range(2):
        row = dict(receipts[index], trial_id=f"b-run:{index}", split="train_inner",
                   task_id=f"b-template-{index}/trial_{index}", template_id=f"b-template-{index}",
                   cluster_id=f"b-cluster-{index}", status="VERIFIED_SUCCESS", environment_won=True,
                   environment_done=True, success=True)
        row["receipt_sha256"] = manifest_digest({k: v for k, v in row.items() if k != "receipt_sha256"})
        viability.append(row)
    return manifest, split_manifest, viability, [dict(row) for row in receipts], evidence


def test_t2_inputs_use_train_dev_sources_and_complete_inner_budget_only():
    manifest, split, viability, sources, evidence = learning_fixture()
    plan = validate_learning_inputs(manifest, split, viability, sources, evidence)
    assert plan["status"] == "READY_FOR_APPROVAL"
    assert plan["B_independent_wins"] == 2
    assert plan["B_verified_failures"] == 0
    assert plan["planned_provider_calls"] == 62
    assert plan["outer_pairs"] == 0
    assert len(plan["paired_clusters"]["panel_a"]) == 6
    assert len(plan["paired_clusters"]["panel_b"]) == 6


def test_identified_resolved_train_dev_source_may_be_nonconfirmatory_but_inner_may_not():
    manifest, split, viability, sources, evidence = learning_fixture()
    assert all(row["confirmatory_eligible"] is False for row in split["task_membership"]
               if row["split"] == "train_dev")
    validate_learning_inputs(manifest, split, viability, sources, evidence)
    inner = manifest["plan"]["panel_a_pairs"][0]
    task = next(row for row in split["task_membership"] if row["task_id"] == inner["task_id"])
    task["confirmatory_eligible"] = False
    manifest["split_manifest_sha256"] = manifest_digest(split)
    evidence = replace(evidence, train_split_manifest_sha256=manifest["split_manifest_sha256"])
    with pytest.raises(ValueError, match="not confirmatory-eligible"):
        validate_learning_inputs(manifest, split, viability, sources, evidence)


def test_train_dev_source_agent_bytes_remain_independent_of_new_controller_commit():
    manifest, split, viability, sources, evidence = learning_fixture()
    manifest["controller_commit"] = "new-phase4-controller-commit"
    manifest["source_commit"] = "new-phase4-controller-commit"
    assert manifest["agent_sha"] == sources[0]["agent_revision_sha256"]
    assert manifest["controller_commit"] != manifest["agent_sha"]
    validate_learning_inputs(manifest, split, viability, sources, evidence)


def test_b_gate_counts_semantic_clusters_not_directory_templates():
    manifest, split, viability, sources, evidence = learning_fixture()
    viability[1]["cluster_id"] = viability[0]["cluster_id"]
    with pytest.raises(ProviderBudgetExceeded, match="B_viability_gate_not_satisfied"):
        validate_learning_inputs(manifest, split, viability, sources, evidence)


def test_source_receipt_hash_is_recomputed_and_tampering_is_rejected():
    manifest, split, viability, sources, evidence = learning_fixture()
    sources[0]["tampered"] = True
    with pytest.raises(ValueError, match="receipt hashes"):
        validate_learning_inputs(manifest, split, viability, sources, evidence)


def test_t2_manifest_cannot_smuggle_outer_pairs_under_inner_approval():
    manifest, split, viability, sources, evidence = learning_fixture()
    manifest["plan"]["outer_pairs"] = [{"split": "train_outer"}]
    manifest["plan_sha256"] = manifest_digest(manifest["plan"])
    with pytest.raises(ProviderBudgetExceeded, match="separate T3"):
        validate_learning_inputs(manifest, split, viability, sources, evidence)


def test_panel_b_excludes_incomplete_triplets_and_reports_three_common_cluster_comparisons():
    pairs = [{"pair_id": f"B:p{index}"} for index in range(4)]
    rows = []
    for index in range(4):
        arms = {"off": True, "v1": True, "v2": True}
        if index == 0:
            arms.pop("v2")
        for arm, success in arms.items():
            rows.append({"panel": "B", "pair_id": f"B:p{index}", "arm": arm,
                         "success": success, "input_tokens": 70 if arm == "v2" else 100, "completion_tokens": 0,
                         "cost_usd": "0.01" if arm != "v2" else "0.005"})
    selected = select_inner_candidate(rows, "SUCCESS_ONLY_INDUCTION", pairs)
    assert selected["panel_b_complete_triplets"] == 3
    assert selected["panel_b_incomplete_triplet_ids"] == ["B:p0"]
    assert selected["selected"] == "no_skill"
    assert {item["valid_pairs"] for item in selected["pairwise_comparisons"].values()} == {3}
    assert selected["candidate_screens"]["v1"]["regressions"] == 0


def test_v2_regression_is_rejected_and_frozen_candidate_rolls_back_to_v1():
    rows = []
    pairs = [{"pair_id": f"B:p{index}"} for index in range(6)]
    for index in range(6):
        for arm, success, tokens in (("off", True, 100), ("v1", True, 80), ("v2", index != 0, 20)):
            rows.append({"panel": "B", "pair_id": f"B:p{index}", "arm": arm,
                         "success": success, "input_tokens": tokens, "completion_tokens": 0,
                         "cost_usd": str(tokens / 10000)})
    selected = select_inner_candidate(rows, "SUCCESS_ONLY_INDUCTION", pairs)
    assert selected["candidate_screens"]["v2"]["regressions"] == 1
    assert not selected["candidate_screens"]["v2"]["eligible_for_outer_pilot"]
    assert selected["selected"] == "v1"


def test_wrong_native_gamefile_cannot_be_labeled_as_a_different_planned_task():
    manifest, split, viability, sources, evidence = learning_fixture()
    manifest["plan"]["panel_a_pairs"][0]["gamefile"] = "/data/json_2.1.1/train/other/trial_1/game.tw-pddl"
    manifest["plan_sha256"] = manifest_digest(manifest["plan"])
    with pytest.raises(ValueError, match="gamefile"):
        validate_learning_inputs(manifest, split, viability, sources, evidence)


def test_old_agent_source_evidence_cannot_be_used_as_current_agent_evidence():
    manifest, split, viability, sources, evidence = learning_fixture()
    with pytest.raises(ValueError, match="different frozen agent"):
        validate_learning_inputs(manifest, split, viability, sources, replace(evidence, agent_hash="old"))


def test_real_pair_adapter_uses_production_episode_and_persists_both_arms(tmp_path):
    from experiments.skill_evolution.real_learning import _run_one_pair
    manifest, _, _, _, _ = learning_fixture()
    pair = manifest["plan"]["panel_a_pairs"][0]
    closed, persisted, insertions = [], [], []

    class Environment:
        def seed(self, seed):
            assert seed == pair["seed"]

        def reset(self):
            return ["Your task is to: inspect the room."], {
                "extra.gamefile": [pair["gamefile"]], "extra.expert_plan": [["EXPERT_SECRET"]],
                "admissible_commands": [["look", "inventory"]]}

        def step(self, actions):
            assert type(actions) is list and len(actions) == 1
            return ["Visible result."], [0], [True], {"won": [actions[0] == "look"]}

        def close(self):
            closed.append(True)

    def environment_factory(gamefile, *, max_steps, seed):
        assert gamefile == pair["gamefile"] and max_steps == 2 and seed == pair["seed"]
        return Environment()

    class Client:
        budget = None
        def choose_action(self, state, admissible, skill, progress, reason):
            assert "EXPERT_SECRET" not in repr(state) + progress
            insertions.append(skill)
            return {"raw_output": "Action: look" if skill else "Action: inventory",
                    "input_tokens": 10, "output_tokens": 3}

    candidate = SimpleNamespace(body="Verify visible outcomes.", digest="f" * 64, version=1)
    off, on = _run_one_pair(tmp_path, Client(), pair, candidate, split="train_inner", max_steps=2,
                           run_id="fixture", agent_revision="agent", benchmark_revision="benchmark",
                           provenance=manifest, on_arm=lambda arm, row: persisted.append((arm, row)),
                           environment_factory=environment_factory)
    assert off["success"] is False and on["success"] is True
    assert [arm for arm, _ in persisted] == pair["arm_order"]
    assert insertions == [candidate.body if arm == "on" else None for arm in pair["arm_order"]]
    assert len(closed) == 2
    assert on["skill_version_sha256"] == candidate.digest
    assert on["split"] == "train_inner" and on["trajectory"][0]["selected_action"] == "look"


def test_panel_b_reserves_full_triplet_budget_before_first_environment_reset(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from core.acre.resource_budget import ApprovedProviderBudget
    from experiments.skill_evolution.real_learning import _run_panel_cluster

    manifest, _, _, _, _ = learning_fixture()
    manifest.update(study_id="fixture", run_id="triplet", controller_commit="fixture",
                    model="deepseek-flash", agent_revision="agent", benchmark_revision="benchmark")
    approval = {"provider_enabled": True, "study_id": "fixture", "run_id": "triplet",
        "controller_commit": "fixture", "model": "deepseek-flash",
        "approved_manifest_sha256": manifest_digest(manifest),
        "approved_split_manifest_sha256": manifest["split_manifest_sha256"],
        "approved_by": "fixture", "approval_reference": "fixture",
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "expiration_utc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "permitted_stages": ["skill_evolution"], "api_base_allowlist": ["https://api.deepseek.com"],
        "max_provider_calls": 100, "max_input_tokens": 10000, "max_completion_tokens": 10000,
        "per_episode_max_steps": 2, "per_request_max_completion_tokens": 100,
        "max_usd_per_episode": "1.25", "max_calls_per_episode": 60,
        "max_episode_wall_seconds": 3600,
        "max_total_usd": 1, "max_failed_retries": 0, "max_wall_seconds": 3600}
    pricing = {"model": "deepseek-flash", "currency": "USD", "pricing_source_url": "fixture",
        "checked_at_utc": "fixture", "price_period": "fixture", "estimate_quality": "fixture",
        "input_hit_per_million": "0.006", "input_miss_per_million": "0.3", "output_per_million": "1.2"}
    budget = ApprovedProviderBudget(manifest, approval, pricing, controller_commit="fixture",
                                    stage="skill_evolution", model="deepseek-flash",
                                    base_url="https://api.deepseek.com")
    events = []
    original_ensure = budget.ensure_capacity
    def ensure_capacity(calls, input_tokens, completion_tokens):
        events.append(("budget", calls, input_tokens, completion_tokens))
        original_ensure(calls, input_tokens, completion_tokens)
    monkeypatch.setattr(budget, "ensure_capacity", ensure_capacity)
    episode_limits = []
    original_begin_episode = budget.begin_episode
    def begin_episode(**caps):
        episode_limits.append(caps)
        events.append(("episode", caps))
        original_begin_episode(**caps)
    monkeypatch.setattr(budget, "begin_episode", begin_episode)

    class Client:
        max_input_per_request = 100
        max_completion_tokens = 100
        def __init__(self): self.budget = budget
        def choose_action(self, *args):
            return {"raw_output": "Action: look", "input_tokens": 1, "output_tokens": 1}

    class Env:
        def reset(self):
            events.append(("reset",))
            return ["Your task is to: look."], {"extra.gamefile": [pair["gamefile"]],
                                                "admissible_commands": [["look"]]}
        def step(self, actions): return ["Done."], [0], [True], {"won": [True]}
        def close(self): pass

    pair = manifest["plan"]["panel_b_pairs"][0]
    candidate = SimpleNamespace(body="Use visible feedback.", digest="f" * 64, version=1)
    _run_panel_cluster(tmp_path, Client(), pair, {"off": None, "v1": candidate, "v2": candidate},
        split="train_inner", max_steps=2, run_id="triplet", agent_revision="agent",
        benchmark_revision="benchmark", provenance=manifest, on_arm=lambda *_: None,
        environment_factory=lambda *_args, **_kwargs: Env())
    assert events[0] == ("budget", 6, 600, 600)
    assert episode_limits == [{"max_usd": "1.25", "max_calls": 60, "max_wall_seconds": 3600}] * 3
    assert events[1][0] == "episode" and events[2] == ("reset",)
    assert all(events[index][0] == "episode" and events[index + 1] == ("reset",)
               for index in (1, 3, 5))


def test_success_only_bundle_projects_visible_fields_and_rejects_B_sources():
    from experiments.skill_evolution.real_learning import source_bundle_from_receipts
    manifest, split, _, sources, _ = learning_fixture()
    rows = sources[:2]
    for row in rows:
        row["trajectory"] = [{"observation": "You see a soapbar 2 at cabinet 1.", "selected_action": "look",
                              "reasoning_content": "PRIVATE_REASONING", "high_pddl": "EXPERT_SECRET"}]
        row["receipt_sha256"] = manifest_digest({k: v for k, v in row.items() if k != "receipt_sha256"})
    bundle = source_bundle_from_receipts(rows, split, source_mode="SUCCESS_ONLY_INDUCTION")
    assert bundle.failed_trajectory_summaries == ()
    assert "PRIVATE_REASONING" not in repr(bundle) and "EXPERT_SECRET" not in repr(bundle)
    assert "cabinet 1" not in repr(bundle) and "soapbar 2" not in repr(bundle)
    rows[0]["split"] = "train_inner"
    with pytest.raises(ValueError, match="train_dev"):
        source_bundle_from_receipts(rows, split, source_mode="SUCCESS_ONLY_INDUCTION")


@pytest.mark.parametrize("patch_reply", ["valid", "noop", "illegal"])
def test_production_initial_inner_feedback_patch_and_separate_outer_rollback(tmp_path, patch_reply):
    """Actual orchestration and Agent code, injected fixtures, zero network."""
    import json
    from datetime import datetime, timedelta, timezone
    from core.acre.resource_budget import ApprovedProviderBudget
    from core.skills import SkillContract
    from experiments.skill_evolution.real_learning import run_live_learning, run_frozen_outer
    manifest, split, viability, sources, evidence = learning_fixture()
    outer_game = "/data/json_2.1.1/train/outer-template/trial_1/game.tw-pddl"
    outer_member = {"task_id": "outer-template/trial_1", "template_id": "outer-template", "cluster_id": "outer-cluster",
                    "gamefile": outer_game, "split": "train_outer", "identity_status": "IDENTIFIED"}
    split["task_membership"].append(outer_member)
    manifest.update(study_id="SYNTHETIC_FIXTURE", run_id="inner-fixture", controller_commit="fixture",
                    agent_revision="agent", benchmark_revision="benchmark", model="deepseek-flash",
                    source_mode="SUCCESS_ONLY_INDUCTION", split_manifest_sha256=manifest_digest(split))
    sources = sources[:2]
    evidence = replace(evidence, source_mode="SUCCESS_ONLY_INDUCTION", failed_trajectory_summaries=(),
        verified_failure_patterns=(), source_task_ids=tuple(r["task_id"] for r in sources),
        source_template_ids=tuple(r["template_id"] for r in sources), source_cluster_ids=tuple(r["cluster_id"] for r in sources),
        source_trace_ids=tuple(r["trial_id"] for r in sources), source_trace_hashes=tuple(r["receipt_sha256"] for r in sources),
        train_split_manifest_sha256=manifest_digest(split))
    parent = SkillContract(skill_id="fixture.skill", version=1, task_family="clean_place", subtask="clean",
        intent="Complete visible task.", applicability=("clean_place",), procedure=("Inspect visible state.", "Transform and verify."),
        procedure_step_ids=("check", "do"), required_procedure_step_ids=("check",),
        preconditions=("Goal is visible.",), postconditions=("Check official completion.",),
        protected_semantics=("Use only admissible actions.",), source_trial_ids=("scaffold",))
    pricing = {"model": "deepseek-flash", "currency": "USD", "pricing_source_url": "https://fixture.test", "checked_at_utc": "fixture",
               "price_period": "fixture", "estimate_quality": "SYNTHETIC_FIXTURE",
               "input_hit_per_million": "0.006", "input_miss_per_million": "0.3", "output_per_million": "1.2"}

    def approval_for(m):
        return {"provider_enabled": True, "study_id": m["study_id"], "run_id": m["run_id"], "controller_commit": "fixture",
            "model": "deepseek-flash", "approved_manifest_sha256": manifest_digest(m),
            "approved_split_manifest_sha256": m["split_manifest_sha256"], "approved_by": "SYNTHETIC_FIXTURE",
            "approval_reference": "SYNTHETIC_FIXTURE", "approved_at_utc": datetime.now(timezone.utc).isoformat(),
            "expiration_utc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "permitted_stages": [m["stage"]], "api_base_allowlist": ["https://api.deepseek.com"], "max_provider_calls": m["max_calls"],
            "max_input_tokens": 10000, "max_completion_tokens": 10000, "per_episode_max_steps": 2,
            "per_request_max_completion_tokens": 100, "max_usd_per_episode": "1.25",
            "max_calls_per_episode": 60, "max_episode_wall_seconds": 3600,
            "max_total_usd": 1, "max_failed_retries": 0, "max_wall_seconds": 3600}

    sequence, resets = [], []
    class FakeClient:
        evidence_origin = "SYNTHETIC_FIXTURE"
        response_identity = ("deepseek-flash", "fixture")
        def __init__(self, *args, budget, max_completion_tokens, **kwargs):
            self.budget, self.max_completion_tokens = budget, max_completion_tokens
            self.max_input_per_request = 100
        def complete_text(self, messages):
            assert str(self.budget.episode_limits["max_usd"]) == "0.05"
            assert self.budget.episode_limits["max_calls"] == 1
            data = json.loads(messages[1]["content"].split("constraints_and_evidence_json=", 1)[1])
            if "parent_digest" not in data:
                sequence.append("initial")
                value = {"source_trace_ids": data["source_trace_ids"], "intent": parent.intent,
                    "applicability": list(parent.applicability), "procedure": list(parent.procedure),
                    "procedure_step_ids": list(parent.procedure_step_ids), "required_procedure_step_ids": ["check"],
                    "preconditions": list(parent.preconditions), "postconditions": list(parent.postconditions),
                    "protected_semantics": list(parent.protected_semantics), "failure_hypothesis": "Efficiency hypothesis.",
                    "expected_benefit": "Hypothesis only.", "possible_regressions": ["Extra tokens."]}
            else:
                sequence.append("patch")
                assert data["evidence"]["paired_execution_metrics"]
                assert data["evidence"]["source_split"] == "train_inner"
                assert len(data["source_trace_ids"]) == 12
                expected_a = {f"{manifest['run_id']}:{pair['pair_id']}:{arm}"
                              for pair in manifest['plan']['panel_a_pairs'] for arm in ('off', 'on')}
                assert set(data["source_trace_ids"]) == expected_a
                assert all("Visible feedback." in metric for metric in data["evidence"]["paired_execution_metrics"])
                if patch_reply == "noop":
                    value = {"status": "NO_OP", "parent_digest": data["parent_digest"],
                             "source_trace_ids": data["source_trace_ids"]}
                else:
                    value = {"parent_digest": data["parent_digest"], "source_trace_ids": data["source_trace_ids"],
                             "edit_operation": {"op": "replace", "section": "procedure", "index":
                                                (1 if patch_reply == "valid" else 999),
                                                "new_text": "Transform once and verify."},
                             "failure_hypothesis": "Reduce repeated checks.", "expected_benefit": "Hypothesis only.",
                             "possible_regressions": ["Insufficient checks."]}
            self.budget.reserve(20, 20); self.budget.record_usage(20, 20)
            return {"raw_output": json.dumps(value), "finish_reason": "stop", "input_tokens": 20, "output_tokens": 20,
                    "model_response_identity": "deepseek-flash"}
        def choose_action(self, state, admissible, skill, *args):
            sequence.append("on-v2" if skill and "Transform once" in skill else "on-v1" if skill else "off")
            output = 20 if skill and "Transform once" in skill else 40 if skill else 80
            before = self.budget.usd_used
            self.budget.reserve(10, 100); self.budget.record_usage(10, output)
            return {"raw_output": "Action: look", "finish_reason": "stop", "input_tokens": 10, "output_tokens": output,
                    "reasoning_tokens": output - 5, "estimated_provider_usd": str(self.budget.usd_used - before)}

    class Env:
        def __init__(self, game): self.game = game
        def reset(self):
            resets.append(self.game)
            return ["Your task is to: look."], {"extra.gamefile": [self.game], "admissible_commands": [["look"]]}
        def step(self, actions):
            # A real on-arm regression in the simulated outer must cause rollback.
            return ["Visible feedback."], [0], [True], {"won": [not ("outer-template" in self.game and sequence[-1].startswith("on"))]}
        def close(self): pass
    factory = lambda game, **kwargs: Env(game)
    approval = approval_for(manifest)
    budget = ApprovedProviderBudget(manifest, approval, pricing, controller_commit="fixture", stage="skill_evolution",
                                    model="deepseek-flash", base_url="https://api.deepseek.com")
    report = run_live_learning(manifest, split, viability, sources, evidence, parent, budget,
        data_root=tmp_path, output=tmp_path / "inner", api_key="SYNTHETIC_FIXTURE", base_url="https://api.deepseek.com",
        client_factory=FakeClient, environment_factory=factory)
    if patch_reply != "valid":
        expected = "PATCH_NOOP" if patch_reply == "noop" else "REJECTED_PATCH"
        attempt_state = "PATCH_NOOP" if patch_reply == "noop" else "PATCH_REJECTED"
        assert report["status"] == expected and report["patch_attempt_state"] == attempt_state
        assert not (tmp_path / "inner/candidate_v2.json").exists()
        assert not any(row.get("panel") == "B" for row in report.get("paired_receipts", []))
        return
    assert sequence[0] == "initial" and sequence[13] == "patch"
    assert sequence.count("off") == 12 and sequence.count("on-v1") == 12 and sequence.count("on-v2") == 6
    assert len(resets) == 30 and report["inner_selection"]["selected"] == "v2"
    assert report["outer_status"] == "NOT_RUN_REQUIRES_SEPARATE_T3_APPROVAL"
    candidate = SkillContract.from_dict(json.loads((tmp_path / "inner/candidate_v2.json").read_text()))
    initial = SkillContract.from_dict(json.loads((tmp_path / "inner/candidate_v1.json").read_text()))
    inner_rows = report["inner_v1_and_v2_receipts"]
    assert report["panel_b_plan"] == manifest["plan"]["panel_b_pairs"]
    assert report["selected_frozen"]["candidate_digest"] == candidate.digest
    assert {row["panel"] for row in inner_rows} == {"A", "B"}
    assert all({row["arm"] for row in inner_rows if row["panel"] == "B"
                and row["pair_id"] == pair_id} == {"off", "v1", "v2"}
               for pair_id in {row["pair_id"] for row in inner_rows if row["panel"] == "B"})
    extension = {"parent_manifest_sha256": manifest_digest(split), "child_membership": [dict(outer_member,
                  parent_split="train_outer", child_split="outer_skill_validation")]}
    outer = {**manifest, "run_id": "outer-fixture", "stage": "skill_outer_validation", "max_calls": 4,
             "parent_split_manifest_sha256": manifest_digest(split), "split_extension_sha256": manifest_digest(extension),
             "frozen_candidate_sha256": candidate.digest, "selected_inner_version": report["selected_frozen"]["version"],
             "selected_frozen_sha256": manifest_digest(report["selected_frozen"]),
             "inner_receipts_sha256": manifest_digest({"receipts": inner_rows}), "output_dir": str(tmp_path / "outer"),
             "outer_consumption_journal": str(tmp_path / "consumed.jsonl"),
             "plan": {"pair_order_seed": 19, "pairs": [dict(outer_member, pair_id="outer-pair", split="outer_skill_validation", seed=19,
                                     arm_order=["off", "on"], task_type_id=3)]}}
    outer_report = run_frozen_outer(outer, split, extension, candidate, initial, inner_rows,
        report["panel_b_plan"], report["selected_frozen"], approval_for(outer), pricing,
        revision="fixture", base_url="https://api.deepseek.com", api_key="SYNTHETIC_FIXTURE",
        data_root=tmp_path, output=tmp_path / "outer", client_factory=FakeClient, environment_factory=factory)
    assert outer_report["status"] == "REAL_OUTER_REJECTED" and not outer_report["canonical_promotion"]
    assert len(json.loads((tmp_path / "outer/outer_receipts.json").read_text())) == 2
    from experiments.skill_evolution.real_learning import validate_outer_inputs
    with pytest.raises(ValueError, match="consumed"):
        validate_outer_inputs(outer, split, extension, candidate, initial, inner_rows,
            panel_b_pairs=report["panel_b_plan"], selected_frozen=report["selected_frozen"])


def test_pair_order_is_metadata_only_deterministic_and_balanced():
    from experiments.skill_evolution.real_learning import freeze_pair_orders, freeze_triplet_orders
    pairs = [{"pair_id": str(i)} for i in range(6)]
    result = freeze_pair_orders(pairs, 31)
    assert result == freeze_pair_orders(pairs, 31)
    assert sum(row["arm_order"] == ["off", "on"] for row in result) == 3
    changed = [dict(row, future_outcome=False, high_pddl="not a sampler input") for row in pairs]
    assert [row["arm_order"] for row in result] == [row["arm_order"] for row in freeze_pair_orders(changed, 31)]
    triplets = freeze_triplet_orders(pairs, 31)
    assert triplets == freeze_triplet_orders(pairs, 31)
    assert {tuple(row["arm_order"]) for row in triplets} == {
        ("off", "v1", "v2"), ("v1", "v2", "off"), ("v2", "off", "v1"),
        ("off", "v2", "v1"), ("v2", "v1", "off"), ("v1", "off", "v2"),
    }


@pytest.mark.parametrize("pair_count", [6, 8])
def test_long_trace_feedback_stays_bounded_without_dropping_costs(pair_count):
    from core.skills import PairReceipt
    from experiments.skill_evolution.real_learning import _feedback_bundle
    _, _, _, _, source = learning_fixture()
    pairs, raw = [], []
    for index in range(pair_count):
        for arm in ("off", "on"):
            pairs.append(PairReceipt(trial_id=f"inner-{index}-{arm}", pair_id=f"pair-{index}",
                task_id=f"inner-task-{index}", seed=1, arm=arm, split="train_inner",
                skill_digest="a" * 64 if arm == "on" else None, agent_revision="agent", benchmark_revision="benchmark",
                success=True, utility=1.0, verifier_status="verified", hard_gates_passed=True,
                template_id=f"inner-template-{index}", cluster_id=f"inner-cluster-{index}", receipt_sha256="b" * 64))
            raw.append({"pair_id": f"pair-{index}", "arm": arm, "success": True,
                        "input_tokens": 12345, "completion_tokens": 6789, "reasoning_tokens": 6000,
                        "cost_usd": "0.01", "steps": 60, "actions": 60,
                        "feedback_events": [{"observation": "visible feedback " * 50, "selected_action": "look"}
                                             for event in range(60)]})
    feedback = _feedback_bundle(source, pairs, raw)
    feedback.validate(patch_feedback=True)
    assert len(feedback.paired_execution_metrics) == pair_count * 2
    assert all("input_tokens=12345" in item and "full_event_count=60" in item for item in feedback.paired_execution_metrics)
    assert all("visible feedback" in item and "usd=0.01" in item for item in feedback.paired_execution_metrics)
    assert sum(map(len, feedback.paired_execution_metrics)) <= 12000
