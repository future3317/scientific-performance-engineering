from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.acre.resource_budget import ProviderBudgetExceeded, manifest_digest
from core.skills import EvidenceBundle
from experiments.skill_evolution.real_learning import validate_learning_inputs


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
            "identity_status": "IDENTIFIED", "confirmatory_eligible": True,
        })

    pairs = []
    for split, suffix in (("train_inner", "inner"),):
        task_id = f"{suffix}-template/trial_1"
        gamefile = f"/data/json_2.1.1/train/{task_id}/game.tw-pddl"
        row = {"task_id": task_id, "template_id": f"{suffix}-template", "gamefile": gamefile,
               "cluster_id": f"{suffix}-cluster", "split": split,
               "identity_status": "IDENTIFIED", "confirmatory_eligible": True}
        membership.append(row)
        pairs.append({
            "pair_id": f"pair-{suffix}", "task_id": task_id, "template_id": row["template_id"],
            "cluster_id": row["cluster_id"], "split": split, "task_type_id": 3,
            "gamefile": gamefile, "seed": 19, "arm_order": ["off", "on"],
        })

    split_manifest = {"task_membership": membership}
    plan = {"pairs": pairs}
    manifest = {
        "stage": "skill_evolution", "split": "train_inner", "thinking": "enabled",
        "max_retries": 0, "split_manifest_sha256": manifest_digest(split_manifest),
        "plan": plan, "plan_sha256": manifest_digest(plan),
        "viability_receipts_sha256": "b" * 64, "source_receipts_sha256": "c" * 64,
        "max_steps": 2, "max_input_per_request": 100, "max_completion_per_request": 100,
        "proposal_completion_calls": 1, "patch_completion_calls": 1, "max_calls": 10, "agent_sha": "a" * 64,
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
    assert plan["planned_provider_calls"] == 10
    assert plan["outer_pairs"] == 0
    assert plan["paired_clusters"] == {"train_inner": ["inner-cluster"]}


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


def test_wrong_native_gamefile_cannot_be_labeled_as_a_different_planned_task():
    manifest, split, viability, sources, evidence = learning_fixture()
    manifest["plan"]["pairs"][0]["gamefile"] = "/data/json_2.1.1/train/other/trial_1/game.tw-pddl"
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
    pair = manifest["plan"]["pairs"][0]
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
    assert [arm for arm, _ in persisted] == ["off", "on"]
    assert insertions == [None, candidate.body]
    assert len(closed) == 2
    assert on["skill_version_sha256"] == candidate.digest
    assert on["split"] == "train_inner" and on["trajectory"][0]["selected_action"] == "look"


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


def test_production_initial_inner_feedback_patch_and_separate_outer_rollback(tmp_path):
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
            "per_request_max_completion_tokens": 100, "max_total_usd": 1, "max_failed_retries": 0, "max_wall_seconds": 3600}

    sequence, resets = [], []
    class FakeClient:
        evidence_origin = "SYNTHETIC_FIXTURE"
        response_identity = ("deepseek-flash", "fixture")
        def __init__(self, *args, budget, max_completion_tokens, **kwargs):
            self.budget, self.max_completion_tokens = budget, max_completion_tokens
            self.max_input_per_request = 100
        def complete_text(self, messages):
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
                value = {"parent_digest": data["parent_digest"], "source_trace_ids": data["source_trace_ids"],
                         "edit_operation": {"op": "replace", "section": "procedure", "index": 1, "new_text": "Transform once and verify."},
                         "failure_hypothesis": "Reduce repeated checks.", "expected_benefit": "Hypothesis only.", "possible_regressions": ["Insufficient checks."]}
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
    assert sequence == ["initial", "off", "on-v1", "patch", "off", "on-v2"]
    assert len(resets) == 4 and report["inner_selection"]["selected"] == "v2"
    assert report["outer_status"] == "NOT_RUN_REQUIRES_SEPARATE_T3_APPROVAL"
    candidate = SkillContract.from_dict(json.loads((tmp_path / "inner/candidate_v2.json").read_text()))
    initial = SkillContract.from_dict(json.loads((tmp_path / "inner/candidate_v1.json").read_text()))
    inner_rows = report["inner_v1_and_v2_receipts"]
    extension = {"parent_manifest_sha256": manifest_digest(split), "child_membership": [dict(outer_member,
                  parent_split="train_outer", child_split="outer_skill_validation")]}
    outer = {**manifest, "run_id": "outer-fixture", "stage": "skill_outer_validation", "max_calls": 4,
             "parent_split_manifest_sha256": manifest_digest(split), "split_extension_sha256": manifest_digest(extension),
             "frozen_candidate_sha256": candidate.digest, "selected_inner_version": "v2",
             "inner_receipts_sha256": manifest_digest(inner_rows), "output_dir": str(tmp_path / "outer"),
             "outer_consumption_journal": str(tmp_path / "consumed.jsonl"),
             "plan": {"pairs": [dict(outer_member, pair_id="outer-pair", split="outer_skill_validation", seed=19,
                                     arm_order=["off", "on"], task_type_id=3)]}}
    outer_report = run_frozen_outer(outer, split, extension, candidate, initial, inner_rows,
        approval_for(outer), pricing, revision="fixture", base_url="https://api.deepseek.com", api_key="SYNTHETIC_FIXTURE",
        data_root=tmp_path, output=tmp_path / "outer", client_factory=FakeClient, environment_factory=factory)
    assert outer_report["status"] == "REAL_OUTER_REJECTED" and not outer_report["canonical_promotion"]
    assert len(json.loads((tmp_path / "outer/outer_receipts.json").read_text())) == 2
    from experiments.skill_evolution.real_learning import validate_outer_inputs
    with pytest.raises(ValueError, match="consumed"):
        validate_outer_inputs(outer, split, extension, candidate, initial, inner_rows)
