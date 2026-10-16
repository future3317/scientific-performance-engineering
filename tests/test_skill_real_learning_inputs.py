from dataclasses import replace

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
