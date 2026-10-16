import hashlib
import json

import pytest

from core.skills import EvidenceBundle, ModelSkillProposer, PairReceipt, Patch, SkillContract, verify_inner_pairs


SHA = "a" * 64


def evidence(**updates):
    values = {
        "source_split": "train_dev",
        "source_mode": "CONTRASTIVE_REPAIR",
        "source_task_ids": ("task-source-1", "task-source-2"),
        "source_template_ids": ("template-source-1", "template-source-2"),
        "source_cluster_ids": ("cluster-source-1", "cluster-source-2"),
        "source_trace_ids": ("trace-source-1", "trace-source-2"),
        "source_trace_hashes": (SHA, "b" * 64),
        "train_split_manifest_sha256": "c" * 64,
        "task_family": "clean_place",
        "agent_hash": "agent-r1",
        "successful_trajectory_summaries": ("Verified completion after checking visible state.",),
        "failed_trajectory_summaries": ("Verified failure after an unconfirmed transformation.",),
        "verified_failure_patterns": ("unconfirmed transformation",),
        "redaction_report_sha256": "d" * 64,
    }
    return EvidenceBundle(**(values | updates))


class MockCompletion:
    max_completion_tokens = 128

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def complete_text(self, messages):
        self.calls += 1
        return {
            "raw_output": json.dumps(self.response),
            "reasoning_content": "must never be parsed as proposal text",
            "input_tokens": 10,
            "output_tokens": 20,
            "model_response_identity": "mock-model-v1",
            "finish_reason": "stop",
        }


def initial_response():
    return {
        "source_mode": "CONTRASTIVE_REPAIR",
        "source_trace_ids": ["trace-source-1"],
        "intent": "Complete cleaning tasks with visible checks.",
        "applicability": ["ALFWorld clean_place tasks"],
        "procedure": ["Locate the object.", "Acquire and verify it.", "Clean and verify it.", "Place and verify it."],
        "procedure_step_ids": ["locate", "acquire", "clean", "place"],
        "required_procedure_step_ids": ["locate", "acquire", "clean", "place"],
        "preconditions": ["Goal and admissible actions are visible."],
        "postconditions": ["Environment verifier confirms task completion."],
        "protected_semantics": ["Use only admissible actions."],
        "failure_hypothesis": "A transform may be skipped without checking state.",
        "expected_benefit": "Hypothesis only.",
        "possible_regressions": ["Extra checks may consume steps."],
    }


def parent():
    return SkillContract(
        skill_id="alfworld.clean_place", version=1, task_family="clean_place",
        subtask="clean_and_place", intent="Complete cleaning tasks with visible checks.",
        applicability=("ALFWorld clean_place tasks",),
        procedure=("Locate the object.", "Acquire and verify it.", "Clean and verify it.", "Place and verify it."),
        procedure_step_ids=("locate", "acquire", "clean", "place"),
        required_procedure_step_ids=("locate", "acquire", "clean", "place"),
        preconditions=("Goal and admissible actions are visible.",),
        postconditions=("Environment verifier confirms task completion.",),
        protected_semantics=("Use only admissible actions.",),
        source_trial_ids=("trace-parent",),
    )


def test_initial_proposal_uses_final_content_and_binds_train_inner_provenance():
    client = MockCompletion(initial_response())
    skill = ModelSkillProposer(client, max_completion_calls=1, mock_mode=True).propose_initial(
        evidence(), skill_id="alfworld.clean_place", subtask="clean_and_place", constraints=("One page.",),
    )
    assert client.calls == 1
    assert skill.source_split == "train_dev"
    assert skill.source_mode == "CONTRASTIVE_REPAIR"
    assert skill.source_task_ids == evidence().source_task_ids
    assert skill.source_template_ids == evidence().source_template_ids
    assert skill.source_trace_hashes == evidence().source_trace_hashes
    assert skill.proposal_prompt_sha256 and skill.proposal_output_sha256
    assert skill.model_identity == "mock-model-v1"
    assert "reasoning" not in skill.body


@pytest.mark.parametrize("changes, message", [
    ({"source_split": "train_outer"}, "train_inner"),
    ({"teacher_only_source": True}, "teacher-only"),
    ({"uses_hidden_task_state": True}, "hidden task state"),
    ({"unknown_usage_count": 1}, "unknown provider usage"),
    ({"success_labels_origin": "model_judgment"}, "environment verifier"),
])
def test_unqualified_evidence_is_rejected_before_completion(changes, message):
    client = MockCompletion(initial_response())
    proposer = ModelSkillProposer(client, max_completion_calls=1, mock_mode=True)
    with pytest.raises(ValueError, match=message):
        proposer.propose_initial(evidence(**changes), skill_id="alfworld.clean_place",
                                 subtask="clean_and_place", constraints=())
    assert client.calls == 0


def test_real_mode_requires_the_shared_approved_budget_client():
    with pytest.raises(ValueError, match="ApprovedProviderBudget"):
        ModelSkillProposer(MockCompletion(initial_response()), max_completion_calls=1)


def test_zero_completion_cap_never_calls_even_a_mock_client():
    client = MockCompletion(initial_response())
    proposer = ModelSkillProposer(client, max_completion_calls=0, mock_mode=True)
    with pytest.raises(ValueError, match="budget is zero"):
        proposer.propose_initial(evidence(), skill_id="alfworld.clean_place",
                                 subtask="clean_and_place", constraints=())
    assert client.calls == 0


def test_patch_rejects_out_of_bundle_source_and_protected_step_deletion():
    poisoned = {
        "parent_digest": parent().digest,
        "source_trace_ids": ["outside-trace"], "failure_hypothesis": "failure",
        "edit_operation": {"op": "delete", "section": "procedure", "index": 0},
        "expected_benefit": "maybe", "possible_regressions": [],
    }
    client = MockCompletion(poisoned)
    recorded = []
    proposer = ModelSkillProposer(client, max_completion_calls=1, mock_mode=True,
                                  on_completion=recorded.append)
    with pytest.raises(ValueError, match="outside its train_inner"):
        proposer.propose_patch(parent(), evidence(source_split="train_inner"))
    assert recorded[0]["status"] == "received_unvalidated"
    assert json.loads(recorded[0]["raw_output"])["source_trace_ids"] == ["outside-trace"]

    poisoned["source_trace_ids"] = ["trace-source-1"]
    client = MockCompletion(poisoned)
    proposer = ModelSkillProposer(client, max_completion_calls=1, mock_mode=True)
    with pytest.raises(ValueError, match="required_procedure_step_removed"):
        proposer.propose_patch(parent(), evidence(source_split="train_inner"))


def test_proposal_rejects_private_paths_instance_ids_and_oracle_terms():
    for text in ("Read /home/private/answer.txt", "Use cabinet 12 as the target", "Copy the oracle plan"):
        response = initial_response()
        response["procedure"][0] = text
        proposer = ModelSkillProposer(MockCompletion(response), max_completion_calls=1, mock_mode=True)
        with pytest.raises(ValueError):
            proposer.propose_initial(evidence(), skill_id="alfworld.clean_place",
                                     subtask="clean_and_place", constraints=())


def test_inner_verifier_rejects_source_and_reused_templates():
    skill = SkillContract.from_dict({
        **parent().to_dict(), "source_split": "train_inner",
        "source_task_ids": ["source-task"], "source_template_ids": ["source-template"],
        "source_cluster_ids": ["source-cluster"],
        "source_trace_hashes": [SHA],
    })

    def pair(template, task="eval-task", pair_id="pair", cluster="eval-cluster"):
        return [
            PairReceipt("off-" + pair_id, pair_id, task, 1, "off", "train_inner", None,
                        "agent-r1", "bench-r1", False, 0.0, template_id=template, cluster_id=cluster),
            PairReceipt("on-" + pair_id, pair_id, task, 1, "on", "train_inner", skill.digest,
                        "agent-r1", "bench-r1", True, 1.0, template_id=template, cluster_id=cluster),
        ]

    with pytest.raises(ValueError, match="overlaps source"):
        verify_inner_pairs(skill, pair("source-template"))
    receipts = pair("fresh-template", pair_id="one") + pair("fresh-template", task="eval-task-2", pair_id="two")
    with pytest.raises(ValueError, match="duplicate independent task template"):
        verify_inner_pairs(skill, receipts)
    with pytest.raises(ValueError, match="semantic cluster"):
        verify_inner_pairs(skill, pair("fresh-template", cluster="source-cluster"))


def test_real_path_shaped_task_ids_are_metadata_not_governance_identifiers():
    contract = SkillContract.from_dict({
        **parent().to_dict(), "source_split": "train_inner",
        "source_task_ids": ["template/trial_0042"],
        "source_template_ids": ["template"], "source_cluster_ids": ["semantic-cluster"],
    })
    assert contract.source_task_ids == ("template/trial_0042",)


def test_viability_only_approval_cannot_authorize_a_skill_proposal():
    from types import SimpleNamespace
    from core.acre.resource_budget import ApprovedProviderBudget
    class ViabilityBudget(ApprovedProviderBudget):
        def __init__(self):
            self.stage = "train_viability"
    client = SimpleNamespace(budget=ViabilityBudget(), max_completion_tokens=100)
    with pytest.raises(ValueError, match="skill_evolution stage"):
        ModelSkillProposer(client, max_completion_calls=1)
