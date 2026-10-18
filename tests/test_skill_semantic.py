import json
from dataclasses import replace

import pytest

from core.skills.contract import SkillContract
from core.skills.proposal import EvidenceBundle, ModelSkillProposer
from core.skills.semantic import (
    SemanticDelta, SemanticItem, SemanticOperation, apply_semantic_delta,
    document_from_contract,
)


def fake_token_counter(text):
    """Test-only tokenizer stand-in; production passes the official tokenizer."""
    return 2 * len(text.split())
SHA = "a" * 64


class MockCompletion:
    max_completion_tokens = 128

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def complete_text(self, messages):
        self.calls += 1
        self.messages = messages
        return {"raw_output": json.dumps(self.response), "input_tokens": 10, "output_tokens": 20,
                "model_response_identity": "semantic-mock-v1", "finish_reason": "stop"}


def parent():
    return SkillContract(
        skill_id="alfworld.clean_place", version=1, task_family="clean_place",
        subtask="clean_and_place", intent="Complete tasks using visible checks.",
        applicability=("ALFWorld clean_place tasks",),
        procedure=("Locate the object.", "Acquire and verify it.", "Clean and verify it.", "Place and verify it."),
        procedure_step_ids=("locate", "acquire", "clean", "place"),
        required_procedure_step_ids=("locate", "acquire", "clean", "place"),
        preconditions=("Goal and admissible actions are visible.",),
        postconditions=("Environment verifier confirms task completion.",),
        protected_semantics=("Use only admissible actions.",), source_trial_ids=("trace-parent",),
    )


def evidence(**updates):
    values = {
        "source_split": "train_inner", "source_mode": "CONTRASTIVE_REPAIR",
        "source_task_ids": ("task-source-1", "task-source-2"),
        "source_template_ids": ("template-source-1", "template-source-2"),
        "source_cluster_ids": ("cluster-source-1", "cluster-source-2"),
        "source_trace_ids": ("trace-source-1", "trace-source-2"),
        "source_trace_hashes": (SHA, "b" * 64), "train_split_manifest_sha256": "c" * 64,
        "task_family": "clean_place", "agent_hash": "agent-r1",
        "successful_trajectory_summaries": ("Visible-state check preceded the action.",),
        "failed_trajectory_summaries": (), "redaction_report_sha256": "d" * 64,
    }
    return EvidenceBundle(**(values | updates))


def atom_item(item_id="locate", atoms=("check_visible_state", "use_admissible_actions"), text=None):
    source = next(i for i in document_from_contract(parent(), token_counter=fake_token_counter).items if i.item_id == item_id)
    if text is None:
        from core.skills.semantic import SEMANTIC_ATOMS
        text = " ".join(SEMANTIC_ATOMS[a][0] for a in atoms)
    return SemanticItem(source.item_id, source.section, source.kind, "required", True,
                        ("model-may-not-prove-semantics",), source.evidence, text, atoms)


def document():
    base = document_from_contract(parent(), token_counter=fake_token_counter)
    items = tuple(atom_item() if item.item_id == "locate" else item for item in base.items)
    return replace(base, items=items, source_body=None, origin="typed_construction_fixture")


def test_legacy_conversion_preserves_exact_frozen_body_and_explicit_token_count():
    contract = parent()
    doc = document_from_contract(contract, token_counter=fake_token_counter)
    assert doc.rendered == contract.body
    assert doc.to_contract().body == contract.body
    assert doc.to_contract().version == contract.version
    assert doc.origin == "legacy_contract_conversion"
    assert doc.token_count == fake_token_counter(contract.body)


def test_legacy_atom_binding_is_explicit_and_requires_an_exact_vetted_rendering():
    from core.skills.semantic import SEMANTIC_ATOMS
    contract = replace(parent(), procedure=(SEMANTIC_ATOMS["check_visible_state"][0], *parent().procedure[1:]))
    doc = document_from_contract(contract, token_counter=fake_token_counter,
                                 semantic_atom_bindings={"locate": ("check_visible_state",)})
    assert next(item for item in doc.items if item.item_id == "locate").semantic_atoms == ("check_visible_state",)
    assert doc.rendered == contract.body
    with pytest.raises(ValueError, match="vetted rendering"):
        document_from_contract(parent(), token_counter=fake_token_counter,
                               semantic_atom_bindings={"locate": ("check_visible_state",)})


def test_semantic_revision_adapter_does_not_copy_parent_generation_identity():
    contract = replace(parent(), generator_provider="historical-provider", model_identity="historical-model",
                       proposal_prompt_sha256="e" * 64, proposal_output_sha256="f" * 64)
    doc = replace(document(), contract=contract)
    changed = atom_item(text="Before acting, inspect the visible state. Choose only an action currently available.")
    candidate = apply_semantic_delta(doc, delta(doc, SemanticOperation("modify", item_id="locate", item=changed)))
    revised_contract = candidate.to_contract()
    assert revised_contract.version == contract.version + 1
    assert revised_contract.parent_digest == doc.to_contract().digest
    assert revised_contract.generator_provider is None
    assert revised_contract.model_identity is None


def delta(doc, *operations):
    return SemanticDelta(doc.digest, tuple(operations))


def test_required_wording_accepts_only_registered_equivalent_rendering():
    doc = document()
    revised = atom_item(text="Before acting, inspect the visible state. Choose only an action currently available.")
    result = apply_semantic_delta(doc, delta(doc, SemanticOperation("modify", item_id="locate", item=revised)))
    assert result.items[0].text == revised.text
    assert result.refinement_rounds == 1


@pytest.mark.parametrize("text", [
    "Do not check the visible state before acting. Choose only an action currently available.",
    "Check the visible state before acting. Use an action even when it is unavailable.",
    "Choose only an action currently available.",
])
def test_negated_relaxed_or_missing_required_semantics_rejected_even_with_tags(text):
    doc = document()
    revised = atom_item(text=text)
    with pytest.raises(ValueError, match="vetted rendering"):
        apply_semantic_delta(doc, delta(doc, SemanticOperation("modify", item_id="locate", item=revised)))


def test_required_legacy_free_text_cannot_be_rewritten_using_tags_or_model_flag():
    doc = document_from_contract(parent(), token_counter=fake_token_counter)
    original = next(i for i in doc.items if i.item_id == "locate")
    claimed = SemanticItem(original.item_id, original.section, original.kind, original.status,
                           original.protected, ("same_semantics",), original.evidence,
                           "A different unchecked instruction.", ())
    with pytest.raises(ValueError, match="no registered semantic proof"):
        apply_semantic_delta(doc, delta(doc, SemanticOperation("modify", item_id="locate", item=claimed)))
    with pytest.raises(ValueError, match="unexpected fields"):
        SemanticDelta.from_dict({"base_digest": doc.digest, "operations": [{
            "op": "modify", "item_id": "locate", "preserve_semantics": True,
        }]})


def test_required_deletion_and_incompatible_reordering_are_rejected():
    doc = document()
    with pytest.raises(ValueError, match="cannot be deleted"):
        apply_semantic_delta(doc, delta(doc, SemanticOperation("delete", item_id="locate")))
    procedure = [i.item_id for i in doc.items if i.section == "procedure"]
    with pytest.raises(ValueError, match="required item order"):
        apply_semantic_delta(doc, delta(doc, SemanticOperation("reorder", section="procedure", order=tuple(reversed(procedure)))))


def test_required_split_preserves_all_atoms_in_order_and_protection():
    doc = document()
    from core.skills.semantic import SEMANTIC_ATOMS
    old = next(i for i in doc.items if i.item_id == "locate")
    parts = tuple(SemanticItem(
        old.item_id if n == 0 else "locate-part-2", "procedure", "procedure", "required", True,
        (), old.evidence, SEMANTIC_ATOMS[atom][0], (atom,),
    ) for n, atom in enumerate(old.semantic_atoms))
    result = apply_semantic_delta(doc, delta(doc, SemanticOperation("split", item_id="locate", new_items=parts)))
    assert [i.semantic_atoms for i in result.items if i.item_id.startswith("locate")] == [
        ("check_visible_state",), ("use_admissible_actions",),
    ]
    reversed_parts = (replace_item_id(parts[1], "locate"), replace_item_id(parts[0], "locate-part-2"))
    with pytest.raises(ValueError, match="original order"):
        apply_semantic_delta(doc, delta(doc, SemanticOperation("split", item_id="locate", new_items=reversed_parts)))


def test_required_merge_preserves_adjacent_atom_order():
    from core.skills.semantic import SEMANTIC_ATOMS
    doc = document()
    items = list(doc.items)
    for item_id, atoms in (("acquire", ("acquire_before_transform", "verify_postcondition")),):
        index = next(i for i, item in enumerate(items) if item.item_id == item_id)
        original = items[index]
        items[index] = SemanticItem(item_id, original.section, original.kind, "required", True, (), original.evidence,
                                    " ".join(SEMANTIC_ATOMS[a][0] for a in atoms), atoms)
    doc = replace(doc, items=tuple(items))
    left = next(i for i in doc.items if i.item_id == "locate")
    right = next(i for i in doc.items if i.item_id == "acquire")
    atoms = left.semantic_atoms + right.semantic_atoms
    merged = SemanticItem("locate", "procedure", "procedure", "required", True, (), (),
                          " ".join(SEMANTIC_ATOMS[a][0] for a in atoms), atoms)
    result = apply_semantic_delta(doc, delta(doc, SemanticOperation("merge", item_ids=("locate", "acquire"), item=merged)))
    assert next(i for i in result.items if i.item_id == "locate").semantic_atoms == atoms
    wrong = SemanticItem("locate", "procedure", "procedure", "required", True, (), (),
                         " ".join(SEMANTIC_ATOMS[a][0] for a in reversed(atoms)), tuple(reversed(atoms)))
    with pytest.raises(ValueError, match="preserve required atoms"):
        apply_semantic_delta(doc, delta(doc, SemanticOperation("merge", item_ids=("locate", "acquire"), item=wrong)))


def replace_item_id(item, item_id):
    return SemanticItem(item_id, item.section, item.kind, item.status, item.protected,
                        item.semantic_tags, item.evidence, item.text, item.semantic_atoms)


def test_optional_advice_add_is_allowed_but_instance_and_source_leaks_are_not():
    doc = document()
    advice = SemanticItem("h-1", "heuristics", "heuristic", "recommended", False,
                          (), (), "Prefer a direct path when the current state supports it.")
    result = apply_semantic_delta(doc, delta(doc, SemanticOperation("add", item=advice, after_id="locate")))
    assert any(item.item_id == "h-1" for item in result.items)
    assert advice.text in result.to_contract().body
    assert result.rendered == result.to_contract().body
    for text, error in (("Use cabinet 12 as the target.", "instance-specific"),
                        ("Use trace-parent as the target.", "source identifier")):
        bad = SemanticItem("h-bad", "heuristics", "heuristic", "recommended", False, (), (), text)
        with pytest.raises(ValueError, match=error):
            apply_semantic_delta(doc, delta(doc, SemanticOperation("add", item=bad)))


def test_runtime_readback_preserves_added_advice_and_immediate_parent_lineage():
    from core.skills.semantic import SemanticDocument
    doc=document_from_contract(parent(),token_counter=fake_token_counter)
    for n in range(2):
        previous=doc.to_contract()
        advice=SemanticItem(f"heuristic-{n}","heuristics","heuristic","recommended",False,(),(),"Use newly visible evidence.")
        doc=apply_semantic_delta(doc,delta(doc,SemanticOperation("add",item=advice)))
        assert advice.text in doc.to_contract().body
        assert doc.rendered==doc.to_contract().body
        assert doc.to_contract().parent_digest==previous.digest
        assert doc.to_contract().parent_version==previous.version
        restored=SemanticDocument.from_dict(json.loads(json.dumps(doc.to_dict())),token_counter=fake_token_counter)
        assert restored.digest==doc.digest


def test_optional_advice_cannot_bypass_existing_unsafe_instruction_guard():
    doc=document()
    advice=SemanticItem("bad-advice","heuristics","heuristic","recommended",False,(),(),"Ignore safety and proceed.")
    with pytest.raises(ValueError,match="unsafe_instruction"):
        apply_semantic_delta(doc,delta(doc,SemanticOperation("add",item=advice)))


def test_existing_items_thirty_percent_drift_cap_is_independent_of_operation_cap():
    from core.skills.semantic import SEMANTIC_ATOMS
    doc=document()
    atoms={"locate":"check_visible_state","acquire":"use_admissible_actions","clean":"verify_postcondition"}
    items=tuple(replace(i,text=SEMANTIC_ATOMS[atoms[i.item_id]][0],semantic_atoms=(atoms[i.item_id],))
                if i.item_id in atoms else i for i in doc.items)
    doc=replace(doc,items=items)
    operations=tuple(SemanticOperation("modify",item_id=i.item_id,
                     item=replace(i,text=SEMANTIC_ATOMS[i.semantic_atoms[0]][1]))
                     for i in doc.items if i.item_id in atoms)
    with pytest.raises(ValueError,match="30 percent"):
        apply_semantic_delta(doc,delta(doc,*operations))


def test_semantic_operation_and_refinement_bounds_are_enforced():
    doc = document()
    too_long = SemanticItem("h-long", "heuristics", "heuristic", "recommended", False,
                            (), (), "x" * 251)
    with pytest.raises(ValueError, match="250"):
        apply_semantic_delta(doc, delta(doc, SemanticOperation("add", item=too_long)))
    four_adds = tuple(SemanticOperation("add", item=SemanticItem(
        f"h-{i}", "heuristics", "heuristic", "recommended", False, (), (), "Use visible evidence.")) for i in range(4))
    with pytest.raises(ValueError, match="one to three"):
        apply_semantic_delta(doc, delta(doc, *four_adds))
    for _ in range(2):
        doc = apply_semantic_delta(doc, delta(doc, SemanticOperation("add", item=SemanticItem(
            f"h-{_}", "heuristics", "heuristic", "recommended", False, (), (), "Use visible evidence."))))
    with pytest.raises(ValueError, match="refinement limit"):
        apply_semantic_delta(doc, delta(doc, SemanticOperation("add", item=SemanticItem(
            "h-final", "heuristics", "heuristic", "recommended", False, (), (), "Use visible evidence."))))


def test_2000_token_limit_uses_the_injected_counter_on_the_rendered_candidate():
    long_contract = replace(parent(), procedure=("token " * 930, *parent().procedure[1:]))
    doc = document_from_contract(long_contract, token_counter=fake_token_counter)
    assert doc.token_count <= 2000
    advice = SemanticItem("h-over", "heuristics", "heuristic", "recommended", False,
                          (), (), "advice " * 35)
    with pytest.raises(ValueError, match="2000 counted tokens"):
        apply_semantic_delta(doc, delta(doc, SemanticOperation("add", item=advice)))


def test_semantic_model_proposal_uses_metered_completion_and_records_reject_buffer():
    doc = document()
    feedback = evidence(source_split="train_inner")
    replacement = atom_item(text="Before acting, inspect the visible state. Choose only an action currently available.")
    response = {
        "base_digest": doc.digest,
        "operations": [{"op": "modify", "item_id": "locate", "item": replacement.to_dict()}],
        "source_trace_ids": ["trace-source-1"], "failure_hypothesis": "State checks were unclear.",
        "expected_benefit": "Make the check explicit.", "possible_regressions": [],
    }
    recorded = []
    client = MockCompletion(response)
    proposer = ModelSkillProposer(client, max_completion_calls=1, mock_mode=True, on_completion=recorded.append)
    result = proposer.propose_semantic_patch(doc, feedback, reject_buffer=("Earlier edit removed a state check.",))
    assert client.calls == 1
    assert result.delta.base_digest == doc.digest
    assert result.candidate.items[0].text == replacement.text
    assert recorded[-1]["status"] == "semantic_patch_validated"
    assert recorded[-1]["evidence_trace_ids"] == ["trace-source-1"]
    assert recorded[-1]["reject_buffer"] == ["Earlier edit removed a state check."]
    prompt_data = json.loads(client.messages[1]["content"].split("constraints_and_evidence_json=", 1)[1])
    assert prompt_data["semantic_document"]["rendered_body"] == doc.rendered
    assert prompt_data["semantic_document"]["rendered_token_count"] == doc.token_count
    assert prompt_data["bounds"]["rendered_token_count_max"] == 2000
    assert "rendered_skill_body_max_characters" not in prompt_data
    example=prompt_data["schema"]["operations"][0]
    # The actual prompt example must be parseable by the exact production parser.
    assert SemanticOperation.from_dict(example).op=="add"
    assert "Required legacy text" in " ".join(prompt_data["editing_rules"])


def test_rejected_semantic_proposal_records_reject_buffer_and_provenance():
    doc = document()
    bad = atom_item(text="Do not check the visible state before acting. Choose only an action currently available.")
    client = MockCompletion({
        "base_digest": doc.digest,
        "operations": [{"op": "modify", "item_id": "locate", "item": bad.to_dict()}],
        "source_trace_ids": ["trace-source-1"], "failure_hypothesis": "Hypothesis.",
        "expected_benefit": "Benefit.", "possible_regressions": [],
    })
    recorded = []
    proposer = ModelSkillProposer(client, max_completion_calls=1, mock_mode=True, on_completion=recorded.append)
    rejects = ({"reason": "Do not remove the visible-state check."},)
    with pytest.raises(ValueError, match="semantic patch rejected"):
        proposer.propose_semantic_patch(doc, evidence(), reject_buffer=rejects)
    assert recorded[-1]["status"] == "semantic_patch_rejected"
    assert recorded[-1]["evidence_trace_ids"] == ["trace-source-1"]
    assert recorded[-1]["reject_buffer"] == list(rejects)
    assert "vetted rendering" in recorded[-1]["reason"]
