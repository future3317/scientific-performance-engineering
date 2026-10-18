import json
from dataclasses import replace

import pytest

from core.skills import SkillContract, certificate_key, export_skill_pack, read_skill_pack


def contract():
    return SkillContract(
        skill_id="alfworld.clean_place", version=2, task_family="clean_place",
        subtask="verify_transform", intent="Verify a visible transformation before delivery.",
        applicability=("ALFWorld clean_place tasks",),
        procedure=("Check the visible carried-object state.", "Continue only when the required state is confirmed."),
        procedure_step_ids=("check", "continue"), required_procedure_step_ids=("check",),
        preconditions=("The task goal is visible.",),
        postconditions=("The environment verifier confirms completion.",),
        protected_semantics=("Use only admissible actions.",), source_trial_ids=("source-trace-1",),
        source_split="train_inner", source_task_ids=("source-task-1",),
        source_template_ids=("source-template-1",), source_trace_hashes=("a" * 64,),
        source_cluster_ids=("source-cluster-1",),
        train_split_manifest_sha256="b" * 64, generator_provider="deepseek",
        model_identity="deepseek-flash-2026-10", proposal_prompt_sha256="c" * 64,
        proposal_output_sha256="d" * 64, runtime_agent_sha="agent-r1",
    )


def test_skill_pack_exports_standard_frontmatter_and_reads_back(tmp_path):
    root = export_skill_pack(
        contract(), tmp_path / "pack", agent_revision="agent-r1",
        model_identity="deepseek-flash-2026-10", verifier_revision="verifier-r2",
        task_population="alfworld-train-inner-v2",
    )
    skill_md = (root / "SKILL.md").read_text(encoding="utf-8")
    assert skill_md.startswith('---\nname: alfworld-clean-place\ndescription: "')
    assert not (root / "scripts").exists()
    assert read_skill_pack(root).digest == contract().digest
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_task_ids"] == ["source-task-1"]
    assert manifest["source_template_ids"] == ["source-template-1"]
    assert manifest["source_cluster_ids"] == ["source-cluster-1"]
    assert manifest["source_trace_hashes"] == ["a" * 64]
    assert manifest["scripts_allowed"] is False


def test_pack_readback_rejects_manifest_or_content_changes(tmp_path):
    root = export_skill_pack(
        contract(), tmp_path / "pack", agent_revision="agent-r1",
        model_identity="deepseek-flash-2026-10", verifier_revision="verifier-r2",
        task_population="alfworld-train-inner-v2",
    )
    (root / "SKILL.md").write_text((root / "SKILL.md").read_text() + "Changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frontmatter or rendered body|digest mismatch"):
        read_skill_pack(root)


def test_pack_refuses_overwrite_and_non_train_candidates(tmp_path):
    destination = tmp_path / "pack"
    export_skill_pack(
        contract(), destination, agent_revision="agent-r1",
        model_identity="deepseek-flash-2026-10", verifier_revision="verifier-r2",
        task_population="alfworld-train-inner-v2",
    )
    with pytest.raises(FileExistsError):
        export_skill_pack(
            contract(), destination, agent_revision="agent-r1",
            model_identity="deepseek-flash-2026-10", verifier_revision="verifier-r2",
            task_population="alfworld-train-inner-v2",
        )
    with pytest.raises(ValueError, match="only candidate"):
        export_skill_pack(
            replace(contract(), status="rejected"), tmp_path / "rejected",
            agent_revision="agent-r1", model_identity="deepseek-flash-2026-10",
            verifier_revision="verifier-r2", task_population="alfworld-sealed-v2",
        )


def test_certificate_identity_changes_with_content_agent_model_or_population():
    skill = contract()
    base = certificate_key(
        skill, agent_revision="agent-r1", model_identity="model-r1",
        verifier_revision="verifier-r1", task_population="population-r1",
    )
    variants = (
        certificate_key(replace(skill, intent=skill.intent + " Confirm before delivery."),
                        agent_revision="agent-r1", model_identity="model-r1",
                        verifier_revision="verifier-r1", task_population="population-r1"),
        certificate_key(skill, agent_revision="agent-r2", model_identity="model-r1",
                        verifier_revision="verifier-r1", task_population="population-r1"),
        certificate_key(skill, agent_revision="agent-r1", model_identity="model-r2",
                        verifier_revision="verifier-r1", task_population="population-r1"),
        certificate_key(skill, agent_revision="agent-r1", model_identity="model-r1",
                        verifier_revision="verifier-r2", task_population="population-r1"),
        certificate_key(skill, agent_revision="agent-r1", model_identity="model-r1",
                        verifier_revision="verifier-r1", task_population="population-r2"),
    )
    assert all(item.digest != base.digest for item in variants)
