from __future__ import annotations

import hashlib

from experiments.external_real.skill_pipeline import FrozenSkill, freeze_skill
from experiments.external_real.support import fit_support_model


def _row(task_id: str, family: str, effect: float, seed: int) -> dict[str, object]:
    return {
        "task_id": task_id,
        "repeat_seed": seed,
        "context": {"task_family": family},
        "effect": effect,
        "execution_valid": True,
        "scientific_gate": True,
    }


def test_frozen_skill_hash_binds_body_scripts_resources_and_semantic_abstract() -> None:
    skill = freeze_skill(
        body="Use the task grammar before acting.",
        scripts=("helper.py",),
        resources=("scienceworld-actions.md",),
        semantic_abstract="scienceworld action planning",
        benchmark="scienceworld",
        generation_seed=2,
    )

    expected = hashlib.sha256(
        b'{"benchmark":"scienceworld","body":"Use the task grammar before acting.","generation_seed":2,"resources":["scienceworld-actions.md"],"scripts":["helper.py"],"semantic_abstract":"scienceworld action planning"}'
    ).hexdigest()
    assert isinstance(skill, FrozenSkill)
    assert skill.skill_hash == expected


def test_support_model_uses_only_train_rows_and_abstains_when_unresolved() -> None:
    rows = [_row("positive", "navigation", 1.0, seed) for seed in range(10)]
    rows += [_row("negative", "measurement", -1.0, seed) for seed in range(10)]
    contexts = {
        "positive": {"task_family": "navigation"},
        "negative": {"task_family": "measurement"},
        "unseen": {"task_family": "chemistry"},
    }

    model = fit_support_model(rows, contexts, delta=0.05)

    assert model.training_task_ids == ("negative", "positive")
    assert model.decide(contexts["unseen"]).deploy is False
    assert model.decide(contexts["unseen"]).reason in {"underidentified", "no_certificate", "predicate_mismatch"}
