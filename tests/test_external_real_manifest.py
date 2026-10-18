from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.external_real.manifest import build_protocol_manifests, write_protocol_manifests


def test_manifest_rejects_final_and_burned_id_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        build_protocol_manifests(
            benchmark="scienceworld",
            benchmark_revision="scienceworld-1.2.3",
            protocol_revision="external-real-20260915",
            model_id="deepseek-v4-flash",
            final_test_ids=("task-1",),
            burned_evaluation_ids=("task-1",),
        )


def test_manifest_freezes_required_experiment_identity_and_writes_json(tmp_path: Path) -> None:
    manifests = build_protocol_manifests(
        benchmark="scienceworld",
        benchmark_revision="scienceworld-1.2.3",
        protocol_revision="external-real-20260915",
        model_id="deepseek-v4-flash",
        final_test_ids=("task-2",),
        burned_evaluation_ids=("task-1",),
        generation_seeds=(0, 1, 2),
        evidence_budgets=(0, 8, 16, 32, 64),
    )

    write_protocol_manifests(tmp_path, manifests)

    protocol = json.loads((tmp_path / "protocol_manifest.json").read_text())
    assert protocol["generation_seeds"] == [0, 1, 2]
    assert protocol["evidence_budgets"] == [0, 8, 16, 32, 64]
    assert protocol["final_test_ids"] == ["task-2"]
    assert json.loads((tmp_path / "burned_evaluation_ids.json").read_text())["ids"] == ["task-1"]
    assert json.loads((tmp_path / "statistics_manifest.json").read_text())["bootstrap_replicates"] == 10000


def test_manifest_defaults_are_explicitly_provider_free() -> None:
    manifests = build_protocol_manifests(
        benchmark="fixture",
        benchmark_revision="fixture-1",
        protocol_revision="fixture-protocol",
        model_id="provider-free-family-environment",
        final_test_ids=(),
        burned_evaluation_ids=(),
    )

    assert manifests["provider_model_manifest"]["provider_calls_authorized"] is False
    assert manifests["statistics_manifest"]["cluster_unit"] == "family_seed"
