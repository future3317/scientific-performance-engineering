from __future__ import annotations

import inspect
import math

import pytest

from experiments.real_intervention_v4 import secondary_compile


def _family(family_id: str, split: str) -> dict:
    qualification = {
        "correctness_status": "PASS",
        "weights_sha256": "a" * 64,
        "inputs_sha256": "b" * 64,
        "reference_path": f"/frozen/references/{family_id}.pt",
        "reference_sha256": "c" * 64,
    }
    return {
        "family_id": family_id,
        "split": split,
        "model_spec": {
            "source": "torchvision", "model": "resnet18", "seed": 31,
            "shape": [3, 224, 224],
        },
        "qualification_by_batch": {"1": dict(qualification), "8": dict(qualification)},
    }


def _roster() -> dict:
    return {
        "schema_version": 1,
        "study_id": "EVOSPE_REAL_INTERVENTION_V4_COMPILE_AMORTIZATION_20261019",
        "intervention_id": "COMPILE_AMORTIZATION_V1",
        "baseline_only": True,
        "timing_opened": False,
        "family_split_frozen": True,
        "order_seed": 101,
        "torchbench_root": "/frozen/torchbench",
        "torchbench_commit": "1" * 40,
        "dependency_overlay": "/frozen/dependencies",
        "correctness": {"rtol": 1e-4, "atol": 1e-5},
        "families": [_family("family-a", "development"), _family("family-b", "calibration"),
                     _family("family-c", "final")],
        "compile_canary_families": ["family-a"],
        "shakedown_contexts": [
            {"family_id": "family-a", "batch": 1, "horizon": 128,
             "deployment": "COLD_NEW_GRAPH"},
            {"family_id": "family-a", "batch": 8, "horizon": 1024,
             "deployment": "PERSISTENT_CACHE_AVAILABLE"},
        ],
        "shakedown_gate": {"minimum_pairs": 2, "minimum_matched_correct_fraction": 0.8},
    }


def _eligible_pair(family_id: str, horizon: int, deployment: str = "COLD_NEW_GRAPH") -> dict:
    return {
        "family_id": family_id, "split": "development", "horizon": horizon,
        "deployment": deployment,
        "classification": {"paired_comparability": "MATCHED", "scientific_correctness": "PASS",
                           "policy_evidence": "ELIGIBLE"},
        "S0": {"worker_result": {"timing": {"measured_forward_seconds": 10.0}}},
        "S1": {"worker_result": {"timing": {"measured_forward_seconds": 5.0}}},
    }


def test_contract_keeps_compile_study_separate_and_freezes_cache_contracts():
    contract = secondary_compile._load_contract()

    assert contract["study_id"] == "EVOSPE_REAL_INTERVENTION_V4_COMPILE_AMORTIZATION_20261019"
    assert contract["intervention_id"] == "COMPILE_AMORTIZATION_V1"
    assert contract["batches"] == [1, 8]
    assert contract["horizons"] == [128, 1024, 4096]
    assert contract["precision"] == "float32"
    assert {row["id"] for row in contract["deployment_contracts"]} == {
        "COLD_NEW_GRAPH", "PERSISTENT_CACHE_AVAILABLE",
    }
    persistent = next(row for row in contract["deployment_contracts"]
                      if row["id"] == "PERSISTENT_CACHE_AVAILABLE")
    assert persistent["declared_reuse_counts"] == [1, 10, 100]
    assert "cannot replace" in contract["prohibition"]


def test_worker_uses_only_the_registered_compile_intervention():
    source = inspect.getsource(secondary_compile._worker)

    assert 'torch.compile(model, backend="inductor", mode="default", dynamic=None)' in source
    assert 'torch.backends.cuda.matmul.allow_tf32 = False' in source
    assert 'torch.backends.cudnn.allow_tf32 = False' in source
    assert 'torch.autocast(device_type="cuda", enabled=False)' in source
    assert '"channels_last_requested": False' in source
    assert 'metrics.generated_kernel_count' in source
    assert 'and generated_kernel_count == 0' in source
    assert 'and generated_kernel_count > 0' in source
    assert "memory_format=torch.channels_last" not in source
    assert "torch.backends.cuda.matmul.allow_tf32 = True" not in source


def test_roster_requires_prelook_family_splits_and_baseline_correctness():
    contract = secondary_compile._load_contract()
    roster = secondary_compile._validate_roster(_roster(), contract)

    assert roster["family_by_id"]["family-a"]["split"] == "development"
    assert roster["family_by_id"]["family-b"]["split"] == "calibration"
    assert roster["family_by_id"]["family-c"]["split"] == "final"
    assert len(roster["families"]) == 3

    changed = _roster()
    changed["timing_opened"] = True
    with pytest.raises(ValueError, match="timing-blind"):
        secondary_compile._validate_roster(changed, contract)

    duplicated = _roster()
    duplicated["families"].append(_family("family-a", "calibration"))
    with pytest.raises(ValueError, match="would inflate independent n"):
        secondary_compile._validate_roster(duplicated, contract)


def test_context_grid_does_not_change_independent_family_count_or_split():
    contract = secondary_compile._load_contract()
    roster = secondary_compile._validate_roster(_roster(), contract)
    plan = secondary_compile._pair_plan(roster, contract, prefix="formal")

    assert len(plan) == 3 * len(contract["batches"]) * len(contract["horizons"]) * 2
    assert {row["family_id"] for row in plan} == {"family-a", "family-b", "family-c"}
    assert all(len(row["order"]) == 2 and set(row["order"]) == {"S0", "S1"} for row in plan)
    assert all(row["split"] == roster["family_by_id"][row["family_id"]]["split"] for row in plan)


def test_cache_state_is_persistent_only_for_compiled_candidate():
    assert secondary_compile._cache_state_for("S0", "PERSISTENT_CACHE_AVAILABLE") == "EMPTY"
    assert secondary_compile._cache_state_for("S1", "COLD_NEW_GRAPH") == "EMPTY"
    assert secondary_compile._cache_state_for("S1", "PERSISTENT_CACHE_AVAILABLE") == "PREFILLED"


def test_persistent_cache_acquisition_is_amortized_only_at_declared_reuse_counts():
    pair = _eligible_pair("family-a", 128, "PERSISTENT_CACHE_AVAILABLE")
    pair["cache_acquisition"] = {"acquisition_cost_seconds": 20.0}

    at_one = secondary_compile._pair_effect(pair, 1)
    at_ten = secondary_compile._pair_effect(pair, 10)
    at_hundred = secondary_compile._pair_effect(pair, 100)

    assert at_one["s1_seconds_per_deployment"] == 25.0
    assert at_ten["s1_seconds_per_deployment"] == 7.0
    assert at_hundred["s1_seconds_per_deployment"] == pytest.approx(5.2)
    assert at_ten["seconds_saved"] == 3.0


def test_effect_summary_averages_contexts_within_family_before_counting_n():
    rows = [
        _eligible_pair("family-a", 128), _eligible_pair("family-a", 1024),
        _eligible_pair("family-b", 128), _eligible_pair("family-b", 4096),
    ]
    summary = secondary_compile._summarize(rows, secondary_compile._load_contract())
    cold = summary["splits"]["development"]["deployments"]["COLD_NEW_GRAPH"]["per_cold_deployment"]

    assert summary["splits"]["development"]["independent_families"] == 2
    assert summary["splits"]["development"]["planned_pairs"] == 4
    assert cold["independent_families_with_valid_pairs"] == 2
    assert cold["family_mean_log_benefit"] == pytest.approx(math.log(2.0))
