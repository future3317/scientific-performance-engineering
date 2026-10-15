from __future__ import annotations

import json

from experiments.skill_evolution.qualification import (
    qualification_config,
    required_evolution_search_budget,
    required_task_execution_budget,
    run_qualification_study,
)
from experiments.skill_evolution import qualification
from experiments.skill_evolution.study import (
    HYPOTHESES,
    PREDEFINED_SEEDS,
    required_fixture_task_execution_budget,
    run_fixture_study,
)
from core.acre.resource_budget import PairReservation


def test_pair_reservation_is_atomic_when_task_budget_cannot_cover_every_pair():
    reservation = PairReservation.reserve_all_pairs(
        (("pair-1", 3.0), ("pair-2", 3.0)),
        task_execution_budget=5.99,
        evolution_search_budget=100.0,
    )

    assert reservation.launch_allowed is False
    assert reservation.reason == "task_execution_budget_insufficient"
    assert reservation.completed_pairs == ()


def test_evolution_spending_is_separate_and_pair_cost_missingness_is_explicit():
    reservation = PairReservation.reserve_all_pairs(
        (("pair-1", 3.0),), task_execution_budget=3.0, evolution_search_budget=2.0
    )
    updated = reservation.record_evolution(1.25).complete_pair("pair-1", None)

    assert updated.evolution_spent == 1.25
    assert updated.evolution_available == 0.75
    assert updated.measured_pair_costs == (("pair-1", None),)
    assert updated.task_execution_budget == 3.0


def test_fixture_emits_frozen_synthetic_results_and_keeps_real_phase_not_run():
    report = run_fixture_study()

    assert report["synthetic_only"] is True
    assert report["provider_calls"] == 0
    assert report["seed_grid"] == list(PREDEFINED_SEEDS)
    assert report["hypotheses"] == HYPOTHESES
    assert len(report["results"]) == 9
    assert report["real_phase_i"]["status"] == "NOT_RUN"
    assert report["cost_accounting"]["input_tokens"] == {
        "value": None,
        "status": "unknown_not_applicable",
    }
    assert report["reservation"]["completed_pairs"]
    assert len(report["reservation"]["pair_costs"]) == 2880


def test_underfunded_fixture_launches_no_pairs_and_has_no_outcome_rows():
    report = run_fixture_study(
        task_execution_budget=required_fixture_task_execution_budget() - 0.01,
        evolution_search_budget=999.0,
    )

    assert report["launch_status"] == "NOT_LAUNCHED_INCOMPLETE_PAIR_RESERVATION"
    assert "results" not in report
    assert report["reservation"]["completed_pairs"] == []


def test_power_grid_is_planning_only_and_uses_declared_independent_unit():
    report = run_fixture_study()

    assert report["power_table"]["status"] == "planning_approximation_not_observed_power"
    assert len(report["power_table"]["rows"]) == 12
    assert all(row["approximate_power"] >= 0.0 for row in report["power_table"]["rows"])
    assert all(row["independent_clusters"] in {20, 40, 80, 160} for row in report["power_table"]["rows"])
    assert all(result["statistical_unit"].startswith("independent task-family cluster") for result in report["results"])
    assert "supported" in report["results"][0]["stratum_contrasts"]["context_gate"]


def test_selective_fixture_gate_keeps_supported_gain_and_avoids_unsupported_harm():
    report = run_fixture_study()
    selective_seed_17 = next(
        row for row in report["results"]
        if row["regime"] == "selective_transfer" and row["seed"] == 17
    )

    assert selective_seed_17["stratum_contrasts"]["context_gate"]["supported"]["estimate"] > 0.0
    assert selective_seed_17["stratum_contrasts"]["context_gate"]["unsupported"]["estimate"] == 0.0
    assert selective_seed_17["stratum_contrasts"]["global_reuse"]["unsupported"]["estimate"] < 0.0


def test_fixture_command_is_provider_free_and_marks_output_synthetic(capsys):
    from experiments.skill_evolution.__main__ import main

    assert main(["--fixture"]) == 0
    output = capsys.readouterr().out
    assert '"synthetic_only": true' in output
    assert '"status": "NOT_RUN"' in output


def test_learned_qualification_uses_disjoint_splits_and_current_acre_decisions():
    report = run_qualification_study()

    assert report["synthetic_only"] is True
    assert report["qualification_status"] == "SYNTHETIC_FIXTURE_COMPLETE"
    assert report["provider_calls"] == 0
    assert report["pre_frozen_config"]["train_evaluation_overlap"] is False
    assert report["pre_frozen_config"]["confidence_allocation"]["family_alpha"] == 0.05
    assert report["pre_frozen_config"]["confidence_allocation"]["measurement_comparisons"] == 720
    assert report["pre_frozen_config"]["confidence_allocation"]["measurement_alpha"] == 0.025
    assert report["pre_frozen_config"]["confidence_allocation"]["regime_classifier_alpha"] == 0.0125
    assert report["pre_frozen_config"]["confidence_allocation"]["cegis_certificate_alpha"] == 0.0125
    assert report["pre_frozen_config"]["max_candidates"] <= 1000
    assert report["power_table"]["status"] == "planning_approximation_not_observed_power"
    selective = next(row for row in report["results"] if row["regime"] == "selective_transfer")
    broad = next(row for row in report["results"] if row["regime"] == "broad_benefit")
    assert selective["training"]["cegis_status"] == "identified"
    assert selective["training"]["support_regime"] == "SELECTIVE_IDENTIFIED"
    assert broad["training"]["support_regime"] == "GLOBAL_VERIFIED"
    assert selective["evaluation"]["activation_rate"]["old_selective"]["estimate"] == 0.5
    assert selective["evaluation"]["activation_rate"]["new_regime"]["estimate"] == 0.5
    assert selective["new_regime_routing"]["api"] == "AcreEngine.route"
    assert selective["new_regime_routing"]["selected_by_stratum"] == {
        "supported": True,
        "unsupported": False,
    }
    assert selective["new_regime_routing"]["canonical_promotion_claim"] is False
    assert selective["evaluation"]["by_stratum"]["supported"]["activation_rate"]["new_regime"]["estimate"] == 1.0
    assert selective["evaluation"]["by_stratum"]["unsupported"]["negative_transfer_rate"]["new_regime"]["estimate"] == 0.0
    assert selective["evaluation"]["by_stratum"]["unsupported"]["negative_transfer_rate"]["global"]["estimate"] > 0.0
    assert report["real_phase_i"]["status"] == "NOT_RUN"
    assert report["reservation"]["task_execution_budget"] == required_task_execution_budget()
    assert report["reservation"]["evolution_search_budget"] == required_evolution_search_budget()
    assert report["reservation"]["evolution_spent"] <= required_evolution_search_budget()
    assert selective["training"]["cegis_certificate_valid"] is True
    assert len(report["predicate_stress"]["rows"]) == 4
    assert all(row["candidate_cap"] <= 1000 for row in report["predicate_stress"]["rows"])
    assert all(row["search_status"] in {"exact", "truncated"} for row in report["predicate_stress"]["rows"])
    assert any(row["search_status"] == "exact" for row in report["predicate_stress"]["rows"])
    assert any(row["search_status"] == "truncated" for row in report["predicate_stress"]["rows"])


def test_learned_qualification_does_not_start_with_incomplete_evaluation_reservation():
    report = run_qualification_study(task_execution_budget=required_task_execution_budget() - 1)

    assert report["qualification_status"] == "NOT_LAUNCHED_INCOMPLETE_PAIR_RESERVATION"
    assert report["reservation"]["completed_pairs"] == []
    assert "results" not in report


def test_qualification_cli_writes_frozen_config_before_running_study(tmp_path, monkeypatch):
    from experiments.skill_evolution import __main__ as cli

    output = tmp_path / "report.json"
    preregistration = tmp_path / "report.preregistration.json"

    def run_after_preregistration(*, task_execution_budget=None):
        assert preregistration.exists()
        assert not output.exists()
        assert json.loads(preregistration.read_text()) == qualification_config()
        return {"synthetic_only": True, "results": []}

    monkeypatch.setattr(cli, "run_qualification_study", run_after_preregistration)
    assert cli.main(["--qualification", "--output", str(output)]) == 0
    assert output.exists()


def test_qualification_config_freezes_confidence_allocation_and_phase_i_status():
    config = qualification_config()
    allocation = config["confidence_allocation"]

    assert allocation["family_alpha"] == 0.05
    assert allocation["measurement_alpha"] + allocation["regime_classifier_alpha"] + allocation["cegis_certificate_alpha"] == 0.05
    assert allocation["measurement_comparisons"] == 720
    assert config["groups_per_stratum_per_seed"] == 60
    assert config["repetitions_per_group"] == 1024
    assert allocation["minimum_repetitions_for_radius_target"] <= config["repetitions_per_group"]
    assert allocation["planned_worst_case_radius"] <= allocation["planned_radius_target"]
    assert config["real_phase_i"]["status"] == "NOT_RUN"
    assert config["synthetic_rule"]["routing_api"].startswith("AcreEngine.route")
    assert config["synthetic_rule"]["state"] == "canonical synthetic fixture only"
    assert config["synthetic_rule"]["canonical_promotion_claim"] is False


def test_low_repetition_broad_fixture_remains_uncertain(monkeypatch):
    monkeypatch.setattr(qualification, "REPETITIONS_PER_GROUP", 80)
    groups, synthesis = qualification._train_regime("broad_benefit")
    decision = qualification.classify_support_regime(
        groups,
        scope=qualification.SUPPORT_SCOPE,
        target_context={"domain": "skillstudy"},
        runtime_versions=qualification.RUNTIME_VERSIONS,
        selective=synthesis,
        fixed_time_complete=True,
        delta=qualification.REGIME_ALPHA,
        min_groups=qualification.MIN_GROUPS_FOR_GLOBAL,
        min_net_benefit=0.0,
        max_harm_probability=0.10,
    )

    assert decision.regime.value == "UNCERTAIN"
