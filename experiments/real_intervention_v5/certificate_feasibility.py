"""Necessary sample-size conditions for the canonical ACRE certificate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from core.acre.budget import StatisticalBudget
from core.acre.experiments import ReplaySequentialCertificate
from core.sequential_stats import minimum_all_successes


def independent_group_count(rows: Iterable[Mapping[str, Any]]) -> int:
    """Count family identities once, regardless of contexts or repeat sessions."""
    return len({str(row["family_id"]) for row in rows if row.get("family_id")})


def _calibration_minima() -> tuple[int, int]:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "real_intervention_v4"
        / "config"
        / "PRIMARY_LAYOUT_STUDY.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    acre = config["acre"]
    return (
        int(acre["minimum_calibration_support_families"]),
        int(acre["minimum_calibration_non_support_families"]),
    )


def feasibility(
    registered_calibration_groups: int,
    eligible_calibration_groups: int | None = None,
) -> dict[str, Any]:
    """Report group-count lower bounds; this never fits or certifies ACRE.

    When eligibility is not yet known, registered groups are used only as an
    optimistic upper bound and the result records that basis explicitly.
    """
    for name, value in (
        ("registered_calibration_groups", registered_calibration_groups),
        ("eligible_calibration_groups", eligible_calibration_groups),
    ):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{name} must be a non-negative integer")
    if eligible_calibration_groups is not None and eligible_calibration_groups > registered_calibration_groups:
        raise ValueError("eligible calibration groups cannot exceed registered groups")

    certificate = ReplaySequentialCertificate(minimum_groups=1)
    budget: StatisticalBudget = certificate.statistical_budget
    all_success_groups = minimum_all_successes(
        certificate.p_min, budget.replay_minimum_delta,
    )
    support_minimum, non_support_minimum = _calibration_minima()
    support_required = max(support_minimum, all_success_groups + 1)
    required_groups = support_required + non_support_minimum
    mixture_threshold = 1.0 / budget.replay_minimum_delta
    all_success_e_value_n_minus_1 = 1.0 / (
        all_success_groups * certificate.p_min ** (all_success_groups - 1)
    )
    all_success_e_value_n = 1.0 / (
        (all_success_groups + 1) * certificate.p_min ** all_success_groups
    )
    counted_groups = (
        registered_calibration_groups
        if eligible_calibration_groups is None
        else eligible_calibration_groups
    )

    return {
        "primary_effect_final_feasible": None,
        "primary_effect_final_feasibility_owner": "orchestrator",
        "eligible_group_count_basis": (
            "registered_upper_bound" if eligible_calibration_groups is None else "eligible_groups"
        ),
        "registered_calibration_groups": registered_calibration_groups,
        "eligible_calibration_groups": eligible_calibration_groups,
        "acre_canonical_certificate_feasible": counted_groups >= required_groups,
        "acre_canonical_certificate_feasibility_is_necessary_only": True,
        "acre_canonical_certificate_fitted": False,
        "acre_diagnostic_comparison_feasible": counted_groups >= support_minimum + non_support_minimum,
        "acre_diagnostic_comparison_minimum_groups": {
            "support": support_minimum,
            "non_support": non_support_minimum,
            "total": support_minimum + non_support_minimum,
        },
        "canonical_certificate_lower_bound": {
            "p_min": certificate.p_min,
            "delta_mix": budget.replay_minimum_delta,
            "minimum_all_success_independent_support_groups": all_success_groups,
            "all_success_mixture_e_value_at_n_minus_1": all_success_e_value_n_minus_1,
            "all_success_mixture_e_value_at_n": all_success_e_value_n,
            "mixture_e_value_threshold": mixture_threshold,
            "all_success_crosses_mixture_threshold_at_n": (
                all_success_e_value_n_minus_1 < mixture_threshold
                and all_success_e_value_n >= mixture_threshold
            ),
            "additional_independent_support_holdout_groups": 1,
            "support_groups_required": support_required,
            "non_support_groups_required": non_support_minimum,
            "required_independent_groups_lower_bound": required_groups,
            "sufficiency": "necessary_not_sufficient",
        },
        "required_independent_groups_lower_bound": required_groups,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-groups", type=int, required=True)
    parser.add_argument("--eligible-calibration-groups", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = feasibility(args.calibration_groups, args.eligible_calibration_groups)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
