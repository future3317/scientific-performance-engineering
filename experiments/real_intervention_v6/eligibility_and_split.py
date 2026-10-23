"""Outcome-blind family eligibility and deterministic V6 cohort allocation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_SEED = 20261023
ACRE_MINIMUM_CALIBRATION_FAMILIES = 42


def _split_sizes(family_count: int) -> dict[str, int]:
    if family_count < 6:
        sizes = {"development": 0, "calibration": 0, "final": 0}
        for index in range(family_count):
            sizes[("development", "calibration", "final")[index % 3]] += 1
        return sizes

    sizes = {"development": 2, "calibration": 2, "final": 2}
    remaining = family_count - 6
    while remaining and (sizes["development"] < 6 or sizes["calibration"] < 6):
        for split in ("development", "calibration"):
            if remaining and sizes[split] < 6:
                sizes[split] += 1
                remaining -= 1
    sizes["final"] += remaining
    return sizes


def _rank(seed: int, family_id: str) -> str:
    return hashlib.sha256(f"{seed}:{family_id}".encode("utf-8")).hexdigest()


def build_cohort(
    snapshot: Sequence[Mapping[str, Any]],
    exposed_families: Sequence[str],
    study: str,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Select each eligible family once and allocate disjoint family splits.

    Only ``eligibility`` and the family identifier determine membership. The
    returned row is deliberately a small qualification-only projection; timing
    and outcome fields are neither consulted nor copied.
    """
    eligible: dict[str, Mapping[str, Any]] = {}
    for source in snapshot:
        if source.get("eligibility") != "ELIGIBLE":
            continue
        family = str(source.get("family_id") or "")
        if family:
            eligible.setdefault(family, source)

    exposed = {str(family) for family in exposed_families}
    ordered = sorted(
        eligible,
        key=lambda family: (family not in exposed, _rank(seed, family), family),
    )
    sizes = _split_sizes(len(ordered))
    assignments: dict[str, str] = {}
    offset = 0
    for split in ("development", "calibration", "final"):
        end = offset + sizes[split]
        assignments.update((family, split) for family in ordered[offset:end])
        offset = end

    safe_fields = (
        "family_id", "candidate_id", "source", "model", "split", "eligibility",
        "baseline_valid", "candidate_correct", "kernel_exposure_valid",
    )
    rows = []
    for family in ordered:
        source = eligible[family]
        row = {key: source[key] for key in safe_fields if key in source}
        row.update({
            "family_id": family,
            "source_split": source.get("split"),
            "split": assignments.get(family),
            "exposure_flags": {"prior_performance_exposure": family in exposed},
        })
        rows.append(row)

    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("development", "calibration", "final")
    }
    pilot_feasible = all(counts[split] >= 2 for split in counts)
    acre_minimum_met = counts["calibration"] >= ACRE_MINIMUM_CALIBRATION_FAMILIES
    return {
        "schema_version": 1,
        "study": str(study),
        "seed": int(seed),
        "eligible_family_count": len(rows),
        "rows": rows,
        "split_counts": counts,
        "sample_feasibility": {
            "independent_pilot": {
                "status": "FEASIBLE" if pilot_feasible else "INFEASIBLE",
                "minimum_families_per_split": 2,
                "feasible": pilot_feasible,
            },
            "original_acre_certification": {
                "status": (
                    "NECESSARY_MINIMUM_MET_NOT_CERTIFIED"
                    if acre_minimum_met else "INFEASIBLE"
                ),
                "minimum_calibration_families": ACRE_MINIMUM_CALIBRATION_FAMILIES,
                "eligible_calibration_families": counts["calibration"],
                "necessary_minimum_met": acre_minimum_met,
                "certified": False,
            },
        },
    }
