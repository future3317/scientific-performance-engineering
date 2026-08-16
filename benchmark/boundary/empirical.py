"""Executable-workload boundary calibration, separate from synthetic CEGIS."""

from __future__ import annotations

from dataclasses import dataclass
import random
from statistics import mean
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class EmpiricalBoundaryCase:
    context_id: str
    context: Mapping[str, Any]


def run_empirical_boundary(
    cases: Sequence[EmpiricalBoundaryCase],
    evaluator: Callable[[Mapping[str, Any], bool, int], float],
    *,
    repetitions: int = 5,
    practical_effect: float = 0.05,
    noise_floor: float = 0.02,
) -> dict[str, Any]:
    """Run paired executable evaluations and classify uncertain boundaries.

    ``evaluator`` is the workload owner: it returns a bounded utility for one
    context, intervention arm, and seed.  No applicability label is supplied
    to this layer; the result is an empirical estimate with an explicit
    ``inconclusive`` state when noise dominates the effect.
    """
    if repetitions < 2 or practical_effect < 0 or noise_floor < 0:
        raise ValueError("repetitions must be >=2 and thresholds non-negative")
    rows: list[dict[str, Any]] = []
    for case in cases:
        deltas = [float(evaluator(case.context, True, seed)) - float(evaluator(case.context, False, seed)) for seed in range(repetitions)]
        estimate = mean(deltas)
        rng = random.Random(f"boundary:{case.context_id}")
        bootstrap = sorted(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas) for _ in range(2000))
        lower = bootstrap[int(0.025 * (len(bootstrap) - 1))]
        upper = bootstrap[int(0.975 * (len(bootstrap) - 1))]
        if upper < noise_floor:
            decision = "not_applicable"
        elif lower > practical_effect:
            decision = "applicable"
        else:
            decision = "inconclusive"
        rows.append({"context_id": case.context_id, "estimate": estimate, "ci_low": lower, "ci_high": upper, "decision": decision, "noise_floor": noise_floor, "repetitions": repetitions})
    return {"case_count": len(rows), "rows": rows, "inconclusive_count": sum(row["decision"] == "inconclusive" for row in rows)}
