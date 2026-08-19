"""Small preregistration helpers for pilot-based power checks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def efficacy_power(*, task_lineages: int, trials: int, effect: float, sigma: float, alpha: float = 0.05, simulations: int = 2000, seed: int = 0) -> dict:
    """Estimate power for the registered task/lineage unit, not arm count."""
    rng = random.Random(seed)
    successes = 0
    estimates = []
    for _ in range(int(simulations)):
        values = [effect + rng.gauss(0.0, sigma) for _ in range(int(task_lineages))]
        estimate = sum(values) / len(values)
        se = sigma / max(1.0, task_lineages ** 0.5)
        estimates.append(estimate)
        if estimate > 1.96 * se:
            successes += 1
    return {"estimand": "sealed_task_lineage", "task_lineages": task_lineages, "outer_trials": trials, "effect": effect, "sigma": sigma, "alpha": alpha, "simulations": simulations, "power": successes / simulations, "mde_approx": 1.96 * sigma / max(1.0, task_lineages ** 0.5)}


def promotion_power(*, success_rate: float, minimum_groups: int, max_groups: int, simulations: int = 2000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    latencies = []
    promoted = 0
    for _ in range(int(simulations)):
        successes = 0
        for group in range(1, max_groups + 1):
            successes += rng.random() < success_rate
            if group >= minimum_groups and successes / group >= 0.8:
                promoted += 1
                latencies.append(group)
                break
        else:
            latencies.append(max_groups)
    return {"success_rate": success_rate, "minimum_groups": minimum_groups, "max_groups": max_groups, "promotion_probability": promoted / simulations, "mean_groups": sum(latencies) / len(latencies), "simulations": simulations}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sigma", type=float, default=None, help="pilot task/lineage SD; required for a numeric power simulation")
    args = parser.parse_args()
    report = {"schema_version": 1, "status": "diagnostic_only", "formal_claim": "withheld"}
    if args.sigma is None or args.sigma < 0:
        report.update({"status": "blocked", "reason": "pilot task/lineage variance is required before numeric power is reported"})
    else:
        report.update({
            "sigma_source": "explicit_pilot_summary",
            "efficacy": [efficacy_power(task_lineages=35, trials=3, effect=e, sigma=args.sigma, seed=i) for i, e in enumerate((0.05, 0.1, 0.2))],
            "promotion": [promotion_power(success_rate=p, minimum_groups=12, max_groups=36, seed=i) for i, p in enumerate((0.85, 0.90, 0.95))],
        })
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
