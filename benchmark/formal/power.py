"""Small preregistration helpers for pilot-based power checks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from benchmark.formal.schedule import PromotionReplayScheduler
from core.acre.experiments import ReplaySequentialCertificate


def efficacy_power(*, task_lineages: int, trials: int, effect: float, sigma: float, alpha: float = 0.05, simulations: int = 2000, seed: int = 0) -> dict:
    """Estimate power for the registered task/lineage unit, not arm count."""
    rng = random.Random(seed)
    successes = 0
    estimates = []
    effective_sigma = float(sigma) / max(1.0, float(trials) ** 0.5)
    for _ in range(int(simulations)):
        values = [effect + rng.gauss(0.0, effective_sigma) for _ in range(int(task_lineages))]
        estimate = sum(values) / len(values)
        se = effective_sigma / max(1.0, task_lineages ** 0.5)
        estimates.append(estimate)
        if estimate > 1.96 * se:
            successes += 1
    return {"estimand": "sealed_task_lineage", "task_lineages": task_lineages, "outer_trials": trials, "effect": effect, "sigma": sigma, "effective_sigma": effective_sigma, "alpha": alpha, "simulations": simulations, "power": successes / simulations, "mde_approx": 1.96 * effective_sigma / max(1.0, task_lineages ** 0.5)}


def promotion_power(*, success_rate: float, minimum_groups: int | None = None, max_groups: int | None = None, simulations: int = 2000, seed: int = 0, p_min: float = 0.8, delta: float = 0.05) -> dict:
    rng = random.Random(seed)
    scheduler = PromotionReplayScheduler(p_min=p_min, delta=delta)
    minimum_groups = scheduler.minimum_groups if minimum_groups is None else int(minimum_groups)
    max_groups = scheduler.max_groups if max_groups is None else int(max_groups)
    latencies = []
    promoted = 0
    for _ in range(int(simulations)):
        cases = []
        certificate = ReplaySequentialCertificate(minimum_groups=minimum_groups, max_groups=max_groups, epsilon=0.0, delta=delta, p_min=p_min)
        for group in range(1, max_groups + 1):
            success = rng.random() < success_rate
            cases.append({"case_id": f"G-{group}", "independence_group": f"G-{group}", "query_type": "representative", "higher_is_better": True, "intervention_measurements": [1.1 if success else 0.99], "baseline_measurements": [1.0]})
            status = certificate.update(cases)
            if status.get("stop"):
                if status.get("status") == "passed":
                    promoted += 1
                latencies.append(group)
                break
        else:
            latencies.append(max_groups)
    return {"success_rate": success_rate, "minimum_groups": minimum_groups, "max_groups": max_groups, "p_min": p_min, "delta": delta, "promotion_probability": promoted / simulations, "mean_groups": sum(latencies) / len(latencies), "simulations": simulations}


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
            "promotion": [promotion_power(success_rate=p, seed=i) for i, p in enumerate((0.85, 0.90, 0.95))],
        })
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
