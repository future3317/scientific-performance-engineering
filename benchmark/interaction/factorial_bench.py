"""Synthetic factorial interaction pilot, separate from formal benchmark scoring."""

from __future__ import annotations

from core.acre.factorial import FactorialBlock, FactorialEngine, simulate_coverage


_CASES = {
    "synergy": {"00": 0.0, "10": 0.1, "01": 0.1, "11": 0.9},
    "antagonism": {"00": 0.0, "10": 0.8, "01": 0.8, "11": 0.2},
    "independence": {"00": 0.0, "10": 0.3, "01": 0.4, "11": 0.7},
    "prerequisite": {"00": 0.0, "10": 0.4, "01": 0.0, "11": 0.9},
}


def run_factorial_benchmark(*, blocks: int = 5000, seed: int = 7) -> dict[str, object]:
    classifications: dict[str, str] = {}
    estimates: dict[str, dict[str, float]] = {}
    for name, values in _CASES.items():
        engine = FactorialEngine()
        for index in range(blocks):
            engine.add_block(FactorialBlock(f"{name}-{index}", values))
        estimate = engine.estimate()
        classifications[name] = estimate.decision
        estimates[name] = {"gamma": estimate.gamma, "gamma_lcb": estimate.gamma_lcb, "gamma_ucb": estimate.gamma_ucb}
    coverage = simulate_coverage(true_interaction=0.2, noise=0.04, blocks=blocks, repetitions=200, seed=seed)
    return {
        "classifications": classifications,
        "estimates": estimates,
        "coverage": coverage.coverage,
        "coverage_repetitions": coverage.repetitions,
    }
