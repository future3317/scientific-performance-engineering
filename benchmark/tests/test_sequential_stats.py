#!/usr/bin/env python3
"""Tests for the time-uniform Beta-Binomial mixture confidence sequence."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.sequential_stats import mixture_e_value, mixture_lower_bound


def main() -> None:
    # The mixture e-value is the Beta-binomial integral divided by the fixed-p
    # Bernoulli likelihood.  This identity catches a posterior/CS substitution.
    value = mixture_e_value(3, 5, 0.5)
    expected = math.exp(
        math.lgamma(3 + 1.0) + math.lgamma(2 + 1.0) - math.lgamma(5 + 2.0)
        - (math.lgamma(1.0) + math.lgamma(1.0) - math.lgamma(2.0))
        - 5 * math.log(0.5)
    )
    assert math.isclose(value, expected, rel_tol=1e-12), (value, expected)

    assert mixture_lower_bound(0, 20, 0.05) == 0.0
    assert 0.7 < mixture_lower_bound(20, 20, 0.05) < 1.0

    # Exhaustive small-n crossing test under p=0.5.  Optional inspection of
    # every prefix must remain below the requested 5% crossing probability.
    horizon = 10
    crossing_probability = 0.0
    for sequence in itertools.product((0, 1), repeat=horizon):
        successes = 0
        crossed = False
        for index, outcome in enumerate(sequence, start=1):
            successes += outcome
            if mixture_lower_bound(successes, index, 0.05) > 0.5:
                crossed = True
                break
        if crossed:
            crossing_probability += 0.5 ** horizon
    assert crossing_probability <= 0.05 + 1e-12, crossing_probability

    print("test_sequential_stats: OK")


if __name__ == "__main__":
    main()
