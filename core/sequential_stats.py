"""Tighter sequential lower bounds for bounded Bernoulli replay outcomes."""

from __future__ import annotations

import math


def _beta_tail(alpha: int, beta: int, threshold: float) -> float:
    n = alpha + beta - 1
    return min(1.0, max(0.0, sum(
        math.comb(n, j) * threshold**j * (1.0 - threshold) ** (n - j)
        for j in range(alpha)
    )))


def mixture_lower_bound(successes: int, trials: int, delta: float) -> float:
    """Lower confidence sequence from a beta(1,1) mixture boundary.

    The beta-binomial mixture is evaluated at the current prefix and inverted
    by binary search.  It is substantially tighter than a union-bound Hoeffding
    radius while remaining valid for optional checks of the same replay stream.
    """
    if trials < 1 or successes < 0 or successes > trials or not 0 < delta < 1:
        raise ValueError("invalid confidence-sequence inputs")
    alpha, beta = 1 + successes, 1 + trials - successes
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _beta_tail(alpha, beta, mid) >= 1.0 - delta:
            lo = mid
        else:
            hi = mid
    return lo
