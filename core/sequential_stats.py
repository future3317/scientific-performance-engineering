"""Time-uniform Bernoulli bounds from a Beta-Binomial mixture e-process."""

from __future__ import annotations

import math


def mixture_e_value(
    successes: int,
    trials: int,
    null_p: float,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> float:
    """Return the Beta-Binomial mixture likelihood ratio at ``null_p``.

    For a fixed Bernoulli null, integrating the alternative likelihood over a
    Beta prior gives a non-negative martingale.  Ville's inequality therefore
    makes the inverted boundary valid at optional inspection times.
    """
    if (
        trials < 0
        or successes < 0
        or successes > trials
        or not 0.0 <= null_p <= 1.0
        or not math.isfinite(prior_alpha)
        or not math.isfinite(prior_beta)
        or prior_alpha <= 0.0
        or prior_beta <= 0.0
    ):
        raise ValueError("invalid mixture e-process inputs")

    n_failures = trials - successes
    log_numerator = (
        math.lgamma(successes + prior_alpha)
        + math.lgamma(n_failures + prior_beta)
        - math.lgamma(trials + prior_alpha + prior_beta)
        - math.lgamma(prior_alpha)
        - math.lgamma(prior_beta)
        + math.lgamma(prior_alpha + prior_beta)
    )
    if successes and null_p == 0.0:
        return math.inf
    if n_failures and null_p == 1.0:
        return math.inf
    log_denominator = 0.0
    if successes:
        log_denominator += successes * math.log(null_p)
    if n_failures:
        log_denominator += n_failures * math.log1p(-null_p)
    return math.exp(log_numerator - log_denominator)


def mixture_lower_bound(successes: int, trials: int, delta: float) -> float:
    """One-sided lower confidence sequence from a Beta(1,1) mixture.

    The lower endpoint is the smallest null value whose mixture e-value is at
    most ``1 / delta``.  The inversion is restricted to the interval below the
    empirical mean, where the e-value is monotone for the lower-tail test.
    """
    if trials < 1 or successes < 0 or successes > trials or not 0 < delta < 1:
        raise ValueError("invalid confidence-sequence inputs")
    if successes == 0:
        return 0.0
    threshold = 1.0 / delta
    hi = successes / trials
    if mixture_e_value(successes, trials, hi) > threshold:
        return 0.0
    lo = 0.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if mixture_e_value(successes, trials, mid) > threshold:
            lo = mid
        else:
            hi = mid
    return lo
