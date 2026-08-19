# Statistical protocol (current authority)

This file is the single statistical authority for evolution and formal
calibration. Other documents may describe motivation, but must not redefine
these quantities.

## Paired measurements and independent groups

Repeated arm measurements are paired within a fixture. One
`independence_group` contributes at most one promotion Bernoulli observation;
repetitions never increase the number of independent promotion trials.
Synthesis, promotion, held-out, and poison pools are disjoint.

## Confidence accounting

Node and relation contrasts use the time-uniform Hoeffding confidence
sequences registered in `core.acre.budget.StatisticalBudget`. Promotion uses
the implemented inverted Beta--Binomial mixture e-process as a diagnostic
probability lower bound over independent groups. The probability bound is not
a posterior claim. Utility effects retain their bounded-mean confidence
sequence and are the only quantities eligible for routing.

## Context estimand

The current promotion scheduler samples the preregistered deterministic family
lattice. Its claim is finite-lattice coverage, not an iid population
generalization probability. Any future distribution-sampling protocol must be
registered before use.

## Missingness and reruns

Protocol-valid task failures are efficacy-eligible with score zero. Agent
timeout, crash, or budget exhaustion are outcome failures. Executor, hardware,
and protocol failures are infrastructure-invalid and are the only rerunnable
class. Every rerun records its reason, count, and source run digest.

## Confirmatory aggregation

`CLAIMS.yaml` freezes the primary D-B estimand and weighting. Formal aggregation
accepts sealed, efficacy-eligible, complete matrices only; public/dev and
family/difficulty slices are exploratory. If readiness is withheld, all
condition-level effects are withheld.
