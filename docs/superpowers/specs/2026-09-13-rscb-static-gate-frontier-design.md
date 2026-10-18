# RSCB, Static-Test Gate, and Support Frontiers

## Contents

- [Scope and invariants](#scope-and-invariants)
- [Policy interface and shared evaluator](#policy-interface-and-shared-evaluator)
- [RSCB-LCB](#rscb-lcb)
- [Support-decision regret](#support-decision-regret)
- [Static-Test Gate](#static-test-gate)
- [Frontiers and abstention analysis](#frontiers-and-abstention-analysis)
- [Higher-order bundle screening](#higher-order-bundle-screening)
- [Cost profile and artifacts](#cost-profile-and-artifacts)
- [Claim decision table](#claim-decision-table)

## Scope and invariants

This revision adds provider-free comparison policies and frozen frontier
analysis. It does not modify Full ACRE, its existing calibration/evaluation
split, utility definition or margin `delta`, hidden-oracle access, the existing
202-context evaluation, or the five frozen replication seeds. No hosted,
formal, sealed, or C--D run is permitted.

The policy layer may see only the public typed context and calibration evidence.
`expected_applicable`, `scientific_truth`, held-out effects, counterexample
labels, and evaluation outcomes are evaluator-only fields. The data flow is:

```text
raw public context z_raw
        -- executor/runtime abstraction h --> typed context x_typed
        -- policy --> Decision(context)
        -- offline evaluator + registered scoring fields --> metrics
```

ACRE and each new baseline expose the same decision object. A policy never
receives the evaluator's hidden scoring record.

## Policy interface and shared evaluator

`Decision` is an immutable record with:

```text
action: "reuse" | "abstain"
score: float                 # probability or deterministic admission score
uncertainty: float           # posterior/CI width; 0 for deterministic gates
reason: str                  # auditable, policy-owned reason
```

The shared evaluator consumes `(policy, calibration, evaluation, oracle_score)`
and computes, with the same code for every policy:

- `deployed = count(action == reuse)` and `coverage = deployed / n`;
- `unsupported_all = unsupported / n`;
- `unsupported_deployed = unsupported / deployed`, with an explicit undefined
  value when `deployed == 0`;
- recall over oracle-supported contexts;
- support-decision regret;
- abstention decomposition and policy-specific traces.

Raw numerators and both denominators are persisted. Family and seed are the
resampling units; contexts are not treated as independent Bernoulli trials.
The evaluator is the only place allowed to consume held-out oracle scoring
fields.

## RSCB-LCB

RSCB is a fair binary reuse/abstain adaptation of the cited risk-sensitive
contextual gating idea; it is not a line-by-line reproduction of an agent action
space. For each family, features are generated from the registered public
predicate grammar paths used by ACRE. Numeric fields are standardized using
calibration mean and standard deviation (zero variance maps to zero); categorical
fields use a calibration-only sorted vocabulary plus an `<UNK>` column; booleans
are `0/1`; missing values use a deterministic missing indicator and zero value.
Opaque identifiers, split labels, outcome fields, oracle fields, and
counterexample labels are excluded. The encoder schema and calibration moments
are frozen before evaluation.

The calibration label is observable support evidence:

```text
y_i = 1 iff the paired calibration effect lower bound clears delta
      and all public scientific gates pass; otherwise y_i = 0.
```

For feature vector `phi(x)`, fit a regularized Bayesian logistic model with
independent Gaussian prior `beta_j ~ N(0, 1)` for non-intercept coefficients and
`beta_0 ~ N(0, 100)`. The posterior is the Laplace approximation at the
deterministic L2-regularized MAP solution, found by Newton updates with at most
100 iterations and tolerance `1e-9`. If `Sigma` is the inverse observed
negative Hessian plus prior precision, define:

```text
p_mean(x) = sigmoid(phi(x)^T beta_hat)
u(x)      = p_mean(1-p_mean) * sqrt(phi(x)^T Sigma phi(x))
LCB_k(x)  = max(0, p_mean(x) - k * u(x))
```

The default is `k = 1.64485362695147` (one-sided 95% Gaussian posterior
lower bound) and `theta = 0.90`; deploy iff `LCB_k(x) >= theta`.
`RSCB-LCB-matched` chooses only `theta` from the fixed grid
`{0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95}` to minimize absolute calibration
coverage distance from frozen Full ACRE calibration coverage, breaking ties
toward the larger threshold. It never reads evaluation outcomes. The default
and matched operating points are frozen before the 202-context or replication
evaluation.

The frontier uses the pre-registered penalty grid
`k in {0, 0.67448975, 1.28155157, 1.64485363, 2.32634787}` and the threshold
grid above. It is a display of frozen policies, not evaluation-set tuning.

## Support-decision regret

For context `i`, let `tau_i` be the registered paired utility effect and `S_i`
the scientific-gate result. Abstention has fallback utility `u_fallback = 0`.
The admissible oracle action is reuse iff `S_i = 1` and `tau_i > delta`, and
abstain otherwise. Actual reuse utility is `tau_i` when `S_i = 1`; when a
scientific gate fails, reuse is inadmissible and receives `u_fallback` for the
support-decision estimand. Thus:

```text
U_i*(reuse)   = tau_i       if S_i = 1 and tau_i > delta
                0           otherwise
U_i*(abstain) = 0
regret_i      = U_i*(oracle_action) - U_i*(policy_action)
```

The evaluator also reports unsupported deployment separately, so a policy that
reuses on `S_i = 0` is not silently treated as a successful utility outcome.
This is a cross-context support-decision estimand and is distinct from the
longitudinal lifecycle regret, which includes episode history and evolution
cost.

## Static-Test Gate

The Static-Test Gate builds one global admission decision from a fixed,
promotion/calibration-disjoint correctness and scientific regression suite. It
does not consume typed counterexample predicates, new-context paired utility,
or evaluation outcomes. A suite pass admits reuse for every evaluation context;
a suite failure abstains globally. The evaluator reports it as a deterministic
policy with `uncertainty = 0` and reason `static_suite_pass` or
`static_suite_fail`. Its scientific question is whether static correctness and
regression evidence alone establish cross-context deployment permission.

## Frontiers and abstention analysis

ACRE uses the fixed confidence grid
`alpha in {0.20, 0.10, 0.05, 0.025, 0.01}` without changing the registered
utility margin or utility transform. RSCB uses its fixed `(k, theta)` grid.
Nearest experience, global/no-counterexample, and Static-Test Gate are single
control points. Evaluation only realizes frozen policies.

The frontier panels are coverage versus unsupported/all risk and coverage
versus support-decision regret; counterexample dose remains the third panel.
For every ACRE abstention, offline scoring classifies `correct_abstention` or
`missed_supported_deployment`, then records the first blocking reason among
`A`, `E`, `S`, `L`, and unresolved bundle relation. It reports accepted and
missed-supported CI widths. The manuscript may call the latter epistemic
uncertainty only if the missed-supported width is demonstrably larger under the
registered family-cluster analysis; otherwise it reports representation error
or conservative opportunity cost.

## Higher-order bundle screening

The appendix describes a conservative deployment protocol:

```text
if any required singleton is uncertified: abstain
if any possible pair is antagonistic, incompatible, or unresolved: abstain
if all pairs pass but no explicit higher-order certificate exists: abstain
otherwise deploy only with an explicit higher-order certificate
```

This is a policy rule, not a safety theorem. Pairwise failure rejects a
candidate superset conservatively; pairwise success does not certify a higher
order bundle. Worst-case higher-order certification remains combinatorial.

## Cost profile and artifacts

The supplementary instrument records measured CEGIS synthesis time, router
decision latency excluding workload execution, counterexamples until stable
predicate, paired-replay/confidence-sequence stopping horizon, serialized rule
store bytes, bytes per active rule/relation, and active/specialized/retired
revision counts. Timing uses one declared machine, warm-up, and at least 20
deterministic repetitions, reporting median/p90/max. Missing online stopping
traces are reconstructed offline and labeled as such; they are never presented
as online measurements.

Expected artifacts are `results/rscb_static_frontier.json`,
`results/rscb_static_frontier_manifest.json`, `results/cost_profile.json`, and
`evidence/cost_profile_manifest.json`. All include benchmark revision, exact
code commit, split identity, seed list, parameter freeze, environment metadata,
provider call count, and SHA256 values.

## Claim decision table

| Result | Permitted manuscript claim |
|---|---|
| ACRE has lower risk at matched coverage | Explicit counterexample-defined support boundary outperforms smooth probabilistic gating on the evaluated boundary shifts. |
| Risks are comparable, ACRE is more interpretable | ACRE is competitive with risk-sensitive gating while exposing auditable support boundaries. |
| RSCB is better | ACRE's contribution is structured certification, explicit support boundaries, and lifecycle composition; the result is reported without hiding it. |
| Static gate has high coverage but unsupported reuse | Static validation alone does not establish context-indexed deployment permission. |
| Missed-supported uncertainty is not wider | Do not claim epistemic concentration; report support-boundary or representation opportunity cost. |

None of these provider-free comparisons changes the current hosted efficacy,
formal, sealed, or C--D status.
