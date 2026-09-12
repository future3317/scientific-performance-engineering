# Context-Indexed Reuse P0 Evidence Design

## Scope

This addendum narrows the manuscript to context-indexed reuse permission and
counterexample-defined support boundaries. It adds only provider-free analyses
and does not change Full ACRE, the historical 202-context evidence, the frozen
replication, any hosted/formal/sealed status, the scientific gate, or the
utility margin.

## Fixed analysis contract

All policies implement `Decision(context) -> {action, score, uncertainty,
reason}` and are scored by `benchmark.boundary.policy_eval.evaluate_policy`.
The evaluator alone consumes held-out effects, scientific gates, and hidden
applicability truth. The primary resampling units remain family and generator
seed; contexts are not independent bootstrap units.

The selective-risk comparator is `LTTLogisticPolicy`: a logistic scorer is fit
on synthesis rows only. A fixed threshold grid is evaluated on the calibration
rows, and the selected threshold is the lowest score threshold whose
one-sided 95% upper confidence bound for unsupported/deployed is at most 5%,
which maximizes calibration coverage under the registered risk constraint.
If no threshold satisfies the target, the policy abstains everywhere. The
feature schema, score model, threshold grid, confidence level, and target risk
are serialized before evaluation; evaluation rows never choose a threshold.

Counterexample-dose analysis holds family, positive anchors, grammar, utility
margin, and evaluation contexts fixed. Only the prefix of the registered
counterexample sequence changes: 0%, 25%, 50%, and 100%. In addition to
unsupported risk and coverage it reports support precision, support recall,
F1/IoU, predicate description length, and raw counts. Hidden applicability is
used only by the offline scoring function.

Lifecycle comparison uses the existing deterministic episode stream and four
policies: raw reuse, no reuse, frozen certificate, and Full ACRE. Every row
records the action, blocking clause, revision, utility, oracle utility, and
evidence cost. Lifecycle longitudinal regret remains distinct from
support-decision regret.

## External audit boundary

The public SkillsBench aggregate is read only. It is eligible for an external
audit only when the downloaded snapshot exposes task metadata sufficient for a
predeclared stratified split. If only aggregate scores are available, the
artifact is recorded as an external-data availability result and is not used as
evidence for an ACRE superiority claim.

## Claim routing

| Result | Allowed claim |
|---|---|
| ACRE lower risk at matched coverage | Counterexample-defined support is better on the evaluated boundary shifts. |
| Comparable risk, better auditability | ACRE is competitive with risk-sensitive gating while exposing explicit boundaries. |
| LTT lower risk | Report the result; keep ACRE's contribution to structured support identification and lifecycle composition. |
| CE precision/recall improves with dose | Counterexamples identify support boundaries; if recall falls materially, report the risk--coverage trade-off instead. |
| Missed-supported uncertainty is not wider | Do not call it epistemic concentration; report conservative opportunity cost. |
| External metadata unavailable | Do not claim public-agent external validity. |
