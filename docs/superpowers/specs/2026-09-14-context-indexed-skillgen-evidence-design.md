# Context-Indexed SkillGen Evidence Design

## Objective

Reframe the paper around the empirical support of a fixed intervention in a
given context, and add the minimum evidence needed to compare ACRE with
retrieval, positive-only, flexible selective-risk, and semantic-routing
alternatives. The existing registered artifacts remain immutable; every new
analysis receives a new revision and manifest.

## Evidence layers

1. **Controlled support frontier (E1/E2):** reuse the existing family
   generators, fixed synthesis/calibration/evaluation splits, and shared
   `Decision`/`evaluate_policy` contract. Add nearest experience,
   matched-positive, depth-matched CART, CatBoost-LTT, and a provider-backed
   semantic-router adapter when credentials are available. Counterexample dose
   and counterexample-label permutation are separate mechanism analyses.
2. **Frozen replication (E3):** apply only policies whose fit and threshold
   are serialized from the registered synthesis/calibration protocol to the
   existing 1,018-context replication. No replication outcome can change a
   policy parameter.
3. **External fixed-skill audit (E4):** use the public SkillGen ScienceWorld
   protocol. Freeze the original SkillGen-verified skill artifact before
   routing. ACRE may observe only paired verification outcomes and
   pre-execution context fields; held-out outcomes are scoring-only. Compare
   global reuse, SkillGen's public semantic router, positive-only ACRE,
   CatBoost-LTT, and Full ACRE. If provider credentials or the exact public
   split are unavailable, emit a blocked manifest rather than substituting a
   different benchmark.

## Common evaluator contract

Each policy exposes `Decision(context) -> Decision(action, score,
uncertainty, reason)`. The evaluator alone consumes held-out `tau`, scientific
gate, and hidden support truth after decisions are serialized. Primary
uncertainty units are family × generator seed for E1–E3 and task instance
with execution seed nested within instance for E4. Report coverage,
unsupported/deployed, unsupported/all, support precision/recall, support
decision regret, retained repairs, and regressions with raw counts.

## Leakage and freezing invariants

- Feature schema and baseline hyperparameters are fixed before evaluation.
- Thresholds are selected only on calibration; evaluation is never used to
  tune an operating point.
- Hidden truth and held-out outcomes are stored in a scoring-only view and
  joined after decisions are serialized.
- New artifacts record code commit, input artifact digests, model/provider
  configuration, seeds, split identifiers, and exact parameter values.
- Existing Full ACRE, historical formal evidence, hosted feasibility status,
  and sealed C/D status are not overwritten.

## Claim routing

- E1/E2 can support controlled claims about counterexample-defined support
  boundaries and their comparison with fixed alternatives.
- E3 can support frozen-rule reproducibility only.
- E4 can support a public external fixed-skill deployment claim only if the
  public protocol is reproduced and its results are complete; it is not a
  claim about prospective hosted-agent efficacy unless the experiment truly
  executes the agent prospectively.
- A semantic router matching or beating ACRE changes the wording to
  evidence-grounded and auditable support identification; it is not hidden by
  selective reporting.
