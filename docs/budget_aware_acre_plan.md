# Budget-Aware ACRE and Hosted-Execution Feasibility Plan

> **Status:** saved before production changes on 2026-09-02. This is the
> execution plan for the current revision; historical formal evidence remains
> immutable.

## Goal

Extend EVOSPE/ACRE so that evidence governance is resource-aware without
changing the deployability certificate, scientific gates, causal semantics, or
the interpretation of the historical 228-cell campaign. Evolution may become
frozen, but a scheduled task may not disappear because evolution consumed its
budget. The implementation must first produce reproducible provider-free
evidence; a new hosted campaign is considered only after that evidence and a
condition-blind calibration split pass.

## Current root causes and abstraction mapping

1. `benchmark/formal/run_campaign.py` accumulates D-side usage in
   `benchmark/formal/budget.EvolutionComputeBudget`. When any limit is exceeded,
   the whole `stream_id` is put into `blocked_streams`; later scheduled cells
   are written as `resource_blocked` without invocation. This contradicts
   `classify_failure()`'s registered budget-exhaustion outcome semantics.
2. `core/acre/acquisition.py` has a decision-aware surrogate based on
   information gain, sensitivity, risk, and novelty. It has no explicit link
   to unresolved certificate clauses, so it can spend on queries that cannot
   change a deployment decision.
3. `AcreMaintainer.execute_relation_experiment()` eagerly builds and executes a
   `FactorialEngine` for every supplied pair. Existing factorial confidence
   sequences and `RelationIdentifier` remain the estimator; the new policy
   decides whether a pair is eligible to reach them.
4. Existing append-only evidence and versioned rule/relation states already
   provide the natural cache substrate. Reuse will be a dependency comparison
   over these fields, not a second evidence store or a new certificate gate.
5. The hosted funnel's public task contract, verifier, and controller are
   separate from ACRE semantics. Public preflight and near-budget finalization
   are symmetric harness behavior and must not expose hidden oracle, verifier,
   anti-cheat, or performance-target details.

The mapping is therefore:

| Requested concept | Existing abstraction used | Minimal addition |
|---|---|---|
| execution reserve | `EvolutionComputeBudget`, scheduled `plan`, task `Budget` | `ExecutionReserveBudget` ledger and explicit frozen state |
| critical acquisition | `AcquisitionQuery`, `_dynamic_uncertainty`, existing confidence sequences | clause criticality and cost-normalized score |
| lazy relation | `ConservativeCausalRouter` pair requests, `FactorialEngine` | eligibility predicate before scheduler/executor |
| sequential stopping | `ReplaySequentialCertificate`, `FactorialEngine.estimate()` | first-conclusive helper and reserve-aware stop |
| certificate reuse | `RuleSpec`/`RelationSpec` provenance and `RuleState`/`RelationState` versions | dependency fingerprint comparison and invalidation result |
| public preflight | public task schema, `sandbox_preflight`, verifier boundary | public-only validator invoked symmetrically |
| calibration split | `FamilySpec`/family generator and existing pilot entry point | seed-disjoint, promotion-disjoint, evidence-disjoint manifest |

## Architecture and invariants

### 1. Execution reservation

Introduce one small ledger in `core/acre/resource_budget.py` for policy-level
experiments and a matching adapter in `benchmark/formal/budget.py` for hosted
scheduling. For a task position `t`,

```text
execution_reserve(t) = sum(minimum_execution_cost[j] for j >= t)
evolution_available = total_budget - evolution_spent - execution_reserve(t)
```

An evolution query is admitted only when its cost is no greater than
`evolution_available`. The ledger records accepted/rejected queries and
conserves total usage. When no affordable query remains it reports
`evolution-frozen`; it never marks a future scheduled task missing. Existing
budget limits remain hard limits, and exhaustion cannot mutate tasks,
conditions, gates, or scientific policy.

### 2. Certificate-critical acquisition

Extend `AcquisitionQuery` with an optional certificate clause and use the
existing observable uncertainty quantities. The default new policy selects

```text
critical(query) * uncertainty(query) / (cost(query) + epsilon)
```

only among queries that can change `deploy`, `abstain`, `specialize`,
`revalidate`, or `retire`. The old `DECISION_AWARE` path remains available as a
historical ablation/reference. `S` is a hard gate: failure short-circuits
acquisition rather than relaxing risk.

### 3. Lazy relation and sequential stopping

Add a pure eligibility function before `FactorialEngine`: both endpoints must
already satisfy the necessary individual gates, the router must have a
realistic bundle containing both, the relation result must be able to alter
that bundle, and the evolution ledger must admit the next block. An ineligible
pair returns an underidentified/abstain decision with no factorial invocation.
Existing relation labels, confidence-sequence mathematics, and router
conservatism remain unchanged. First conclusive pass/fail/equivalent stops a
sequence; a block rejected by the reserve returns underidentified.

### 4. Dependency-aware reuse

Represent dependencies as a stable mapping of context fields, model/data and
objective identity, hardware/software identity, logical-update semantics,
relation partner set, and scientific policy. A prior certificate is reusable
only when all dependencies relevant to its clauses match. Changed dependencies
invalidate only affected clauses. Reuse emits a provenance receipt referencing
the original evidence and the dependency comparison; historical evidence is
never edited.

### 5. Hosted feasibility

Add a public-only contract preflight on the same worker-visible inputs for all
conditions. Allow at most one public-contract repair when the existing task
budget permits it. Near the token/turn boundary, finalize, preflight, and
submit instead of starting another exploratory turn. These changes are
prospectively scoped to a new benchmark/method revision and must be frozen
before any hosted run.

## Tests (TDD order)

Write each focused failing test before its production change and verify the
failure for the intended reason. Required behaviors:

- future scheduled cells still launch after evolution budget exhaustion;
- no reserve violation and conserved ledger accounting;
- deterministic replay of ledger decisions;
- budget exhaustion cannot mutate scientific policy;
- unavailable evidence causes abstention, not task cancellation;
- critical acquisition ignores non-critical queries and uses existing
  confidence-sequence uncertainty;
- an irrelevant pair never invokes `FactorialEngine`;
- single-rule routing never pays pairwise relation cost;
- conclusive interval stops early;
- reserve-limited relation returns safe abstention;
- unchanged dependencies reuse a certificate with a provenance receipt;
- changed software/hardware/context invalidates only dependent clauses;
- public preflight is symmetric and hidden fields are not exposed;
- near-budget finalization does not exceed the frozen turn/token budget;
- calibration split has disjoint seed/promotion/evidence identities.

## Provider-free experimental plan

After targeted tests pass, run existing provider-free family, lifecycle, and
relation fixtures plus a new compact ablation with fixed contexts/evidence:

1. legacy ACRE;
2. execution-reserve accounting;
3. certificate-critical acquisition;
4. lazy relation;
5. full budget-aware ACRE (the composition).

Report evidence cost, query count, relation blocks, certificate decisions,
unsafe rate, recall/coverage, regret, cache/reuse hits, observation-completion
simulation, and compute frontiers. Preserve all negative rows. Use a
seed-disjoint calibration split to set feasibility budgets and finalize policy;
it never enters a treatment estimate.

## Hosted decision rule

Do not rerun historical 228 cells or call a provider before provider-free and
calibration closure. If those stages pass, freeze a new benchmark/method
revision, model snapshot, task order, seeds, controller, public preflight,
budgets, and conditions A/B/C/D. Set the cell count from calibration event
rate/variance rather than copying the old schedule. Compare D-C primarily and
A/B secondarily under equal task-execution budget and equal total compute.
If calibration shows the funnel is still infeasible, stop with a root-cause
report rather than lowering scientific gates or repeatedly tuning tasks.

## Evidence and paper policy

Every new result records exact commit, benchmark revision, seed list,
configuration, environment metadata, command receipt, and SHA256 digest. The
old `dce4d19` campaign and all current provider-free evidence stay immutable.
Only supported claims enter the paper: provider-free results are mechanism or
policy evidence; old 228 cells motivate treatment-coupled observability; a new
hosted run can support end-to-end efficacy only if its measured positive events
and pre-specified analysis justify that claim.

## Rollback strategy

Each stage is committed separately on a clearly named branch and can be
reverted without touching historical results. If a targeted test fails, stop
that stage and retain the previous implementation. If a new hosted run has a
generic infrastructure failure, preserve its root and stop diagnosis; do not
rerun automatically. If any change alters model-facing semantics or scientific
policy, abandon the prospective run and create a new revision rather than
relabeling evidence.

## Completion criteria

The work is complete only when targeted and full server tests, deterministic
replay, provider-free experiments, calibration, and the LaTeX build have fresh
evidence; the final report records exact commits/digests and distinguishes
supported, unsupported, and underidentified claims. The automation then stops
at `PAUSED`.
