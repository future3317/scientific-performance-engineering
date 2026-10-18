# Public ACRE Treatment and Evolving-Skill Evaluation Design

## Contents

- [Status and scope](#status-and-scope)
- [Frozen protocol before held-out outcomes](#frozen-protocol-before-held-out-outcomes)
- [Shared policy and evaluator contract](#shared-policy-and-evaluator-contract)
- [Fixed-skill public audit](#fixed-skill-public-audit)
- [Online evolving-skill treatment](#online-evolving-skill-treatment)
- [Failure handling and stopping](#failure-handling-and-stopping)
- [Artifact and claim decision table](#artifact-and-claim-decision-table)

## Status and scope

This design starts a new evidence revision, `public-acre-treatment-20260915-v1`.
It does not overwrite the provider-free freeze, historical formal evidence, or
the 2026-09-14 fixed-skill calibration. The experiment produces new manifests,
receipts, and result artifacts bound to the implementation commit, benchmark
revisions, model/configuration, and task split.

The evidence package has two related tracks:

1. **Fixed-skill public audit.** Reproduce the comparable SkillGen-style paired
   setting on the selected public agent benchmarks. The generated skill is
   immutable during held-out evaluation. This track tests whether context-indexed
   ACRE is useful beyond the controlled substrate and compares against the public
   semantic router and selective baselines.
2. **Online evolving-skill treatment.** Compare no persistent skill (A), raw
   experience reuse (C), and ACRE-governed evolution (D) on a sequential stream.
   Skill versions are immutable records, but D may create `v1`, `v2`, ... from
   prior experience. The primary estimand is policy-level net utility,
   `Delta_D-C = U(D) - U(C)`, under equal task-execution budget and under equal
   total compute.

The fixed-skill track is not used as evidence for the evolving-policy estimand;
the evolving track is not described as a fixed-intervention effect.

## Frozen protocol before held-out outcomes

Before the first held-out outcome is visible, write a protocol manifest containing:

- SkillGen repository commit and benchmark/data commits or release identifiers;
- exact benchmark task lists and train/calibration/test split digests;
- model snapshot, provider/model configuration, prompt/template revision,
  temperature, maximum output, tool and wall-clock budgets;
- public context schema and scoring-only fields;
- policy implementation commit and parameter grid/defaults;
- random seeds, task order, execution-replicate count, and retry rule;
- evaluator schema, bootstrap unit, and all primary/secondary metrics;
- campaign identity and output paths.

The test split is task-disjoint from synthesis/calibration. A task's outcome,
judge result, hidden score, trajectory, and future task outcome are scoring-only.
Every policy emits an append-only decision ledger entry before those fields are
joined. Any infrastructure-invalid cell may be replayed once with the same seed
under the pre-registered rule; correctness, contract, scientific-gate, or
negative-transfer outcomes are never retried as infrastructure failures.

The staged execution rule is part of the protocol: first run one ScienceWorld
smoke to validate the complete path; only if the smoke is structurally valid run
the public suite and online treatment. A blocked or incomplete adapter writes a
blocked manifest and does not create efficacy rows.

## Shared policy and evaluator contract

Each policy implements:

```text
Decision(context, available_evidence, skill_snapshot)
    -> {reuse, score, uncertainty, reason, certificate_id, skill_version}
```

`context` contains only execution-before-visible fields: benchmark/task identity,
public category/difficulty, tool/interface metadata, interaction mode, model and
harness identity, and the current public observation where the benchmark makes
it available. `available_evidence` is restricted to prior synthesis/calibration
evidence and earlier sequential tasks. The evaluator serializes all decisions,
then joins paired outcomes and computes every policy's metrics from the same rows.

The evaluator reports, per benchmark/model/condition and macro-averaged over
benchmark-by-model cells:

- official score and paired score delta;
- deployment coverage, unsupported/deployed, unsupported/all, and abstention;
- repair recall, negative-transfer rate, correctness and measurement reach;
- unsafe deployment rate and support-decision regret;
- promotion/rejection counts, immutable skill-version count and cache/reuse hits;
- task compute, evolution compute, total compute, cost per useful improvement;
- observation completion and cumulative net utility/longitudinal regret.

The unit for uncertainty is task instance, with execution seeds nested inside an
instance and benchmark/model cells retained for macro aggregation. Paired
bootstrap resamples instances, not individual rollout rows.

## Fixed-skill public audit

The initial public suite is ScienceWorld, Mind2Web, ToolBench, and MCP-Bench,
using the exact SkillGen-compatible preparation/splits that can be verified from
their pinned source. The first implementation uses the available provider-backed
model configuration and adds a second backbone only when its pinned weights,
runtime, and evaluator are available under the same contract; the manifest must
record a blocked cell rather than silently substitute a model.

The intervention is generated and globally verified before held-out routing, then
stored immutably. Policies are Global Skill, the public SkillGen semantic router,
nearest/kNN evidence, CatBoost-LTT over the pre-registered public context, and
Full ACRE. Positive-only ACRE is an ablation. All policies see identical cached
no-skill/skill paired rollouts and execution replicates.

Primary external claim is conditional: if ACRE reduces regression risk while
retaining useful repairs at a non-degenerate operating point, the paper may say
that intervention-grounded context certification improves held-out public skill
deployment. If semantic routing is comparable, the claim is auditability and
certificate semantics rather than superiority. If ACRE abstains broadly or a
selective baseline dominates, the paper reports that result and narrows the
claim.

## Online evolving-skill treatment

The sequential stream is group-split by task family, mechanism, lineage, and
mutation template. The visible acquisition segment may update C/D stores; same-
family transfer, cross-family transfer, drift, poisoned-experience, and recovery
segments remain held out according to the existing benchmark split contract.

- **A:** no persistent skill or experience.
- **C:** raw prior experience retrieval under the same evolution token/tool budget;
  it cannot promote, specialize, retire, or revalidate canonical skill versions.
- **D:** experience capture plus ACRE candidate admission, paired replay,
  deployability certification, promotion, specialization, revalidation, and
  retirement. Optional acquisition obeys the execution reserve and can return
  unresolved/abstain.

Every skill version is content-addressed and append-only with parent version,
creation task index, evidence IDs, context predicate, certificate clauses,
model/data/objective identity, policy decision, and provenance receipt. No task
may read its own outcome or any future outcome. A decision ledger entry is written
before execution; the outcome and subsequent maintenance event are appended only
after verification.

The primary contrast is paired D minus C. Secondary contrasts are A minus C and
D minus A. Report both equal task-execution budget and equal total compute. No
composite headline metric is used unless its bounded utility weights are frozen
in the protocol; cumulative net utility remains a reported trajectory.

## Failure handling and stopping

Provider/network/runtime failures are classified as infrastructure-invalid and
can be replayed once under the frozen retry rule. Public contract failure,
correctness failure, scientific-gate failure, negative transfer, abstention, and
unresolved evidence are valid scientific outcomes. A missing receipt invalidates
the cell; it does not become a zero score. The campaign stops before the next
phase if the smoke does not establish decision serialization, outcome separation,
paired execution, and complete resource receipts.

## Artifact and claim decision table

| Result | Permitted paper claim |
|---|---|
| Smoke only | execution-path readiness; no efficacy |
| Fixed-skill public paired result | external held-out fixed-intervention deployment claim, conditional on observed metrics |
| Evolving D-C with complete paired observations | policy-level net-utility/treatment claim, with both resource views |
| Zero/non-degenerate measurement or incomplete observations | no efficacy claim; report feasibility boundary and preserve raw artifacts |
| Selective baseline matches or beats ACRE | retain auditable certification contribution; remove superiority wording |
| Positive result only on controlled SPE-EvoBench | mechanism evidence, not public-agent efficacy |

Paper changes are deferred until the artifacts, manifests, receipts, tests, and
analysis are complete. The final claim matrix is generated from observed results,
not from this design.
