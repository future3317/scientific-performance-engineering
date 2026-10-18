# Public-Benchmark Prospective Evaluation Design

## Contents

- [Goal](#goal)
- [Scope and evidence boundary](#scope-and-evidence-boundary)
- [Reused abstractions](#reused-abstractions)
- [Data flow](#data-flow)
- [Core data contracts](#core-data-contracts)
- [Information-flow invariants](#information-flow-invariants)
- [Splits and skill lifecycle](#splits-and-skill-lifecycle)
- [Policy freeze and shared outcomes](#policy-freeze-and-shared-outcomes)
- [Metrics and statistics](#metrics-and-statistics)
- [Failure handling and gates](#failure-handling-and-gates)
- [Artifacts](#artifacts)
- [Claim policy](#claim-policy)

## Goal

Build one `experiments/external_real/` production path for benchmark-native,
prospective evaluation of benchmark-specific frozen skills and ACRE decisions.
The first implementation target is ScienceWorld; tau3-bench and StableToolBench
remain follow-on adapters and are not started until the ScienceWorld smoke gate
passes.

## Scope and evidence boundary

- The existing generic fixed-intervention external audit is diagnostic only and
  is not extended or used to create a new efficacy claim.
- The old hosted campaign is audited for root causes but is not repaired or
  rerun as an efficacy experiment.
- No historical formal/provider-free artifact is overwritten.
- Every new external run has a new protocol, benchmark, skill, and evidence
  identity.
- No provider call occurs before the native ScienceWorld smoke protocol,
  decision-freeze checks, and leakage checks pass locally/server-side.

## Reused abstractions

The new path reuses the existing ACRE policy/evidence objects under `core/` and
`benchmark/boundary/`, the formal executor and receipt contract under
`benchmark/formal/`, and the existing paired outcome/statistical conventions.
It does not duplicate predicate learning, certificate semantics, sandbox
execution, or paired bootstrap code. `experiments/external_real/` owns only
external protocol orchestration and benchmark adapters.

## Data flow

```text
official train
  -> deterministic construction / verification / synthesis / calibration split
  -> benchmark-native SkillGen candidate
  -> execution viability
  -> global skill verification
  -> freeze skill
  -> frozen skill manifest
  -> ACRE support identification
  -> frozen policy decisions + hash
  -> shared test Y0/Y1 native executions
  -> native scoring join
  -> deploy skill OR fallback to base agent
  -> policy metrics and report
```

### Practicality and scope addendum

Skill construction and ACRE deployment are separate stages. A candidate must
first pass benchmark-native execution viability and a global verification gate
on construction/verification data before it can become a frozen skill. ACRE
does not repair an invalid skill, API syntax failure, executor instability, or
an agent's inability to solve the underlying task. It only identifies where an
already verified, frozen skill is admissible.

The two rejection outcomes are explicit:

```text
WITHHOLD_SKILL          # candidate fails viability/global verification
FALLBACK_TO_BASE_AGENT  # frozen skill is not supported for this task
```

For a valid paired outcome, the policy potential outcome is
`Y_policy = Y_skill` when the decision is deploy and `Y_policy = Y_base`
otherwise. The primary policy-level utility includes
`incremental_policy_utility = mean(Y_policy - Y_base)`. Consequently, a
zero-coverage policy has incremental utility exactly zero; abstention is not a
performance or safety win by itself.

`PolicyDecision` is serialized and hashed before any test `PairedOutcome` is
read by the scorer. The scorer asserts that decision input fields do not
contain `ground_truth`, `baseline_score`, `skill_score`, `treatment_effect`,
`oracle_support`, `judge_result`, or `post_execution_trace`, and asserts that
the decision timestamp precedes the scoring-join timestamp.

## Core data contracts

```python
@dataclass(frozen=True)
class ExternalTask:
    task_id: str
    benchmark: str
    public_context: dict[str, object]
    native_payload: dict[str, object]

@dataclass(frozen=True)
class PairedOutcome:
    task_id: str
    repeat_seed: int
    baseline_score: float
    skill_score: float
    baseline_success: bool
    skill_success: bool
    native_metrics: dict[str, object]
    execution_valid: bool
    failure_reason: str | None

@dataclass(frozen=True)
class PolicyDecision:
    task_id: str
    policy_name: str
    deploy: bool
    reason: str
    decision_hash: str
    decision_timestamp: str
```

`deploy=False` decisions use `FALLBACK_TO_BASE_AGENT`; `WITHHOLD_SKILL` is
the admission result recorded by the skill-verification stage and does not
create a deployable skill. The scorer never treats a withheld candidate as a
deployed skill.

The exact runtime representations may use the repository's existing dataclass
or JSON contracts when they are equivalent. Native adapters must expose task
loading, public-context extraction, no-skill/skill execution, and official
metric scoring without converting the benchmark into generic QA.

## Information-flow invariants

Policy training may read only construction/verification/synthesis/calibration
data and public task context. Test decisions may read only frozen training
artifacts and pre-execution public context. The following fields are forbidden
before decisions are serialized:

```text
ground_truth, baseline_score, skill_score, treatment_effect,
oracle_support, judge_result, post_execution_trace
```

Native metadata such as task family, operation type, domain, and typed public
task fields is allowed only when available before execution. If parsing is
needed, the parser prompt, model, temperature=0, schema, and parser revision
are frozen in the protocol manifest.

The ContextAdapter exposes two explicit modes: `native_structured`, which
selects typed fields already present in the benchmark task record, and
`raw_task_to_typed`, which requires a frozen parser model/version, prompt,
schema, and parser revision. The latter is a P1 comparison and cannot be used
to bypass the native ScienceWorld P0 smoke. Both modes run the same
information-flow check before a policy sees the context.

## Splits and skill lifecycle

Splits are assigned by canonical task ID and a fixed hash seed. When an official
train/dev split exists it is respected first; the train pool is then allocated
to construction, verification, synthesis, and calibration. The official test
set remains untouched. A benchmark-specific skill is generated and verified
using only its benchmark's training data. After admission, body, scripts,
resources, semantic abstraction, and content hash are frozen. ACRE can decide
where to deploy the skill but cannot modify it.

Upstream benchmark verification is evidence, not a second independent
execution by ACRE. When same-instance baseline/skill scores, repair or
regression labels, and provenance satisfy the paired-outcome contract, they
are converted to `SupportEvidence` with
`evidence_source=upstream_verification`. Only evidence acquired after that
conversion is labeled `evidence_source=acre_additional_query`. Reports must
separate reused pairs, additional pairs, total paired evidence, and marginal
provider cost due to ACRE.

## Policy freeze and shared outcomes

Before test execution, the runner materializes decisions for global reuse,
SkillGen semantic routing, nearest-evidence routing, CatBoost-LTT, positive-only
ACRE, and full ACRE. It writes `test_policy_decisions.jsonl` and its SHA256.
No policy is allowed to trigger a second test execution. Every test task gets
the same potential outcomes `Y0` (no skill) and `Y1` (frozen skill); each policy
selects one outcome after scoring.

## Metrics and statistics

Each adapter reports its official native metric first. Shared metrics are:

- deployment coverage;
- regression/deployed, defined as baseline success to skill failure among
  deployed tasks;
- repair recall, defined as deployed repairs over all repairs;
- retained repairs and induced regressions;
- paired score gain and policy utility;
- incremental policy utility, `mean(Y_policy - Y_base)`;
- official benchmark score and fallback rate;
- additional provider/router cost.

The statistical cluster is task ID; repeated executions are within-task
observations. The default uncertainty procedure is paired task-cluster
bootstrap with 10,000 replicates. Metrics are reported per benchmark before
any cross-benchmark summary.

ACRE evidence budgets are pre-registered as `B = [0, 8, 16, 32, 64]` when
the train pool supports those absolute budgets, otherwise as
`[0%, 10%, 25%, 50%, 100%]` of the available extra train evidence. Acquisition
variants are verification-only, random, uncertainty-first, and
decision-boundary/counterexample-seeking. They can read train-side pools only;
test decisions are frozen once per budget and then all variants share the same
test potential outcomes.

## Failure handling and gates

Native execution failures are recorded as invalid outcomes and never converted
to zero scientific scores. ACRE coverage of zero is a diagnostic stop, not a
threshold-tuning prompt. The ScienceWorld smoke may proceed to full ScienceWorld
only if native environment validity is at least 0.95, decisions are frozen
before outcomes, no forbidden field reaches policy input, and the official
evaluator runs successfully with nonzero ACRE support coverage. A failed gate
produces an audit artifact and stops the next benchmark.

If a task lacks a valid frozen skill, the outcome is `WITHHOLD_SKILL` and the
base agent remains the only executable policy. If a frozen skill is valid but
the support certificate is unresolved or does not match, the policy outcome is
`FALLBACK_TO_BASE_AGENT`. Underidentified support therefore means
`abstain -> optional acquire/revalidate when admissible -> otherwise remain
unresolved`; it never authorizes ACRE to repair or mutate the skill.

## Artifacts

The production output directory is `results/external_real/` and contains the
protocol, information-flow audit, skill manifest, train outcomes, support model,
calibration report, pre-frozen test decisions and digest, test outcomes, native
metrics, aggregate metrics, bootstrap results, failure audit, and `REPORT.md`.
Figures and tables are generated only from these machine-readable artifacts.

## Claim policy

The experiment may support a bounded public-benchmark prospective result only
when native execution, frozen decisions, and held-out scoring are all valid.
It does not by itself support cross-benchmark generalization, universal safety,
SkillGen generation efficacy, or hosted-agent efficacy outside the evaluated
native benchmark and fixed model/harness revision.

An optional ACRE-R operating mode is deferred until a benchmark-specific
verified skill exists, support evidence is adequate, and strict ACRE coverage
is below 0.10. If enabled later, `alpha in [0.05, 0.10, 0.20]` is frozen in
the protocol. ACRE-R is an operating mode, not a new theory contribution.
