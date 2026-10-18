# Public-Benchmark Prospective Evaluation Implementation Plan

## Contents

- [Global Constraints](#global-constraints)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and validate a benchmark-native prospective evaluation path, starting with a ScienceWorld smoke and stopping safely when its pre-registered gates fail.

**Architecture:** `experiments/external_real/` owns protocol orchestration, immutable policy freezing, native adapter boundaries, scoring, and reporting. Existing ACRE policy/evidence code, formal executor/receipt code, and paired statistics remain the single sources of truth.

**Tech Stack:** Python in the server `equivcompiler` environment, existing repository dependencies, pytest, JSONL/JSON artifacts, and existing plotting/reporting tools.

**Spec:** `docs/superpowers/specs/2026-09-15-external-real-design.md`

## Global Constraints

- Execute tests, validators, and evidence-producing commands only on `dbcloud` in `/home/workspace/lrh/CODE/scientific-performance-engineering` with `/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python`.
- Preserve current uncommitted user changes; do not reset, stash, checkout, or overwrite them.
- Do not modify historical formal/provider-free artifacts or rerun the old generic hosted campaign.
- Do not call a provider before the ScienceWorld smoke protocol and information-flow tests pass.
- Do not use test outcomes, oracle support, or hidden evaluator results to tune policy thresholds.
- Use one production path under `experiments/external_real/`; reuse existing ACRE and executor abstractions.
- A candidate must pass execution viability and global verification before it is frozen; ACRE never repairs or mutates a skill.
- Reject outcomes are `WITHHOLD_SKILL` at admission and `FALLBACK_TO_BASE_AGENT` at deployment.
- Report `incremental_policy_utility = mean(Y_policy - Y_base)`; zero coverage has utility exactly zero.
- Reuse valid upstream verification pairs as `SupportEvidence`; report them separately from ACRE additional queries.
- Do not implement ACRE-R until the explicit verified-skill, evidence-adequacy, and strict-coverage conditions in the spec hold.

---

### Task 1: Phase-0 failure audit

**Files:**
- Create: `scripts/audit_external_failure.py`
- Create: `results/external_failure_audit.json` (server-generated, not committed if repository policy excludes results)
- Create: `results/external_failure_audit.md` (server-generated, not committed if repository policy excludes results)
- Test: `tests/test_external_failure_audit.py`

**Interfaces:**
- `audit_skillgen_suite(path: Path) -> dict[str, object]`
- `audit_hosted_campaign(path: Path) -> dict[str, object]`
- `write_failure_audit(out_dir: Path, suite_paths: Mapping[str, Path], hosted_path: Path | None) -> None`

- [ ] Write tests for four benchmark summaries, the six root-cause labels, and hosted A/B/C/D funnel conservation.
- [ ] Run the new tests on the server and verify they fail because the audit module is absent.
- [ ] Implement artifact discovery and explicit “missing artifact” records without silently substituting current `main` or unrelated revisions.
- [ ] Implement benchmark-level counts: train evidence, anchors, counterexamples, unresolved, repairs, regressions, both-pass, both-fail, feature combinations, candidate predicates, and rejection reasons.
- [ ] Implement the hosted funnel with raw-receipt denominators and A/B/C/D strata; preserve unknown fields instead of inferring them.
- [ ] Classify root causes only from observed evidence into `protocol_failure`, `intervention_failure`, `context_schema_failure`, `evidence_sparsity`, `executor_failure`, or `treatment_no_effect`.
- [ ] Run tests again and generate the audit from existing local/server artifacts without provider calls.

### Task 2: Native protocol and immutable contracts

**Files:**
- Create: `experiments/external_real/__init__.py`
- Create: `experiments/external_real/protocol.py`
- Create: `experiments/external_real/models.py`
- Create: `experiments/external_real/freeze.py`
- Create: `tests/test_external_real_protocol.py`

**Interfaces:**
- `ExternalTask`, `PairedOutcome`, `PolicyDecision`
- `ProtocolManifest.from_dict(payload: Mapping[str, object]) -> ProtocolManifest`
- `assign_split(task_id: str, seed: str, phase_sizes: tuple[int, int, int, int]) -> str`
- `freeze_decisions(path: Path, decisions: Sequence[PolicyDecision]) -> str`
- `assert_information_flow(policy_input: Mapping[str, object]) -> None`

- [ ] Write failing tests for task-ID deterministic splits, forbidden policy fields, decision hashes, decision-before-scoring timestamps, and artifact conservation.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Implement the smallest immutable dataclasses and JSON serialization compatible with current repository conventions.
- [ ] Implement decision hashing over canonical JSON and a single writer that writes JSONL before outcome files can be consumed.
- [ ] Implement information-flow assertions that reject forbidden fields rather than deleting them.
- [ ] Run focused tests and inspect serialized examples.

### Task 3: Paired runner and native scoring interface

**Files:**
- Create: `experiments/external_real/paired_runner.py`
- Create: `experiments/external_real/scorer.py`
- Create: `experiments/external_real/report.py`
- Create: `tests/test_external_real_runner.py`
- Create: `tests/test_external_real_scorer.py`

**Interfaces:**
- `NativeAdapter.load_tasks(split: str) -> Sequence[ExternalTask]`
- `NativeAdapter.run_pair(task: ExternalTask, skill: FrozenSkill, seed: int) -> PairedOutcome`
- `score_policies(decisions, outcomes, native_scorer) -> dict[str, object]`
- `paired_task_bootstrap(rows: Sequence[Mapping[str, object]], reps: int = 10000, seed: int = 0) -> dict[str, object]`

- [ ] Write tests proving each task is executed once per condition and policy selection happens after shared outcomes are available.
- [ ] Write tests for official native metric precedence, invalid execution handling, repair/regression denominators, and task-cluster bootstrap.
- [ ] Run focused tests and verify they fail before implementation.
- [ ] Implement runner/scorer interfaces with no generic QA conversion and no hidden fields in policy input.
- [ ] Implement metric aggregation per benchmark and explicit failure reasons.
- [ ] Run focused tests and inspect a synthetic native adapter fixture.

### Task 4: ScienceWorld native adapter

**Files:**
- Create: `experiments/external_real/context/scienceworld.py`
- Create: `experiments/external_real/adapters/scienceworld_native.py`
- Create: `tests/test_scienceworld_native.py`

**Interfaces:**
- `ScienceWorldAdapter.protocol_manifest() -> dict[str, object]`
- `ScienceWorldAdapter.load_tasks(split: str) -> Sequence[ExternalTask]`
- `ScienceWorldAdapter.public_context(task: ExternalTask) -> Mapping[str, object]`
- `ScienceWorldAdapter.run_pair(task, frozen_skill, seed) -> PairedOutcome`
- `ScienceWorldAdapter.native_score(raw_result) -> dict[str, object]`

- [ ] Write fixture tests for public-context extraction and rejection of goal/reference trajectory fields.
- [ ] Write fixture tests for native evaluator invocation and invalid executor receipts.
- [ ] Run focused tests and verify they fail before the adapter exists.
- [ ] Implement the adapter against the installed official simulator/evaluator discovered on the server; fail explicitly if the pinned dependency is unavailable.
- [ ] Record benchmark commit/tag, simulator version, evaluator identity, model/harness identity, and environment metadata in the manifest.
- [ ] Run focused adapter tests without provider calls.

### Task 5: SkillGen train-side pipeline and ACRE support model

**Files:**
- Create: `experiments/external_real/skill_pipeline.py`
- Create: `experiments/external_real/support.py`
- Create: `tests/test_external_real_skill_pipeline.py`

**Interfaces:**
- `build_benchmark_skill(adapter, split_manifest, generation_seed) -> FrozenSkill`
- `verify_skill(adapter, skill, verification_tasks) -> VerificationReport`
- `fit_support_model(train_outcomes, public_contexts) -> SupportModel`
- `decide_policies(support_model, test_tasks) -> Sequence[PolicyDecision]`

- [ ] Write tests for benchmark isolation, frozen skill hashes, admission only after verification, and train-only evidence.
- [ ] Run focused tests and verify the expected failures.
- [ ] Reuse existing SkillGen release/admission and ACRE predicate semantics where present; do not create a generic fixed intervention.
- [ ] Implement the registered generation seed budget of at most three and persist every attempted seed.
- [ ] Implement the viability report with both-pass, repair, regression, both-fail, effect distribution, and context-group counts.
- [ ] Implement all registered policy baselines and full ACRE from training evidence only.
- [x] Gate every candidate through execution viability and global verification before freezing; record `WITHHOLD_SKILL` without invoking ACRE repair.
- [x] Convert contract-valid upstream verification pairs into `SupportEvidence(evidence_source="upstream_verification")` and label new ACRE queries separately.
- [x] Keep ACRE support identification train-side only; unresolved support falls back to the base agent.
- [ ] Run focused tests with offline fixtures, not provider calls.

### Task 5A: Evidence-budget and fallback accounting

**Files:**
- Modify: `experiments/external_real/models.py`
- Modify: `experiments/external_real/scorer.py`
- Modify: `experiments/external_real/support.py`
- Test: `tests/test_external_real_scorer.py`
- Test: `tests/test_external_real_skill_pipeline.py`

**Interfaces:**
- `SupportEvidence.evidence_source: Literal["upstream_verification", "acre_additional_query"]`
- `score_policies(...)` reports `incremental_policy_utility`, `official_score`, and `fallback_rate`.
- `evidence_budget_grid(available_extra: int | None) -> tuple[int, ...]`

- [x] Write failing tests for fallback selecting baseline score, exact zero utility at zero coverage, source-separated evidence counts, and fixed budget grids.
- [x] Run those focused tests on the server and verify they fail for the expected missing-contract reasons.
- [x] Implement the smallest source-tagged evidence and scorer changes without duplicating native execution.
- [x] Run focused tests and inspect metric denominators.

### Task 6: ScienceWorld smoke protocol

**Files:**
- Create: `experiments/external_real/run_scienceworld.py`
- Create: `experiments/external_real/manifest.py`
- Create: `tests/test_scienceworld_smoke_manifest.py`

**Interfaces:**
- `build_smoke_manifest(benchmark_revision: str, model_id: str, seeds: Sequence[int]) -> ProtocolManifest`
- `run_scienceworld_smoke(manifest, out_dir) -> dict[str, object]`
- `evaluate_smoke_gate(report) -> tuple[bool, list[str]]`

- [ ] Write tests that reject missing decision digest, outcome-before-decision timestamps, zero support coverage, and invalid native evaluator results.
- [ ] Run focused tests and verify they fail before the smoke runner exists.
- [ ] Implement the fixed 20 construction / 10 verification / 20 support / 20 test smoke manifest and fixed seeds.
- [ ] Freeze decisions and hash them before any test pair execution.
- [ ] Run the smoke on the server only after provider-use authorization is represented in the manifest and all local/server regression tests pass.
- [ ] Evaluate environment validity, decision ordering, information-flow audit, evaluator validity, and ACRE coverage.
- [ ] If any gate fails, persist diagnostics and stop before full ScienceWorld or another benchmark.

### Task 6A: ContextAdapter and deferred ACRE-R contract

**Files:**
- Modify: `experiments/external_real/protocol.py`
- Create: `experiments/external_real/context/adapter.py`
- Test: `tests/test_external_real_protocol.py`

**Interfaces:**
- `ContextAdapter(mode: Literal["native_structured", "raw_task_to_typed"], ...)`
- `ContextAdapter.extract(task: ExternalTask) -> Mapping[str, object]`

- [x] Write failing tests for native structured extraction, parser metadata freeze, and evaluator-field rejection.
- [x] Implement native structured mode first; keep raw-task parsing unavailable until its model, prompt, schema, and revision are in the manifest.
- [x] Record the comparison as P1 and do not let it block the native ScienceWorld P0 smoke.
- [x] Add an explicit deferred ACRE-R manifest validator requiring verified skill, adequate support evidence, strict coverage `< 0.10`, and `alpha in {0.05, 0.10, 0.20}`.

### Task 7: Tables, figures, report, and paper boundary update

**Files:**
- Modify: `experiments/external_real/report.py`
- Create: `tables/public_benchmark_main.csv`
- Create: `tables/public_benchmark_per_skill.csv`
- Create: `figures/public_risk_coverage.pdf`
- Create: `figures/public_repairs_regressions.pdf`
- Create: `results/external_real/REPORT.md`
- Modify: paper only after smoke/full-run artifacts are valid

**Interfaces:**
- Reports consume only frozen machine-readable artifacts and include source
  digests, split identities, decision digest, native evaluator identity, and
  benchmark-specific metrics.

- [ ] Write report tests for missing artifacts, denominator labels, and claim-boundary text.
- [ ] Run report tests and verify failures before implementation.
- [ ] Implement report generation and figure/table generation from artifacts only.
- [ ] Run the report generator on synthetic fixture artifacts.
- [ ] Update the paper only after the actual smoke/full ScienceWorld outcome and gate status are known.

### Task 8: Verification and integration

- [ ] Run focused external-real tests on the server.
- [ ] Run the existing provider-free test suite on the server; record failures without weakening tests.
- [ ] Run the ScienceWorld smoke gate and inspect all produced manifests and raw receipts.
- [ ] Run LaTeX build and paper cross-reference checks only if valid new evidence exists.
- [ ] Review git diff and preserve unrelated dirty user changes.
- [ ] Commit only the new implementation/docs/tests and explicitly exclude private/local analysis files.
- [ ] Push only the benchmark repository branch if the remote and branch policy permit it; do not invent a paper remote.
