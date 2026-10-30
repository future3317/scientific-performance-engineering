# Formal Release Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 28 review findings without changing ACRE semantics, then produce auditable active-30 calibration, independent approval, sealed-35 materialization, contamination PASS, and a real 50-slot formal dry-run before any efficacy claim.

**Architecture:** Keep one population source, one approval validator, one formal release manifest, and one formal scheduling path. Public pilot metadata remains public; sealed packages live outside the repository under a private root and are referenced only by digests. Formal mode consumes only the materialized frozen release manifest and the campaign config.

**Tech Stack:** Python 3.11, pytest, JSON/YAML manifests, existing benchmark harness, `core.acre`, ReferenceExecutor/bwrap, and the existing server calibration environment.

**Spec:** [local task attachment]

## Global Constraints

- Do not change ACRE, CEGIS, router, governance, lifecycle, or evidence semantics.
- Keep `formal-50 = not generated` and efficacy withheld until approval and dry-run gates pass.
- No synthetic fallback for calibration, sandbox, contamination, or formal execution.
- Active pilot population is explicit and includes 30 tasks; retired tasks are historical only.
- Formal mode accepts only `materialized_frozen` release manifests from the sealed root.

### Task 1: Explicit active population and package digests

**Files:** `benchmark/pilot_population.json`, `benchmark/formal/attest.py`, population validator/tests, existing population artifacts.

- [ ] Add the 30-task active manifest and make population validation consume it instead of directory enumeration.
- [ ] Mark H2D-23 historical/retired and add a same-family counterexample replacement entry without counting the retired task.
- [ ] Replace task-only hashing with deterministic package/Merkle digest over all task files required by the formal contract.
- [ ] Add tests for retired-task exclusion, package mutation changing the digest, and exact active count.

### Task 2: Calibration schema, unified artifact, and read-only readiness

**Files:** calibration generator/validator, `benchmark/calibration/pilot_calibration.json`, readiness scripts/tests, CI workflow.

- [ ] Require real revision, environment, outer trials, noise control, oracle CI, semantic gates, anti-cheat, and measurement fields for every active task.
- [ ] Build the unified artifact from validated calibration evidence and verify each nested digest.
- [ ] Make readiness validation read-only and require the frozen empirical artifact explicitly.
- [ ] Make every active task `eligible`; `pending` and `blocked` fail closed.

### Task 3: Single approval and release manifest

**Files:** `benchmark/formal/approval.py`, `benchmark/formal/release_manifest.py`, formal entry/claim gate, tests.

- [ ] Implement `validate_calibration_approval()` and route readiness, formal entry, and claim gate through it.
- [ ] Bind approval to code, claims, statistical protocol, active population, pilot calibration, and replacement digests.
- [ ] Add `formal_release_manifest.json` with commit, approval, preregistration, materialized population, contamination report, config, claims/protocol, and executor image.

### Task 4: Materialized sealed-35 and formal schedule/config

**Files:** `benchmark/manifests/v1.0-50-slots.json`, materialization/audit scripts, `benchmark/formal/run_campaign.py`, `benchmark/formal/schedule.py`, campaign config/tests.

- [ ] Add `preregistered_content_withheld → materialized_frozen` transitions with slot/package/visibility/split metadata.
- [ ] Require `--sealed-tasks-root`; formal mode must source schedule exclusively from the materialized frozen manifest.
- [ ] Make campaign config authoritative; duplicate CLI values must match or fail.
- [ ] Randomize A/B/C/D at task-level blocks and persist independent condition stores.
- [ ] Require slot_id and visibility on every formal record; missing visibility fails.

### Task 5: Confirmatory statistics and missingness

**Files:** `benchmark/formal/aggregate.py`, `CLAIMS.yaml`, `benchmark/formal/budget.py`, driver/tests.

- [ ] Use `aggregate_confirmatory()` for formal outputs only.
- [ ] Implement `primary_db_estimator()` matching `equal_task_then_equal_lineage` exactly.
- [ ] Decide and encode D-C as exploratory (headline narrowed to D-B) unless an explicit multiplicity plan is added.
- [ ] Route all exits through `classify_failure()` and preserve protocol-valid failures as score-zero efficacy cells.
- [ ] Accumulate and enforce `EvolutionComputeBudget` on every D compute transition.
- [ ] Align finite-lattice claims with the actual inference gate and update protocol text.

### Task 6: A_CTX, transfer, provenance, QA, and contamination

**Files:** condition builders, driver, `benchmark/formal/sandbox_preflight.py`, contamination audit, `benchmark/formal/power.py`, grader tests, manifests.

- [ ] Remove skill and treatment labels from A_CTX; generate a tokenizer-count-matched placebo context.
- [ ] Implement real frozen-transfer snapshots and independent held-out probes, or keep the claim withheld.
- [ ] Replace power shortcuts with the actual replay certificate/scheduler and outer-trial variance.
- [ ] Make contamination checks cover exact, AST, repair-pattern, lineage, and public/sealed overlap; non-PASS exits non-zero.
- [ ] Run sandbox preflight inside the real namespace for every formal worker invocation.
- [ ] Add adversarial grader QA and reject placeholder formal provenance.

### Task 7: Freeze, calibrate, approve, seal, and dry-run

**Files:** calibration and release artifacts, server private sealed root, reports.

- [ ] Freeze code/protocol after all focused tests pass.
- [ ] Run real active-30 calibration in small resource-bounded batches and preserve raw evidence.
- [ ] Run independent approval against the frozen commit and artifacts.
- [ ] Materialize sealed-35 only in the private root, run contamination audit to PASS, and create the release manifest.
- [ ] Run the real 50-slot formal dry-run with expected-cell/visibility checks; keep efficacy withheld until reviewed.

