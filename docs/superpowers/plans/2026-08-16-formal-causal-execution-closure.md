# Formal Causal Execution Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining formal execution invariants without claiming external-executor evidence before an executable verifier path exists.

**Architecture:** Keep semantics in `core`, family truth and executable synthetic calibration in `benchmark/families`, and orchestration in the formal harness. Raw source patches are realized and verified before semantic action classification; only typed paired contrasts, persisted lifecycle transitions, and router-owned relation/higher-order certificates can affect governance or routing.

**Tech Stack:** Python 3.11, pytest, JSON stores, existing bounded/sequential statistics.

## Global Constraints

- Keep formal-50 results unclaimed and do not start an efficacy campaign.
- Preserve one production path; remove pre-execution semantic classification and fail-open fallbacks.
- New behavior is test-first and uses canonical public context shape.

### Task 1: Raw realization and pending candidate execution

**Files:** `core/models.py`, `benchmark/formal/run_campaign.py`, `benchmark/tests/test_formal_treatment.py`

- [ ] Add a typed raw realization record and make the formal D path materialize and verify the patch before constructing `ActionSpec`.
- [ ] Rehydrate a collecting candidate's persisted action/certificate and continue replay without worker causal fields.
- [ ] Add tests for raw-patch-first ordering and autonomous collecting-candidate continuation.

### Task 2: Paired contrast evidence and replay statistics

**Files:** `core/acre/experiments.py`, `core/acre/evidence.py`, `scripts/run_rule_replay.py`, `benchmark/formal/schedule.py`

- [ ] Emit one group-level `PairedContrastEvidence` envelope while retaining arm events for audit only.
- [ ] Use maximum preregistered replay budget, correct futility bound, and independent group seeds with shared on/off streams.
- [ ] Add tests for contrast-only assessment, replay recovery after failures, and group seed separation.

### Task 3: Canonical context and lifecycle/store transitions

**Files:** `core/public_context.py`, `benchmark/taskgen/*`, `core/mutation_journal.py`, `core/state_store.py`, `core/lifecycle.py`, validators/tests

- [ ] Flatten family parameters into `public_context.workload` and require exact equality in population validation.
- [ ] Validate journal paths relative to the store, verify mutable artifacts as transition chains, and enforce CAS on state transitions.
- [ ] Preserve immutable canonical parents for revalidate/quarantine/specialize/retire and update registry semantics.

### Task 4: Router-owned relation and higher-order execution

**Files:** `core/acre/router.py`, `benchmark/formal/schedule.py`, `benchmark/formal/run_campaign.py`, `core/acre/engine.py`, tests

- [ ] Derive pair experiment endpoint/version/family references from the canonical registry.
- [ ] Persist relation certificates and promote through typed relation governance; remove synthetic outcomes from formal promotion.
- [ ] Consume required higher-order experiments, persist certificates, and fail closed when revision refs are incomplete.

### Task 5: Executable poison/promotion replay and CI

**Files:** `benchmark/formal/run_campaign.py`, `benchmark/formal/schedule.py`, `.github/workflows/validate.yml`, `requirements-ci.txt`, end-to-end tests

- [ ] Separate synthetic family replay from executable workspace/verifier replay; only the latter can authorize formal promotion.
- [ ] Require action-specific activation instrumentation and exactly-one semantic action match.
- [ ] Add full lifecycle E2E coverage and pin NumPy in CI.

### Verification

- [ ] Run focused closure tests first.
- [ ] Run `pytest -q`, benchmark contract tests, ACRE/evolution/skill validators, population/leakage checks, and `compileall`.
- [ ] Inspect diff/status and only then commit and push.
