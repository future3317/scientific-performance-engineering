# ACRE Formal Authority Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make formal action activation, family surfaces, statistical budgets, evidence roles, required experiments, and mutation authority match the causal evaluation contract without weakening the existing fail-closed behavior.

**Architecture:** Keep Core as the sole owner of semantics. Family catalog owns public surfaces, action-level policies, and executable/calibration contracts; formal harness consumes router requests and verifier evidence. One statistical budget and one digest definition are threaded through synthesis, replay, validation, and governance.

**Tech Stack:** Python 3.11, dataclasses, pytest, existing ACRE/FDR/replay modules.

## Global Constraints

- Do not generate formal-50 slots or claim formal efficacy results.
- Synthetic family/oracle execution remains calibration-only.
- Preserve one production path: worker proposals do not define relations or applicability.
- Every new change must have a focused regression test before broad verification.

### Task 1: Family and Action semantic closure

**Files:** `benchmark/families/catalog.py`, `core/models.py`, `benchmark/harness/verifier.py`, family benchmark modules, family tests.

- Add repair-level action metadata and public predicate grammars for all declared families.
- Make S4 activation require an exactly-one registered action match with family/action validator output.
- Remove family-default action classification and align task/anchor scientific policy references.

### Task 2: Frozen surfaces and CEGIS state

**Files:** `benchmark/families/catalog.py`, `benchmark/formal/schedule.py`, `core/acre/cegis.py`, `benchmark/formal/run_campaign.py`.

- Materialize one frozen decision lattice with disjoint synthesis, promotion, and validation IDs.
- Make synthesis always return a persisted `SynthesisResult`, including underidentified version space.
- Use the same lattice partitions for acquisition, promotion, and validation.

### Task 3: Statistical budget and evidence roles

**Files:** `core/acre/budget.py`, `core/acre/experiments.py`, `core/acre/evidence.py`, `core/models.py`, replay/governance callers, tests.

- Introduce a typed preregistered alpha ledger and deterministic group spending.
- Thread group/mix/synthesis/validation deltas instead of hard-coded 0.05 values.
- Separate synthesis, promotion representative, adversarial, and validation evidence in assessment and replay.

### Task 4: Required experiment execution and higher-order gates

**Files:** `benchmark/formal/run_campaign.py`, `benchmark/formal/schedule.py`, `core/acre/factorial.py`, `core/acre/router.py`, formal tests.

- Remove worker relation proposal input and execute router-owned pair/triple requests.
- Require joint-family execution for cross-family pairs and per-arm scientific gates.
- Make higher-order certificates require all eight gates, residual CS, exact revisions, and context digest.
- Keep formal promotion fail-closed unless an external executor receipt exists.

### Task 5: Governance integrity and lifecycle E2E

**Files:** `core/mutation_journal.py`, `core/acre/router.py`, `core/governance.py`, formal E2E tests, CI regeneration.

- Enforce journal/store diff equality and complete CAS transitions.
- Separate semantic and artifact digests; reject context-dependent parent routing.
- Add one lifecycle E2E covering activation → CEGIS → disjoint external replay → promotion → restart → pair/triple requests → journal replay.
- Regenerate canonical AST skeleton hashes and run local/CI-equivalent verification.
