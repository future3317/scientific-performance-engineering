# Evolution Diagnostic Effect Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make self-evolution calibration produce genuine reuse evidence and make boundary/interaction diagnostics stop only with certified evidence.

**Architecture:** Keep FamilyEnvironment, FormalConditionAdapter, StatisticalCEGIS, and RelationIdentifier as the single production paths. Change only the episode fixture selection, the interaction diagnostic stopping predicate, and boundary context selection; all promotion and confidence semantics remain unchanged.

**Tech Stack:** Python, pytest, existing benchmark harness and ACRE statistical modules.

**Spec:** `docs/superpowers/specs/2026-08-20-evolution-diagnostic-effect-design.md`

## Global Constraints

- Do not modify ACRE promotion thresholds, router semantics, or scientific contracts.
- Do not generate formal-50 or claim efficacy.
- Use tests first and run the smallest targeted tests before broader validation.
- Keep ``bounded_log_speedup_v1`` and its conservative utility LCB unchanged;
  only repair evidence scale consistency and the router's token-cost units.

## Contents

- [Global Constraints](#global-constraints)
- [Task 1: Prevent premature interaction stopping](#task-1-prevent-premature-interaction-stopping)
- [Task 2: Exercise a real positive evolution transfer cell](#task-2-exercise-a-real-positive-evolution-transfer-cell)
- [Task 3: Make boundary diagnostics honor the declared horizon](#task-3-make-boundary-diagnostics-honor-the-declared-horizon)
- [Task 4: Close replay scale and governed reuse](#task-4-close-replay-scale-and-governed-reuse)
- [Task 5: Verification and evidence update](#task-5-verification-and-evidence-update)
- [Self-review](#self-review)

---

### Task 1: Prevent premature interaction stopping

**Files:**
- Modify: `benchmark/interaction/factorial_bench.py`
- Test: `benchmark/tests/test_pilot_calibration.py`

**Interfaces:**
- `run_family_factorial_benchmark(...)` keeps its existing signature and report schema.

- [ ] **Step 1: Write the failing regression test**

Add a test that runs one family interaction surface and asserts that an
`underidentified_context_relation` result is not selected at the first block:

```python
def test_interaction_diagnostic_does_not_stop_on_underidentified_context() -> None:
    report = run_family_factorial_benchmark(count=1, seed=3)
    row = report["surface_results"][0]
    assert row["stopping_blocks"] is None or row["stopping_blocks"] >= 128
    assert row["predicted_relation"] in {"underidentified_context_relation", "unresolved"} or row["stopping_blocks"] >= 128
```

- [ ] **Step 2: Run the regression test and verify it fails**

Run `pytest benchmark/tests/test_pilot_calibration.py::test_interaction_diagnostic_does_not_stop_on_underidentified_context -q`.
The current implementation fails because it stops at block 8 on the
`underidentified_context_relation` sentinel.

- [ ] **Step 3: Implement the minimal stopping predicate**

In `run_family_factorial_benchmark`, accept a stopping decision only when it
is not one of `{"unresolved", "underidentified_context_relation"}`. Preserve
the final scheduled estimate when no decision is certified.

- [ ] **Step 4: Run the regression test and existing interaction tests**

Run `pytest benchmark/tests/test_pilot_calibration.py benchmark/interaction -q`.

- [ ] **Step 5: Commit**

Commit as `fix: prevent premature interaction diagnostic stopping`.

### Task 2: Exercise a real positive evolution transfer cell

**Files:**
- Modify: `benchmark/tasks/EVOL-EQUIVARIANT-SPECIALIZE-30/episodes/equivariant_specialization_episode.yaml`
- Modify: `benchmark/tasks/EVOL-COMPILER-DRIFT-20/episodes/compiler_drift_episode.yaml`
- Test: `benchmark/tests/test_pilot_calibration.py`

**Interfaces:**
- Episode manifests remain valid miniyaml and continue through `run_episode`.

- [ ] **Step 1: Change only the episode task sequence**

For the compiler episode, use `CORE-COMPILE-DYNAMIC-11` (or an equivalent
positive dynamic-shape context) in `same_family_transfer`, keep
`CORE-COMPILE-TINY-12` as a counterexample in a separate phase, and retain
`CORE-KERNEL-FUSION-09` as cross-family negative. For the equivariant episode,
use `SCIML-EQUIV-LOWORDER-26` in `same_family_transfer` and keep
`SCIML-EQUIV-RECOMPUTE-06` as the explicit counterexample. No rule trigger or
FamilyEnvironment effect is changed.

- [ ] **Step 2: Run the focused evolution test**

Run `pytest benchmark/tests/test_pilot_calibration.py::test_evolution_episode_has_transfer_and_regret_evidence -q`.
The matched transfer context must be present in the raw episode record. If
`rule_reuse_utility` remains null, retain that result: the router's valid
utility-effect LCB is still negative and the episode is not efficacy evidence.

- [ ] **Step 5: Commit**

Commit as `fix: exercise governed rule reuse in evolution episodes`.

### Task 3: Make boundary diagnostics honor the declared horizon

**Files:**
- Modify: `benchmark/boundary/families.py`
- Test: `benchmark/tests/test_pilot_calibration.py`

**Interfaces:**
- `family_cases(...)` keeps its existing return keys and `BoundaryCase` type.

- [ ] **Step 1: Write the failing test**

Add tests that pass the requested surface horizon to a SciML family and that
the compile grammar can express its four-literal public action boundary.

```python
report = run_boundary_family("equivariant_head", surface_count=100, seed=7)
assert report["pool_sizes"]["representative_pool"] > 8
compile_report = run_boundary_family("compile", surface_count=100, seed=7)
assert compile_report["sealed_errors"] == 0
```

- [ ] **Step 2: Run it and verify it fails**

Run the two boundary regression tests and confirm the current implementation
fails because it silently falls back to the 24-surface view and clamps the
compile grammar to three literals.

- [ ] **Step 3: Implement the minimal diagnostic fixes**

Make `run_boundary_family` pass `surface_count` for every canonical family,
scale the evidence slice with the requested horizon, and allow four literals
only for the compile family. Do not alter `BoundaryObservation`, CEGIS, or
sealed error logic.

- [ ] **Step 4: Run boundary tests and a bounded diagnostic**

Run `pytest benchmark/tests/test_pilot_calibration.py benchmark/boundary -q`,
then run the underidentified families at `surface_count=24` and save the
diagnostic output without changing approval artifacts.

- [ ] **Step 5: Commit**

Commit as `fix: target declared family boundary thresholds`.

### Task 4: Close replay scale and governed reuse

**Files:**
- Modify: `benchmark/harness/evolution.py`
- Modify: `core/acre/router.py`
- Test: `benchmark/tests/test_pilot_calibration.py`

- [x] Episode replay uses `UTILITY_LOG_SCALE` rather than a local scale.
- [x] Sparse positive strata collect three preregistered slices before CEGIS.
- [x] Router prompt cost is normalized by the active token budget while the
      hard token-budget feasibility check remains unchanged.
- [x] Compiler episode regression asserts positive transfer/reuse and zero
      negative-transfer rate.

### Task 5: Verification and evidence update

**Files:**
- Modify only if required by measured results: `benchmark/calibration/staged-probes/2026-08-20/summary.json`, `benchmark/population_report.json`

- [x] **Step 1: Run focused and full tests**

Run `pytest benchmark/tests/test_pilot_calibration.py benchmark/interaction benchmark/boundary -q`, then `pytest -q`.

- [x] **Step 2: Run the short evolution and diagnostic probes**

Run the existing `run_drift_poison` probe and the bounded boundary/interaction
diagnostics. Do not run active-30, formal-50, or a long campaign.

- [ ] **Step 3: Record only observed outcomes**

Update calibration artifacts only with measured reuse/transfer and diagnostic
status. Keep `calibration_approval=blocked` unless all existing gates pass.

- [ ] **Step 4: Commit**

Commit as `docs: record evolution and diagnostic closure results` only if
artifact updates are necessary.

---

## Self-review

- Evolution transfer, interaction stopping, and threshold-aware boundary
  selection each have a dedicated test and preserve the existing interfaces.
- No task changes statistical definitions, promotion thresholds, or router
  semantics.
- No formal-50 or efficacy claim is produced.
