# Evolution 600-Second Calibration Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve or isolate the observed evolution episode timeout, produce declared-repetition evidence, and allow population approval only when all 30 active tasks pass their existing calibration gates.

**Architecture:** Keep the existing episode verifier, task contracts, population eligibility rules, and strict-formal validator as the single production paths. First classify the timeout with stage-level evidence and compare it with the passing poison episode; then make one minimal code or fixture change only if the evidence identifies a real defect. Re-run affected evolution tasks with their declared three repetitions, run the unchanged 30-task population, and derive/review approval from those artifacts.

**Tech Stack:** Python, PyTorch, existing benchmark harness, pytest, JSON artifacts, SSH alias `dbcloud`, server environment `equivcompiler`.

**Spec:** `benchmark/BENCHMARK_DESIGN.md`, `benchmark/taskgen/validate_population.py`, and the evolution task manifests.

## Global Constraints

- Use the server project environment `equivcompiler`; do not use Conda `base`.
- Preserve the 600-second task contract; do not raise or reinterpret it to hide a timeout.
- Preserve eligibility thresholds, repetition requirements, ACRE/promotion/router semantics, and scientific task meaning.
- A timeout remains an invalid protocol result until a matched rerun proves the root cause was corrected.
- Do not hand-edit result, pilot-calibration, or approval JSON to make a gate pass.

### Task 1: Preflight and classify the current 30-task evidence

**Files:**
- Read: `benchmark/taskgen/validate_population.py`
- Read: `scripts/run_active30_calibration.py`
- Read: `benchmark/harness/verifier.py`
- Read: server `population_report.json`, `empirical.json`, and `pilot_calibration.json`

- [ ] Confirm the server revision, environment, GPU/process topology, and current artifact paths.
- [ ] Extract every blocked task's rejection flags and the exact gate that blocks it; distinguish invalid execution from scientific/noise/semantic rejection.
- [ ] Confirm each evolution manifest declares `measurement.repetitions: 3` and record the runner's current outer-trial mapping.
- [ ] Record the current timeout stage, wall time, and subprocess command for `EVOL-COMPILER-DRIFT-20` and `EVOL-EQUIVARIANT-SPECIALIZE-30`.

### Task 2: Reproduce and profile the timeout before changing code

**Files:**
- Read: `benchmark/harness/evolution.py`
- Read: `benchmark/harness/verifier.py`
- Read: `benchmark/formal/reference_executor.py`
- Create outside Git: bounded server profiling artifacts under `~/RESULTS/SPE/`

- [ ] Run each failing evolution task separately with one outer trial and the existing 600-second limit, preserving raw stdout/stderr and process cleanup evidence.
- [ ] Measure materialization, phase replay, store transitions, governance/promotion, and candidate/baseline subprocess time for the failing and passing episode.
- [ ] Compare cache state, CPU/GPU utilization, process/thread topology, and fixture/task sequence between the failing full campaign and the passing single-task probe.
- [ ] State one falsifiable root-cause hypothesis and test it with the smallest bounded experiment; do not patch until the failing behavior is reproduced or the external resource cause is documented.

### Task 3: Add a regression test and implement one minimal root-cause fix

**Files:**
- Test: the smallest existing harness test covering the identified stage
- Modify: only the implementation or episode manifest named by the confirmed root cause

- [ ] Write a failing regression test that reproduces the observed timeout mechanism or incorrect resource/fixture behavior.
- [ ] Run the regression test and verify it fails for the expected reason.
- [ ] Implement the smallest single fix while preserving the 600-second contract and fail-closed timeout classification.
- [ ] Run the targeted regression plus affected harness/eligibility tests; if the hypothesis fails, return to Task 2 with a new hypothesis instead of stacking fixes.

### Task 4: Generate declared-repetition evolution evidence

**Files:**
- Create outside Git: server raw/evidence directory for the frozen revision

- [ ] Run all three evolution tasks with `outer_trials=3`, matching their declared `measurement.repetitions` and using one isolated GPU/process topology.
- [ ] Verify each task has three complete paired episode records, valid delta vectors, semantic gates, noise-control records, and no protocol timeout.
- [ ] Re-run only a task that fails for a confirmed execution defect; retain genuine scientific/noise rejection as blocked evidence.

### Task 5: Complete population calibration and approval review

**Files:**
- Create outside Git: final server `population_report.json`, `pilot_calibration.json`, and `calibration_approval.json`
- Read: `benchmark/taskgen/validate_population.py`, approval schema, and `benchmark/BENCHMARK_DESIGN.md`

- [ ] Run the unchanged active-30 calibration with the required repetition settings and persist all raw records.
- [ ] Generate `calibration_approval.json` through the repository's approval path, binding the exact population/evidence/revision metadata.
- [ ] Review approval status, eligible/blocked IDs, rejection reasons, evolution repetition counts, timeout flags, and artifact references against raw evidence.
- [ ] Run `validate_population --strict-formal` with the final paths and report the actual exit code and any remaining blocking task; do not claim 30/30 unless the validator says so.

## Self-review

- The plan treats timeout, repetition, eligibility, approval, and strict-formal as separate gates.
- No threshold, timeout, statistical rule, scientific contract, or promotion semantics are changed.
- A genuine blocker remains visible rather than being converted to a positive or omitted result.
