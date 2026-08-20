# Campaign Readiness Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish whether the current 30-task calibration can proceed by running bounded real executor trials, fixing only observed execution/calibration blockers, and preserving fail-closed efficacy claims.

**Architecture:** Use the existing `scripts/run_active30_calibration.py` and `benchmark.formal` paths on the server in the `equivcompiler` environment. Start with one-task and two-task probes, persist raw results and fingerprints, classify failures before reuse, and expand only after execution-valid gates pass. No new scheduler, store, executor abstraction, or ACRE method semantics.

**Tech Stack:** Python, PyTorch, existing ReferenceExecutor/bubblewrap, pytest, JSON calibration artifacts, SSH alias `dbcloud`.

**Spec:** `benchmark/BENCHMARK_DESIGN.md` and the current calibration gate in `benchmark/calibration/calibration_approval.json`.

## Global Constraints

- Use the server Conda environment `equivcompiler`; never use Conda `base` or the obsolete `equicomp` name.
- Do not generate formal-50 or claim efficacy during this plan.
- Keep ACRE, CEGIS, router, governance, lifecycle, evidence semantics, statistical thresholds, and benchmark task definitions unchanged unless a test exposes a concrete bug.
- Stop a stream on protocol/infrastructure failure; do not convert failures into scientific negatives.
- Record hardware/software/executor provenance and keep no-op or inconclusive outcomes separate from efficacy evidence.

### Task 1: Preflight current implementation and server capacity

**Files:**
- Read: `scripts/run_active30_calibration.py`
- Read: `benchmark/formal/reference_executor.py`
- Read: `benchmark/calibration/calibration_approval.json`
- Read: `benchmark/pilot_population.json`

**Interfaces:**
- Consumes: current `main` revision and existing calibration artifacts.
- Produces: a bounded-run command, selected GPU/process budget, and a preflight record outside Git.

- [ ] Verify local tree is clean and server is on the pushed revision with `equivcompiler` active.
- [ ] Confirm target GPU has sufficient free memory and no competing project workloads.
- [ ] Run `python scripts/run_active30_calibration.py --help` and inspect the existing task filter/outer-trial options.
- [ ] Select one representative Dynamic task and one non-compile positive/evolution task for the first probe; do not launch all 30 tasks yet.

### Task 2: Run a bounded real calibration probe

**Files:**
- Create outside Git: server result directory under `~/RESULTS/SPE/`.

**Interfaces:**
- Consumes: Task 1 command and current executor contract.
- Produces: raw result, noise-control artifact, envelope, summary, and process-cleanup evidence.

- [ ] Run one Dynamic task with `outer_trials=1` through the existing calibration runner.
- [ ] If it is execution-valid, run one representative evolution/SciML task with `outer_trials=1`.
- [ ] Capture return code, timeout/protocol flags, receipt validity, surviving PIDs, scientific gates, and wall time.
- [ ] Classify each result as reusable, rerun, blocked-requires-revision, or inconclusive using the existing classifier; never aggregate a non-reusable result.

### Task 3: Fix only observed blockers and regress them

**Files:**
- Modify only the concrete implementation file named by the failing traceback.
- Test: the smallest existing test module covering that path, plus one regression test for the observed failure.

**Interfaces:**
- Consumes: failing artifact/traceback from Task 2.
- Produces: minimal patch with no semantic redesign.

- [ ] For an execution/protocol failure, reproduce it with the smallest task and write a failing regression test first.
- [ ] Implement the minimal fix; preserve fail-closed classification and receipt requirements.
- [ ] Run the targeted regression and campaign reliability tests.
- [ ] Re-run the same bounded probe and compare only execution validity and the failed contract.

### Task 4: Expand to staged active-30 calibration

**Files:**
- Create outside Git: server calibration result directory and readiness summary.
- Read: `benchmark/taskgen/validate_population.py` and `benchmark/formal/aggregate.py`.

**Interfaces:**
- Consumes: a passing bounded probe and unchanged population manifest.
- Produces: active-30 calibration artifacts with `outer_trials=1` first, then `outer_trials=3` only if execution-valid.

- [ ] Run the existing active-30 runner with `outer_trials=1` and a bounded wall-time/process budget.
- [ ] Validate every persisted envelope, raw digest, fingerprint, task result classification, and population report.
- [ ] If all required execution gates pass, rerun the same frozen population with `outer_trials=3`; otherwise repair the observed blocker and repeat only the affected stage.
- [ ] Keep calibration approval blocked until an independent review artifact is explicitly produced; do not modify approval JSON by hand.

### Task 5: Update current documentation and handoff evidence

**Files:**
- Modify: `README.md`, `benchmark/README.md`, `benchmark/BENCHMARK_DESIGN.md`, and the paper only after results are validated.

**Interfaces:**
- Consumes: validated server summaries and exact revision/environment provenance.
- Produces: concise readiness status with execution, calibration, and efficacy claims separated.

- [ ] Record only reproducible facts: task count, valid/invalid/inconclusive counts, blockers, runtime, and artifact locations.
- [ ] State whether calibration remains blocked, formal-50 remains ungenerated, and efficacy remains unclaimed.
- [ ] Run documentation diff checks and the paper build if manuscript text changes.
- [ ] Commit only the intended code/docs paths and push the canonical `main` branch after tests pass.

