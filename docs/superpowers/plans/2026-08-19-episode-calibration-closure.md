# Episode Calibration Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the reviewed episode-verification, calibration-provenance, estimator, and formal-release blockers without changing ACRE semantics or starting a campaign.

**Architecture:** Keep one harness-owned episode execution path. Candidate workspaces return only declarative actions; the verifier runs the episode, normalizes all gates, applies the registered score/effect contracts, and records immutable cell identity. Resume and release checks reuse existing digest readers and validators rather than adding a second store or transaction layer.

**Tech Stack:** Python 3, pytest, existing benchmark harness, JSON/YAML manifests.

**Spec:** `benchmark/BENCHMARK_DESIGN.md` and the attached review requirements.

## Global Constraints

- Do not modify ACRE, CEGIS, router, governance, promotion thresholds, or statistical estimators other than the explicitly required D-B outer-trial grouping.
- Do not run active-30, formal-50, or efficacy campaign.
- Reuse `runner.normalize_gates()`, `stats.read_noise_control()`, and existing release/approval paths.

### Task 1: Harden episode score authority and gates

**Files:** `benchmark/harness/verifier.py`, `benchmark/harness/runner.py`, evolution task `benchmark.py` files and episode workspace/oracle solutions, targeted tests.

- Make episode candidates return only an action/policy; compute `run_episode`, metrics, score, and gates in the harness.
- Normalize tuple/dict/bool gates through `runner.normalize_gates()`.
- Set evolution task score to zero whenever any scientific gate fails.
- Add preregistered `oracle.expected_delta_range` and require all outer-trial deltas to satisfy it.
- Preserve S0/S6 harness hashes and execute each episode arm under an OS process-group timeout.

### Task 2: Make calibration cells non-replayable

**Files:** `benchmark/formal/attest.py`, `scripts/run_active30_calibration.py`, `scripts/audit_calibration_resume.py`, `benchmark/harness/verifier.py`, targeted tests.

- Bind `task_id`, `outer_trial_id`, `seed`, and `measurement_class` into raw results and envelopes.
- Add and verify `envelope_digest`.
- Recompute noise artifact digests with `stats.read_noise_control(expected=...)` during resume and cell reuse.
- Reuse only valid measurements and valid negative/inconclusive evidence; rerun infrastructure/resource failures and block protocol failures.

### Task 3: Close population, estimator, and formal release gates

**Files:** `benchmark/taskgen/validate_population.py`, `benchmark/formal/aggregate.py`, `benchmark/formal/approval.py`, `benchmark/formal/release_manifest.py`, `benchmark/formal/run_campaign.py`, tests.

- Validate AST/package/oracle integrity for every track, including evolution.
- Group D-B by `(task, lineage, outer_trial)`, average trials within task, then tasks within lineage, then lineages.
- Require `required_cells` in estimator validation.
- Add one `validate_formal_release()` that recomputes release digest and checks git/population/approval/claims/protocol/config/executor bindings; use it at formal entry and claim gate.
- Compare campaign-config population paths canonically relative to the repository root.

### Task 4: Verify

- Run focused episode/provenance/aggregate/population/release tests.
- Run the full repository test suite and existing `run_all.py`.
- Confirm no campaign process was started and the working tree is clean before delivery.
