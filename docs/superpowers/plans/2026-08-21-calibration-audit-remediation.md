# Calibration Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the active-30 calibration pipeline provenance-complete, deadline-bounded, topology-aware, and strict-formal verifiable from one current evidence chain.

**Architecture:** Keep the existing runner/verifier path as the single production path. Add the missing identity fields at the CLI boundary, validate every raw/noise/envelope cell before aggregating statistics, and bind the frozen trial protocol and immutable empirical artifact into the existing approval chain. Use the existing subprocess isolation and fingerprint mechanisms rather than introducing a second execution framework.

**Tech Stack:** Python 3.11, stdlib subprocess/JSON/hashlib, pytest, GitHub Actions, existing SPE harness and formal validators.

**Spec:** `审计.md`

## Global Constraints

- Active population remains the explicit 30-task manifest.
- Atomic outer count is the calibration protocol value; task repetitions remain verifier-internal paired measurements.
- Evolution outer count equals declared `measurement.repetitions`.
- Missing, stale, invalid, timed-out, or provenance-incomplete evidence remains fail-closed.
- No checked-in calibration artifact is edited to manufacture approval.

---

### Task 1: Complete calibration provenance and baseline control identity

**Files:**
- Modify: `benchmark/harness/cli.py`
- Modify: `benchmark/harness/verifier.py`
- Modify: `benchmark/harness/stats.py`
- Modify: `scripts/run_active30_calibration.py`
- Test: `benchmark/tests/test_calibration_runner.py`, `benchmark/tests/test_noise_control.py`

- [ ] Add CLI arguments for `task_package_digest` and `population_manifest_digest` to both calibration commands and pass all identity fields into verifier/noise APIs.
- [ ] Make noise calibration always use `task_dir/workspace/<entrypoint>` and record `control_implementation="baseline"` plus its package identity.
- [ ] Pass the real digests from the active runner and immediately re-read the produced artifact with the full expected identity before writing an envelope.
- [ ] Add regression tests proving raw/envelope identity and baseline-vs-baseline selection.

### Task 2: Freeze trial protocol and shared deadlines

**Files:**
- Create: `benchmark/calibration/calibration_protocol.json`
- Modify: `scripts/run_active30_calibration.py`
- Modify: `scripts/audit_calibration_resume.py`
- Modify: `benchmark/harness/verifier.py`
- Test: `benchmark/tests/test_calibration_runner.py`, `benchmark/tests/test_population.py`

- [ ] Define atomic and evolution outer-trial rules in one protocol artifact.
- [ ] Load the protocol from runner, resume audit, and population validation; reject mismatched counts.
- [ ] Give evolution baseline/candidate arms one shared deadline and pass remaining time to the second arm.
- [ ] Add tests for mixed `--outer-trials=1` behavior and a second arm receiving only remaining budget.

### Task 3: Validate every calibration cell and classify failures honestly

**Files:**
- Modify: `benchmark/taskgen/validate_population.py`
- Modify: `scripts/run_active30_calibration.py`
- Modify: `benchmark/harness/runner.py`
- Modify: `scripts/audit_calibration_resume.py`
- Test: `benchmark/tests/test_population.py`, `benchmark/tests/test_calibration_runner.py`

- [ ] Validate raw identity, noise identity/digest, envelope self-digest, raw digest, fingerprint, and required trial coverage before empirical statistics.
- [ ] Mark broken cells `protocol_invalid` and do not aggregate them as noise failures.
- [ ] Replace quarantine filename reuse with artifact-digest directories.
- [ ] Return a machine-readable timeout cleanup receipt from subprocess execution.

### Task 4: Freeze campaign topology and produce timing evidence

**Files:**
- Modify: `benchmark/harness/fingerprint.py`
- Modify: `benchmark/harness/runner.py`
- Modify: `scripts/run_active30_calibration.py`
- Modify: `benchmark/taskgen/validate_population.py`
- Test: `benchmark/tests/test_calibration_runner.py`, `benchmark/tests/test_noise_control.py`

- [ ] Capture torch thread settings and include execution topology in comparability fields.
- [ ] Add selected-GPU foreign-process preflight that resource-blocks a cell without killing unrelated work.
- [ ] Bound isolated post-run task validation and compile projection subprocesses.
- [ ] Emit per-cell timing and cleanup fields covering materialization, noise, verifier, post-validation, total, reuse, quarantine, and timeout.

### Task 5: Complete strict-formal empirical digest chain and workflow input

**Files:**
- Modify: `benchmark/taskgen/validate_population.py`
- Modify: `benchmark/formal/approval.py`
- Modify: `.github/workflows/validate.yml`
- Test: `benchmark/tests/test_formal_freeze_contract.py`, `benchmark/tests/test_population.py`

- [ ] Require and digest the supplied empirical artifact in strict-formal mode.
- [ ] Bind report, pilot, empirical, protocol, claims, statistical protocol, and current revision in approval validation.
- [ ] Pass `--empirical` in the formal-readiness workflow from an immutable calibration bundle input.
- [ ] Verify strict-formal rejects a mismatched empirical artifact and a non-HEAD approval.

### Task 6: Verify the fixed revision and prepare the next campaign

- [ ] Run targeted unit/regression tests.
- [ ] Run structural population validation and inspect the exact errors.
- [ ] Confirm the fixed code is committed and pushed before launching a new remote campaign.
- [ ] Re-run the active-30 campaign only from the fixed revision; generate approval only after all 30 cells are complete and eligible.

