# Formal Statistics, Driver Closure, and Paper Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the formal benchmark driver, sequential inference, hierarchical aggregation, budget accounting, and manuscript claims so every reported quantity has a runnable and auditable contract.

**Architecture:** Keep the existing v1.0-20 population and production paths. Replace the mislabeled posterior bound with an inverted Beta-Binomial mixture e-process, add a direct post-task transition in the formal driver, and make trial validity depend on machine-readable usage. Aggregate raw task score effects separately from log median-speedup effects with family/lineage/task/trial resampling. Update only the manuscript sections and bibliography needed to describe the implemented protocol and current empirical status.

**Tech Stack:** Python 3.11 standard library, existing benchmark harness, JSON/YAML manifests, ICLR 2027 LaTeX.

## Contents

- [Task 1: Replace the sequential bound with a true mixture e-process](#task-1-replace-the-sequential-bound-with-a-true-mixture-e-process)
- [Task 2: Make formal aggregation hierarchical and unit-correct](#task-2-make-formal-aggregation-hierarchical-and-unit-correct)
- [Task 3: Close the post-task transition and store attestation loop](#task-3-close-the-post-task-transition-and-store-attestation-loop)
- [Task 4: Enforce hard machine-readable usage budgets](#task-4-enforce-hard-machine-readable-usage-budgets)
- [Task 5: Align manuscript and bibliography with implemented claims](#task-5-align-manuscript-and-bibliography-with-implemented-claims)
- [Task 6: Full verification and integration](#task-6-full-verification-and-integration)

## Global Constraints

- Do not generate the remaining 30 tasks or claim formal benchmark results that have not been run.
- Preserve the scientific contract and existing v1.0-20 population.
- New production behavior requires a failing test before implementation.
- Missing or over-budget agent usage makes a trial invalid/inconclusive and excludes it from headline aggregation.

---

### Task 1: Replace the sequential bound with a true mixture e-process

**Files:**
- Modify: `core/sequential_stats.py`
- Modify: `benchmark/tests/test_stats.py`
- Modify: `scripts/evolution_utility_tests.py`

**Interfaces:** `mixture_lower_bound(successes, trials, delta)` remains the public API; add `mixture_e_value(successes, trials, null_p, prior_alpha=1.0, prior_beta=1.0)` for deterministic tests.

- [ ] Write tests for the e-value identity, all-success/all-failure bounds, and exhaustive small-n crossing probability.
- [ ] Run the focused tests and confirm they fail against the current posterior implementation.
- [ ] Implement the log-stable Beta-Binomial mixture e-process and invert it on the lower-tail interval.
- [ ] Run the focused tests and then the existing evolution utility checks.

### Task 2: Make formal aggregation hierarchical and unit-correct

**Files:**
- Modify: `benchmark/formal/aggregate.py`
- Modify: `benchmark/tests/test_formal.py`

**Interfaces:** `aggregate_trials(records)` continues to return paired task-score effects; add `hierarchical_effects` and `paired_log_speedups` fields.

- [ ] Add a test showing that performance uses `log(candidate_speedup)-log(control_speedup)` while task-score effects remain linear.
- [ ] Add a test fixture with family, lineage, task, and outer-trial identifiers and assert deterministic hierarchical intervals.
- [ ] Implement nested resampling and raw-speedup extraction from formal trial records.
- [ ] Exclude invalid trials and keep counterexample/abstention records separate.

### Task 3: Close the post-task transition and store attestation loop

**Files:**
- Modify: `benchmark/harness/conditions.py`
- Modify: `benchmark/formal/run_campaign.py`
- Modify: `benchmark/tests/test_formal.py`

**Interfaces:** add `conditions.store_digest`, `conditions.refresh_attestation`, and `run_campaign.post_task_update(...)` returning pre/post digests, evidence IDs, and transition status.

- [ ] Add a test that invokes the transition for C and D and checks the audit record and context-specific writes.
- [ ] Run the focused test and confirm the transition API is absent/fails.
- [ ] Implement raw evidence capture for C and replay/governance handoff for D using the existing harness path, followed by policy validation and refreshed attestation.
- [ ] Integrate the transition immediately after verification/scoring and persist it in each trial record.

### Task 4: Enforce hard machine-readable usage budgets

**Files:**
- Modify: `benchmark/formal/budget.py`
- Modify: `benchmark/formal/run_campaign.py`
- Modify: `benchmark/formal/experiment.schema.json`
- Modify: `benchmark/schema/result.schema.json`
- Modify: `benchmark/tests/test_formal.py`

**Interfaces:** add `Budget.validate_usage(usage)` and require `SPE_AGENT_USAGE_PATH` with input/output token counts, tool calls, and wall time for agent-backed trials.

- [ ] Add tests for missing usage, malformed usage, and over-budget usage.
- [ ] Run the focused tests and confirm the current permissive checker fails them.
- [ ] Implement normalization, validation, invalid-trial marking, and schema fields.
- [ ] Ensure dry-run behavior remains plan-only.

### Task 5: Align manuscript and bibliography with implemented claims

**Files:**
- Modify: `E:/PAPER/EvoSPE/sections/method.tex`
- Modify: `E:/PAPER/EvoSPE/sections/evaluation.tex`
- Modify: `E:/PAPER/EvoSPE/sections/related.tex`
- Modify: `E:/PAPER/EvoSPE/sections/problem.tex`
- Modify: `E:/PAPER/EvoSPE/sections/appendix.tex`
- Modify: `E:/PAPER/EvoSPE/refs.bib`

- [ ] Replace the Hoeffding proposition with the Beta-Binomial mixture e-process definition and Ville-based lower-bound statement.
- [ ] Define a scalar normalized utility policy from the raw outcome vector and gates.
- [ ] Document post-task transitions, hard usage accounting, hierarchical aggregation, and the v1.0-20 calibration status without reporting results.
- [ ] Expand related work with the named benchmark and skill-maintenance families using non-anonymous bibliographic records where available.
- [ ] Compile the ICLR manuscript and inspect the resulting PDF.

### Task 6: Full verification and integration

**Files:** existing repository and paper outputs only.

- [ ] Run focused statistical/formal tests.
- [ ] Run the repository benchmark test suite and validators.
- [ ] Run `bibtex` and two `pdflatex` passes for the paper.
- [ ] Review the diff for forbidden internal/process language and confirm no formal results were invented.
- [ ] Commit the bounded changes to `main` only after fresh verification.
