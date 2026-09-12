# Context-Indexed Reuse P0 Evidence Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the smallest provider-free evidence needed to test selective-risk comparison, counterexample support-boundary recovery, lifecycle conservatism controls, and the public-data audit boundary.

**Architecture:** Reuse the shared `Decision`/evaluator contract and canonical family generators. Add one risk-calibrated logistic policy module, one support-boundary analysis driver, one lifecycle receipt reader only if existing receipts expose the required action fields, and one read-only public-audit loader. Results are derived artifacts with explicit claim routing; no hosted or formal execution path is touched.

**Tech Stack:** Existing Python environment, NumPy/SciPy already available on `dbcloud`, pytest, existing LaTeX/figure sources.

**Spec:** `docs/superpowers/specs/2026-09-13-context-indexed-reuse-p0-design.md`

## Global Constraints

- Do not call providers or run hosted, formal, sealed, unblinded, or C--D experiments.
- Do not modify historical formal evidence, Full ACRE evidence, hidden oracle, scientific gate, utility margin, or frozen evaluation splits.
- Run Python/tests/evidence-producing commands only on `dbcloud` with `/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python`.
- Preserve all existing dirty files; do not reset, stash, checkout, or overwrite them.
- Keep `unsupported/deployed` and `unsupported/all` separate and record family/seed units.

### Task 1: Risk-calibrated selective baseline

**Files:**
- Create: `benchmark/boundary/ltt.py`
- Create: `benchmark/boundary/test_ltt.py`

**Interface:** `LTTLogisticPolicy.fit(synthesis_rows, calibration_rows, schema, config) -> LTTLogisticPolicy`; `.decide(context) -> Decision`; `.to_dict() -> dict`.

- [ ] Write tests for calibration-only threshold selection, exact denominator handling, deterministic fit, and all-abstain when the target risk cannot be certified.
- [ ] Run the focused server tests and confirm the new tests fail before implementation.
- [ ] Implement a deterministic logistic scorer using the existing typed schema and one-sided exact binomial upper bounds from SciPy; do not expose evaluation rows to fitting or threshold selection.
- [ ] Run the focused tests and the existing boundary tests.

### Task 2: Support-boundary dose analysis

**Files:**
- Create: `scripts/run_support_boundary_analysis.py`
- Create: `benchmark/boundary/test_support_boundary_analysis.py`

**Interface:** `run_support_boundary_analysis(surface_count, generator_seeds, doses) -> dict`.

- [ ] Write tests for fixed evaluation IDs across doses, hidden truth consumed only by scoring, and support precision/recall raw counts.
- [ ] Run tests red on the server.
- [ ] Implement deterministic 0/25/50/100% counterexample prefixes, Full ACRE, RSCB, and LTT rows using the shared evaluator plus offline support metrics.
- [ ] Run the bounded provider-free analysis on the server and write a new revisioned artifact under `results/`.

### Task 3: Lifecycle evidence audit

**Files:**
- Create: `scripts/audit_lifecycle_controls.py`
- Create: `benchmark/boundary/test_lifecycle_controls.py`

- [ ] Inspect existing poison/compiler-drift/specialization receipts for action, utility, and evidence-cost fields before writing code.
- [ ] If receipts contain all required fields, derive four-policy summaries without rerunning cells; otherwise emit an explicit insufficient-data artifact and do not infer the missing controls.
- [ ] Test that lifecycle longitudinal regret is not merged with support-decision regret.

### Task 4: Public aggregate audit boundary

**Files:**
- Create: `scripts/audit_public_skillsbench.py`
- Create: `benchmark/boundary/test_public_audit.py`

- [ ] Read the public aggregate snapshot into a separate data directory only; do not add it to the benchmark source tree.
- [ ] Require task metadata for the preregistered stratified split; if unavailable, write an availability/blocked artifact and keep it out of manuscript claims.
- [ ] If metadata is available, run the frozen split and common evaluator, recording the public snapshot revision and no provider calls.

### Task 5: Paper and figures

**Files:**
- Modify: `E:\PAPER\EvoSPE\main.tex` and relevant `sections/*.tex` only after artifact verification.
- Modify/create: existing figure-generation source only after the chart-vs-table preflight.

- [ ] Update ACRE expansion and abstract/intro wording to context-indexed certification; do not state hosted efficacy.
- [ ] Add only verified selective-risk/support-boundary/lifecycle numbers.
- [ ] Rebuild LaTeX, inspect changed pages and figures at final size, and check undefined references/citations and overfull boxes.

### Task 6: Final verification

- [ ] Run focused provider-free tests, existing boundary/lifecycle tests, and deterministic replay on the server.
- [ ] Verify artifact manifests, claim routing, git diffs, and preserved hosted/formal status.
- [ ] Report unsupported or blocked parts rather than filling gaps with assumptions.

