# RSCB, Static-Test Gate, and Support Frontier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add frozen provider-free RSCB-LCB and Static-Test Gate baselines, a shared policy evaluator, support-decision frontiers, abstention decomposition, and measured cost-profile artifacts without changing Full ACRE or any hosted/formal status.

**Architecture:** Add one shared decision/evaluation module under `benchmark/boundary`, one deterministic typed-feature/RSCB module, and one static-gate module. Existing ACRE and historical baseline functions are adapted at the evaluator boundary only; their scientific logic and persisted evidence remain unchanged. A provider-free driver writes frozen manifests and derived JSON artifacts, while paper edits consume only those artifacts after verification.

**Tech Stack:** Python 3.10+ standard library plus dependencies already used by the repository; pytest; existing family generators, `BoundaryObservation`, paired evidence and clustered bootstrap utilities; LaTeX for the paper.

**Spec:** `docs/superpowers/specs/2026-09-13-rscb-static-gate-frontier-design.md`

## Global Constraints

- Do not reset, stash, checkout, overwrite, or delete existing user changes.
- Do not modify Full ACRE, its existing split, utility policy, hidden oracle, 202-context evidence, or five frozen replication seeds.
- Do not call any provider or run hosted, formal, sealed, unblinded, or C--D experiments.
- Policy APIs receive only public typed context and calibration evidence; held-out oracle fields stay in the evaluator.
- All benchmark tests, scripts, and evidence-producing commands run on `dbcloud` with `/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python`.
- Keep conditional `unsupported/deployed` and `unsupported/all` denominators separate; family and seed are bootstrap units.

### Task 1: Shared decision and metric contract

**Files:**
- Create: `benchmark/boundary/policy_eval.py`
- Create: `benchmark/boundary/test_policy_eval.py`

**Interfaces:**
- `Decision(action: str, score: float, uncertainty: float, reason: str)`
- `evaluate_policy(policy, calibration, evaluation, score_records) -> dict`
- `support_decision_regret(action, tau, scientific_gate, delta) -> float`
- `clustered_frontier(rows, cluster_keys) -> dict`

- [ ] **Step 1: Write failing contract tests** for both denominators, undefined conditional risk at zero deployment, scientific-gate regret, and policy isolation from scoring fields.
- [ ] **Step 2: Run the focused tests on the server** and confirm failure is due to the missing evaluator contract.
- [ ] **Step 3: Implement the immutable `Decision`, shared metric calculations, deterministic support-decision regret, and family/seed clustered summaries.
- [ ] **Step 4: Re-run the focused tests and then the existing boundary tests.**
- [ ] **Step 5: Verify `git diff` contains no changes outside the new module/test.**

### Task 2: Typed feature encoder and split leakage tests

**Files:**
- Create: `benchmark/boundary/typed_features.py`
- Create: `benchmark/boundary/test_typed_features.py`

**Interfaces:**
- `TypedFeatureSchema.from_calibration(family, calibration_contexts) -> TypedFeatureSchema`
- `TypedFeatureSchema.transform(context) -> tuple[float, ...]`
- `TypedFeatureSchema.to_dict() -> dict`

- [ ] **Step 1: Write failing tests** for sorted grammar-path features, calibration-only categorical vocabulary, unknown-category handling, deterministic numeric standardization, and rejection of hidden/evaluation fields.
- [ ] **Step 2: Run the tests and confirm the encoder does not yet exist.**
- [ ] **Step 3: Implement the schema using the family’s registered public predicate grammar paths; exclude opaque IDs, split labels, outcomes, oracle labels, and counterexample labels.**
- [ ] **Step 4: Add leakage tests** that inject hidden fields into evaluation records and prove policy features are unchanged.
- [ ] **Step 5: Run the focused feature tests on the server.**

### Task 3: Static-Test Gate

**Files:**
- Create: `benchmark/boundary/static_gate.py`
- Create: `benchmark/boundary/test_static_gate.py`

**Interfaces:**
- `StaticTestGate.from_calibration_suite(suite_records) -> StaticTestGate`
- `StaticTestGate.decide(context) -> Decision`
- `StaticTestGate.to_dict() -> dict`

- [ ] **Step 1: Write failing tests** for suite-pass global reuse, suite-fail global abstention, zero uncertainty, and absence of hidden-label/context-effect access.
- [ ] **Step 2: Run the tests and confirm failure.**
- [ ] **Step 3: Implement the fixed suite admission decision with auditable reason strings and no context-dependent re-certification.**
- [ ] **Step 4: Run static-gate and leakage tests together.**

### Task 4: RSCB posterior and LCB

**Files:**
- Create: `benchmark/boundary/rscb.py`
- Create: `benchmark/boundary/test_rscb.py`

**Interfaces:**
- `RSCBConfig.default() -> RSCBConfig`
- `RSCBModel.fit(calibration_rows, schema, config) -> RSCBModel`
- `RSCBModel.decide(context) -> Decision`
- `RSCBModel.to_dict() -> dict`

- [ ] **Step 1: Write failing tests** for deterministic MAP fitting, Laplace covariance, monotonic LCB penalty, fixed default threshold, and calibration-only matched threshold selection.
- [ ] **Step 2: Run the tests and confirm failure before implementation.**
- [ ] **Step 3: Implement the registered Gaussian-prior logistic model, Newton MAP solver, Laplace covariance, delta-method uncertainty, `LCB_k`, and binary decision.
- [ ] **Step 4: Implement matched threshold selection using only calibration coverage against frozen Full ACRE calibration coverage; add tie-breaking toward the larger threshold.**
- [ ] **Step 5: Run RSCB tests, including reproducibility and finite-value checks.**

### Task 5: Frozen provider-free frontier driver

**Files:**
- Modify: `scripts/run_pilot_surface_experiments.py`
- Create: `scripts/run_rscb_static_frontier.py`
- Create: `benchmark/boundary/test_frontier_driver.py`

**Interfaces:**
- `run_frozen_frontier(surface_count, seed, replication_seeds) -> dict`
- `write_frontier_artifacts(output_root, report, manifest) -> None`

- [ ] **Step 1: Write failing driver tests** proving no evaluation record is passed to threshold selection and every policy emits the shared `Decision` record.
- [ ] **Step 2: Run the tests and confirm failure.**
- [ ] **Step 3: Implement calibration fitting for RSCB default/matched, Static-Test Gate admission, frozen ACRE alpha grid, RSCB `(k, theta)` grid, and single-point controls.**
- [ ] **Step 4: Run the driver against the existing provider-free family generator only, preserving the existing split and seeds.**
- [ ] **Step 5: Persist `results/rscb_static_frontier.json` and `results/rscb_static_frontier_manifest.json` with raw counts, family/seed summaries, calibration freeze, and environment metadata.**

### Task 6: Abstention decomposition and frontier checks

**Files:**
- Modify: `benchmark/boundary/policy_eval.py`
- Create: `benchmark/boundary/test_abstention_frontier.py`

- [ ] **Step 1: Write failing tests** for correct abstention, missed supported deployment, reason precedence `A/E/S/L/R`, and CI-width comparisons without a preselected interpretation.
- [ ] **Step 2: Run the tests and confirm failure.**
- [ ] **Step 3: Implement decomposition and family-cluster difference summaries.**
- [ ] **Step 4: Add assertions that the report emits facts even when uncertainty is lower for missed-supported contexts.**
- [ ] **Step 5: Run focused evaluator/frontier tests.**

### Task 7: Cost-profile instrumentation

**Files:**
- Create: `scripts/collect_cost_profile.py`
- Create: `benchmark/boundary/test_cost_profile.py`
- Create: `evidence/cost_profile_manifest.json` only after the server run

- [ ] **Step 1: Write failing tests** for warm-up exclusion, at least 20 samples, median/p90/max, offline stopping-horizon labeling, and revision-count accounting.
- [ ] **Step 2: Run the tests and confirm failure.**
- [ ] **Step 3: Implement deterministic lightweight instrumentation around existing CEGIS, router, replay, and rule-store operations; do not add provider calls.**
- [ ] **Step 4: Run the instrument on the server and persist `results/cost_profile.json` plus its manifest with raw timing samples.**
- [ ] **Step 5: Validate that no online stopping time is claimed when only offline replay exists.**

### Task 8: Higher-order screening protocol and documentation

**Files:**
- Modify: `core/acre/router.py` only if the existing route lacks the conservative protocol
- Create or modify: `benchmark/tests/test_higher_order_screening.py`
- Modify: `docs/budget_aware_acre_completion_report.md` only for the new provider-free comparison

- [ ] **Step 1: Write failing tests** for pairwise fail/unresolved rejection and pairwise-pass-without-higher-order-certificate abstention.
- [ ] **Step 2: Run the tests and confirm the missing protocol behavior.**
- [ ] **Step 3: Implement the smallest policy-level conservative screening path, preserving existing higher-order certificate semantics.**
- [ ] **Step 4: Run higher-order and router regression tests.**

### Task 9: Results, paper, and final verification

**Files:**
- Modify: `results`/`evidence` manifests only with verified server artifacts
- Modify: `E:\PAPER\EvoSPE\sections\evaluation.tex`, `sections\related.tex`, `sections\appendix.tex`, figures and claim matrix only after result freeze

- [ ] **Step 1: Write result-generation tests** for frozen artifact schema and claim-decision table routing.
- [ ] **Step 2: Run provider-free focused suites and deterministic replay on the server.**
- [ ] **Step 3: Inspect frontier JSON, denominators, uncertainty units, and claim boundaries before editing the paper.**
- [ ] **Step 4: Update paper text/figures without adding hosted efficacy claims or renaming ordinary held-out contexts as OOD.**
- [ ] **Step 5: Rebuild LaTeX, inspect PDF pages/figures, verify references and overfull boxes, then report exact commits and hashes.**
