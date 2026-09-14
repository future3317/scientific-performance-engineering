# Context-Indexed SkillGen Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement this plan task-by-task.

**Goal:** Add the attachment-requested controlled baseline comparison, counterexample leakage control, frozen replication, and fixed-skill SkillGen audit, then update the paper only from verified artifacts.

**Architecture:** Extend the existing boundary policy/evaluator layer with provider-free policies and a single artifact-producing driver. Keep semantic routing behind a provider-backed adapter that records decisions and can emit a blocked result when credentials are unavailable. Generate figures and manuscript tables from machine-readable reports, never from hand-entered values.

**Tech Stack:** Python 3.11, existing benchmark modules, pytest, SkillGen public repository, LaTeX/PGFPlots.

**Spec:** `docs/superpowers/specs/2026-09-14-context-indexed-skillgen-evidence-design.md`

## Global Constraints

- Do not overwrite registered `results/support_boundary_analysis.json`, `results/support_boundary_sensitivity.json`, frozen replication, or historical formal evidence.
- Run repository tests and evidence-producing Python commands only on `dbcloud` in `/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python`.
- Use the same family × seed split and evaluator for E1–E3; the 1,018-context replication is frozen-rule evaluation only.
- Hidden applicability truth and held-out outcomes are scoring-only and never enter policy fitting or routing.
- Every new artifact records the new commit, input artifact digest, exact parameters, seeds, split, and claim boundary.
- Do not claim hosted efficacy or external ACRE superiority from an incomplete or blocked E4.

### Task 1: Baseline policy implementations

**Files:**
- Modify: `benchmark/boundary/baselines.py`
- Create: `benchmark/boundary/tree_baselines.py`
- Create: `benchmark/boundary/test_tree_baselines.py`

**Interfaces:**
- `fit_cart_policy(synthesis_rows, calibration_rows, schema, *, max_depth, seed) -> policy`
- `fit_catboost_ltt_policy(synthesis_rows, calibration_rows, schema, *, depth, trees, learning_rate, seed) -> policy`
- `fit_nearest_policy(visible, heldout) -> Mapping[str, bool]`
- `fit_positive_only_policy(visible, heldout) -> Mapping[str, bool]`

- [ ] Write tests proving fitting uses synthesis/calibration only, decisions are deterministic, feature schema is public typed context only, and policies return the shared `Decision` contract.
- [ ] Run focused tests on `dbcloud` and confirm they fail for the missing interfaces.
- [ ] Implement the smallest deterministic CART and CatBoost-LTT adapters using installed dependencies; if CatBoost is unavailable, emit a machine-readable `not_available` configuration rather than silently substituting another model.
- [ ] Run focused tests and existing boundary tests.

### Task 2: Unified E1/E2/E3 analysis driver

**Files:**
- Modify: `scripts/run_support_boundary_analysis.py`
- Create: `scripts/run_context_indexed_frontier.py`
- Create: `benchmark/boundary/test_context_indexed_frontier.py`

**Interfaces:**
- `run_context_indexed_frontier(surface_count, generator_seeds, families, doses, permutation_reps) -> dict`
- `evaluate_registered_replication(report, replication_input) -> dict`

- [ ] Write tests for common evaluation IDs, frozen threshold/model receipts, permutation-label isolation, family × seed aggregation, and separate unsupported/all versus unsupported/deployed denominators.
- [ ] Run tests red on the server.
- [ ] Implement the fixed policy registry: global reuse, positive-only, nearest experience, matched positive-anchor, CART, CatBoost-LTT, and Full ACRE. Keep legacy LTT/RSCB rows for provenance but do not make them the only baselines.
- [ ] Implement 100 fixed counterexample-label permutations with deterministic seeds and report the null distribution without changing the registered main artifact.
- [ ] Run the provider-free E1–E3 analysis on the server, write a revisioned artifact and manifest, and verify no existing artifact changed.

### Task 3: Semantic router adapter and provider readiness

**Files:**
- Create: `scripts/run_semantic_router_frontier.py`
- Create: `benchmark/boundary/test_semantic_router_frontier.py`
- Create: `docs/skillgen_external_protocol.md`

- [ ] Document the exact SkillGen commit, ScienceWorld split, skill artifact identity, router model, temperature, prompt inputs, cache key, and scoring-only fields.
- [ ] Add a provider-backed adapter that accepts only typed context plus intervention text and caches decisions by exact input; it must record token usage and return `blocked` when required credentials or data are absent.
- [ ] Run provider-free contract tests and a no-call readiness check.
- [ ] If the server has no credentials, transfer only the user-authorized provider environment through stdin for the bounded run; never write keys to the repository or logs.

### Task 4: SkillGen E4 fixed-skill audit

**Files:**
- Create: `scripts/run_skillgen_external_audit.py`
- Create: `scripts/summarize_skillgen_external_audit.py`
- Create: `benchmark/boundary/test_skillgen_external_audit.py`

- [ ] Clone the public SkillGen repository at the recorded commit and prepare the official ScienceWorld train/test data without mixing folds.
- [ ] Freeze the first SkillGen-verified skill artifact and its verification receipt before held-out routing.
- [ ] Run global reuse, SkillGen semantic router, positive-only ACRE, CatBoost-LTT, and Full ACRE on the same held-out instances; use three execution seeds only if the public protocol and budget permit it, otherwise record the exact smaller design.
- [ ] Persist per-instance decisions, paired outcomes, raw counts, nested seed metadata, provider/model usage, and claim status.
- [ ] Produce a blocked or complete artifact with no post-outcome cell selection. Do not start C/D or alter EVOSPE certificate semantics.

### Task 5: Paper, figures, and citation update

**Files:**
- Modify: `E:/PAPER/EvoSPE/sections/intro.tex`
- Modify: `E:/PAPER/EvoSPE/sections/problem.tex`
- Modify: `E:/PAPER/EvoSPE/sections/method.tex`
- Modify: `E:/PAPER/EvoSPE/sections/evaluation.tex`
- Modify: `E:/PAPER/EvoSPE/sections/discussion.tex`
- Modify: `E:/PAPER/EvoSPE/sections/related.tex`
- Modify: `E:/PAPER/EvoSPE/figures/evospe_overview.tex`
- Modify: `E:/PAPER/EvoSPE/figures/support_boundary_p0.tex`
- Create or modify: `E:/PAPER/EvoSPE/figures/skillgen_external.tex` only if E4 is complete.

- [ ] Move lifecycle, bundle, legacy frontiers, and SkillsBench audit to supplementary references unless an artifact shows they are needed for a main claim.
- [ ] Rewrite the main method around `E[tau] > 0` not implying context-specific support and define positive/counterexample/unresolved from the actual implementation.
- [ ] Add only verified baseline/E1–E3 numbers; add E4 numbers only when the artifact is complete and its claim boundary permits them.
- [ ] Rebuild Figure 1/2 and add Figure 3 only from artifact data; use conditional unsupported/deployed on risk–coverage plots and keep unsupported/all as annotations/table values.
- [ ] Add missing citations for SkillGen, SEU/PATH-Bench, SkillCommit, VaG, and the correctly identified SafeEvolve work.

### Task 6: Verification and integration

- [ ] Run focused provider-free tests, then the full server test suite.
- [ ] Re-run deterministic E1–E3 replay and compare manifests.
- [ ] Build LaTeX twice, check undefined references/citations and overfull boxes, and visually inspect changed pages and figures at final size.
- [ ] Write a concise completion report with exact commits, artifact paths, statuses, and supported/unsupported claims.
- [ ] Push the feature branch only after verification; do not merge into `main` without an explicit integration choice.
