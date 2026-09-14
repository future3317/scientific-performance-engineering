# Public ACRE Treatment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auditable public fixed-skill audit and an online evolving-skill A/C/D treatment campaign, then update the paper only from the resulting evidence.

**Architecture:** Reuse the existing `EvolutionDecisionLedger`, condition materialization, ACRE engine, and formal task verifier. Add a small external-evaluation layer with one decision-ledger schema, one common evaluator, immutable skill-version records, and two runners: SkillGen-compatible fixed-skill pairing and sequential evolving-skill treatment. All runtime/evidence commands execute on `dbcloud` in `equivcompiler`.

**Tech Stack:** Python 3.11, existing stdlib/PyTorch benchmark harness, OpenAI-compatible provider adapter, JSONL append-only ledgers, pytest, LaTeX/Poppler for paper QA.

**Spec:** `docs/superpowers/specs/2026-09-15-public-acre-treatment-design.md`

## Global Constraints

- Create a new evidence revision `public-acre-treatment-20260915-v1`; do not overwrite provider-free, historical formal, or 2026-09-14 artifacts.
- Every held-out decision is serialized before score, judge, trajectory, or future-task outcome is joined.
- Skill versions are immutable; D may create new versions only from prior evidence.
- Infrastructure-invalid cells may be replayed once with the same seed; scientific failures are never retried.
- Report equal task-execution budget and equal total compute; do not use a composite headline metric.
- Provider calls occur only after the protocol manifest and smoke gate are complete.
- Local Windows is for editing/static inspection; tests and experiments run through `ssh dbcloud` with `/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python`.
- Paper claims are updated only after artifact manifests, receipts, tests, and analyses are complete.

### Task 1: Freeze protocol and shared schemas

**Files:**
- Create: `docs/public_acre_treatment_protocol_v1.md`
- Create: `benchmark/external/__init__.py`
- Create: `benchmark/external/contracts.py`
- Create: `benchmark/external/manifest.py`
- Create: `tests/test_public_acre_contracts.py`

**Interfaces:**
- `DecisionRecord.from_mapping(mapping) -> DecisionRecord`
- `PublicTreatmentManifest.create(...) -> dict[str, Any]`
- `validate_public_context(context, scoring_only_fields) -> None`
- `validate_no_outcome_fields(decision_record) -> None`

- [ ] **Step 1: Write failing contract tests** for required decision fields, scoring-only field rejection, task-disjoint split IDs, and manifest identity fields.
- [ ] **Step 2: Run focused tests on the server** and verify they fail for missing modules/schema.
- [ ] **Step 3: Implement the dataclasses/validators** using the existing canonical JSON/digest conventions; do not duplicate evidence storage.
- [ ] **Step 4: Add the full protocol** with four public benchmarks, the available DeepSeek configuration, optional second-backbone blocked-cell semantics, fixed seeds, three paired repeats, budgets, retry rule, and A/C/D estimands.
- [ ] **Step 5: Run focused tests and `git diff --check`; commit** `protocol: freeze public ACRE treatment contract`.

### Task 2: Add immutable skill versions and pre-outcome decision ledger

**Files:**
- Create: `benchmark/external/skill_versions.py`
- Create: `benchmark/external/decision_ledger.py`
- Modify: `benchmark/harness/evolution_ledger.py` only where a direct reusable helper is required
- Create: `tests/test_public_acre_ledgers.py`

**Interfaces:**
- `SkillVersionStore.create(parent_id, task_index, content, evidence_ids, certificate, provenance) -> SkillVersion`
- `SkillVersionStore.get(version_id) -> SkillVersion`
- `DecisionLedger.append(decision: DecisionRecord) -> None`
- `DecisionLedger.append_outcome(decision_id, outcome) -> None`
- `DecisionLedger.rows() -> list[dict[str, Any]]`

- [ ] **Step 1: Write failing tests** for immutable content, parent/version lineage, append-only decision-before-outcome ordering, current-task outcome rejection, and future-outcome rejection.
- [ ] **Step 2: Run the focused tests** and confirm failures identify missing behavior rather than environment failures.
- [ ] **Step 3: Implement version records** with exact content digest, parent, task index, evidence IDs, predicate/certificate, identity fields, and provenance receipt.
- [ ] **Step 4: Implement the ledger** as a digest-chained JSONL projection that writes decisions before execution and appends outcomes only afterward; reuse existing digest-chain code where possible.
- [ ] **Step 5: Add deterministic replay tests** and commit `feat: add immutable public treatment ledgers`.

### Task 3: Implement common evaluator and policy adapters

**Files:**
- Create: `benchmark/external/evaluator.py`
- Create: `benchmark/external/policies.py`
- Create: `benchmark/external/metrics.py`
- Create: `tests/test_public_acre_evaluator.py`

**Interfaces:**
- `evaluate_paired_rows(rows, policy_names, bootstrap_seed, bootstrap_reps=10000) -> dict[str, Any]`
- `support_decision_regret(row, fallback_utility) -> float`
- `summarize_policy(rows, policy_name) -> dict[str, Any]`
- `macro_average(cell_summaries) -> dict[str, Any]`
- `GlobalPolicy`, `SemanticRouterPolicy`, `NearestEvidencePolicy`, `CatBoostLTTPolicy`, `AcrePolicy`

- [ ] **Step 1: Write failing tests** for unsupported/deployed versus unsupported/all denominators, paired task bootstrap, macro aggregation, regret, and decision serialization before outcomes.
- [ ] **Step 2: Run focused tests** and record the expected failures.
- [ ] **Step 3: Implement the evaluator** so all policies share identical rows and task-instance bootstrap units; keep official score, paired delta, coverage, repair recall, negative transfer, unsafe rate, cost, and observation completion separate.
- [ ] **Step 4: Implement adapters** over the existing semantic router and ACRE logic; the adapter interface may not inspect hidden outcome fields.
- [ ] **Step 5: Run tests and commit** `feat: add common public deployment evaluator`.

### Task 4: Build fixed-skill public benchmark adapter and ScienceWorld smoke

**Files:**
- Create: `benchmark/external/skillgen_adapter.py`
- Create: `scripts/run_public_acre_smoke.py`
- Create: `scripts/run_public_acre_fixed_audit.py`
- Create: `tests/test_skillgen_public_adapter.py`
- Modify: `docs/skillgen_external_protocol.md` to point to the new revision without rewriting the 2026-09-14 calibration record

**Interfaces:**
- `load_pinned_benchmark(manifest) -> BenchmarkView`
- `build_public_context(instance) -> dict[str, Any]`
- `run_paired_instance(instance, skill_version, model_config, seed) -> PairedOutcome`
- `run_smoke(protocol_path, out_dir) -> dict[str, Any]`

- [ ] **Step 1: Write failing adapter tests** for public/scoring-only field separation, train/test task disjointness, shared paired rollout identity, and blocked dependency reporting.
- [ ] **Step 2: Run tests on the server.** Do not call a provider yet; the smoke command must support a dry validation mode.
- [ ] **Step 3: Implement pinned source/data discovery** for ScienceWorld first, then the remaining SkillGen-compatible adapters only when their exact source and split are available.
- [ ] **Step 4: Run the no-provider protocol smoke** and produce a manifest with `provider_calls=0`.
- [ ] **Step 5: Run one real ScienceWorld smoke** under the frozen protocol; verify decisions are serialized before score joins, paired outcomes share cache identity, receipts are complete, and the provider/model response is parseable.
- [ ] **Step 6: Commit** `feat: add pinned fixed-skill public audit path` only after smoke tests pass.

### Task 5: Implement online evolving-skill A/C/D campaign

**Files:**
- Create: `benchmark/external/evolving_campaign.py`
- Create: `scripts/run_public_acre_evolving_campaign.py`
- Create: `tests/test_public_acre_evolving_campaign.py`
- Modify: `benchmark/harness/conditions.py` only for an existing condition contract needed by the runner

**Interfaces:**
- `run_sequential_campaign(protocol, condition, task_stream, out_dir) -> CampaignArtifact`
- `make_next_skill_version(previous, evidence, decision) -> SkillVersion`
- `condition_decision(condition, public_context, store, skill_version) -> DecisionRecord`

- [ ] **Step 1: Write failing tests** for A/C/D asymmetry, no future information, equal task budget, immutable skill versions, reserve-aware D acquisition, and raw C retrieval without promotion.
- [ ] **Step 2: Run the focused tests** and confirm the failures are behavioral.
- [ ] **Step 3: Implement the campaign** by reusing existing condition materialization, ACRE engine, budget manager, and evolution ledger; D performs candidate admission/replay/promotion/revalidation/retirement, while C stores and retrieves raw experiences only.
- [ ] **Step 4: Add task-phase accounting** for acquisition, same-family transfer, cross-family transfer, drift, poisoned experience, and recovery; write decisions before each execution and outcomes afterward.
- [ ] **Step 5: Run provider-free sequential replay** over the existing deterministic fixtures and assert no leakage, equal budgets, complete ledger, and conserved accounting.
- [ ] **Step 6: Commit** `feat: add evolving-skill A-C-D treatment runner`.

### Task 6: Run staged provider experiments and freeze artifacts

**Files:**
- Create: `results/public_acre_treatment_20260915_v1/` on the server only
- Create: `docs/public_acre_treatment_completion_v1.md`
- Create: `scripts/analyze_public_acre_treatment.py`
- Create: `tests/test_public_acre_analysis.py`

- [ ] **Step 1: Generate the final protocol manifest** from the exact implementation commit and pin external sources, model configuration, seeds, task order, and split.
- [ ] **Step 2: Run the ScienceWorld real smoke**. If it fails structurally, classify the blocker and stop before held-out suite execution.
- [ ] **Step 3: If smoke passes, run the fixed-skill public suite** with shared paired caches and three execution repeats where valid; record blocked cells rather than substituting models or benchmarks.
- [ ] **Step 4: Run the online A/C/D campaign** on the preregistered sequential stream, first equal task-execution budget and then equal total-compute accounting.
- [ ] **Step 5: Run analysis** for official score, paired delta, coverage, unsupported/deployed, repair recall, negative transfer, unsafe rate, promotions/rejections, skill versions, compute, cost per useful improvement, observation completion, cumulative net utility, and paired D-C confidence intervals.
- [ ] **Step 6: Run analysis tests**, validate every artifact against the manifest, and commit only lightweight manifests/summary metadata; keep raw provider data in the server evidence directory.

### Task 7: Re-run controlled non-degenerate evidence and baseline frontiers

**Files:**
- Create: `scripts/run_non_degenerate_support_evidence.py`
- Create: `scripts/run_support_frontier_grid.py`
- Create: `tests/test_non_degenerate_support_evidence.py`
- Modify: existing controlled analysis only through new revisioned inputs; never overwrite the 849-context artifact

- [ ] **Step 1: Write tests** for randomized paired order, 5–20 repetitions, sequential confidence stopping, positive/counterexample/unresolved rules, and task/family/split-seed disjointness.
- [ ] **Step 2: Implement the new evidence runner** using the existing confidence-sequence machinery and shared scientific gates; preserve unresolved cases instead of relabeling them.
- [ ] **Step 3: Implement full risk-coverage grids** for nearest, CART-LTT, CatBoost-LTT, positive-only, and Full ACRE with threshold selection restricted to synthesis/calibration.
- [ ] **Step 4: Add five fixed hash split seeds**; if a family lacks two independent factor dimensions for compositional OOD, emit `not_applicable` for that family in the manifest rather than inventing a split.
- [ ] **Step 5: Run provider-free controlled experiments on the server**, write a new manifest, and report evidence repetitions, coverage-risk, AURC, matched coverage/risk, precision, recall, regret, and unresolved status.
- [ ] **Step 6: Commit** `exp: add non-degenerate support frontier revision`.

### Task 8: Update paper, figures, and final audit from observed results

**Files:**
- Modify: `E:/PAPER/EvoSPE/main.tex` and relevant `sections/*.tex`
- Modify: `E:/PAPER/EvoSPE/refs.bib`
- Create: `E:/PAPER/EvoSPE/figures/public_deployment.tex`
- Create: `E:/PAPER/EvoSPE/figures/evolving_treatment.tex`
- Modify: `E:/PAPER/EvoSPE/main.tex` to include the two generated figures
- Modify: `E:/PAPER/EvoSPE/evidence/claim_matrix.md`
- Create: `E:/PAPER/EvoSPE/evidence/public_acre_treatment_20260915_v1.md`

- [ ] **Step 1: Build machine-readable tables/figures** from frozen result artifacts; do not hand-enter experimental values.
- [ ] **Step 2: Move public fixed-skill results to the first main experiment section** only if the corresponding manifest is complete; keep SkillsBench retrospective audit supplementary.
- [ ] **Step 3: Put evolving D-C results in the primary efficacy section** only when paired observations and receipts are complete; otherwise report a blocked/feasibility result without efficacy wording.
- [ ] **Step 4: Keep controlled SPE-EvoBench as mechanism evidence**, add non-degenerate uncertainty and frontier only if produced by the new revision, and preserve the registered 849-context artifact unchanged.
- [ ] **Step 5: Update Figure 1/2/3 and captions** with explicit denominator semantics, paired execution funnel, risk-coverage, cumulative net utility, and evolution compute; use publication figure QA before delivery.
- [ ] **Step 6: Build LaTeX**, check undefined citations/references, overfull boxes, page boundary, PDF text, and render representative pages/figures for visual inspection.
- [ ] **Step 7: Update claims only according to the artifact decision table**, record exact commits and hashes, and commit paper changes separately.

### Task 9: Repository and evidence closeout

**Files:**
- Modify: `docs/experiment_registry.md`
- Modify: `docs/paper_claim_matrix.md`
- Modify: `AGENTS.md` only to document the new revision boundary after results are frozen
- Modify: `docs/public_acre_treatment_completion_v1.md` generated by Task 6

- [ ] **Step 1: Run focused unit and targeted campaign tests.**
- [ ] **Step 2: Run the complete server test suite and all relevant validators in `equivcompiler`; record any pre-existing teardown flake separately.**
- [ ] **Step 3: Verify no historical/provider-free artifact was overwritten and all raw results have receipts/manifest bindings.**
- [ ] **Step 4: Push the new benchmark/controller branch and paper branch; create tags only after successful verification, never force-push.
- [ ] **Step 5: Produce the final report** listing supported/unsupported claims, fixed-skill result, evolving D-C result, controlled mechanism result, cost/coverage, observation completion, exact commits, and remaining limitations.
