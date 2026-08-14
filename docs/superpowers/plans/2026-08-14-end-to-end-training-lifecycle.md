# End-to-End Scientific Training Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Extend the skill from a benchmark evidence checker into an end-to-end scientific training systems optimizer without moving lifecycle details into `SKILL.md`.

**Architecture:** Keep `SKILL.md` as a compact router and policy boundary. Put lifecycle guidance in three focused references, make benchmark records structurally executable through a JSON schema plus validator, and use lightweight behavioral fixtures and GitHub Actions for regression protection. No training framework, cache service, or shared-runner GPU gate is added.

**Tech Stack:** Markdown references, JSON Schema (draft 2020-12 subset), Python standard library validators/fixtures, GitHub Actions.

## Global Constraints

- Preserve the scientific contract, work-normalized metrics, CUDA completion proof, raw-run statistics, and existing assessment states.
- Treat missing lifecycle evidence as `inconclusive`; do not silently infer support from an installed package.
- Keep shared CI CPU-only; GPU performance gates require an explicitly labeled self-hosted runner.
- Use the existing `D:\Anaconda\envs\EGNN\python.exe` environment for validation.

### Task 1: Add failing lifecycle contract fixtures

**Files:**
- Modify: `scripts/behavioral_contract_tests.py`
- Test: the new assertions in that file

- [ ] Add failing assertions for preflight, sync census, logical-update DAG, cache/H2D proof, amortized metrics, and checkpoint cursor fields.
- [ ] Run the fixture and confirm it fails because the current record/template lacks those contracts.

### Task 2: Add schema and benchmark-record validator

**Files:**
- Create: `assets/benchmark_record.schema.json`
- Create: `scripts/validate_benchmark.py`
- Modify: `assets/benchmark_record.json`
- Modify: `scripts/validate_skill.py`

- [ ] Implement the minimal required structural schema and standard-library validator.
- [ ] Add template fields for preflight, lifecycle census, cache/H2D proof, optimizer/EMA/validation, failure diagnostics, and resume cursor.
- [ ] Run the targeted fixtures to turn them green.

### Task 3: Extend runtime collection and comparator gates

**Files:**
- Modify: `scripts/collect_env.py`
- Modify: `scripts/compare_benchmarks.py`
- Modify: `scripts/behavioral_contract_tests.py`

- [ ] Collect both `PYTORCH_ALLOC_CONF` and legacy allocator alias plus privacy-safe compile/cache metadata.
- [ ] Require preflight compatibility, lifecycle evidence, sync census, cache/H2D proof, amortized metrics, and resume cursor in accepted records.
- [ ] Add scenario fixtures for changed host state, missing active path, missing quality, and incomplete lifecycle evidence.

### Task 4: Add lifecycle references and route them from the skill

**Files:**
- Create: `references/CODE_AND_RUNTIME_AUDIT.md`
- Create: `references/DATA_AND_TRAINING_LIFECYCLE.md`
- Create: `references/MEMORY_COMPILER_DISTRIBUTED.md`
- Modify: `SKILL.md`
- Modify: `references/MEASUREMENT_CONTRACT.md`
- Modify: `references/PERFORMANCE_PLAYBOOK.md`
- Modify: `references/PATCH_PATTERNS.md`
- Modify: `references/TECHNOLOGY_MATRIX.md`
- Modify: `references/SOURCES.md`

- [ ] Document the preflight→amortized-job state machine, custom-op correctness gate, sync budget, data four-layer model, cache contract, memory/compiler/DDP/checkpoint failure routes, and numerical divergence localization.
- [ ] Keep details out of `SKILL.md`; route only by workload and mode.
- [ ] Add current official/living implementation source links without treating package presence as capability proof.

### Task 5: Add CI with a self-hosted GPU boundary

**Files:**
- Create: `.github/workflows/validate.yml`
- Modify: `README.md`

- [ ] Run structure, schema, behavior, comparator, and material self-tests on ordinary GitHub-hosted Python.
- [ ] Add a documented, opt-in `self-hosted` GPU job shape that is never required by the shared validation job and never hard-codes a 5% speed gate.

### Task 6: Full verification and integration

- [ ] Run all validators, fixtures, self-tests, syntax checks, and `git diff --check`.
- [ ] Verify no generated outputs are tracked and sync the exact repository files to `<LOCAL_USER_HOME>\.agents\skills\scientific-gpu-training-optimizer`.
- [ ] Commit, push, and fast-forward `main` only after the full verification passes.
