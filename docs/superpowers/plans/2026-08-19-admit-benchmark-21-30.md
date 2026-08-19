# Admit SPE-EvoBench Tasks 21–30 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Admit the ten validated authoring packages 21–30 into the canonical 30-task benchmark population while preserving FamilySpec truth, existing scientific gates, and fail-closed formal claims.

**Architecture:** Register the ten task IDs as explicit FamilySpec anchors, move the harness-ready packages into `benchmark/tasks/`, and extend the existing sequential split/population report from 20 to 30 tasks. Reuse the existing validator, harness, action specs, and formal manifest; do not add a second candidate path or weaken any gate.

**Tech Stack:** Python 3.10+, restricted `benchmark.harness.miniyaml`, JSON/YAML manifests, existing SPE-EvoBench validators and tests.

**Spec:** `benchmark/BENCHMARK_DESIGN.md` and the archived candidate handoff at `benchmark/archive/candidate-bundles/README.md`.

## Global Constraints

- Keep ACRE/CEGIS/router/governance/evidence semantics unchanged.
- Do not create formal-50 sealed content or efficacy claims.
- Preserve each candidate's family parameters, digest, AST skeleton, action, polarity, and scientific gates.
- Use `benchmark/tasks/` as the single canonical task location; do not retain a second executable candidate copy.
- Keep H2D task 23 CUDA calibration and evolution task 30 full-harness validation explicitly pending.

### Task 1: Register explicit FamilySpec anchors

**Files:**
- Modify: `benchmark/families/catalog.py`
- Test: `benchmark/families/consistency.py` via the existing consistency command

- [x] Add the ten IDs and exact parameter maps from the candidate packages to the appropriate anchor tuples and `_EXPLICIT_ANCHORS` maps.
- [x] Verify `reconstruct_anchor_instance()` reproduces every candidate `family_parameters`, applicability polarity, and `family_instance_digest`.
- [x] Run `python -m benchmark.families.consistency --tasks-root benchmark/tasks --surface-count 10` before moving packages.

### Task 2: Admit the task packages

**Files:**
- Create: `benchmark/tasks/CORE-SCALAR-SYNC-LOW-CADENCE-21/`
- Create: `benchmark/tasks/CORE-REPEATED-BACKBONE-LOW-REUSE-22/`
- Create: `benchmark/tasks/CORE-H2D-OVERFANOUT-23/`
- Create: `benchmark/tasks/CORE-CHECKPOINT-HIGH-PRESSURE-24/`
- Create: `benchmark/tasks/CORE-AUTOGRAD-VJP-SMALL-25/`
- Create: `benchmark/tasks/SCIML-EQUIV-LOWORDER-26/`
- Create: `benchmark/tasks/SCIML-GRAPH-CACHE-BOUNDARY-27/`
- Create: `benchmark/tasks/SCIML-CRYSTAL-HIGH-GUIDANCE-28/`
- Create: `benchmark/tasks/SCIML-CRYSTAL-STATIC-SAMPLING-29/`
- Create: `benchmark/tasks/EVOL-EQUIVARIANT-SPECIALIZE-30/`

- [x] Extract the harness-ready ZIP and copy only the ten task directories; do not copy the candidate bundle's validation scripts into the production task tree.
- [x] Preserve task YAML, workspace, public/hidden tests, scientific contract, oracle artifacts, and episode/poison files byte-for-byte.
- [x] Run `python -m benchmark.harness.cli validate-task` for all ten packages.

### Task 3: Extend the canonical split and population contract

**Files:**
- Modify: `benchmark/split/sequential.yaml`
- Modify: `benchmark/taskgen/validate_population.py`
- Modify: `benchmark/tests/test_population.py`
- Modify: `benchmark/tests/test_formal.py`

- [x] Place tasks 21–30 into later acquisition/transfer/drift/recovery phases without duplicating `(family, mechanism, source, mutation_template_id)` lineage keys across phases.
- [x] Change the expected population size/counts to 30 / `spe_core=16`, `sciml=11`, `evolution=3`, and report `SPE-EvoBench-v1.0-30-pilot`.
- [x] Update schedule assertions from 20 to 30 while preserving four conditions, reset/carry semantics, and outer-trial structure.
- [x] Keep empirical calibration status pending/blocked unless existing artifacts explicitly certify a task; do not treat authoring smoke values as calibration.

### Task 4: Update benchmark documentation and reports

**Files:**
- Modify: `benchmark/README.md`
- Modify: `benchmark/BENCHMARK_DESIGN.md`
- Modify: `benchmark/TASK_MATRIX.md`
- Modify: `benchmark/manifests/v1.0-50-slots.json`
- Regenerate: `benchmark/population_report.json`

- [x] Describe the canonical population as 30-task pilot/authoring-expanded population and list all ten new tasks with their status and pending calibration boundaries.
- [x] Keep frozen v1.0-50 quotas unchanged; update only the generation gate text to identify the 30-task calibration population.
- [x] State explicitly that formal-50 contents and efficacy claims remain withheld.

### Task 5: Verify and publish

- [x] Run `python -m benchmark.taskgen.validate_population --tasks-root benchmark/tasks --out benchmark/population_report.json`.
- [x] Run `python -m benchmark.families.consistency --tasks-root benchmark/tasks --surface-count 100`.
- [ ] Run `python benchmark/tests/run_all.py` and `python -m pytest -q`.
- [ ] Review `git diff --check`, confirm no generated runtime outputs entered the tree, commit, and push `main`.
