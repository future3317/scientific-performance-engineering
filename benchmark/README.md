# SPE-EvoBench

A benchmark suite for **S**cientific **P**erformance **E**ngineering agents and for
the **evolution** of the `scientific-performance-engineering` skill itself.

**Read [BENCHMARK_DESIGN.md](BENCHMARK_DESIGN.md) first** — it is the normative spec
(task packages §4, verification pipeline §6, threat model §7, scoring §8,
conditions A–D §9, sequential split §10, harness architecture §11).
[INTEGRATION_REQUIREMENTS.md](INTEGRATION_REQUIREMENTS.md) lists the benchmark-local
workarounds for core-skill gaps (no core file is ever modified).

**Status:** the v1.0-20 population-validity pilot contains 18 atomic tasks plus
2 evolution episodes. The formal-eval driver is available for a dry-run or an
explicit agent command, but this repository claims no formal A/B/C/D results.
The target remains frozen as SPE-EvoBench v1.0-50 (24 SPE-Core + 20 SciML + 6
Evolution). The current pilot contains 11 Core, 7 SciML, and 2 Evolution
tasks; 13 Core, 13 SciML, and 4 Evolution slots remain ungenerated and are
gated on pilot calibration.

The workload source of truth is now the Family catalog. The five pilot families
(`compile`, `graph_cache`, `h2d_pipeline`, `checkpoint`, and `scalar_sync`)
define parameter axes, interventions, and legal transformations. Existing
directories under `tasks/` remain materialized canonical anchors; task,
BoundaryBench, InteractionBench, and evolution views refer back to the same
family generator instead of maintaining separate synthetic workloads.

Each `FamilySpec` owns the applicability predicate and scientific truth used by
all views. The 20 materialized tasks are reconstructable anchor instances via
`reconstruct_anchor_instance`; their task metadata is only a projection. Run
`python -m benchmark.families.consistency --surface-count 100` to check anchor,
BoundaryBench, InteractionBench, and evolution lineage agreement. The pilot
surface runner expands the same five families to 100--500 hidden contexts:

```bash
python scripts/run_pilot_surface_experiments.py --surface-count 100 \
  --out benchmark/pilot_surface_report.json
```

This report is a calibration/diagnostic artifact, not an algorithmic-success claim
and not a formal-50 result. Active-boundary rows use an observable posterior
certificate and report multi-seed cost/error curves. Interaction rows use the
8/16/32/64/128 sequential block schedule and retain surface-level hidden versus
predicted relations, confidence intervals, and stopping blocks. Evolution rows
must contain a canonical-rule transfer phase before drift/poison metrics are
interpreted. Evolution Regret and the vector Performance Profile are aggregated
only by `benchmark.formal.aggregate`.

Agent-backed trials must write `agent_usage.json` at the path supplied by
`SPE_AGENT_USAGE_PATH` with `input_tokens`, `output_tokens`, `tool_calls`, and
`wall_time_s`. The driver treats a missing, malformed, or over-budget receipt as
an invalid trial. After verification, each task records an explicit evidence,
maintenance, and store-attestation transition; C remains raw-experience-only,
while D is the only condition allowed to promote governed rules.

## Layout

```
benchmark/
  BENCHMARK_DESIGN.md          # normative spec
  INTEGRATION_REQUIREMENTS.md  # core-skill gap requests + local workarounds
  sources/sources.yaml         # upstream pins/licenses
  schema/                      # documentation schemas (validation is in-code)
    task.schema.json           #   task.yaml contract (§4.1)
    result.schema.json         #   result.json contract (§6 S6, §8.1)
  harness/
    miniyaml.py                # restricted-subset YAML parser (no PyYAML)
    api.py                     # named workspace API contracts
    fingerprint.py             # hw/sw fingerprint (stdlib+torch, psutil optional)
    stats.py                   # median/IQR/MAD, seeded bootstrap CI, noise floor
    anticheat.py               # AST/regex scan, hashing, canaries, probes, tripwire
    scientific_gates.py        # energy/force, gradient, equivariance, validity, moments
    runner.py                  # sandbox + subprocess isolation + paired measurement
    verifier.py                # S0–S6 pipeline orchestrator
    scoring.py                 # per-task + track aggregates (§8.2–8.3)
    evolution.py               # episode runner + evolution metrics (§8.4)
    evolution_ledger.py        # monotonic replay/promotion decision ledger
    conditions.py              # A/B/C/C_STRESS/D materialization + attestation
  taskgen/                     # pilot generator and population validator
  families/                    # canonical family specs, instances, and transformations
  formal/                      # experiment manifest, schedule, driver, aggregation
  population_report.json       # deterministic v1.0-20 population summary
    split.py                   # sequential split + leakage checker
    cli.py                     # the CLI below
  tasks/<task_id>/...          # task packages (§4)
  tests/                       # standalone assert-scripts (no pytest)
```

## Quickstart

Run everything from the repository root (the directory containing `benchmark/`):

```bash
# tests (standalone assert-scripts, repo convention)
python benchmark/tests/run_all.py
# pytest-compatible unit tests (task-package smoke fixtures are excluded by pytest.ini)
python -m pytest -q
python -m benchmark.taskgen.validate_population \
    --tasks-root benchmark/tasks --out benchmark/population_report.json

# CLI examples (run from the repo root)
python -m benchmark.harness.cli validate-task benchmark/tasks/<task_id>
python -m benchmark.harness.cli run-task benchmark/tasks/<task_id> --solution DIR --out result.json \
    [--predict-mechanism scalar_sync,h2d_blocking] [--seed 0]
python scripts/render_skill_view.py SKILL_ROOT SKILL_VIEW_DIR
python -m benchmark.harness.cli materialize-condition {A,B,C,C_STRESS,D} --snapshot SKILL_VIEW_DIR --out DIR --context-mode reset
python -m benchmark.harness.cli run-episode benchmark/tasks/EVOL-EPISODE-POISON-10/episodes/<id>.yaml --condition {C,C_STRESS,D} --out DIR --context-mode reset
python -m benchmark.harness.cli check-leakage benchmark/split/sequential.yaml [--tasks-root DIR]
python -m benchmark.harness.cli score-run RUN_DIR --out scores.json
```

Exit codes: `validate-task`/`check-leakage` return 0 when clean, 1 on findings;
`run-task` returns 0 (pass), 1 (fail), 2 (error), 3 (inconclusive).

## Dependencies

Python ≥ 3.10 stdlib + torch. PyG is optional (SciML tasks ship fallback scatter).
YAML files are restricted to the subset documented in `harness/miniyaml.py`;
`validate-task` enforces it. No network access, no PyYAML, no pytest.
