# SPE-EvoBench

A benchmark suite for **S**cientific **P**erformance **E**ngineering agents and for
the **evolution** of the `scientific-performance-engineering` skill itself.

**Read [BENCHMARK_DESIGN.md](BENCHMARK_DESIGN.md) first** — it is the normative spec
(task packages §4, verification pipeline §6, threat model §7, scoring §8,
conditions A–D §9, sequential split §10, harness architecture §11).
[INTEGRATION_REQUIREMENTS.md](INTEGRATION_REQUIREMENTS.md) lists the benchmark-local
workarounds for core-skill gaps (no core file is ever modified).

**Status:** 10 runnable prototype tasks implemented and individually verified;
see [TASK_MATRIX.md](TASK_MATRIX.md) for the task list, verified speedups, and
expansion plan to 60–100 tasks.

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
    conditions.py              # A/B/C/D materialization + attestation
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

# CLI examples (run from the repo root)
python -m benchmark.harness.cli validate-task benchmark/tasks/<task_id>
python -m benchmark.harness.cli run-task benchmark/tasks/<task_id> --solution DIR --out result.json \
    [--predict-mechanism scalar_sync,h2d_blocking] [--seed 0]
python -m benchmark.harness.cli materialize-condition {A,B,C,D} --snapshot DIR --out DIR
python -m benchmark.harness.cli run-episode benchmark/tasks/EVOL-EPISODE-POISON-10/episodes/<id>.yaml --condition {C,D} --out DIR
python -m benchmark.harness.cli check-leakage benchmark/split/sequential.yaml [--tasks-root DIR]
python -m benchmark.harness.cli score-run RUN_DIR --out scores.json
```

Exit codes: `validate-task`/`check-leakage` return 0 when clean, 1 on findings;
`run-task` returns 0 (pass), 1 (fail), 2 (error), 3 (inconclusive).

## Dependencies

Python ≥ 3.10 stdlib + torch. PyG is optional (SciML tasks ship fallback scatter).
YAML files are restricted to the subset documented in `harness/miniyaml.py`;
`validate-task` enforces it. No network access, no PyYAML, no pytest.
