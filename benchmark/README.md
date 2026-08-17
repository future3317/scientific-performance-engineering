# SPE-EvoBench

A benchmark suite for **S**cientific **P**erformance **E**ngineering agents and for
the **evolution** of the `scientific-performance-engineering` skill itself.

**Read [BENCHMARK_DESIGN.md](BENCHMARK_DESIGN.md) first** — it is the normative spec
(task packages §4, verification pipeline §6, threat model §7, scoring §8,
conditions A–D §9, sequential split §10, harness architecture §11).
[INTEGRATION_REQUIREMENTS.md](INTEGRATION_REQUIREMENTS.md) lists the benchmark-local
workarounds for core-skill gaps (no core file is ever modified).

## Contents

- [Status and calibration scope](#status-and-calibration-scope)
- [Family source of truth](#family-source-of-truth)
- [Layout](#layout)
- [Quickstart](#quickstart)
- [Dependencies](#dependencies)

## Status and calibration scope

**Status:** the v1.0-20 population-validity pilot contains 18 atomic tasks plus
2 evolution episodes. The formal-eval driver is available for a dry-run or an
explicit agent command, but this repository claims no formal A/B/C/D results.
The target remains frozen as SPE-EvoBench v1.0-50 (24 SPE-Core + 20 SciML + 6
Evolution). The current pilot contains 11 Core, 7 SciML, and 2 Evolution
tasks; 13 Core, 13 SciML, and 4 Evolution slots remain ungenerated and are
gated on pilot calibration.

## Family source of truth

The workload source of truth is now the Family catalog. All eleven canonical
families define parameter axes, interventions, scientific policies, and legal
transformations. Existing
directories under `tasks/` remain materialized canonical anchors; task,
BoundaryBench, InteractionBench, and evolution views refer back to the same
family generator instead of maintaining separate synthetic workloads.

Each `FamilySpec` owns the applicability predicate and scientific truth used by
all views. The 20 materialized tasks are reconstructable anchor instances via
`reconstruct_anchor_instance`; their task metadata is only a projection. Run
`python -m benchmark.families.consistency --surface-count 100` to check every
family anchor,
BoundaryBench, InteractionBench, and evolution lineage agreement. The pilot
surface runner expands the canonical families to 100--500 hidden contexts:

```bash
python scripts/run_pilot_surface_experiments.py --surface-count 100 \
  --out benchmark/pilot_surface_report.json
```

This report is a calibration/diagnostic artifact, not an algorithmic-success claim
and not a formal-50 result. Active-boundary rows use an observable posterior
time-uniform Bernoulli confidence sequences (without posterior fallback) and report the registered context
count, multi-seed cost/error curves, and TP/FP/FN/TN trajectories. Interaction
rows use the 8/16/32/64/128 sequential block schedule with predeclared alpha
spending and retain canonical hidden versus predicted relations, confidence
intervals, context variants, and stopping blocks. Evolution rows
are evaluated by a cross-context `RelationIdentifier`; redundancy and
context-dependent relations are reported separately, while higher-order rows
use a measured three-intervention residual. Evolution C retrieves raw
experiences under the matched budget and D routes governed specs through the
same FamilyEnvironment. Evolution rows must contain a canonical-rule transfer
phase before drift/poison metrics are interpreted. Evolution Regret and the vector Performance Profile are aggregated
only by `benchmark.formal.aggregate`.

Family specifications also own the finite predicate grammar, public feature
domains, legal semantic actions, outcome model, and transformation policy.
Boundary and interaction generators therefore do not maintain a second global
grammar or a label-level interaction switch. Promotion replay uses family
representative/query contexts to obtain the preregistered number of independent
groups required by the mixture confidence sequence; the task stream is only the
candidate trigger. Held-out validation records replication, within-predicate
transfer, and boundary-challenge classes separately. Outer formal trials use a
seeded blocked rotation/shuffle and record that order in each trial manifest;
this is not called a Latin square.

The external executor must write an `executor_receipt.json` at
`SPE_EXECUTOR_RECEIPT_PATH` with `network_mode: none`, an explicit mount
allowlist, executor digest, worker identity, skill-view digest (for B/C/D), and
measured usage. The driver treats a missing, malformed, or over-budget receipt
as an invalid trial; worker-authored usage is not authoritative. After verification, each task records an explicit evidence,
maintenance, and store-attestation transition; C remains raw-experience-only,
while D is the only condition allowed to promote governed rules.

Formal worker treatment is namespace-based: an agent run must supply an external
container or namespace executor with `--executor-command` and an allowlisted
`--executor-digest`. B/C/D receive the same
read-only rendered skill view; A receives no skill. The executor mounts only the
materialized public task, solution workspace, public tests, optional skill view,
and pre-task `retrieved_context.json`; the benchmark root, verifier, hidden task
package, and condition store remain outside the worker namespace. Retrieval and
routing use only the task's explicit public context. `reset` is the default
context mode and starts each task without prior trajectory context. The formal
interaction report uses `wrong_relation_rate_among_resolved`, `unresolved_rate`,
and `total_identification_failure_rate`; unresolved decisions are not counted as
wrong relations.

Formal result claims require a separate `calibration_approval.json` beside the
population report. The approval records digests of the population and empirical
calibration artifacts plus the governing review policy; a writable report alone
cannot open the claim gate.

The three compile anchors are intentionally non-interchangeable. `compile_recompile`
measures a graph-break plus cold shape-specialization schedule, `compile_dynamic_shapes`
keeps a tensor-only variable-shape workload and tests dynamic-shape handling, and
`compile_tiny_graphs` is a short-lived counterexample where compile startup should
be rejected. The first two use an end-to-end `schedule_wall_ms` primary metric that
includes first compilation and subsequent registered-shape work; the tiny anchor
uses `cold_shape_schedule_ms` so its non-amortized startup cost remains visible.
First encounters are never hidden in warmup, while steady-state latency is retained
as a diagnostic. The compile-family contract fixes `TORCHINDUCTOR_COMPILE_THREADS=2`
for A/B/C/D and records it in the calibration environment manifest.

## Layout

Promotion evidence is collected independently of boundary evidence. Candidate
proposals are persisted before they are sufficient for synthesis, and later
tasks hydrate the append-only ledger before rerunning CEGIS. Only representative
cases covered by the synthesized predicate enter promotion replay. Held-out
validation is execution-backed, promotion-disjoint, scientifically valid, and
must clear the registered regression tolerance; poison probes must execute a
materialized intervention and be rejected by the observed environment result.
Replay and routing share the versioned bounded_log_speedup_v1 utility policy.

The executable closure is Core-owned: `AcreMaintainer` runs paired plans,
factorial relation experiments, and lifecycle reduction; the formal driver only
materializes public contexts, invokes the verifier/environment, and persists
immutable cases and certificates. Pending replay is executed through the same
FamilyEnvironment path rather than exposed as worker-visible schedule metadata.
Higher-order bundles remain blocked until a typed `RequiredExperiment` and its
2^3 certificate are available.

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
`validate-task` enforces it. The standalone benchmark suite uses only the
standard library; the repository-level pytest suite is optional. Campaign
workers are expected to run with network disabled.
