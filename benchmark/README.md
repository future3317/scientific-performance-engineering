# SPE-EvoBench

A benchmark suite for **S**cientific **P**erformance **E**ngineering agents and for
the **evolution** of the `scientific-performance-engineering` skill itself.

**Read [BENCHMARK_DESIGN.md](BENCHMARK_DESIGN.md) first** — it is the normative spec
(task packages §4, verification pipeline §6, threat model §7, scoring §8,
conditions A–D §9, sequential split §10, harness architecture §11). The former
integration-requirements note is archived; its resolved boundaries and remaining
optional integrations are summarized below.

Formal readiness is fail-closed: the checked-in 30-task population remains a
pilot until `population_report.json`, `calibration/pilot_calibration.json`, and
`calibration/calibration_approval.json` are reviewed and
`validate_population --strict-formal` passes. The v1.0-50 file is a content-free
50-slot preregistration; confirmatory headline D−B uses sealed-35 only. Public
dev slots are calibration/debugging and secondary reporting. Claims and
statistics are frozen in the repository-root `CLAIMS.yaml` and
`references/STATISTICAL_PROTOCOL.md`.

## Contents

- [Status and calibration scope](#status-and-calibration-scope)
- [Family source of truth](#family-source-of-truth)
- [Layout](#layout)
- [Quickstart](#quickstart)
- [Pilot modules](#pilot-modules)
- [Integration boundary](#integration-boundary)
- [Dependencies](#dependencies)

## Status and calibration scope

**Status:** the v1.0-30 population-validity pilot contains 27 atomic tasks plus
3 evolution episodes. The formal-eval driver is available for a dry-run or an
explicit agent command, but this repository claims no formal A/B/C/D results.
The target remains frozen as SPE-EvoBench v1.0-50 (24 SPE-Core + 20 SciML + 6
Evolution). The current pilot contains 16 Core, 11 SciML, and 3 Evolution
tasks; 8 Core, 9 SciML, and 3 Evolution slots remain ungenerated and are
gated on pilot calibration.

`CORE-COMPILE-DYNAMIC-11` has historical same-host oracle evidence with an
outward-rounded expected speedup range `[2.56, 3.34]` and observed control
floors of 51.87--61.22%. That evidence is stale for the current task digest,
so it does not confer current calibration eligibility. The task remains in the
pilot and requires current-revision calibration before any formal use.

The ten-task authoring bundle (21–30) is retained under
[`archive/candidate-bundles/`](archive/candidate-bundles/). It expands
applicability boundaries, SciML positive/counterexample pairs, and evolution
specialization and is now materialized in the canonical v1.0-30 population.
The archive keeps the source bundle and provenance; it is not a second
executable task source. The tasks remain calibration candidates and do not
produce efficacy claims or sealed formal-50 content.

The current H2D counterexample candidate is
`CORE-H2D-OVERFANOUT-23R2`, an independent replacement that requires fresh
target-runtime calibration; it remains a pilot candidate and does not confer
formal eligibility. The evolution episode `EVOL-EQUIVARIANT-SPECIALIZE-30` completed the full C/D
harness: C produced no canonical promotion, while D produced one validated
promotion with replay and poison checks passing. These are execution and
governance observations only, not efficacy or superiority claims.

The server-side staged probes from 2026-08-19 are recorded under
[`calibration/staged-probes/2026-08-19/`](calibration/staged-probes/2026-08-19/).
They include a blocked SciML oracle (observed noise 320.85%), a task-level H2D
replacement probe (valid paired result, observed noise 40.92%, activation not
declared), and a D evolution governance episode (one canonical rule and poison
survival 1.0). These artifacts extend the audit trail only; they do not update
the population approval gate or create formal-50 content.

## Recent external-execution closure

The current server attestation was run from commit `c103257` in the
`equivcompiler` environment with the real `ReferenceExecutor`/bubblewrap path
on GPU 4. Python boot, network blocking, read-only task mounts, writable
solution/result mounts, hidden host/repository paths, and process cleanup all
passed. The executor digest was
`af662c55cd85178a58da083220a9348c4a7d3c24333fd0bc7badb18c93392987`.

The subsequent B/D execution smoke covered four tasks in reset context (eight
cells): all cells were execution-valid, all receipts were attested, no cell was
resource-blocked, and no process survived cleanup. The worker was deliberately
a no-op smoke worker, so every scientific outcome was `inconclusive`. A separate
three-task D restart smoke completed two mutable tasks, was interrupted, and
resumed from the contiguous prefix; all three final cells were execution-valid,
attested, and produced valid store transitions. These runs establish executor,
resume, and governance plumbing only. They do not establish C/D efficacy,
approve calibration, or generate formal-50. The operator-side evidence remains
under `RESULTS/SPE` on the server.

## Family source of truth

The workload source of truth is now the Family catalog. All eleven canonical
families define parameter axes, interventions, scientific policies, and legal
transformations. Existing
directories under `tasks/` remain materialized canonical anchors; task,
BoundaryBench, InteractionBench, and evolution views refer back to the same
family generator instead of maintaining separate synthetic workloads.

Each `FamilySpec` owns the applicability predicate and scientific truth used by
all views. The 30 materialized tasks are reconstructable anchor instances via
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

## Pilot modules

The pilot modules are calibration diagnostics over the same family catalog:

- **BoundaryBench** (`benchmark/boundary/`) evaluates typed predicate synthesis
  after certified counterexamples. `core/acre/cegis.py` owns synthesis; the
  benchmark only constructs disjoint representative/query/sealed pools and
  scores the returned predicate. Hidden sealed truth is an offline score, never
  an acquisition stopping signal.
- **InteractionBench** (`benchmark/interaction/`) runs factorial relation,
  acquisition, and routing pilots. Cross-context relations, normalized
  three-way residuals, alpha-spent contrast confidence sequences, and
  effect-strength × noise power curves are calibration evidence only.
- **Real-artifact feasibility** (`benchmark/formal/real_artifacts/`) contains
  two tiny offline packaging prototypes. They are not part of v1.0-30, do not
  produce benchmark results, and require a network-disabled evaluator with the
  pinned upstream checkout supplied externally.

## Integration boundary

The benchmark consumes the core skill as-is and never edits core files during a
run. The old integration note is archived because its resolved items are now in
the production path: rendered skill views provide the worker boundary, replay
uses the versioned `bounded_log_speedup_v1` utility, and the harness owns the
benchmark fingerprint and condition semantics. The remaining items are
non-blocking: a convenience bridge from telemetry JSON into
`benchmark_record.json`, and a single schema-version index if the core schemas
change. Neither is required for the v1.0-30 population-validity pilot.

The three compile anchors are intentionally non-interchangeable. `compile_graph_break`
isolates a graph break on a fixed-shape schedule, `compile_dynamic_shapes` keeps a
tensor-only variable-shape workload and uses targeted `torch._dynamo.mark_dynamic`
annotation, and `compile_tiny_graphs` is an eight-step counterexample where compile
startup should be rejected. All three use end-to-end `schedule_wall_ms`, including
the required cold compilation; steady-state latency, graph-break count, recompile
count, unique graphs, compile count, and compile time remain diagnostics. The
compile-family contract fixes `TORCHINDUCTOR_COMPILE_THREADS=2` for A/B/C/D and
records it in the calibration environment manifest. Family `logical_steps`, public
context, and measured schedule horizon are validated to be identical, and
`graph_size = hidden_dim * (num_blocks + 1)` is the executable projection.
An optional `scripts/run_compile_transfer_check.py` probes one installed
TorchBench model as calibration-only evidence; missing TorchBench is reported
as blocked and never substituted with a synthetic workload or formal evidence.
Recompile horizon selection is preregistered over 64/128/256/512/1024 steps;
the checked-in graph-break anchor is retained as a calibrated counterexample; a
future positive graph-break workload must be a new family point with its own cold
CI before it can be promoted. Dynamic-mode comparisons are
automatic, targeted annotation, and global diagnostic mode; only targeted is
eligible as the canonical oracle.

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
  archive/                     # historical integration and pilot notes
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
    conditions.py              # A/A_CTX/B/C/C_STRESS/D materialization + attestation
  taskgen/                     # pilot generator and population validator
  families/                    # canonical family specs, instances, and transformations
  formal/                      # experiment manifest, schedule, driver, aggregation
  population_report.json       # deterministic v1.0-30 population summary
    split.py                   # sequential split + leakage checker
    cli.py                     # the CLI below
  tasks/<task_id>/...          # task packages (§4)
  tests/                       # pytest suite; run_all.py is a thin compatibility wrapper
```

## Quickstart

Run everything from the repository root (the directory containing `benchmark/`):

```bash
# tests (single repository authority)
python benchmark/tests/run_all.py
# equivalent direct entrypoint (task-package smoke fixtures are excluded by pytest.ini)
python -m pytest -q
python -m benchmark.taskgen.validate_population \
    --tasks-root benchmark/tasks --out benchmark/population_report.json

# CLI examples (run from the repo root)
python -m benchmark.harness.cli validate-task benchmark/tasks/<task_id>
python -m benchmark.harness.cli run-task benchmark/tasks/<task_id> --solution DIR --out result.json \
    [--predict-mechanism scalar_sync,h2d_blocking] [--seed 0]
python scripts/render_skill_view.py SKILL_ROOT SKILL_VIEW_DIR
python -m benchmark.harness.cli materialize-condition {A,A_CTX,B,C,C_STRESS,D} --snapshot SKILL_VIEW_DIR --out DIR --context-mode reset
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
