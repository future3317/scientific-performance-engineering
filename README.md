# Scientific Performance Engineering

Evidence-governed performance engineering for PyTorch and scientific workloads.
The repository contains one reusable skill (`SKILL.md`), its typed Rule OS, and
the executable SPE-EvoBench calibration harness.

## Contents

- [What is current](#what-is-current)
- [Quickstart](#quickstart)
- [Repository map](#repository-map)
- [Benchmark status](#benchmark-status)
- [Recent execution evidence](#recent-execution-evidence)
- [Documentation policy](#documentation-policy)
- [License](#license)

## What is current

The October skill-evolution development study is documented in
[`docs/skill_v2/IMPLEMENTATION_LOG.md`](docs/skill_v2/IMPLEMENTATION_LOG.md), with
the executable provider-free protocol at
[`experiments/skill_evolution/PROTOCOL.md`](experiments/skill_evolution/PROTOCOL.md).
It preserves the September evidence and separates agent, skill-content, and
routing effects. Its synthetic qualification is implementation evidence;
real-task efficacy requires the independent staged study described there.

The production path is:

```text
preflight → contract freeze → lifecycle census → baseline/noise
→ profile/classify → hypothesis/Amdahl → minimal intervention
→ activation proof → layered measurement → scientific/failure gates
→ statistical decision
```

The evolution path is equally explicit:

```text
experience → candidate → independent paired replay → bounded utility/CS gate
→ governance → canonical or retired revision
```

`RuleSpec` (meaning), `EvidenceEvent` (append-only evidence), and `RuleState`
(materialized lifecycle state) are separate typed records. `core/acre/` is the
single ACRE method core; `AcreEngine.route()` is the only governed production
routing path. Retrieval is not promotion, and a worker cannot author
applicability, relation truth, or replay outcomes.

The canonical registry is intentionally small (`registry/rules.json` is empty
until reviewed replay evidence exists). The checked-in v1.0-30 population is a
structural/calibration pilot, not a formal A/B/C/D efficacy result. Formal-50
and efficacy claims remain fail-closed until a preregistered population,
independent calibration approval, and a valid external-executor receipt exist.

Performance-efficacy analysis uses a separate frozen, agent-blind eligibility
manifest. It binds benchmark revision, population/protocol digests, task-package
digests, runtime contract, paired measurement-session IDs, and baseline-only
typical null noise before any A/B/C/D outcome is observed. The full pilot remains
available for attrition, abstention, negative-transfer, poison, drift, and
diagnostic evidence; only the pre-frozen measurement-eligible subset contributes
to confirmatory performance estimands. The primary contrast is D-B, with D-C and
B-A as secondary contrasts. Raw speedup is one performance coordinate, not the
sole self-evolution endpoint.

The target runtime is part of the estimand. Before calibration, a baseline-only
resource-feasibility census freezes the executable population in
`benchmark/formal/resource_domain.py`. Clean, repeated capacity failures remain
in the 30-task accounting and attrition report, but are not assigned zero utility
or included in confirmatory performance estimates. Unresolved admission or
protocol evidence blocks release rather than being silently classified as
capacity attrition.

## Execution environment

The server is the only supported runtime. Run every project test, validator,
CLI command, benchmark, calibration, qualification, timing probe, GPU job,
and evidence-producing command through `ssh dbcloud`, from
`/home/workspace/lrh/CODE/scientific-performance-engineering`, with
`/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python`. Do not run
project commands from the local Windows checkout, local Python, or any local
Conda environment. Local access is for editing, reading, and static inspection
only; local output is not runtime evidence. Do not use Conda `base`.
Qualification and official calibration use the frozen
`SHARED_GPU_MATCHED_SESSION_V1` policy: shared compute is allowed, but the
selected GPU UUID/runtime contract, paired session identity, required
artifacts, and executor lifecycle remain hard validity invariants. Foreign
processes and GPU load telemetry are recorded as nuisance covariates rather
than treated as automatic failures.

## Quickstart

Run on the server from the repository root with the project-specific Python
environment (not Conda `base`):

```bash
ssh dbcloud
cd /home/workspace/lrh/CODE/scientific-performance-engineering
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python scripts/validate_skill.py .
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python scripts/behavioral_contract_tests.py
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python scripts/experience_contract_tests.py
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python scripts/evolution_contract_tests.py
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python scripts/rule_engine_tests.py
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python benchmark/tests/run_all.py
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python -m pytest -q
```

For a release/readiness check on the server, also run the schema, experience,
benchmark, and leakage validators:

```bash
ssh dbcloud
cd /home/workspace/lrh/CODE/scientific-performance-engineering
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python scripts/generate_rule_schemas.py
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python scripts/validate_benchmark.py assets/benchmark_record.json
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python scripts/evolution_utility_tests.py
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python assets/materials_gnn_checks.py --self-test
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python scripts/compare_benchmarks.py --self-test
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python -m benchmark.harness.cli check-leakage benchmark/split/sequential.yaml
```

For a selected GPU workload:

```bash
ssh dbcloud
cd /home/workspace/lrh/CODE/scientific-performance-engineering
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python scripts/run_with_gpu_monitor.py --output monitor.json --gpu 0 -- /home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python train.py
```

The benchmark entry points and exit codes are documented in
[`benchmark/README.md`](benchmark/README.md). Do not treat a smoke test,
static review, or GPU utilization trace as a campaign-level speedup claim.

## Repository map

- [`SKILL.md`](SKILL.md): compact routing policy and evidence/lifecycle contract.
- `core/`: typed models, predicates, governance, ACRE, and retrieval adapters.
- [`references/`](references/): current domain references. Start with
  [`MEASUREMENT_CONTRACT.md`](references/MEASUREMENT_CONTRACT.md) and
  [`EXPERIMENT_WORKFLOW.md`](references/EXPERIMENT_WORKFLOW.md), then load only
  the route relevant to the workload.
- `scripts/`: validators, replay, telemetry, and contract tests.
- `benchmark/`: frozen family catalog, task packages, harness, formal driver,
  boundary/interaction pilots, and task matrix.
- `experience/`, `evolution/`, `rules/`, `registry/`: evidence and rule state.
- `assets/`: schemas, templates, and durable report formats.
- `docs/archive/`: historical implementation plans retained for provenance;
  nothing there is a current execution instruction.

## Benchmark status

SPE-EvoBench v1.0-30 contains 27 atomic tasks and 3 evolution episodes. The
family catalog is the single source of truth for public parameters,
applicability, actions, transformations, and scientific gates. BoundaryBench,
InteractionBench, and evolution views reconstruct from that catalog; they do
not define alternate workload truth.

Conditions A/B/C/D share the public task context and budget. B/C/D receive the
same read-only rendered skill view; C retrieves raw experiences only; D alone
performs governed promotion and routing. Formal execution is invalid without a
real no-network executor receipt. See [`benchmark/README.md`](benchmark/README.md)
for the current calibration boundary and commands.

## Recent execution evidence

The latest server-side closure run used the `equivcompiler` environment and the
real `ReferenceExecutor`/bubblewrap path on GPU 4. The executor attestation
passed Python boot, network blocking, read-only task mounts, writable solution
and result mounts, hidden host/repository paths, and process cleanup. Its
digest was `af662c55cd85178a58da083220a9348c4a7d3c24333fd0bc7badb18c93392987`.

An eight-cell B/D smoke (`4 tasks × 2 conditions`, reset context) completed with
8/8 execution-valid cells, valid receipts, no resource-blocked cells, and no
surviving processes. The worker intentionally performed no optimization patch,
so all scientific outcomes were `inconclusive`; this run validates execution,
store-transition, and governance plumbing only.

A separate three-task D restart smoke completed two mutable tasks, was
intentionally interrupted, and resumed from the contiguous prefix. The resumed
campaign finished 3/3 execution-valid cells with 3/3 attested receipts and
store digest transitions, while keeping `results_claimed=false`. These artifacts
are operator-side evidence under `RESULTS/SPE`; they do not open the empirical
calibration gate, generate formal-50, or support an ACRE-versus-baseline claim.

## Current calibration state

The measurement-protocol v2 revision is present in the server working tree but is not
yet a release revision. It uses a label-symmetric log-ratio median/MAD estimator,
separates cold_start_inclusive compiler timing from steady_state CUDA timing, keeps
raw control pairs and runtime provenance, and reports maximum deviation only as a
diagnostic. Estimator semantics, phase separation, and provenance passed the structural
qualification checks.

A 12-session baseline-only repeatability characterization on GPU2 is recorded outside
the repository at
/home/workspace/lrh/RESULTS/measurement-repeatability-be752e9-gpu2/.
Typical-noise session values remain variable for graph and crystal workloads, so
measurement repeatability is pending. Q3 and RC8 are NO-GO; no candidate/oracle result
is being reinterpreted and no workload or scientific threshold is being changed.

Research-process evidence is maintained outside Git under
/home/workspace/lrh/RESULTS/RESEARCH_LEDGER.md and
/home/workspace/lrh/RESULTS/research_ledger.jsonl. The paper source is maintained
separately outside this repository; paper prose is not benchmark evidence.

## Documentation policy

Normative specifications and active workload references stay at their stable
paths. Superseded plans, resolved integration notes, and duplicate pilot
README files are kept under the corresponding `archive/` directory so history
remains recoverable without creating a second current source of truth.

## License

MIT. Do not commit datasets, checkpoints, runtime logs, private environment
snapshots, or credentials.
