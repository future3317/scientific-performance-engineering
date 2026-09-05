# Scientific Performance Engineering

Evidence-governed performance engineering for PyTorch and scientific workloads.
The repository contains one reusable skill (`SKILL.md`), its typed Rule OS, and
the executable SPE-EvoBench calibration harness.

## Contents

- [What is current](#what-is-current)
- [Quickstart](#quickstart)
- [Repository map](#repository-map)
- [Benchmark status](#benchmark-status)
- [Current freeze and execution evidence](#current-freeze-and-execution-evidence)
- [Documentation and repository policy](#documentation-and-repository-policy)
- [License](#license)

## What is current

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
structural/calibration pilot, not a formal A/B/C/D efficacy result. The
cross-repository freeze is currently provider-free: hosted feasibility is
insufficient, sealed C/D was not run, and no efficacy result is claimed.

The historical formal campaign remains an immutable observation-boundary case.
Its treatment-coupled missingness is analyzed with the registered
partial-identification machinery; its incomplete cells are not relabeled as
zero outcomes and are not used to claim D-B, D-C, or B-A efficacy.

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

## Current freeze and execution evidence

The authoritative cross-repository mapping is
[`docs/EVOSPE_EVIDENCE_FREEZE_20260905.json`](docs/EVOSPE_EVIDENCE_FREEZE_20260905.json).
The manifest SHA256 is
`b978c74f52b0f10a1b595d84c26c84f5ba5ab8278c161fea20d7bf80ad404b0c`.
The benchmark freeze tag targets the docs-inclusive `main` checkout; the
manifest's `benchmark_commit` identifies the evidence-producing benchmark
source commit. The controller and paper repositories have their own local
freeze-tag targets, and the manifest is the cross-repository authority.

The current status is:

```text
HOSTED_EFFICACY_STATUS = UNSUPPORTED
HOSTED_FEASIBILITY_STATUS = INSUFFICIENT
SEALED_C_D_STATUS = NOT_RUN
RESULTS_CLAIMED = FALSE
```

The final server verification on the frozen benchmark path was `420 passed,
8 warnings`. The earlier full-order cleanup flake was fixed with a
child-ready handshake rather than a fixed sleep; the targeted cleanup
regression passed 36 tests. This verifies the repository's test and evidence
plumbing. It does not authorize a hosted run or create an efficacy claim.

The latest hosted feasibility evidence remains diagnostic only: the registry
records the v1/v2/v3 progression, with current hosted feasibility
insufficient and no sealed C/D campaign. Provider calls for the current
freeze are zero. Hosted failures remain separate from provider-free mechanism
evidence and from the immutable historical formal observation-boundary case.

Research-process evidence is maintained outside Git under
/home/workspace/lrh/RESULTS/RESEARCH_LEDGER.md and
/home/workspace/lrh/RESULTS/research_ledger.jsonl. The paper source is maintained
separately outside this repository; paper prose is not benchmark evidence.

## Documentation and repository policy

Normative specifications and active workload references stay at their stable
paths. Superseded plans, resolved integration notes, and duplicate pilot
README files are kept under the corresponding `archive/` directory so history
remains recoverable without creating a second current source of truth.

The freeze manifest, experiment registry, claim matrix, runtime contract, and
revision history are the current documentation entry points. Do not create
parallel `final`, `v2`, or date-suffixed summaries for the same state. Do
not move or delete RESULTS, checkpoints, manifests, receipts, logs, or local
agent state merely because they are ignored by Git. The Windows
`local_analysis/` directory is private and excluded locally; it is not a
benchmark source and must not be staged.

## License

MIT. Do not commit datasets, checkpoints, runtime logs, private environment
snapshots, or credentials.
