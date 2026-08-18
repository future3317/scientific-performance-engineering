# Scientific Performance Engineering

Evidence-governed performance engineering for PyTorch and scientific workloads.
The repository contains one reusable skill (`SKILL.md`), its typed Rule OS, and
the executable SPE-EvoBench calibration harness.

## Contents

- [What is current](#what-is-current)
- [Quickstart](#quickstart)
- [Repository map](#repository-map)
- [Benchmark status](#benchmark-status)
- [Documentation policy](#documentation-policy)
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
until reviewed replay evidence exists). The checked-in v1.0-20 population is a
structural/calibration pilot, not a formal A/B/C/D efficacy result. Formal-50
and efficacy claims remain fail-closed until a preregistered population,
independent calibration approval, and a valid external-executor receipt exist.

## Quickstart

Run from the repository root with the project-specific Python environment (not
Conda `base`):

```powershell
python scripts/validate_skill.py .
python scripts/behavioral_contract_tests.py
python scripts/experience_contract_tests.py
python scripts/evolution_contract_tests.py
python scripts/rule_engine_tests.py
python benchmark/tests/run_all.py
python -m pytest -q
```

For a release/readiness check, also run the schema, experience, benchmark, and
leakage validators:

```powershell
python scripts/generate_rule_schemas.py
python scripts/validate_benchmark.py assets/benchmark_record.json
python scripts/evolution_utility_tests.py
python assets/materials_gnn_checks.py --self-test
python scripts/compare_benchmarks.py --self-test
python -m benchmark.harness.cli check-leakage benchmark/split/sequential.yaml
```

For a selected GPU workload:

```powershell
python scripts/run_with_gpu_monitor.py --output monitor.json --gpu 0 -- python train.py
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

SPE-EvoBench v1.0-20 contains 18 atomic tasks and 2 evolution episodes. The
family catalog is the single source of truth for public parameters,
applicability, actions, transformations, and scientific gates. BoundaryBench,
InteractionBench, and evolution views reconstruct from that catalog; they do
not define alternate workload truth.

Conditions A/B/C/D share the public task context and budget. B/C/D receive the
same read-only rendered skill view; C retrieves raw experiences only; D alone
performs governed promotion and routing. Formal execution is invalid without a
real no-network executor receipt. See [`benchmark/README.md`](benchmark/README.md)
for the current calibration boundary and commands.

## Documentation policy

Normative specifications and active workload references stay at their stable
paths. Superseded plans, resolved integration notes, and duplicate pilot
README files are kept under the corresponding `archive/` directory so history
remains recoverable without creating a second current source of truth.

## License

MIT. Do not commit datasets, checkpoints, runtime logs, private environment
snapshots, or credentials.
