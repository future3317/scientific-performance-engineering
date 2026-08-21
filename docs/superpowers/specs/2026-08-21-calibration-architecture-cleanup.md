# Calibration Architecture Cleanup

## Goal

Converge calibration, task population, harness execution, and formal approval
around one authority per concern while preserving the frozen scientific
contract from `534325f`.

## Non-negotiable invariants

- No changes to active task IDs, workload definitions, repetitions, noise
  floors, effect thresholds, scientific gates, or task package contents.
- The current P0/P1 closure remains a separately published rollback point at
  `534325f`.
- Task packages own scientific contracts; the harness owns one-cell execution;
  calibration owns evidence lifecycle; population owns structural validity;
  formal owns scheduling/statistics; CLI scripts only parse arguments.
- Derived report/pilot/approval JSON is never an evidence authority.
- Cell identity contains task contract, population, protocol, executor/harness,
  and environment identities; taskset identity is campaign-level only.

## Target boundaries

`benchmark/calibration/` becomes the calibration package with these stable
modules:

- `protocol.py`: frozen protocol loading, API execution class, repetition
  policy, and topology contract.
- `identity.py`: cell/taskset identity and digest boundary definitions.
- `execution.py`: the shared `CellExecutor` for bounded subprocess/cleanup and
  atomic/episode/noise cell execution.
- `bundle.py`: raw/noise/envelope persistence, resume compatibility, and
  cleanup receipts.
- `report.py`: deterministic population report/pilot projection from evidence.
- `approval.py`: canonical approval construction and validation entrypoint.

The existing formal and taskgen modules may retain domain-specific algorithms,
but call these public calibration APIs instead of importing script helpers or
private helpers across layers.

## Migration order

1. Freeze and publish the closure point (`534325f`).
2. Move protocol/identity/report/approval authority and replace script-private
   imports with public package calls.
3. Introduce `CellExecutor`; route active calibration first, then formal task
   execution, with behavior-equivalence tests at the boundary.
4. Add the single API execution-class and cell-state serializers; remove
   direct `workspace.api == ...` branching from calibration/formal paths.
5. Split population structural validation and evolution runtime only where
   imports/tests prove an independent responsibility; delete replaced paths.
6. Remove uncalled compatibility branches, materialize runtime stores, and
   move historical calibration artifacts under revision-scoped history without
   changing current evidence semantics.
7. Separate task contract/taskset/executor/formal digests and converge the test
   entrypoint to pytest.

## Acceptance

Each commit must pass the affected focused tests and preserve behavior against
the closure baseline. Before RC freeze run full pytest, structural 30-task
validation, CLI smoke, resume corruption, approval tamper, strict-formal
fail-closed, and `git diff --check`. No active-30 evidence may be generated
until this cleanup series is frozen.
