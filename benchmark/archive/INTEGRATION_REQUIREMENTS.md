# SPE-EvoBench — Integration Requirements for the Core Skill

The benchmark (`benchmark/**`) is implemented against the core skill *as-is*. The
items below are gaps where a core-skill interface would make the benchmark cleaner or
where a core bug affects benchmark runs. **They are requests, not modifications** —
the benchmark works around every one of them locally and never edits core files.

Priority: P0 blocks formal evaluation; P1 blocks clean scoring; P2 nice-to-have.

## R1 (resolved) — Runtime rule-injection interface

The runtime needs a reproducible way to materialize the skill an agent sees,
without exposing benchmark or oracle files.

`scripts/render_skill_view.py` now provides the allowlisted skill-view boundary;
`benchmark/harness/conditions.py` uses it and never copies the repository root.

## R2 (resolved) — `run_rule_replay.py` CLI crash

The historical CLI binding bug would make replay subprocesses fail; the current
implementation binds the manifest result before printing.

The CLI now binds the built manifest before printing and the benchmark imports
the same production `build_manifest()` path. This requirement is resolved.

## R3 (P1) — Bridge from telemetry scripts to `benchmark_record.json`

`collect_env.py` and `run_with_gpu_monitor.py` outputs are not shaped like the
benchmark record's `hardware` / `software` / `hardware.host_contention` fields;
nothing merges them. Agents (and the benchmark harness) must hand-bridge.

- Workaround: `benchmark/harness/fingerprint.py` produces a benchmark-local
  fingerprint block.
- Requested: `scripts/bridge_record.py environment.json monitor.json --record record.json`
  or a documented field mapping.

## R4 (resolved) — Grounded utility measurement for replay and library scoring

`run_rule_replay.py` takes `utility_on/utility_off` as *input* JSON, and
`score_rule_library.py` takes an externally supplied utility mapping; nothing in core
measures utility. The benchmark supplies its own grounded measurements
(task-verified speedups) as the utility source.

- Workaround: `benchmark/harness/evolution.py` generates replay case bundles from
  measured paired runs.
Replay now records `utility_policy_id=bounded_log_speedup_v1` and uses the
bounded dimensionless log-speedup transform, with direction determined by
`higher_is_better` and a versioned log scale.
External measurements must provide a positive versioned scale; promotion and
validation reject other policy IDs or unbounded mean effects.

## R5 (P2) — Condition semantics (frozen / raw-retrieval / governed) as core concepts

Conditions A–D (BENCHMARK_DESIGN.md §9) are benchmark-defined. If the core skill
formalizes "frozen snapshot", "raw-experience retrieval", the `C_STRESS`
append-only ablation, and "governed mode" as named configurations, condition
materialization could delegate to core.

## R6 (P2) — Registry/store schema stability notice

The benchmark's evolution episodes populate `experience/`, `evolution/`, `rules/`,
`tests/rule_cases/`, and `registry/rules.json` in synthetic skill copies. Any core
schema-version bump to the five `assets/*.schema.json` contracts changes what
episodes must generate. Requested: a `SCHEMA_VERSIONS.md` or equivalent single place
listing current schema versions (currently: benchmark_record=4, all others=1).

---

Benchmark-local workarounds mean **no core file changes are required** to run the
prototype. R1, R2, and R4 are resolved in the current repository; remaining
requirements are optional follow-up integrations and do not block the v1.0-20
population-validity pilot.
