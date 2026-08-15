# SPE-EvoBench — Integration Requirements for the Core Skill

The benchmark (`benchmark/**`) is implemented against the core skill *as-is*. The
items below are gaps where a core-skill interface would make the benchmark cleaner or
where a core bug affects benchmark runs. **They are requests, not modifications** —
the benchmark works around every one of them locally and never edits core files.

Priority: P0 blocks formal evaluation; P1 blocks clean scoring; P2 nice-to-have.

## R1 (P0) — Runtime rule-injection interface

`references/EXPERIENCE_EVOLUTION.md` states canonical cards "do not automatically
enter the runtime prompt"; there is no programmatic way to materialize "the skill as
an agent sees it" for a given store state. The benchmark needs a core-supported way
to render a *skill view* (SKILL.md + selected references + a chosen set of canonical
rule cards) into a directory/prompt bundle.

- Workaround: `benchmark/harness/conditions.py` assembles views itself
  (copy SKILL.md + references + rendered cards) and hash-attests the result.
- Requested core interface: e.g. `scripts/render_skill_view.py --rules RULE_ID... --out DIR`.

## R2 (P0) — `run_rule_replay.py` CLI crash

`scripts/run_rule_replay.py: main()` prints `result["outcome"]` but `result` is only
defined inside `build_manifest()` — the CLI writes the manifest then raises
`NameError` (exit != 0), which breaks the evolution episode runner's subprocess calls.

- Workaround: the benchmark imports `build_manifest()` directly.
- Requested fix: bind the return value in `main()` before printing.

## R3 (P1) — Bridge from telemetry scripts to `benchmark_record.json`

`collect_env.py` and `run_with_gpu_monitor.py` outputs are not shaped like the
benchmark record's `hardware` / `software` / `hardware.host_contention` fields;
nothing merges them. Agents (and the benchmark harness) must hand-bridge.

- Workaround: `benchmark/harness/fingerprint.py` produces a benchmark-local
  fingerprint block.
- Requested: `scripts/bridge_record.py environment.json monitor.json --record record.json`
  or a documented field mapping.

## R4 (P1) — Grounded utility measurement for replay and library scoring

`run_rule_replay.py` takes `utility_on/utility_off` as *input* JSON, and
`score_rule_library.py` takes an externally supplied utility mapping; nothing in core
measures utility. The benchmark supplies its own grounded measurements
(task-verified speedups) as the utility source.

- Workaround: `benchmark/harness/evolution.py` generates replay case bundles from
  measured paired runs.
- Requested: a documented contract for what "utility" means numerically (units,
  noise handling) so external measurements are admissible.

## R5 (P2) — Condition semantics (frozen / append-only) as core concepts

Conditions A–D (BENCHMARK_DESIGN.md §9) are benchmark-defined. If the core skill
formalizes "frozen snapshot", "append-only experience mode", and "governed mode" as
named configurations, condition materialization could delegate to core.

## R6 (P2) — Registry/store schema stability notice

The benchmark's evolution episodes populate `experience/`, `evolution/`, `rules/`,
`tests/rule_cases/`, and `registry/rules.json` in synthetic skill copies. Any core
schema-version bump to the five `assets/*.schema.json` contracts changes what
episodes must generate. Requested: a `SCHEMA_VERSIONS.md` or equivalent single place
listing current schema versions (currently: benchmark_record=4, all others=1).

---

Benchmark-local workarounds mean **no core file changes are required** to run the
prototype. Items R1/R2 should land before formal evaluation campaigns are claimed in
the paper.
