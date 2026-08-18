# SPE-EvoBench: Benchmark Design

A paper-grade benchmark for **S**cientific **P**erformance **E**ngineering agents and for the
**evolution** of the `scientific-performance-engineering` skill itself.

Status: **v1.0-20 population-validity and calibration pilot implemented** (18
atomic tasks plus 2 evolution episodes). The formal-eval driver supports a
reproducible dry-run and explicit agent-command execution, but no formal A/B/C/D
result or algorithmic-success claim is made by this repository. The formal target is frozen as SPE-EvoBench v1.0-50
(24 SPE-Core + 20 SciML + 6 Evolution). The current pilot contains 11 Core,
7 SciML, and 2 Evolution tasks; the remaining 13 Core, 13 SciML, and 4
Evolution slots are gated on empirical calibration.

## Contents

- [1. Purpose and scope](#1-purpose-and-scope)
- [2. Design principles](#2-design-principles)
- [2.1 Family views and frozen anchors](#21-family-views-and-frozen-anchors)
- [3. Tracks](#3-tracks)
- [4. Task package anatomy](#4-task-package-anatomy)
- [5. Controlled-mutation taxonomy](#5-controlled-mutation-taxonomy)
- [6. Verification architecture](#6-verification-architecture)
- [7. Anti-reward-hacking threat model](#7-anti-reward-hacking-threat-model)
- [8. Scoring](#8-scoring)
- [9. Experimental conditions A–D](#9-experimental-conditions-ad)
- [10. Sequential split and leakage control](#10-sequential-split-and-leakage-control)
- [11. Harness architecture and CLI](#11-harness-architecture-and-cli)
- [12. Pilot task set](#12-pilot-task-set)
- [13. Reproducibility and environment](#13-reproducibility-and-environment)
- [14. External sources](#14-external-sources)
- [15. Limitations](#15-limitations)

## 1. Purpose and scope

SPE-EvoBench measures three things that existing benchmarks measure only in isolation:

1. **SPE-Core** — whether an agent can do *end-to-end* PyTorch performance engineering
   (host overhead, synchronization, launch fragmentation, H2D/data pipeline, graph
   construction, repeated computation, autograd overhead, memory pressure,
   compile/recompilation, checkpointing, DDP synchronization, thread contention),
   not only CUDA kernel generation (KernelBench's scope).
2. **SciML** — whether optimizations preserve *scientific semantics*: energy/force
   consistency, gradients, equivariance, physical validity, and time-to-quality on
   graph/materials/crystal-generation workloads derived from FAIRChem/OC20 and
   CDVAE/MP-20 structures, but self-contained (no dataset downloads).
3. **Evolution** — whether a skill that *evolves* over a sequential task stream
   (rule acquisition, reuse, specialization, conflict resolution, retirement, drift
   recovery, poisoning survival) outperforms frozen, raw-retrieval, and no-skill
   conditions under identical budgets; the old append-only arm is retained as
   `C_STRESS` only.

The benchmark is a **skill-evaluation** benchmark: the unit of comparison is a paired
condition (§9), and the headline result is a *verified* performance/evolution delta,
never a raw trajectory judgment. No LLM-as-judge is used for scoring anywhere.

Non-goals: CUDA/runtime correctness debugging, install/build failures, generic
distributed correctness, and pure kernel-authoring skill without a systems context.

## 2. Design principles

P1. **Executable before literary.** Every claim an agent can make is checked by running
    code. Prose in a report is never scored.
P2. **Correctness and scientific gates precede speed.** A fast wrong answer scores 0.
    Gates are deterministic programs in `hidden_verifier/`, not prompts.
P3. **Verified speedup, not measured speedup.** A speedup is *verified* only when the
    paired bootstrap CI lower bound clears `max(min_improvement, noise_floor)` on the
    same artifact that passed correctness — the exact callable that was timed is the
    callable that was verified (TorchBench's split-pass hole is closed).
P4. **Real workload + controlled mutation.** Tasks are minimal but structurally real
    training/inference/sampling loops; bottlenecks are injected via documented mutation
    templates (`mutation_template_id`), and oracle patches exist for calibration.
P5. **Counterexamples are first-class.** Every mechanism family ships tasks where the
    "obvious" optimization is wrong (slower, or scientifically invalid), so pattern
    matching is penalized, and the correct action can be *do not apply rule R*.
P6. **Group-split by lineage.** Acquisition/evolution-visible tasks and hidden test
    tasks never share `(family, mechanism, source_lineage, mutation_template)` groups.
P7. **Paired everything.** Baseline/candidate, skill/no-skill, condition A/B/C/D run
    interleaved under identical seeds, order, budgets, and hardware fingerprints.
P8. **Small and deterministic beats large and flaky.** Tiny self-contained fixtures
    (seeded synthetic data, bundled small data only) with hard runtime budgets.
P9. **The harness is immutable and attestable.** Harness files are hashed; the hidden
    verifier and oracle never enter the agent-visible sandbox.
P10. **Raw dimensions preserved.** Composite scores are reported *alongside* every raw
    metric; no dimension is hidden by aggregation.

## 2.1 Family views and frozen anchors

`benchmark/families/` is the single workload-family source of truth. A
`FamilySpec` defines parameter axes, legal interventions, and family
transformations; `FamilyInstance` is a deterministic point in that space. The
spec also owns the applicability predicate and scientific-truth projection used
by every view. `reconstruct_anchor_instance` rebuilds each of the 20 anchors
from its declared family rather than copying a second workload definition. A
cross-view validator checks the anchor, boundary, interaction, and evolution
lineage before pilot experiments are run.
existing packages under `benchmark/tasks/` are retained as canonical anchor
instances with their original verifiers and oracle calibration. `taskgen`
materializes future slots from the same specs, while BoundaryBench derives
representative, active-query, and sealed-boundary pools and InteractionBench
combines family interventions into factorial surfaces. Evolution episodes record
the same family transformations (software, hardware, scale, scientific-regime,
or harness drift) rather than introducing a separate workload generator.
The pilot surface runner expands all eleven canonical families to 100--500 hidden
contexts for calibration. Active-boundary stopping uses a time-uniform Bernoulli
confidence sequence and remains truth-blind during acquisition; InteractionBench
is sequential over 8/16/32/64/128 blocks with predeclared alpha spending;
evolution episodes require canonical transfer evidence
before drift/poison metrics are interpreted. Relation applicability is learned
with the same typed predicate grammar as boundary applicability, and three-way
interaction residuals are reported on a normalized [-1,1] scale alongside their
raw inclusion--exclusion value. These surfaces are not formal-50 slots or formal
results.

Formal authority is explicit. Worker submissions contain only solution artifacts
and typed intervention hypotheses; replay cases, paired effects, scientific
gates, confidence sequences, applicability predicates, and relation certificates
are produced by the harness and Core. Boundary counterexamples reduce the CEGIS
version space but never enter representative promotion evidence. Promotion,
held-out validation, poisoning probes, and relation endpoints are disjoint and
are checked against their current canonical revisions. A required pair or
three-way experiment is blocked unless an external verifier returns every
declared arm, scientific gate, and an attested execution receipt. Mutation
journals account for every governed rule, relation, registry, and state artifact;
an unjournaled store change invalidates the campaign.

## 3. Tracks

### 3.1 SPE-Core

End-to-end PyTorch performance-engineering tasks. The bottleneck catalog (§5) spans:
CPU/Python overhead, scalar synchronization (`.item()`/host reads), launch
fragmentation, H2D/data pipeline (pinning, prefetch, worker fan-out), repeated graph
construction, repeated computation, autograd/VJP overhead, memory pressure,
`torch.compile` graph breaks and dynamic-shape recompilation, activation
checkpointing, DDP synchronization, validation/checkpoint cadence, and host/process/
thread contention. Kernel-only tasks are explicitly out of scope for this track
(they live in the kernel/compiler prototype, track `spe_core` family `compiler`).

### 3.2 SciML

Scientific-ML workloads with executable semantic gates:

- **Graph/materials** (OC20/FAIRChem-shaped): radius-graph + PBC construction,
  message passing, energy/force (`F = -∂E/∂x`) heads, equivariant tensor properties.
  Gates: energy/force consistency, gradient agreement, equivariance error
  (rank-3 check adapted from `assets/materials_gnn_checks.py`), conservation/range
  checks.
- **Crystal generation** (CDVAE/MP-20-shaped): diffusion/score sampling loops on
  tiny deterministic fixtures. Gates: structural validity (minimum interatomic
  distance, lattice-parameter ranges, charge-balance surrogate on synthetic
  compositions), distribution-moment checks for samplers, and **time-to-quality**
  (wall-clock to reach a fixed validity threshold under a step budget).

Workloads are *shape-faithful, data-tiny*: the tensor structure, loop structure, and
gate structure mirror the upstream systems; the data is seeded synthetic or bundled
small data, so every prototype runs with zero external downloads.

### 3.3 Evolution

A sequential episode protocol over task streams (§10) that measures the skill's
maintenance machinery, not the agent's raw ability: rule acquisition, rule reuse,
specialization, conflict handling, negative transfer, redundant-rule accumulation,
retirement, version/runtime drift, misleading experience, poisoned experience, and
recovery after bad rules. Scored by the evolution metrics in §8.4, always as paired
deltas between conditions C/D (and controls A/B).

## 4. Task package anatomy

```
tasks/<task_id>/
  task.yaml               # machine-readable contract (schema: schema/task.schema.json)
  metadata.json           # lineage, split group, difficulty, authoring info
  benchmark.py            # harness-side measurement entry (immutable during eval)
  scientific_contract.py  # harness-side scientific gates (immutable during eval)
  workspace/              # AGENT-VISIBLE: the code under optimization
    solution.py           #   entrypoint the agent edits (API fixed by task.yaml)
    README.md             #   task instruction given to the agent
    ...                   #   supporting modules, fixtures loaders
  public_tests/           # AGENT-VISIBLE smoke tests (may be run by the agent)
  hidden_verifier/        # HARNESS-ONLY: correctness + scientific + anti-cheat gates
  oracle/                 # DEV-ONLY, never shipped to an evaluation sandbox
    bottleneck.json       #   injected-bottleneck description (mechanism, location)
    reference_patch.diff  #   one acceptable fix (for calibration, not scoring)
    expected_mechanism.json # gold diagnosis label(s) for diagnosis accuracy
```

Rules:

- `benchmark.py` exposes `main()` and the functions `load_solution(path)`,
  `make_fixtures(seed)`, `run_correctness(solution, fixtures)`,
  `run_scientific_gates(solution, fixtures)`, `run_performance(solution, fixtures)`
  — the harness drives these in a fixed order (§6).
- During formal evaluation the agent sandbox receives **only** `workspace/` and
  `public_tests/`. `benchmark.py`, `scientific_contract.py`, `hidden_verifier/`,
  `oracle/`, `task.yaml`, and `metadata.json` stay outside the writable sandbox.
- `task.yaml` is the single source of truth for measurement parameters; a task whose
  code and `task.yaml` disagree fails `validate-task`.

### 4.1 `task.yaml` (normative sketch)

```yaml
schema_version: 1
task_id: CORE-SCALAR-SYNC-01        # ^[A-Z0-9-]+$ ; unique
track: spe_core                     # spe_core | sciml | evolution
family: training_loop_overhead      # workload family (split key)
mechanism: scalar_sync              # bottleneck mechanism id (§5, split key)
kind: positive                      # positive | counterexample | do_not_apply
lineage:
  source: synthetic                 # synthetic | fairchem-shaped | cdvae-shaped | kernelbench-shaped
  mutation_template_id: MT-SCALAR-SYNC-V1
  mutation_seed: 1234
title: "Per-step scalar synchronization in a materials GNN training loop"
requires_cuda: false
time_budget_s: 600
workspace:
  entrypoint: solution.py
  api: train_loop_v1                # named API contract (see harness/api.py)
measurement:
  primary_metric: step_ms_p50       # step_ms_p50 | throughput_units_per_s | time_to_quality_s
  higher_is_better: false
  warmup_iterations: 5
  measured_iterations: 30
  repetitions: 5                    # paired baseline/candidate repetitions
  min_improvement_percent: 5.0
  noise_floor_percent: 2.0
  speedup_tripwire: 20.0            # >tripwire => audit flag, not auto-pass
correctness:
  num_fresh_inputs: 3               # fresh seeded input draws per verdict
  reference: fp64_recompute         # fp64 live recompute, never stored goldens
  tolerance: {rtol: 1.0e-5, atol: 1.0e-6}
scientific_gates: []                # gate names implemented in scientific_contract.py
diagnosis:
  enabled: true                     # agent must name the mechanism before/without seeing oracle
  choices: [scalar_sync, h2d_blocking, repeated_compute, compile_graph_break, ...]
oracle:
  expected_speedup_range: [1.2, 6.0]  # calibration range from reference patch
```

## 5. Controlled-mutation taxonomy

Each mutation template has an id `MT-*`, a mechanism id, and a polarity (it can
generate both positive and counterexample instances). The catalog (v1):

| Mechanism id | Mutation templates (examples) | Counterexample form |
|---|---|---|
| `scalar_sync` | inject `.item()`/host reads per step; per-step Python `float()` stats | sync removal is noise on CPU-bound tiny loops → do-not-apply |
| `h2d_blocking` | drop pinning/`non_blocking`, sync copies in loop, no prefetch | pinned+async slower for tiny batches on CPU-only hosts |
| `ragged_loops` | replace batched ops with per-item Python loop | loop is faster below a size threshold (batched kernel launch overhead) |
| `launch_fragmentation` | split fusible pointwise chain into many tiny kernels | fusion/compile slower for tiny tensors (fixed overhead) |
| `repeated_compute` | recompute shared backbone/features per head/step | caching invalid under changing inputs (semantic trap) |
| `graph_rebuild` | rebuild radius graph / neighbor list every step though structure fixed | rebuild required when positions actually change (do-not-apply for MD) |
| `compile_graph_break` | `.item()`-dependent control flow inside a compiled region | graph-break repair is distinct from dynamic-shape handling |
| `autograd_overhead` | per-sample VJP loop instead of batched backward; redundant `backward()` calls | — |
| `checkpoint_cadence` | checkpoint+validation every step; blocking saves | — |
| `memory_pressure` | needless fp64 intermediates, retained graphs, no checkpointing | activation checkpointing slower when memory is ample (do-not-apply) |
| `ddp_sync` | unnecessary `all_reduce`/barrier per micro-step | (multi-process; prototype phase: simulated single-process telemetry) |
| `dataloader_fanout` | pathological `num_workers`/`prefetch` for the workload size | more workers worse for tiny in-memory data |
| `cpu_preprocess` | per-sample Python/NumPy preprocessing in the hot loop | — |

Polarity is recorded in `task.yaml: kind`; scoring treats `counterexample` and
`do_not_apply` tasks symmetrically: the agent is rewarded for *not* applying the
tempting rule and for producing the evidence that justifies abstention.

## 6. Verification architecture

Pipeline (all deterministic, seeded; implemented by `harness/verifier.py` driving the
task's `benchmark.py` + `hidden_verifier/`):

```
S0 sandbox        -> copy workspace/ + public_tests/ to a fresh sandbox dir; hash harness files
S1 static scan    -> AST/regex anti-cheat scan of the candidate diff (§7); hard-fail or flag
S2 correctness    -> scientific work executed on num_fresh_inputs fresh seeded inputs;
                     outputs vs fp64 live-recomputed reference within tolerance ladder;
                     output checksums recorded (anti-caching evidence)
S3 scientific     -> task-declared gates from scientific_contract.py (energy/force,
                     gradient, equivariance, validity, sampling-law); all must be True
S4 activation     -> where relevant: compile counters (no fallback), cache-hit evidence,
                     sync-count evidence; a source patch must match exactly one
                     registered repair-level ActionSpec, otherwise attribution is rejected
S5 performance    -> paired interleaved measurement (§6.1); only runs on the artifact
                     that passed S2–S4
S6 verdict        -> emit result.json (schema/result.schema.json): gates, verified
                     speedup + CI, diagnosis, cost, fingerprints, anti-cheat findings
```

### 6.1 Measurement protocol

- Warmup `warmup_iterations`, then `measured_iterations` timed iterations per run;
  `repetitions` paired runs, **interleaved** baseline/candidate ordering with the order
  itself seeded and recorded (`run_order`).
- CUDA tasks: `torch.cuda.synchronize` bracketing, CUDA events *and* host wall clock
  (cross-checked; divergence beyond tolerance → inconclusive), L2-cache thrash between
  trials for kernel-level tasks, steady-state only (first measured iteration dropped
  from statistics, kept in raw).
- Statistics (mirroring `scripts/compare_benchmarks.py` semantics): per-run paired
  improvement %, median/IQR/MAD, bootstrap CI (seeded `random.Random(0)`,
  `bootstrap_samples=2000`, confidence 0.95), noise floor from a baseline-vs-baseline
  control measurement run on the same host.
- **Verified speedup** = CI lower bound ≥ `max(min_improvement_percent,
  noise_floor_percent)`. Otherwise `inconclusive` (not zero-speedup).
- Formal campaign aggregation keeps this task-level measurement separate from
  evolution promotion: performance effects use paired log ratios of raw median
  speedups, while task-score effects remain linear differences.
- `speedup_tripwire`: verified speedup above the tripwire is flagged for audit
  (possible semantic skip) — flagged results are excluded from headline aggregates
  unless the audit trail explains them.
- Hardware/software fingerprint (CPU, governor-ish info where available, GPU name/
  UUID policy per privacy default, torch/CUDA/PyG versions, thread env) is recorded
  into every result; results with mismatched fingerprints are never compared.

### 6.2 Time-to-quality (SciML)

For generation/sampling tasks, the primary metric can be `time_to_quality_s`:
wall-clock to first reach a declared quality threshold (e.g., validity rate ≥ τ over a
seeded evaluation batch), capped by `time_budget_s`; failures-to-reach score as the
cap with a `reached=false` flag (never silently treated as the cap).

## 7. Anti-reward-hacking threat model

Adversary: a capable agent optimizing the score, with write access to its sandbox,
read access to public tests, and no network (formal setting). Threats and defenses:

| Threat | Defense |
|---|---|
| Skip scientific work (return cached/fake outputs) | Fresh seeded inputs per verdict (S2); output checksums; input-dependent recompute; result-reuse adversarial probes (KernelBench-style: zeroed-input probe, mutated-input probe) |
| Change task/batch semantics (smaller batch, fewer steps, weaker validation) | Frozen fixtures; work-unit counters inside fixtures (forward/backward call counts, atoms/edges processed) must match baseline exactly; validator cadence asserted |
| Monkey-patch timing/randomness/torch internals | Static AST scan (hard fail on `time.`, `torch.cuda._`, `random.seed` tampering, `sys.modules` mutation of harness modules); runtime module-integrity check before/after |
| Modify the benchmark harness | Harness files hashed at S0 and re-hashed at verdict; hidden verifier lives outside the sandbox; sandbox has no write path to it |
| Read the oracle / hidden tests | Oracle + hidden verifier never copied into sandbox; canary strings in oracle files trip a finding if they appear in the candidate diff or logs |
| Network lookup of reference solutions | Formal runs: network disabled at the driver level; static scan flags `socket`/`urllib`/`requests` imports in candidate code |
| Git-history leakage | Evaluation sandboxes are exported trees (no `.git`); harness asserts absence of VCS metadata |
| Specialize to public tests (teach-to-the-test) | Public tests are smoke-level only; hidden verifier uses disjoint fresh inputs and additional gates; counterexample tasks punish over-application |
| Overfit fixed shapes/seeds | Fixtures re-drawn per repetition from seeded generators; shape jitter within the task's declared family where semantics allow |
| Excessive speedup via semantic skip | Tripwire + audit; work-unit counters; distribution checks on outputs (not just allclose to reference) |
| Reward farming via retries | One verdict per (task, condition, seed-set); retries recorded in cost, not hidden |

Known limitations of the prototype harness: process isolation is by subprocess with
timeouts (no container); resource isolation (cgroup-level) is driver responsibility in
formal runs; the static scan is defense-in-depth, not a proof.

## 8. Scoring

### 8.1 Per-task result (raw dimensions, all preserved)

- `correctness_pass` (bool), `scientific_gates` (map name→bool)
- `verified_speedup`: median paired speedup, CI low/high, `verified` bool, `inconclusive` bool
- `time_to_quality_s` + `reached` (where applicable)
- `diagnosis`: predicted mechanism(s) vs `expected_mechanism.json` → `diagnosis_correct`
- `cost`: wall time, token/tool counts (filled by the outer agent driver)
- `anticheat`: findings list, `tripwired` bool

### 8.2 Per-task score (prototype defaults)

- Gates failed → task score 0 (and the gate failure is reported).
- Else `task_score = w_perf · perf_term + w_diag · diagnosis_correct` with
  `w_perf=0.8, w_diag=0.2`; `perf_term` for positive tasks is
  `min(verified_speedup, cap) / cap` (`cap = oracle.expected_speedup_range[1]`, so a
  full reference-class fix scores 1.0; over-tripped results excluded); for
  counterexample/do-not-apply tasks `perf_term = 1` iff the agent abstained with a
  defensible record (no regression beyond noise floor), else 0.
- `inconclusive` scores 0.5 × the achieved terms and is counted separately in
  aggregates (never silently as pass or fail).

### 8.3 Track aggregates (SPE-Core, SciML)

- `pass_rate` = fraction of all tasks with all gates passed (denominator = ALL tasks,
  fast_p-style).
- `verified_optimization_rate` = verified positive optimizations / correctness-valid
  positive tasks.
- `geomean_speedup_all_valid` over every correctness-valid positive task with a
  measured positive speedup, including unverified or sub-parity outcomes.
- `semantic_failure_rate` = tasks failing correctness/scientific/anti-cheat gates /
  all tasks.
- `verified_speedup_geomean` remains a secondary diagnostic over verified positives;
  it is never the sole headline because it is vulnerable to survivorship bias.
- `paired_log_speedup_effect` = `log(s_on) - log(s_off)` for paired raw median
  speedups; confidence intervals resample family/lineage, then task, then outer
  trial. Invalid trials are excluded before aggregation.
- `mean_time_to_quality` ratio vs baseline (SciML generation tasks).
- `diagnosis_accuracy`, `mean_cost`, `inconclusive_rate`.
- Secondary composite = `pass_rate × verified_speedup_geomean`, reported with the raw
  tuple and never used as the paper headline.

### 8.4 Evolution-track metrics

Measured per condition over the sequential stream (§10):

- `transfer_gain`: Δ task score on held-out same-family/cross-family tasks vs the
  no-evolution control, computed on paired seeds.
- `rule_reuse_utility`: mean bounded utility delta for applications explicitly
  marked `reused`; missing reuse telemetry is reported as `null`, never zero.
- `negative_transfer_rate`: fraction of rule applications that regress the paired
  control beyond the noise floor.
- `rule_precision`: admitted rules that survive held-out regression ÷ admitted rules.
- `library_growth`: canonical rule count + description-length over episodes
  (rate-distortion view, mirroring `score_rule_library.py`).
- `utility_per_rule` and `utility_per_token`: transfer gain / rules / prompt tokens.
- `conflict_rate`: conflicting canonical pairs ÷ canonical pairs (should be 0 under
  governance; `validate_evidence.py` fails otherwise).
- `drift_recovery_latency`: episodes between a version/runtime-drift break and the
  rule library's return to pre-drift utility.
- `poisoning_survival_rate`: fraction of poisoned/misleading experiences that fail to
  reach canonical status (D) or that cause measurable regressions (C).
- `evolution_regret`: cumulative hindsight utility gap plus weighted experiment
  cost, decomposed into acquisition, negative-transfer, interaction, and
  drift/recovery terms. This longitudinal quantity is reported across the
  family, boundary, interaction, and evolution views; it is not collapsed into
  the track score.
- Evolution results report the raw metric vector; no composite is used as a
  headline until its bounded utility policy and weighting are preregistered.
- Promotion uses the bounded `bounded_log_speedup_v1` policy and a
  Beta--Binomial mixture e-process. A candidate is eligible only when the
  one-sided lower confidence sequence clears `p_min` under optional inspection;
  a fixed-sample posterior interval is not treated as a confidence sequence.

Promotion evidence has three statistical levels: paired repetitions form a
within-context effect interval; one independent context group contributes one
promotion Bernoulli trial only when its effect LCB clears the practical floor;
independent family contexts then form the generalization confidence sequence.
Boundary counterexamples remain in the CEGIS version space but never enter the
promotion case set. Held-out validation is partitioned into replication,
within-predicate transfer, and boundary challenge, with all entries executed by
the verifier or the registered family environment and kept disjoint from
promotion evidence.

Interaction contrasts use one normalized decision scale: the factorial
interaction is divided by four, conditional effects by two, and redundancy by
two, all in ``[-1, 1]``.  Raw contrasts are retained only as reporting fields;
relation gates compare normalized confidence sets against the preregistered
margin.

## 9. Experimental conditions A–D

Four paired conditions over the *same* model, agent framework, tool budget, task
order, seeds, and hardware; only the skill/evolution condition changes:

- **A. no-skill**: agent runs without the skill.
- **B. frozen-skill**: the initial skill snapshot, mounted read-only; experience and
  evolution machinery disabled.
- **C. raw-experience retrieval**: skill + raw `experience/inbox/` capture and
  retrieval under the same token budget as D; no RuleSpec abstraction, replay
  promotion, specialization, or retirement. `C_STRESS` retains the old
  append-only/inject-everything ablation.
- **D. governed self-evolving**: the full pipeline — experience capture, candidate
  cards, paired replay (`run_rule_replay.py`), governance audit
  (`validate_evolution.py`), promotion to canonical, maintenance/retirement
  (`score_rule_library.py`), and usage telemetry (`capture_rule_usage.py`).

Harness support: `harness/conditions.py materialize A|B|C|C_STRESS|D` builds an isolated skill
copy per condition from a pinned snapshot, with the appropriate read-only/writable
bits and injection policy; rule injection into the agent context is performed by the
harness from the condition's store (the core skill has no runtime retrieval interface
— see `README.md#integration-boundary`). All condition copies are hash-attested so a
run can prove which skill bits were visible.

Equal-budget controls: identical max tool calls/tokens/wall time per task; C and D get
identical retrieval/token budgets for evolution activities; skill-text context length
is matched between B/C/D via a filler arm when comparing against A. The primary
benchmark uses `context_mode=reset`; `carry` is an explicit sequential adaptation
control, never silently mixed into reset results.

## 10. Sequential split and leakage control

### 10.1 Stream phases

1. **acquisition** — tasks visible to evolution (experience capture allowed; C/D may
   learn rules here).
2. **same-family transfer** — held-out tasks, same families, *different* mutation
   templates and lineages.
3. **cross-family transfer** — held-out families.
4. **drift** — runtime/version drift (e.g., changed torch version semantics simulated
   via fixture/flag changes, new hardware fingerprint) invalidating specific rules.
5. **misleading/poisoned experience** — injected experiences/rules that are plausible
   but wrong (globally over-broad rules, subtly wrong thresholds, rules that
   hallucinate evidence).
6. **recovery/retirement** — tasks measuring recovery latency and library hygiene.

### 10.2 Group split

Split key = `(family, mechanism, lineage.source, lineage.mutation_template_id,
generator_family_id, oracle_fix_pattern_id, scientific_contract_id,
workspace_ast_skeleton_hash)`. The population validator also rejects explicit
lineage reuse and near-duplicate repair patterns.
`harness/split.py check-leakage` verifies: no split key appears in both the
evolution-visible set (phase 1) and any held-out set (phases 2–6); oracle files are
not readable from any agent sandbox; and the split manifest (`split/sequential.yaml`)
is hash-pinned into each run record. Hidden test tasks are additionally excluded from
any public listing the agent can enumerate.

## 11. Harness architecture and CLI

```
benchmark/
  BENCHMARK_DESIGN.md  TASK_MATRIX.md  README.md
  sources/sources.yaml
  schema/{task.schema.json, result.schema.json}
  harness/
    cli.py             # entry points below
    api.py             # named workspace API contracts (train_loop_v1, sampler_v1, ...)
    runner.py          # sandbox materialization + benchmark.py driving + subprocess isolation
    verifier.py        # S0–S6 pipeline
    stats.py           # median/IQR/MAD, seeded bootstrap CI, noise floor, paired runs
    anticheat.py       # AST/regex scan, harness hashing, canary, tripwires, probes
    scientific_gates.py# shared gate library (energy/force, equivariance, validity, moments)
    fingerprint.py     # hw/sw fingerprint (stdlib + torch; psutil/nvidia-smi optional)
    scoring.py         # per-task + track aggregates (§8.1–8.3)
    evolution.py       # episode runner + evolution metrics (§8.4)
    conditions.py      # A/B/C/D materialization + attestation
    split.py           # sequential split + leakage checker
  taskgen/             # v1.0-20 generator and population validator
  formal/              # dry-run/agent campaign driver and lineage-aware aggregation
  manifests/           # frozen v1.0-50 slot quotas (no sealed task contents)
  tasks/<task_id>/...
  tests/               # standalone assert-scripts (repo convention; no pytest dependency)
```

The formal outer driver lives in `benchmark/formal/`. Its default plan is
`A/B/C/D × reset × 3` independent outer trials over all 20 pilot tasks. A
dry-run writes only `campaign.json` and `schedule.json` with
`results_claimed=false`; an agent command is required before trial result
files are produced. Each trial manifest records the benchmark revision,
allowlisted skill-view and task-manifest digests, model/configuration,
condition, `context_mode`, task order, outer-trial ID, budgets, and hardware /
software fingerprints. Inner baseline/candidate repetitions remain verifier
measurements and are not substituted for independent outer trials.
After verification, the driver performs an explicit post-task transition:
evidence capture, condition-specific maintenance, policy validation, and store
attestation. C can append only raw experience; D alone can invoke replay and
governance. Agent-backed trials must emit `agent_usage.json` with input/output
tokens, tool calls, and wall time; missing or over-budget receipts are invalid
and excluded from headline aggregates.

CLI:

```
python -m benchmark.harness.cli validate-task tasks/<id>          # schema + self-consistency:
                                                                  # baseline fails speedup gate,
                                                                  # oracle patch passes all gates,
                                                                  # fixtures deterministic
python -m benchmark.harness.cli run-task tasks/<id> --solution DIR --out result.json
python -m benchmark.harness.cli materialize-condition {A,B,C,D} --snapshot DIR --out DIR
python -m benchmark.harness.cli run-episode episodes/<id>.yaml --condition {C,D} --out DIR
python -m benchmark.harness.cli check-leakage split/sequential.yaml
python -m benchmark.harness.cli score-run RUN_DIR --out scores.json
```

Dependencies: Python ≥3.10 stdlib + torch. PyG optional (SciML graph tasks ship a
fallback scatter implementation so CPU-CI can run them). YAML parsing: a tiny
built-in subset parser (no PyYAML dependency) — `task.yaml` files are restricted to
the supported subset, enforced by `validate-task`.

## 12. Pilot task set

The pilot contains 20 packages (18 atomic tasks plus 2 evolution episodes),
runnable with zero external downloads; CPU-capable unless noted:

| # | task_id | track | family | mechanism | kind |
|---|---|---|---|---|---|
| 1 | CORE-SCALAR-SYNC-01 | spe_core | training_loop_overhead | scalar_sync | positive |
| 2 | CORE-REPEATED-BACKBONE-02 | spe_core | repeated_compute | repeated_compute | positive (+ semantic-trap counterexample variant in-task) |
| 3 | CORE-H2D-PIPELINE-03 | spe_core | data_pipeline | h2d_blocking | positive; requires_cuda (degrades to inconclusive on CPU) |
| 4 | CORE-COMPILE-RECOMPILE-04 | spe_core | compiler | compile_graph_break | counterexample; cold calibration found no positive CI through 1024 steps |
| 5 | SCIML-GNN-RAGGED-05 | sciml | graph_energy_force | ragged_loops + autograd_overhead | positive |
| 6 | SCIML-EQUIV-RECOMPUTE-06 | sciml | equivariant_head | repeated_compute | counterexample (caching-equivariant-basis vs changing positions) |
| 7 | SCIML-CRYSTAL-DIFFUSION-07 | sciml | crystal_generation | scalar_sync + launch_fragmentation | positive, time-to-quality |
| 8 | SCIML-GRAPH-REBUILD-08 | sciml | crystal_sampling | graph_rebuild | do_not_apply (positions change every step — caching is wrong) |
| 9 | CORE-KERNEL-FUSION-09 | spe_core | compiler | launch_fragmentation | positive (KernelBench-style verified fusion; CPU-capable) |
| 10 | EVOL-EPISODE-POISON-10 | evolution | episode | (all above) | 6-phase episode over held-out variants + poisoned experience |
| 11 | CORE-COMPILE-DYNAMIC-11 | spe_core | compiler | compile_dynamic_shapes | positive |
| 12 | CORE-COMPILE-TINY-12 | spe_core | compiler | compile_tiny_graphs | counterexample |
| 13 | CORE-MEM-RETAINED-GRAPH-13 | spe_core | memory | retained_graph | positive |
| 14 | CORE-CHECKPOINT-AMPLE-MEM-14 | spe_core | memory | checkpoint_ample_memory | counterexample |
| 15 | CORE-AUTOGRAD-BATCHED-VJP-15 | spe_core | autograd | batched_vjp | positive |
| 16 | CORE-DATALOADER-FANOUT-16 | spe_core | data_pipeline | dataloader_worker_fanout | positive |
| 17 | SCIML-GNN-STATIC-GRAPH-CACHE-17 | sciml | graph_energy_force | static_graph_cache | positive |
| 18 | SCIML-GNN-DYNAMIC-GRAPH-18 | sciml | graph_energy_force | dynamic_graph_rebuild | counterexample |
| 19 | SCIML-FORCE-AUTOGRAD-19 | sciml | graph_energy_force | force_autograd | positive |
| 20 | EVOL-COMPILER-DRIFT-20 | evolution | episode | compile_dynamic_shapes + runtime_drift | positive |

Each atomic task ships with baseline/oracle validation metadata, fresh-input
verification, an anti-cheat probe, a deterministic fixture, a declared noise
floor, and lineage identifiers. Each task ships with an oracle patch that
`validate-task` proves passes all gates, and a baseline that `validate-task` proves does *not* meet the verified-speedup bar
(positive tasks) or *would* regress if the tempting rule were applied (counterexample
tasks, proven via an oracle "tempting-patch" that the verifier rejects).

Pilot calibration is a release gate, not an automatic promotion: measured
oracle confidence intervals, control noise, semantic gates, platform effects,
and anti-cheat findings are recorded in `population_report.json`. Tasks that
fail the calibration gate are retired or rewritten before any v1.0-50 task is
generated. The frozen v1.0-50 slot manifest records only track quotas and
public/sealed allocation; it intentionally contains no sealed task content.

Formal result claims additionally require a digest-attested
`calibration_approval.json` beside the population report. The approval binds
the population artifact, empirical calibration artifact, and review policy;
editing the report alone cannot satisfy the claim gate.

## 13. Reproducibility and environment

- Python ≥ 3.10, torch ≥ 2.7 (CI floor 2.7.1 CPU); PyG optional with fallback.
- Deterministic fixtures: every generator takes an explicit seed; fixture hashes are
  recorded in results.
- External cache: large third-party artifacts never enter this repository; they live
  under `$SPE_BENCH_CACHE` (for example, `./.cache/spe-evobench` locally). Download tooling is
  dry-run by default; anything >2 GB requires an explicit `--i-understand-large-download`
  flag and is represented in prototypes by manifests/instructions only.
- Formal-evaluation hardening: the worker is run through an externally supplied namespace
  executor with a no-network receipt, read-only task/skill mounts, an observed isolation
  canary, and harness-owned usage accounting. Without that receipt, required experiments
  and formal promotion remain blocked.

## 14. External sources

All upstream references, pinned commits, licenses, sizes, subsets, and cache
locations are recorded in `sources/sources.yaml`. Design lessons may be kept in an
author-local notes directory outside this repository. Headline borrowings: KernelBench's correctness-gated fast_p and
adversarial probes; TorchBench's fp64 live-recomputed reference and subprocess
isolation; MLPerf-HPC's quality-gated time-to-quality and repetition rules;
SkillsBench's paired arms and leakage gate; ContinualSkillBench's sequential
independent-vs-sequential pairing; PERFOPT-Bench's bottleneck-injection construction
and anti-cheat contract; FAIRChem's fake-dataset blueprint for tiny scientific
fixtures; CDVAE's hard validity gates and recall+precision pairing.

## 15. Limitations

- Pilot scale (20 tasks) calibrates population validity, difficulty, and noise
  floors; it is not a formal v1.0-50 result.
- DDP/contention mechanisms are represented by simulated single-process telemetry in
  the prototype; multi-process tasks arrive in the expansion matrix.
- Local subprocess execution is not a formal security boundary; formal runs require
  the attested external namespace executor described in §6.1.
- The formal worker receives only a public task view (contract, workspace, and
  public tests) from an isolated working directory. Oracle, hidden-verifier,
  and repository-root paths are not passed to the worker. Its condition-store
  view is read-only; the harness appends evidence after verification. Synthetic
  Boundary/Interaction calibration remains diagnostic, while executable-workload
  calibration uses the empirical boundary adapter and may be inconclusive when
  its paired interval is inside the measured noise floor.
- Speedups are hardware-relative; cross-hardware claims require re-measured noise
  floors (fingerprints gate comparability).
