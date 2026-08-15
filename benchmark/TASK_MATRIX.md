# SPE-EvoBench Task Matrix

This matrix documents the **10 runnable prototype tasks** and the planned
the frozen SPE-EvoBench v1.0-50 target. All prototypes run with zero external downloads;
planned tasks keep the same data discipline (synthetic fixtures, bundled small
data, or deterministic generators).

## Contents

- [Legend](#legend)
- [Prototype tasks (10)](#prototype-tasks-10--implemented-and-verified)
- [Verification meaning](#verification-meaning)
- [Expansion plan](#expansion-plan)
- [Task-family coverage](#task-family-coverage)

## Legend

- **Track**: `spe_core` (PyTorch performance engineering), `sciml`
  (scientific-ML semantic constraints), `evolution` (sequential skill stream).
- **Kind**: `positive` (optimization is correct and beneficial),
  `counterexample` (tempting optimization is wrong / slower / invalid),
  `do_not_apply` (correct action is to abstain and justify).
- **Source**: `synthetic` (seeded torch-only fixture),
  `fairchem-shaped` (tensor/loop structure derived from FAIRChem/OC20),
  `cdvae-shaped` (structure derived from CDVAE/MP-20),
  `kernelbench-shaped` (fused-op structure inspired by KernelBench).

## Prototype tasks (10) — implemented and verified

| # | Task ID | Track | Family | Mechanism(s) | Kind | Source | Verified oracle speedup | Status |
|---|---------|-------|--------|--------------|------|--------|------------------------|--------|
| 1 | CORE-SCALAR-SYNC-01 | spe_core | training_loop_overhead | scalar_sync | positive | synthetic | **1.51×** | ✅ |
| 2 | CORE-REPEATED-BACKBONE-02 | spe_core | repeated_compute | repeated_compute | positive | synthetic | **2.38×** | ✅ |
| 3 | CORE-H2D-PIPELINE-03 | spe_core | data_pipeline | h2d_blocking | positive | synthetic | **4.77×** (CUDA) | ✅ |
| 4 | CORE-COMPILE-RECOMPILE-04 | spe_core | compiler | compile_break | positive | synthetic | **1.16×** | ✅ |
| 5 | SCIML-GNN-RAGGED-05 | sciml | graph_energy_force | ragged_loops + autograd_overhead | positive | fairchem-shaped | **6.23×** | ✅ |
| 6 | SCIML-EQUIV-RECOMPUTE-06 | sciml | equivariant_head | repeated_compute | counterexample | fairchem-shaped | abstain (≈1.0×); tempting fails equivariance gate | ✅ |
| 7 | SCIML-CRYSTAL-DIFFUSION-07 | sciml | crystal_generation | scalar_sync + launch_fragmentation | positive | cdvae-shaped | **21.3×** time-to-quality | ✅ |
| 8 | SCIML-GRAPH-REBUILD-08 | sciml | crystal_sampling | graph_rebuild | do_not_apply | fairchem-shaped | abstain (≈1.0×); tempting fails scientific gate | ✅ |
| 9 | CORE-KERNEL-FUSION-09 | spe_core | compiler | launch_fragmentation | positive | kernelbench-shaped | **3.42×** | ✅ |
| 10 | EVOL-EPISODE-POISON-10 | evolution | episode | all above | positive | synthetic | **2.0×** episode score (C vs D) | ✅ |

Verification meaning: “✅” = `validate-task` passes, baseline is
`inconclusive` (no verified speedup), oracle/reference is `pass` with a real
measured speedup, and counterexample/do_not_apply tempting patches are rejected.

## Sequential split (prototype)

See `benchmark/split/sequential.yaml`.

| Phase | Name | Tasks | Purpose |
|-------|------|-------|---------|
| 1 | acquisition | 01, 03 | Skill learns scalar-sync and H2D rules from visible examples. |
| 2 | same-family transfer | 02, 04 | Held-out tasks in related families (repeated compute, compiler). |
| 3 | cross-family transfer | 07, 09 | Material generation and kernel tasks test generalization. |
| 4 | drift | *(experience only)* | Injected experience: H2D async rule specialized to CPU-only regime. |
| 5 | poisoned experience | 08 | Misleading “cache neighbor lists” + “sync every step” rules; must be rejected. |
| 6 | recovery | 05, 06 | SciML tasks measure recovery after poison; 06 is a counterexample anchor. |

Leakage rule: no `(family, mechanism, source, mutation_template_id)` key
appears in both phase 1 and any later phase. `check-leakage` enforces this.

## Expansion plan to the frozen 50-task release

The expansion keeps the same task-package structure and harness, varying only
fixtures, model shapes, and mutation polarity.

### SPE-Core (target 24–32 tasks)

| Family | Mechanisms | Planned tasks | Variation axes |
|--------|-----------|---------------|----------------|
| training_loop_overhead | scalar_sync | 4 | `.item()` on loss/grads/metrics; logging cadence; mixed-precision sync; CPU vs CUDA prominence. |
| repeated_compute | repeated_compute | 3 | Shared backbone, cached embeddings, repeated feature extraction; changing-input traps. |
| data_pipeline | h2d_blocking | 3 | pin_memory/non_blocking, worker fan-out, prefetch, CPU-only inconclusive path. |
| compiler | compile_break, launch_fragmentation | 5 | graph breaks, dynamic shapes, recompilation, pointwise fusion, custom op fallback. |
| memory_pressure | checkpoint_cadence, activation_memory | 3 | checkpoint granularity, retained graphs, OOM-edge cases. |
| distributed | ddp_sync, process_contention | 3 | all_reduce timing, gradient bucketing, host thread contention (simulated or DDP when available). |
| autograd | autograd_overhead, unbatched_vjp | 3 | repeated backward, per-sample VJP, higher-order overhead. |

### SciML (target 20–30 tasks)

| Family | Mechanisms | Planned tasks | Source lineage |
|--------|-----------|---------------|----------------|
| graph_energy_force | ragged_loops, graph_rebuild, autograd_overhead | 6 | fairchem-shaped synthetic crystals; radius-graph PBC; energy/force consistency gates. |
| equivariant_head | repeated_compute, launch_fragmentation | 4 | rank-2/rank-3 tensor prediction; equivariance gates; SO(3) rotation probes. |
| crystal_generation | scalar_sync, launch_fragmentation | 5 | cdvae-shaped diffusion/sampler; validity + distribution-moment gates; time-to-quality. |
| crystal_sampling | graph_rebuild, scalar_sync | 5 | Langevin/MCMC samplers; do-not-apply traps; poisoned-rule anchors. |

### Evolution episodes (target 4–6 episodes)

| Episode | New mechanism emphasis | # phases | New stresses |
|---------|------------------------|----------|--------------|
| Compiler-drift | compile_break + torch version drift | 6 | Old `torch.compile` flags become invalid after version bump. |
| Multi-property GNN | rule specialization by property head | 6 | A rule valid for energy fails for forces; specialization required. |
| Curriculum scaling | repeated_compute → distributed | 6 | Rules must retire as workload scale changes. |
| Adversarial poisoning | misleading experience + conflicting rules | 6 | Two plausible rules conflict; governance must reject one. |

## Data and dependency discipline

- **No dataset downloads**: all tasks use seeded synthetic tensors, tiny bundled
  structures, or deterministic generators.
- **Optional PyG**: SciML graph tasks implement scatter with plain
  `torch.index_add` / `torch.scatter` so they run without PyG.
- **External cache**: any future real-structure extracts >2 MB go to
  `$SPE_BENCH_CACHE`, never into the repository.
- **License-safe**: no OC20/OC22/MP-20 data is bundled; only structural
  *shapes* from the upstream code are mirrored.

## Implementation priority for expansion

1. Harden the harness on the 10 prototypes (noise floors, CI stability,
   anti-cheat tripwires).
2. Add SPE-Core compiler and memory-pressure tasks next — they are the cheapest
   to author and validate.
3. Add 4–6 more SciML graph/crystal tasks with varied validity gates.
4. Author the second evolution episode (compiler drift) once the task pool
   supports all 6 phases.
5. Run a small population-validity pilot (20 tasks) to calibrate difficulty
   before scaling to the frozen 50-task release.
