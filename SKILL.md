---
name: scientific-gpu-training-optimizer
description: Use when a PyTorch scientific training or inference decision is primarily about end-to-end latency, throughput, memory, host/GPU utilization, launch gaps, multi-task/autograd overhead, PyG/e3nn kernels, diffusion sampling, compilation, or distributed scaling. Do not use for CUDA/runtime correctness bugs, installation/build failures, generic distributed correctness, or numerical instability unless a performance decision is the primary task.
---

# Scientific GPU Training Optimizer

Act as an end-to-end scientific training systems workflow, not a list of GPU tricks. Preserve the scientific contract and accept a candidate only when the record and validators prove compatibility, lifecycle coverage, comparability, active-path use, statistics, timing completion, and correctness.

## Route and select mode

- **Review/diagnose:** explain evidence; do not edit without an explicit fix request.
- **Optimize/benchmark:** use the lifecycle below and `compare_benchmarks.py` as the judge.
- **Architecture/algorithm experiment:** require explicit authorization; label it `algorithmic_experiment`, separate from systems optimization.
- **Static-only:** when representative runtime/data is unavailable, report hypotheses and runnable commands, never a measured speedup.
- **Long-run gate:** do not launch the full campaign horizon until precompute cost, campaign resource topology, and a short staged horizon have passed the contract, quality, and time-to-quality gates.
- **Evidence scope:** declare `comparison_class` (`systems`, `scaling`, or `algorithmic`) and `evidence_level` (`static`, `micro`, `module`, `logical_update`, `amortized_job`, or `time_to_quality`). Only the fields applicable to that scope are gates; a micro/module result is not required to pretend it is a full campaign.

Do not use this skill for CUDA/runtime correctness bugs, installation/build failures, generic distributed correctness, or numerical instability unless performance is the primary decision.

Load only the relevant reference: `MEASUREMENT_CONTRACT.md` first; then route to `CODE_AND_RUNTIME_AUDIT.md`, `DATA_AND_TRAINING_LIFECYCLE.md`, `MEMORY_COMPILER_DISTRIBUTED.md`, `PERFORMANCE_PLAYBOOK.md`, `PATCH_PATTERNS.md`, `TECHNOLOGY_MATRIX.md`, `GNN_PREDICTION_WORKLOADS.md`, `CRYSTAL_GENERATION.md`, `EQUIVARIANT_OPERATOR_DESIGN.md`, or `REPOSITORY_NOTES.md`.

## Policy and evidence lifecycle

Use this sequence:

`Preflight -> Contract Freeze -> Lifecycle Census -> Baseline/Noise -> Profile/Classify -> Hypothesis/Amdahl -> Minimal Intervention -> Activation Proof -> Micro/Module/Logical Update/Amortized Job -> Scientific/Resume/Distributed/Failure Gates -> Statistical Gate`

1. **Preflight:** record model construction order, rank/device mapping, topology, CPU/NUMA/thread state, allocator, and the compatibility matrix for compile, checkpointing, DDP/FSDP, CUDA Graphs, custom ops, dynamic shapes, and higher-order autograd. Reject known-incompatible combinations before a long run.
2. **Contract Freeze:** freeze model/data/sampler/objective/effective batch/seed/world size, initialization checkpoint, anchor provenance and scope, constraint stack, auxiliary cadence/masks, cache keys, timing boundaries, and acceptance policy. A code change is `base_revision + benchmark_harness_hash + candidate_patch_hash + declared_change_set`; `--allow-difference` cannot bypass scientific fields.
3. **Lifecycle Census:** enumerate precompute/cache construction, campaign orchestration, the logical-update DAG, and the synchronization census before focusing on a kernel. Include fetch, CPU/GPU preprocessing, H2D, forward/heads, loss, auxiliary derivatives, backward, gradient transforms, clipping, communication, optimizer, scheduler, EMA/SWA, metrics, checkpoint, and validation; record the full process/worker/thread topology for concurrent seeds or endpoints. In the benchmark record, explicitly classify `startup`, `precompute`, `logical_update`, `evaluation_sampling`, `checkpoint_resume`, `teardown`, and `failure_retry`; an included stage needs measured time and evidence, while an excluded stage needs an explicit reason.
4. **Baseline/Noise:** collect cold/warm cache state, repeated randomized windows, host/GPU state, and the noise floor.
5. **Profile/Classify:** reconcile named timing buckets and classify CPU, data, synchronization, memory, compiler, distributed, or scientific bottlenecks. Record kernel-launch density and scalar-sync evidence for fragmented traces; place `data_ready` after actual readiness.
6. **Hypothesis/Amdahl:** write one attributable intervention (or declared coupled bundle with ablation), expected movement, semantic risk, falsification test, and Amdahl ceiling. Mark changes to loss weights, anchors, gradient projection/caps, task cadence, supervision membership, solver/NFE, or sampling law as `algorithmic_experiment` when they change gradient statistics or the scientific objective.
7. **Minimal Intervention:** preserve an eager/reference path and change only the declared lever.
8. **Activation Proof:** prove the candidate dispatch, compile/cache state, active path, and absence of silent fallback.
9. **Layered Measure:** compare Micro -> Module -> Logical Update -> Amortized Job on identical scientific work. Report steady-state train-step and cadence-amortized throughput/time-to-quality.
10. **Gates:** run numerical/gradient/physics/quality, resume, distributed, OOM, and failure diagnostics before statistics. For conditional or multitask work, include supervision coverage, fixed-condition/composition controls, route/realization probability checks, task-level gradient ownership, optimizer-aware thinning controls, and non-inferiority of auxiliary objectives.
11. **Statistical Gate:** require raw runs, median/IQR/MAD, bootstrap confidence, noise floor, and complete timing accounting.

The record is the executable contract. Compare frozen contract fields exactly; declare intervention fields such as cache/H2D/synchronization/compiler changes; treat measured evidence such as sync counts, cache hits, overlap, launch count, memory, and latency as expected-to-change observations. Missing evidence required by the declared scope is `inconclusive`, never an accepted speedup. Host contention is a materiality confounder, not a byte-for-byte identity field. A training-contract mismatch or an unbudgeted CPU/process fan-out outranks a performance gain; stop at the shortest useful horizon and run a matched ablation before spending the full campaign budget. Unsupported hypothetical cases remain non-blocking.

## Experience boundary

Treat self-evolution as a separate, auditable maintenance workflow. During a task, the practitioner may record a reusable surprise, falsified hypothesis, negative result, rule boundary, or hidden synchronization in `experience/inbox/` using `assets/experience_record.json`; an experience is evidence, not a rule. Use `scripts/capture_experience.py` to validate and store it without overwriting an existing case. Do not let runtime work edit `SKILL.md`, canonical references, rule status, or acceptance semantics. A maintainer may consolidate cases into a candidate rule card, but `scripts/validate_evolution.py` must show replay passed, regression cases, and human review before a candidate becomes canonical. Read [EXPERIENCE_EVOLUTION.md](references/EXPERIENCE_EVOLUTION.md) when capturing or maintaining experience.

## Required records and tools

- Copy `assets/benchmark_record.json` and `assets/performance_report.md` for durable work.
- Run `scripts/collect_env.py` (privacy-safe by default; use `--include-sensitive-host-metadata` only when needed).
- Run `scripts/run_with_gpu_monitor.py --gpu <index-or-UUID> ...`; it records GPU mapping and trainer/worker/process-tree memory, not just monitor RSS.
- Run `scripts/validate_benchmark.py` for the lifecycle/schema contract.
- Run `scripts/validate_experience.py` and `scripts/validate_evolution.py` for the experience and rule-promotion contracts.
- Run `scripts/compare_benchmarks.py`; inspect its separate `comparison_class`, `evidence_level`, `comparability`, and `decision` fields. `assessment` is retained as a compatibility/debug classification; the decision is `accepted`, `rejected`, or `inconclusive`.
- Run `scripts/validate_skill.py`, `scripts/behavioral_contract_tests.py`, and the material/GNN self-test before publishing.

## Claim boundaries

- Static review and microbenchmarks support hypotheses only; GPU utilization alone proves nothing.
- Do not call a candidate faster when code/data/work/timing/host state differs outside its declared change set.
- A configured compiler/kernel/precision feature is not active until runtime evidence proves it and proves no silent fallback.
- `opcheck`/FakeTensor prove operator contracts, not mathematics; use output/gradient/physics gates separately.
- Keep comparison class, evidence level, comparability, and decision separate; include commands, records, evidence, gates, rejected hypotheses, rollback, and limitations.

## Detailed references

- [Measurement contract](references/MEASUREMENT_CONTRACT.md)
- [Experience-driven evolution](references/EXPERIENCE_EVOLUTION.md)
- [Code and runtime audit](references/CODE_AND_RUNTIME_AUDIT.md)
- [Data and training lifecycle](references/DATA_AND_TRAINING_LIFECYCLE.md)
- [Memory, compiler, and distributed routes](references/MEMORY_COMPILER_DISTRIBUTED.md)
- [Performance playbook](references/PERFORMANCE_PLAYBOOK.md)
- [Patch patterns](references/PATCH_PATTERNS.md)
- [GNN prediction workloads](references/GNN_PREDICTION_WORKLOADS.md)
- [Crystal generation](references/CRYSTAL_GENERATION.md)
- [Equivariant operator design](references/EQUIVARIANT_OPERATOR_DESIGN.md)
