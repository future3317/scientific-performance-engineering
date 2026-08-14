---
name: scientific-gpu-training-optimizer
description: Use when a PyTorch scientific training or inference decision is primarily about end-to-end latency, throughput, memory, host/GPU utilization, launch gaps, multi-task/autograd overhead, PyG/e3nn kernels, diffusion sampling, compilation, or distributed scaling. Do not use for CUDA/runtime correctness bugs, installation/build failures, generic distributed correctness, or numerical instability unless a performance decision is the primary task.
---

# Scientific GPU Training Optimizer

Act as a performance decision workflow, not a list of GPU tricks. Preserve the scientific contract and accept a candidate only when the measurement record and validator prove comparability, active-path use, statistics, timing completion, and correctness gates.

## Contents

- [Route and select mode](#route-and-select-mode)
- [Policy and evidence state machine](#policy-and-evidence-state-machine)
- [Required records and tools](#required-records-and-tools)
- [Claim boundaries](#claim-boundaries)
- [Detailed references](#detailed-references)

## Route and select mode

- **Review/diagnose:** explain evidence; do not edit without an explicit fix request.
- **Optimize/benchmark:** use the state machine below and the comparator as the裁判.
- **Architecture/algorithm experiment:** require explicit authorization; label it `algorithmic_experiment`, separate from systems optimization.
- **Static-only:** when representative runtime/data is unavailable, report hypotheses and runnable commands, never a measured speedup.

Do not use this skill for the negative-routing cases in the description unless performance is the primary decision.

Do not use for CUDA/runtime correctness bugs, installation/build failures, generic distributed correctness, or numerical instability unless a performance decision is the primary task.

Load only the relevant reference: `MEASUREMENT_CONTRACT.md` first; then `PERFORMANCE_PLAYBOOK.md`, `PATCH_PATTERNS.md`, `TECHNOLOGY_MATRIX.md`, `GNN_PREDICTION_WORKLOADS.md`, `CRYSTAL_GENERATION.md`, `EQUIVARIANT_OPERATOR_DESIGN.md`, or `REPOSITORY_NOTES.md` as routed by the workload.

## Policy and evidence state machine

Use five explicit phases: **Explore → Plan → Execute → Integrate → Review**.

1. **Explore:** inspect the real hot path and collect environment/host state before editing.
2. **Plan:** write a hypothesis card with measured bottleneck share, one independently attributable intervention (or a declared coupled bundle plus ablation plan), expected movement, semantic risk, falsification test, and Amdahl ceiling.
3. **Execute:** compare at three levels—Micro → Module → End-to-end—on the same representative work. Keep the eager/reference output and one known-good path.
4. **Integrate:** retain only independently evidenced candidates; keep an experiment ledger with rejected and inconclusive entries.
5. **Review:** block reachable correctness, deadlock, OOM, fallback, reproducibility, API, or scientific-quality risks according to severity. Unsupported hypothetical cases are non-blocking.

Freeze model/data/sampler/objective/effective batch/seed/world size, hardware/software, benchmark harness, timing boundaries, and acceptance policy. A code change is represented by `base_revision + benchmark_harness_hash + candidate_patch_hash + declared_change_set`; do not bypass immutable scientific fields with `--allow-difference`.

For each step, run a timing bucket audit (`batch_fetch`, graph build, H2D, model/heads, physical features, `autograd.grad`, loss, backward, optimizer, DDP) and place `data_ready` after actual readiness. A CUDA bucket needs a paired event or synchronized completion proof; NVTX/`record_function` is attribution only. Build a task census and report unaccounted time, logical update work, host contention, and raw repeated windows.

Before accepting a result, run correctness gates in order: shape/finite values → numerical output/loss → forward and gradient agreement → invariance/equivariance/physical constraints → task/validation quality → distributed and resume equivalence. The comparator requires raw runs, median/IQR/MAD, bootstrap confidence, quality gates, active-path evidence, and timing accounting; missing evidence is `inconclusive`, not success.

Stochastic thinning must skip the entire unselected forward/gradient path and define global versus rank-local masks, loss normalization, used-parameter/DDP behavior, zero-selected windows, and clipping. Checkpointing must state `use_reentrant`, `preserve_rng_state`, higher-order/autograd support, accumulation boundary, and restored RNG/optimizer/sampler state.

Change coupled kernel/layout/compile/dtype pieces only when they are one independently attributable intervention; declare the bundle and ablate it when the result matters.

## Required records and tools

- Copy `assets/benchmark_record.json` and `assets/performance_report.md` for durable work.
- Run `scripts/collect_env.py` (privacy-safe by default; use `--include-sensitive-host-metadata` only when needed).
- Run `scripts/run_with_gpu_monitor.py --gpu <index-or-UUID> ...`; it records GPU mapping and trainer/worker/process-tree memory, not just monitor RSS.
- Run `scripts/compare_benchmarks.py`; `gates_passed` is reserved for a complete accepted record. `inconclusive`, `algorithmic_experiment`, `incomparable`, and `gates_failed` are non-accepting states.
- Run `scripts/validate_skill.py`, `scripts/behavioral_contract_tests.py`, and the material/GNN self-test before publishing.

## Claim boundaries

- Static review and microbenchmarks support hypotheses only; GPU utilization alone proves nothing.
- Do not call a candidate faster when code/data/work/timing/host state differs outside its declared change set.
- A configured compiler/kernel/precision feature is not active until runtime evidence proves it and proves no silent fallback.
- Keep the result `accepted`, `rejected`, `inconclusive`, `algorithmic_experiment`, or `static-only`; include commands, records, evidence, gates, rejected hypotheses, rollback, and limitations.

## Detailed references

- [Measurement contract](references/MEASUREMENT_CONTRACT.md)
- [Performance playbook](references/PERFORMANCE_PLAYBOOK.md)
- [Patch patterns](references/PATCH_PATTERNS.md)
- [GNN prediction workloads](references/GNN_PREDICTION_WORKLOADS.md)
- [Crystal generation](references/CRYSTAL_GENERATION.md)
- [Equivariant operator design](references/EQUIVARIANT_OPERATOR_DESIGN.md)
