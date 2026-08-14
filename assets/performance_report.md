# GPU/CPU training performance report

## Contents

- [Decision](#decision)
- [Reproducible baseline](#reproducible-baseline)
- [Timing bucket audit](#timing-bucket-audit)
- [Acceptance contract](#acceptance-contract)
- [Correctness and scientific gates](#correctness-and-scientific-gates)
- [Reproduction](#reproduction)

## Decision

- Status: accepted / rejected / inconclusive / algorithmic_experiment
- Evidence level: static-only / microbenchmark / end-to-end / scientific gate
- Primary bottleneck:
- Material result:
- Active fast path and fallback status:
- Optimization objective:

## Reproducible baseline

- Commit and dirty state:
- Base revision, benchmark harness hash, candidate patch hash, and declared change set:
- Hardware, CPU affinity/NUMA, and storage:
- Driver/CUDA or ROCm/PyTorch/kernel-library versions:
- Command/config and relevant environment variables:
- Host load/available memory/swap/worker RSS during the run:
- GPU selection and logical-to-physical UUID mapping:
- Environment privacy mode and whether sensitive host metadata was enabled:
- Data panel, seed, sampler/order, and batch composition:
- Scientific contract ID and intentionally allowed differences:
- Work unit and timing boundaries:
- CUDA timing proof (paired events or synchronized boundary, stream):
- Per-bucket clock/stream/completion proof, bucket sum, end-to-end step, and unaccounted ratio:
- Raw run/window metrics, randomized run order, median/IQR/MAD, bootstrap CI, and noise floor:
- Warmup, compile warmup, measured steps, repetitions:
- Benchmark levels: micro / module / end-to-end
- Logical update and accumulation definition:
- Task composition and work counts per logical update:

## Evidence

- Profiler trace(s):
- Nsight Systems/Compute report(s), if used:
- Dominant CPU/CUDA/communication events:
- Data wait, H2D, synchronization, graph-break, recompilation, or fallback evidence:

## Timing bucket audit

`data_ready` must follow actual batch readiness. Record the measured interval and code/NVTX range for each bucket; do not infer semantics from a label.

| Bucket | Range/definition | p50 | p95 | Notes |
|---|---|---:|---:|---|
| Batch fetch/CPU graph build | | | | |
| H2D | | | | |
| Structural/model | | | | |
| Mechanism | | | | |
| Property heads | | | | |
| Physical features / autograd.grad | | | | |
| Loss | | | | |
| Backward/optimizer | | | | |
| DDP communication | | | | |
| Unaccounted step time | | | | |

Task census: structures/graphs/atoms/edges per rank and globally; mechanism/property/structural/physical task counts; Python loop iterations; auxiliary forward and `autograd.grad` calls; skipped-task calls; shared-backbone reuse.

## Change

- Files and code path:
- Hypothesis:
- One changed lever:
- Amdahl ceiling for the targeted region:
- Active capability/kernel and version:
- Scientific risk and mitigation:
- Rollback/disable setting:
- Explicitly authorized algorithmic differences, if any:
- Stochastic-thinning policy, selection probability, loss reweighting, and seed/rank semantics (if applicable):
- DDP thinning contract: global/rank-local mask, used-parameter set, `find_unused_parameters`/`static_graph`, zero-selected windows, normalization, clip fraction:
- Activation-checkpoint contract: `use_reentrant`, `preserve_rng_state`, `autograd.grad`/higher-order support, `no_sync()` support, resume boundary and restored state:

## Acceptance contract

- Primary objective metric and threshold:
- Required quality gates from `acceptance.required_quality_gates`:
- Guardrails (p95 latency / memory / host contention):
- Quality/non-inferiority gate:

## Before/after

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Work-normalized throughput | | | |
| Secondary work throughput | | | |
| Step p50 | | | |
| Step p95 | | | |
| Peak allocated memory | | | |
| Peak reserved/external memory | | | |
| Host RSS | | | |
| Data wait | | | |
| H2D | | | |
| Forward/loss | | | |
| Backward | | | |
| Optimizer | | | |
| Communication/overlap | | | |
| Scaling efficiency | | | |

## Compiler and kernel evidence

- Cold compile/autotune time versus steady state:
- Graph breaks/recompiles/dynamic guards:
- Compiled-region coverage and fallback:
- Kernel/backend/layout/dtype used:
- GPU busy, SM/memory, or occupancy evidence (supporting evidence only):

## Correctness and scientific gates

Run these gates in order; do not treat a speedup that fails an earlier gate as an optimization.

| Check | Result | Tolerance/gate |
|---|---|---|
| Unit/regression tests | | |
| Numerical output comparison | | |
| Loss/component comparison | | |
| Gradient norm/cosine | | |
| Finite values/overflow | | |
| Invariance/equivariance/physical constraints | | |
| Validation/sampling/rollout quality | | |
| Resume/data-order equivalence | | |
| Distributed/checkpoint equivalence | | |

For numerical-sensitive kernels or dtype changes, record stress cases covering the reachable small/large, high-coordination, short-distance or near-singular, and small/large-value regimes.

## Rejected hypotheses and caveats

-

## Reproduction

- Environment record: `environment.json`
- Benchmark record: `benchmark_record.json`
- Exact baseline command:
- Exact candidate command:
- Benchmark comparison command:
