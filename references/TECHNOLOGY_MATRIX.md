# Technology and capability matrix

Last reviewed: 2026-08-04. This is a selection guide, not a minimum dependency list. The installed runtime, GPU architecture, CPU topology, workload shape, and scientific contract decide whether a feature is eligible.

## Rules for using the matrix

- Probe availability at runtime and record the result. Use `hasattr`, import checks, library version checks, and a small path assertion; never infer activation from a config flag.
- Keep an eager, full-precision or project-default baseline. Every accelerated path is a candidate until its steady-state benefit and scientific gates pass.
- Treat release snapshots as context only. APIs and defaults can change; consult the installed version's documentation before emitting code.
- Do not add a dependency solely for a benchmark. Prefer functionality already in the project's lockfile or environment.
- Distinguish `installed`, `importable`, `training-dispatchable`, and `inference-dispatchable`. A third-party acceleration package is not evidence that every mode is supported.
- Record API generation, capability probe, supported operator signature, layout/dtype, and backward support for fast-changing kernel libraries.

## Snapshot and recommended order

| Layer | 2026 snapshot or option | Use when | Required guardrail |
|---|---|---|---|
| Baseline | PyTorch eager, standard ATen ops | Always | Same work, environment, seed, and measurement |
| CPU | oneDNN/OpenMP/MKL/BLAS plus explicit Torch thread counts | CPU transforms, collation, preprocessing, or CPU inference dominate | Thread/NUMA affinity and oversubscription sweep |
| Data | DataLoader, PyG PrefetchLoader, DynamicBatchSampler | Loader wait or variable graph cost dominates | Pinning, RSS, ordering, sampling, and H2D overlap |
| GPU math | BF16/FP16, TF32/math precision, SDPA dispatch | Tensor-core or attention compute dominates | Dtype, backend, overflow, output/gradient/quality gates |
| GPU kernels | cuEquivariance, OpenEquivariance when installed, native PyG ops, FlashAttention/custom kernels | Exact kernel signature is a bottleneck | Layout, irreps, masks, fallback, forward/backward benchmark |
| Compiler | `torch.compile`/TorchInductor, regional compilation | Stable repeated regions have launch/fusion overhead | Graph breaks, recompiles, cold start, cache state, dynamic shapes |
| CUDA execution | CUDA Graphs or compiler CUDA Graph Trees | Fixed addresses, shapes, control flow, and allocations | Capture/replay correctness and memory lifetime |
| Low precision | torchao 0.18.0 or Transformer Engine 2.17 Float8/FP8 | Hardware and recipe support it and precision budget allows | Scaling/calibration, overflow, optimizer/checkpoint, quality |
| Distributed | DDP, FSDP2 `fully_shard`, device mesh, DCP | Replication, model memory, or scale is limiting | Per-rank timing, overlap, effective batch, resume semantics |
| Lifecycle | Optimizer/EMA/SWA/scheduler, logging, validation, async DCP | Amortized job cost or time-to-quality dominates | Cadence, state/bytes, host/pinned memory, resume equivalence |

## Hardware and backend boundaries

- CUDA, ROCm, CPU, and other backends do not share identical kernels or compiler coverage. Record `torch.version.cuda`, `torch.version.hip`, device properties, and backend-specific availability.
- BF16, TF32, FP8, tensor cores, FlashAttention, cuEquivariance, and OpenEquivariance depend on compute capability, driver/toolkit, library build, shapes, alignment, layout, dtype, and operation coverage. Use feature probes and exact traces.
- FP8/Float8 is not a drop-in replacement for BF16. Hopper/Blackwell-class support does not guarantee that a scientific operator, reduction, eigensystem, covariance, or custom kernel is safe to quantize.
- CPU speed depends on physical cores, SMT, NUMA, cache, memory bandwidth, oneDNN/BLAS implementation, process affinity, and competing workers. Record topology and thread settings instead of reporting only the CPU model.

## What to measure for each layer

- CPU/data: loader wait, transform/collate time, H2D time, worker RSS, CPU utilization, affinity, storage/cache state, and work-normalized throughput.
- GPU/kernel: kernel duration distribution, achieved work, memory traffic, occupancy or tensor-core evidence when relevant, launch gaps, peak allocated/reserved/external memory, and forward/backward behavior.
- Compiler: cold compile time, steady-state time, graph breaks, recompiles, dynamic guards, cache fingerprint/hits/misses, compiled-region coverage, and fallback behavior.
- Distributed: per-rank compute and input time, NCCL duration and overlap, max/min rank time, communication volume, scaling efficiency, and checkpoint/resume time.
- Scientific: output/loss/gradient differences, finite values, invariance/equivariance/physical constraints, convergence, calibration/coverage, sampler or rollout quality, and deterministic resume.

## Features that need extra skepticism

- Global precision changes, Float8, quantization, pruning, approximation, changed reduction order, and altered sampler grids are algorithmic experiments unless the contract explicitly allows them.
- `torch.compile(mode="max-autotune")`, CUDA Graphs, FSDP2, asynchronous prefetch, and allocator changes can improve one workload while increasing cold start, memory, synchronization, or failure risk.
- Private APIs such as `torch._foreach_*` or undocumented environment variables need version pinning and a public fallback. A silent fallback is a failed activation, not a successful optimization.
