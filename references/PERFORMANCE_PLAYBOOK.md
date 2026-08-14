# Performance playbook

Use this profiling-led decision tree for scientific PyTorch, PyG, e3nn, cuEquivariance, diffusion, and structured probabilistic training. Start with the installed capability map in `TECHNOLOGY_MATRIX.md`; do not infer support from a package name alone.

## Contents

1. Define the work and contract
2. Establish a reproducible baseline
3. Profile in layers
4. Diagnose CPU and input pipeline
5. Diagnose hot paths, kernels, and ragged data
6. Validate precision, math modes, and quantization
7. Diagnose optimizer and memory
8. Validate compilation and CUDA Graphs
9. Validate distributed execution and checkpointing
10. Apply scientific acceptance gates

## 1. Define the work and contract

Freeze the scientific and measurement contract before touching code:

- model outputs, loss components, invariance/equivariance/physical constraints, sampler law, and checkpoint semantics;
- precision policy and FP32/FP64 islands;
- data membership/order/augmentation, seed, microbatch, effective batch, accumulation, optimizer, schedule, and world size;
- work unit: graphs, atoms, edges, samples, transitions, tokens, or another domain unit;
- measured region: data wait, H2D, forward/loss, backward, optimizer, logging, checkpoint, validation, and compilation;
- quality metrics, numerical tolerances, memory budget, and minimum material improvement.

For variable-size graphs, report graphs/s together with atoms/s or edges/s. A larger graphs/s value is not evidence of an improvement if the candidate receives cheaper graphs.

## 2. Establish a reproducible baseline

Use the same base revision, benchmark harness, hardware, driver, CUDA/ROCm, PyTorch, PyG, e3nn, cuEquivariance, Triton, Transformer Engine/torchao, NCCL, CPU affinity, storage path, data panel, seed, worker configuration, and batch composition. Represent the candidate as a declared patch/change set; a changed commit is not by itself an incomparability.

Warm up CUDA libraries, allocator pools, compiled kernels, autotuning, and caches. Reset peak memory after warmup. Store every independent run/window and its randomized or A/B order. Report median, IQR/MAD, and a bootstrap confidence interval; a point estimate clearing 5% is inconclusive when its improvement interval does not clear the objective threshold and noise floor.

Use paired CUDA events on the relevant stream for CUDA component timing, or synchronize at the phase boundary. `record_function`/NVTX ranges are attribution labels only; a CPU timestamp around asynchronous launches is not a GPU completion time. Record clock/stream/completion proof for every bucket and reconcile the bucket sum with end-to-end step time; an unaccounted ratio above the acceptance policy is inconclusive. Include compile, checkpoint, validation, and data startup costs when they recur in the actual workflow; otherwise label them separately.

## 3. Profile in layers

Instrument coarse named ranges around `next_batch`, collation, H2D, forward, loss, backward, gradient transforms/clipping, optimizer, scheduler, logging, checkpoint, and communication.

Start with a scheduled `torch.profiler` window after warmup. Keep shapes, stacks, and memory off for the first low-overhead pass; enable them in focused traces because they perturb runtime. Use `torch.utils.benchmark` for isolated CPU/kernel comparisons, not as a substitute for the end-to-end trace.

Use Nsight Systems when the trace cannot explain CPU gaps, CUDA API/launch latency, stream interaction, OS scheduling, H2D overlap, or NCCL overlap. Use a short Nsight Compute run when a specific kernel needs occupancy, instruction, memory-throughput, or roofline evidence. Use PyG profiling APIs for loader and GNN-specific attribution when available.

| Signature | Likely class | Next check |
|---|---|---|
| Wide CPU gaps before H2D/forward | input/CPU bound | workers, I/O, collation, affinity, pinning, NUMA |
| Many tiny kernels with host launch gaps | Python/launch bound | loops, scalar extraction, fusion, stable compile region, CUDA Graphs |
| Long GEMM/attention/tensor-product kernels | compute bound | dtype, tensor-core eligibility, layout, batch, kernel backend |
| Low arithmetic intensity and high memory traffic | bandwidth/allocation bound | materialization, dtype, layout, fusion, buffer reuse |
| Synchronize/D2H spans | host-device synchronization | `.item()`, `.cpu()`, tensor-to-Python control flow, logging |
| Repeated compile regions or recompiles | dynamic/compile bound | graph breaks, guards, bucketing, regional compile |
| NCCL spans not overlapped or long-tail rank | communication/imbalance | buckets, unused params, rank input time, workload balance |

### 3.1 Audit timing buckets and task composition

Before changing a kernel, reconcile the measured wall-clock step with named ranges. Use distinct ranges for `batch_fetch`, JSON/graph construction, H2D, structural/model work, mechanism, property heads, physical-feature construction, `autograd.grad`, loss, backward, optimizer, and DDP communication. A `data_ready` mark belongs after the batch is actually ready; if mechanism or property work runs before it, split those ranges and rename the bucket. Report unaccounted time rather than forcing it into `data`.

Build a task census for the same representative step: per-rank/global graphs, structures, atoms, edges, each auxiliary task/head, Python loop iterations, auxiliary forward calls, `autograd.grad` calls, FP64/CPU sections, and shared-backbone consumers. Many short CPU-launched kernels with 0–2% SM duty cycle are a CPU/launch/synchronization hypothesis, not evidence of insufficient GPU FLOPs.

For multi-head scientific training, compare in this order:

1. reuse one backbone/features for compatible heads;
2. combine compatible vector-Jacobian products or use a batched VJP when the installed autograd API and higher-order requirements allow it;
3. vectorize per-structure reconstruction/occupation/constraint work with stacked, bucketed, segmented, or masked operations;
4. only then consider a custom kernel for the measured remaining hotspot.

For every step, record whether each auxiliary task was actually executed. Stochastic thinning is valid only when an unselected task skips its forward and gradient path, with explicit selection probability, loss reweighting, seed/rank semantics, and quality gates. A computed-then-masked loss is not thinning. In DDP, specify a global mask (shared step/global seed or broadcast) versus rank-local masks; test used-parameter sets, `find_unused_parameters`, `static_graph`, reducer behavior, and `no_sync()` for the chosen policy. Include zero-selected windows, cross-rank loss/gradient normalization, and global-norm clip fraction.

Use `data_seconds` only for the interval it actually measures. Cache deterministic parsing/topology/geometry and prefetch through the existing loader only when the trace identifies CPU preparation as material; verify worker RSS, synchronization, and ordering. Distinguish microbatch, accumulation window, optimizer update, and DDP reduction, and report work per logical update. A changed graph/task mixture or effective batch is a separate data/objective experiment, not a free throughput normalization.

Record host load average, available memory, swap use, and worker RSS beside GPU telemetry. Material host contention invalidates a direct before/after claim unless both runs are comparable.

## 4. CPU and input pipeline

1. Measure loader wait, transform, collation, H2D, and training separately. Sweep `num_workers` around 0, 2, 4, and 8 or a hardware-appropriate range; CPU count is not a setting.
2. Enable `persistent_workers` only with workers and datasets safe to retain. Sweep `prefetch_factor`; excess prefetch raises RSS and can worsen random I/O. Treat `in_order=False` as a contract change unless data order and class/size balance are proven irrelevant.
3. Verify `pin_memory=True` reaches every tensor in custom/PyG batches and that `non_blocking=True` copies overlap useful CPU work. Do not assume a custom type is pinnable. Do not initialize CUDA in workers.
4. Bound total parallelism: DDP ranks, DataLoader workers, Torch intra-op threads, inter-op threads, BLAS/OpenMP threads, and native kernel threads. Record `torch.get_num_threads()` and `torch.get_num_interop_threads()`.
5. On multi-socket systems, test rank and worker placement by NUMA node. Use process/worker affinity and `OMP_PROC_BIND`/`OMP_PLACES` only when traces and topology justify them. Avoid sharing physical cores between loader and compute threads.
6. Prefer vectorized parsing/transforms, local shard access, deterministic cache of topology/bases/parsed records, and batch-size-aware bucketing. Preserve stochastic augmentation and sampling provenance.
7. For PyG, compare ordinary loaders with `PrefetchLoader`, `DynamicBatchSampler`, and CPU affinity on the actual graph distribution. Include padding, transfer, worker RSS, and batch-cost variance in the result.

## 5. Hot path, kernels, and ragged data

Look for `.item()`, `int(cuda_tensor)`, `.cpu()`, host indexing, Python branches on CUDA tensors, synchronous metrics, per-graph loops, repeated `eye`/`arange`/mask/basis creation, dtype/device conversion, sorting, and allocation.

Keep metadata such as graph sizes and offsets on CPU when it is immutable and transferring it would synchronize. Otherwise compare, in increasing invasiveness:

1. cached metadata and immutable tensors;
2. vectorized or segmented reductions;
3. size bucketing and grouped execution;
4. padded batched kernels with measured padding waste;
5. a compiled stable subregion or CUDA Graph candidate.

For attention, begin with PyTorch SDPA backend dispatch. Try FlashAttention or a custom backend only when the exact mask, sequence length, head dimension, layout, dtype, and backward path select the intended kernel. For GNN/equivariant code, benchmark exact irreps, edge counts, layouts, multiplicities, precision, and forward/backward. Layout transposes and silent fallbacks can erase a kernel gain.

## 6. Precision, math modes, and quantization

Use autocast around the real forward and loss region and exit before backward. Keep sensitive algebra outside autocast:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
    prediction = model(batch)
    fast_terms = compute_fast_terms(prediction, batch)

with torch.autocast(device_type="cuda", enabled=False):
    stable_term = stable_linear_algebra(prediction["matrix"].float(), target.float())

loss = fast_terms.float() + stable_term
loss.backward()
```

Validate FP32 versus BF16/FP16/FP8 with output max/mean error and cosine, component/total loss error, finite checks, gradient norm ratio/cosine, short trajectory, and downstream quality. For FP16, unscale before clipping. For BF16, do not assume a scaler is needed. Record `torch.set_float32_matmul_precision`, TF32, deterministic settings, and backend selection.

Treat torchao Float8/quantization and Transformer Engine FP8 as opt-in recipes, not global switches. Check GPU architecture, supported shapes/layouts, scaling and calibration, optimizer state, overflow handling, checkpoint compatibility, and whether the scientific contract permits quantization. Reject a faster candidate when quality, stability, or reproducibility fails.

## 7. Optimizer and memory

- Keep `zero_grad(set_to_none=True)` unless material zero gradients are part of the contract.
- Benchmark optimizer `foreach` and `fused` modes on the target parameter structure, dtype/device mix, sparse/None gradients, and capture path. Do not assume fused is faster.
- Group gradient transforms with foreach only after checking sparse gradients, bucket views, higher-order autograd, and numerical equivalence. Prefer public APIs when available; pin versions when an underscored API is unavoidable.
- Measure activation checkpointing as a throughput/compute tradeoff, not only a memory win. Before combining it with `autograd.grad`, higher-order gradients, DDP accumulation, or `no_sync()`, test `use_reentrant` and `preserve_rng_state` explicitly; reentrant checkpointing is not a drop-in path for every autograd API. Record whether checkpoints resume only at optimizer boundaries or also mid-accumulation. A mid-window contract must restore pending gradients, optimizer/scheduler/GradScaler, sampler cursor, and Python/CUDA RNG; otherwise state the boundary-only limitation. Preserve validation gradients when the scientific quantity requires them.
- Distinguish allocated, reserved, active, and external-library memory. Inspect `memory_stats`, memory snapshots, allocator configuration, and host RSS. Tune `cudaMallocAsync` or allocator split/rounding only after fragmentation evidence.
- Never call `empty_cache()` per step and never accumulate validation outputs on GPU without a bounded plan.

## 8. Compilation and CUDA Graphs

Start from eager evidence. Compile a stable repeated block before a ragged end-to-end trainer.

- Compare supported `torch.compile` modes and backends on identical work. `max-autotune` can trade compile time and memory for steady-state speed; `reduce-overhead` may use CUDA Graphs. Measure both separately and use only modes present in the installed version.
- Enable `TORCH_LOGS=graph_breaks,recompiles,dynamic` for diagnosis. Record graph breaks, recompiles, guards, compiled-region coverage, cold compile time, cache behavior, and steady-state time.
- Treat Python metadata, dynamic lists, custom ops, tensor-to-Python conversion, ragged shapes, and stage-dependent branches as likely breakpoints. Use bucketing, dynamic shapes, regional compilation, or explicit graph boundaries only when the contract permits.
- Do not claim a compiled path when errors are suppressed or the hot function is not entered. `torch.export` and AOTInductor are primarily graph/export/deployment tools; do not use their existence as training evidence.
- CUDA Graphs and compiler CUDA Graph Trees require stable shapes/control flow, addresses, allocations, streams, and dependencies. Ragged PyG batches and changing stage branches need a proven capture design.

## 9. Distributed and checkpointing

Measure per-rank data wait, forward, backward, optimizer, NCCL communication, overlap, total time, graph/atom/edge counts, and max/min rank latency. Select GPUs explicitly by index/UUID and record logical-to-physical UUID mapping. Report aggregate throughput and scaling efficiency against the same single-GPU per-device work.

- Keep gradient accumulation and DDP `no_sync()` boundaries mathematically correct, including short final windows and graph-count weighting.
- Use `find_unused_parameters=True` only when needed. If used-parameter sets and control flow are invariant, inspect DDP logging and test `static_graph=True`; a wrong assumption can hang or produce incorrect gradients.
- With stochastic thinning, rank-local masks can change the used-parameter set from step to step. Do not enable `static_graph=True` until the mask schedule proves that set is invariant; otherwise keep the needed unused-parameter detection and test reducer completion on zero-selected windows.
- Test `gradient_as_bucket_view=True` and bucket tuning only when the trace shows copy or overlap issues. Do not set NCCL tuning variables without a measured reason.
- Use FSDP2 `fully_shard`/device mesh when model or optimizer memory requires sharding; include resharding and communication in the measurement. Use `torch.distributed.checkpoint` for scalable save/load only when its state-dict and resume semantics are verified.
- Consider tensor/pipeline parallelism only when the model and workload have a stable partitioning plan. Do not trade away scientific batch semantics to obtain a scaling graph.

## 10. Scientific acceptance

Store required quality gates in `acceptance.required_quality_gates`; CLI flags may add gates but cannot replace or omit the recorded policy. Missing quality results are inconclusive, never accepted. Require the repository's existing gates. Typical gates include unit/regression tests; rotation/reflection/permutation/translation/periodic/gauge covariance; positive-volume, SPD, conservation, exact-count, and finite-gradient constraints; FP32/candidate output and gradient comparisons; calibration/coverage; rollout or sampler non-inferiority; deterministic resume/data-order; and checkpoint compatibility.

A faster sampler with degraded distributional quality is a failed optimization unless an explicit algorithmic tradeoff was authorized. If no repository gate exists, select the primary acceptance metric from the recorded optimization objective and define guardrails before measurement; apply the familiar 5% work-normalized throughput threshold only for a throughput objective.
