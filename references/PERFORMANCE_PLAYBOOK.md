# Performance playbook

Use this profiling-led decision tree for scientific PyTorch, PyG, e3nn, cuEquivariance, diffusion, and structured probabilistic training. Start with the installed capability map in `TECHNOLOGY_MATRIX.md`; do not infer support from a package name alone.

## Contents

1. Define the work and contract
2. Establish a reproducible baseline
3. Profile in layers
4. Diagnose CPU and input pipeline
5. Diagnose hot paths, kernels, and ragged data
6. Route scope-specific mechanics
7. Apply scientific acceptance gates

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

### 3.2 Execution-graph fragmentation and launch density

When a trace contains many short kernels, host gaps, or frequent scalar synchronizations, record kernel launches per logical update, scalar-sync events, and tensor-to-Python branches as first-class evidence. Keep hot-path decisions and telemetry on device, stack metrics into one aggregate transfer at the declared cadence, and remove Python control flow only after confirming the scientific monitoring contract. Fuse a stable pure-tensor chain such as periodic displacement → lattice transform → basis/encoding → segmented reduction only when the trace identifies it as material; compile that regional boundary or write a custom kernel, rather than compiling an entire ragged runner. Compare launch count, graph breaks/recompiles, tail latency, memory, and forward/backward/higher-order gates; a lower launch count without end-to-end or scientific improvement is inconclusive.

When two compatible auxiliary conditions share graph features and a backbone, first test one mixed/block-diagonal head pass instead of duplicated paired/gap-only forwards. Preserve task counts, masks, normalization, gradients, and peak memory; a changed condition mixture or supervision membership is an objective experiment, not a free fusion.

For every step, record whether each auxiliary task was actually executed. Stochastic thinning is valid only when an unselected task skips its forward and gradient path, with explicit selection probability, loss reweighting, seed/rank semantics, and quality gates. A computed-then-masked loss is not thinning. In DDP, specify a global mask (shared step/global seed or broadcast) versus rank-local masks; test used-parameter sets, `find_unused_parameters`, `static_graph`, reducer behavior, and `no_sync()` for the chosen policy. Include zero-selected windows, cross-rank loss/gradient normalization, and global-norm clip fraction.

Use `data_seconds` only for the interval it actually measures. Cache deterministic parsing/topology/geometry and prefetch through the existing loader only when the trace identifies CPU preparation as material; verify worker RSS, synchronization, and ordering. Distinguish microbatch, accumulation window, optimizer update, and DDP reduction, and report work per logical update. A changed graph/task mixture or effective batch is a separate data/objective experiment, not a free throughput normalization.

Record host load average, available memory, swap use, and worker RSS beside GPU telemetry. Material host contention invalidates a direct before/after claim unless both runs are comparable.

Before a long run, audit precompute and campaign fan-out as first-class performance stages. Measure time to first usable batch and amortized cache-build cost; do not optimize only the steady-state forward while a one-time graph/feature build consumes hours. For multi-seed or multi-endpoint execution, record the process-tree upper bound (`concurrent jobs × endpoints × loader sets × workers`) and add intra-op/inter-op/BLAS/native threads, CPU/NUMA placement, host memory/swap, and GPU assignment. A worker count that works for one seed can oversubscribe the host when multiplied across seeds; use a measured concurrency/worker sweep and preserve separate artifact roots.

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

## 6. Route scope-specific mechanics

Keep this playbook as the common decision tree. Route detailed policy to one canonical owner:

- `DATA_AND_TRAINING_LIFECYCLE.md`: storage/dataset/collation/transfer, cache keys, worker topology, optimizer-adjacent work, logging, validation, checkpoint cadence, resume, and time-to-quality.
- `CODE_AND_RUNTIME_AUDIT.md`: synchronization census, custom operators, higher-order autograd, active dispatch, and numerical localization.
- `MEMORY_COMPILER_DISTRIBUTED.md`: memory forensics, precision/allocator interactions, compiler/CUDA Graphs, DDP/FSDP, collectives, and checkpoint mechanics.
- `GNN_PREDICTION_WORKLOADS.md`, `CRYSTAL_GENERATION.md`, and `EQUIVARIANT_OPERATOR_DESIGN.md`: workload-specific scientific and representation contracts.

Do not repeat a route's detailed compatibility or acceptance policy here; link the owner and apply only the scope declared in the benchmark record.

## 7. Scientific acceptance

Store required quality gates in `acceptance.required_quality_gates`; CLI flags may add gates but cannot replace or omit the recorded policy. Missing quality results are inconclusive, never accepted. Require the repository's existing gates. Typical gates include unit/regression tests; rotation/reflection/permutation/translation/periodic/gauge covariance; positive-volume, SPD, conservation, exact-count, and finite-gradient constraints; FP32/candidate output and gradient comparisons; calibration/coverage; rollout or sampler non-inferiority; deterministic resume/data-order; and checkpoint compatibility.

A faster sampler with degraded distributional quality is a failed optimization unless an explicit algorithmic tradeoff was authorized. If no repository gate exists, select the primary acceptance metric from the recorded optimization objective and define guardrails before measurement; apply the familiar 5% work-normalized throughput threshold only for a throughput objective.
