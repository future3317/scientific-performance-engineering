# Memory, Compiler, and Distributed Failure Routes

Use this reference when an optimization changes allocator behavior, compilation, CUDA Graphs, rank balance, collectives, or checkpoint/resume.

## Memory forensics

Classify OOM as capacity, activation peak, optimizer state, retained autograd graph, temporary workspace, allocator fragmentation, external-library allocation, or host pinned-memory pressure. Capture `memory_stats`, a CUDA memory snapshot, peak allocated/reserved/active values, and device-total versus PyTorch-visible discrepancy before changing allocator settings. `PYTORCH_ALLOC_CONF` is the current name; record the legacy alias when present. Treat `max_split_size_mb`, `expandable_segments`, and `cudaMallocAsync` as evidence-gated candidates.

## Compiler and CUDA Graphs

Record compile cache state (`cold`, `warm`, `disabled`), cache fingerprint, hit/miss evidence, cold compile time, graph breaks, guards, recompiles, compiled-region coverage, and steady-state time. Start with specialized/static shape buckets when shapes are concentrated; use `dynamic=None`/bounded dynamic dimensions when recompiles demonstrate shape variation; use `dynamic=True` only when the measured regime benefits. Regional compile and intentional graph breaks are valid when data-dependent regions are isolated.

Compile repeated, shape-stable message blocks, RBF/MLP regions, or fixed-interface scalar/vector updates before attempting a whole ragged runner. Keep periodic catalogues, routers, Python control flow, and data-dependent shape changes eager unless a trace proves a stable boundary. A regional candidate needs matched steady-state p50/p95, cold-start amortization, graph-break/recompile evidence, and output/gradient/loss equivalence; a local compile speedup without end-to-end movement is inconclusive. For higher-order derivatives, benchmark a small compiled functional region or Compiled Autograd separately rather than hiding it inside a full-run result.

CUDA Graph capture requires stable shapes, control flow, addresses, allocations, streams, dependencies, DDP/NCCL path, and host callbacks. Verify input-copy and replay lifetimes after capture. Its objective is reducing launch/host overhead, not merely reacting to low GPU utilization.

## Distributed diagnostics

Report max-rank step latency and rank distributions, not only means. Preserve uneven-input semantics; use `join()` only for a proven uneven-input contract. Treat `static_graph=True` as conditional on an invariant used-parameter set. Route failures by symptom: slow collective → profiler/Nsight; rank desync/hang → ProcessGroupNCCL Flight Recorder/sequence diagnostics; topology/interface issue → NCCL topology diagnostics. Do not treat every distributed issue as an NCCL environment-variable tuning problem.

## Checkpoint and resume

Benchmark synchronous, asynchronous, and asynchronous-with-pinned-staging modes separately. Include staging memory and one-outstanding-checkpoint limits in the contract. A resume gate fails if dataloader cursor, shard position, order/augmentation RNG, EMA/SWA, or pending gradients are not restored at the declared boundary. A compile cache may be rebuilt, but the record must say so and prove the resumed scientific trajectory remains equivalent.
