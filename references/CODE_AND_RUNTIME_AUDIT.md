# Code and Runtime Audit

Use this reference when the candidate changes runtime topology, a custom kernel/operator, compiler region, synchronization pattern, or numerical behavior.

## Preflight before profiling

Record model construction order, device/rank mapping, GPU UUID, NUMA/CPU affinity, PCIe/NVLink/NIC topology, container shared memory, thread environment, allocator configuration, and the active combination of compile, activation checkpointing, DDP/FSDP, CUDA Graphs, custom ops, and higher-order autograd. Mark the combination `pass`, `fail`, or `inconclusive` before launching a long run. A known-incompatible combination is a preflight failure, not a slow benchmark.

Treat `torch.compile`/CUDA Graphs, dynamic shapes, higher-order gradients, DDP unused parameters, checkpointing, and custom ops as a compatibility matrix. Do not wait for a hang to discover a rejected combination. FSDP2 parameter sharding must be applied before constructing an optimizer because it changes parameter representation.

## Performance-code correctness gate

For a custom C++/CUDA/Triton/Python operator, preserve one reference implementation and an exact-signature benchmark. Register schema, mutation/aliasing behavior, FakeTensor/meta behavior, and the training autograd path. Run `torch.library.opcheck` where available, then separate `assert_close`, `gradcheck`, and `gradgradcheck` when higher-order derivatives are part of the contract. Include empty, non-contiguous, edge-shape, dtype/device, compile, and actual forward/backward dispatch cases. `opcheck` validates the operator contract; it does not prove mathematical correctness.

When a numerical gate fails, localize rather than only rejecting: compare eager/reference and candidate dispatched operations with DebugMode or equivalent op traces, record the first divergent op and tensor metadata (dtype/layout/backend), then make a focused reproduction.

## Synchronization census

Count `.item()`, `.cpu()`, tensor-to-Python branches, gradient-norm reads, metric reductions, progress updates, validation statistics, checkpoint staging, barriers, allocator calls, logging/flush/network, host indexing, and explicit synchronizations. Classify each as `required`, `removable`, `amortizable`, or `overlappable`. Reuse device-resident aggregates and one already-issued D2H copy when possible; do not remove scientific monitoring without recording the replacement cadence.

For a host-driven hot loop, keep CUDA scalars and control flow on device: do not use `.item()`, `.cpu()`, or Python `if` to decide per-graph/per-task work, and do not log one scalar at a time. Stack metrics and move one aggregate to CPU at the declared cadence. `torch.cuda.set_sync_debug_mode("warn")` is a diagnostic detector, not a production optimization; confirm the call stack and overlap with Nsight Systems before changing semantics.

For conservative energy→force/stress→parameter derivatives, preserve the derivative contract and isolate a pure functional region before optimizing. Compare the existing `autograd.grad(create_graph=True)` path with `torch.func.grad` and compiled/Compiled Autograd candidates on identical outputs, gradients, higher-order requirements, and work. Never remove `create_graph` merely to make a trace shorter.

## Acceptance

The preflight result, active-path proof, numerical localization evidence, and sync census belong in the benchmark record. Missing evidence is `inconclusive`; a package being installed is not proof that a training backend or operator mode is active.
