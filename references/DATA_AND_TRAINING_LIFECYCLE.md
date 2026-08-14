# Data and Training Lifecycle

Use this reference when the trace labels work as `data`, when long-run throughput differs from step throughput, or when caching, logging, validation, or checkpointing may change the campaign result.

## Four-layer data model

Separate `storage → dataset → collate/preprocess → transfer`. Locate the bottleneck before changing `num_workers`: file read, parse, graph construction, collation, pinning, or H2D. For map-style datasets, benchmark batched `__getitems__(indices)` against per-sample `__getitem__`; keep ordering, augmentation RNG, shard identity, and graph provenance unchanged. Cache only immutable topology/parsed artifacts with an explicit key containing dataset/sample identity, cutoff, PBC/cell convention, augmentation, species mapping, graph-builder version, dtype/layout, and basis version.

Benchmark `cold-cache`, `warm-cache`, and the intended long-run cache hit regime separately. A warm-cache result is not a first-epoch claim. For custom batches, prove `is_pinned()`, `non_blocking=True`, copy stream, source lifetime, consumer-stream dependency, and an overlap timeline; the flag alone is not proof.

## Precompute and campaign resource census

Treat preprocessing, graph/geometry construction, feature materialization, and cache building as measured workloads, not invisible setup. Record wall time to first usable batch, CPU/GPU time, peak host/device/pinned memory, reuse count, invalidation/rebuild triggers, and amortized cost per logical update. If precompute is material to time-to-first-use or campaign time, profile and optimize it before launching a long training run; compare vectorized/batched construction, reuse, and bounded parallelism without changing provenance.

For concurrent seeds or endpoints, budget the whole process tree before starting: concurrent jobs × endpoints per process × train/validation/test loader sets × workers per loader, plus intra-op/inter-op/BLAS/native threads. Compare the resulting worker/thread count with physical cores, NUMA placement, host memory, swap, and GPU assignment. A single-seed setting that is healthy does not authorize multiplying its workers by the campaign concurrency. Start with one seed or a measured low-worker topology, then increase concurrency only when GPU duty cycle, loader wait, host load, RSS, and result isolation remain within the contract. Keep artifact roots and seed identities separate when stopping or restarting a campaign.

## Logical-update DAG

Account for every stage: fetch, CPU preprocessing, H2D, GPU preprocessing, backbone/heads, loss, auxiliary derivatives, backward, unscale/gradient transforms, clipping, communication, optimizer, scheduler, EMA/SWA, metrics, checkpoint staging, and validation trigger. `step_ms` is not the whole job. Record steady-state train-step throughput and an amortized training throughput that includes the real logging, validation, checkpoint, and sampling cadence. For a fixed quality target, report time-to-quality as a campaign metric.

## Training contract and objective cadence

Record the exact initialization checkpoint and every auxiliary reference used during training. An anchor or proximal term must point to an explicitly declared frozen reference (normally the initialization trunk for a successor), with parameter scope and coefficient recorded; an anchor to a different stage/checkpoint is a contract mismatch until a matched ablation or documented objective proves it is intended. Freeze the complete constraint stack (shared learning-rate scaling, anchors, gradient projection, norm caps, and task weights) before comparing throughput or quality.

When several constraints are stacked, use a small matched ablation matrix (current stack, remove one constraint, or relax one cap) with the same initialization, horizon, and gates. Do not introduce a new combined recipe before knowing which constraint changes the observed regression or quality trade-off.

Separate systems changes from objective changes. Changing loss weights, task membership, supervision packing, auxiliary update cadence, gradient statistics, solver/NFE, or sampling law is an `algorithmic_experiment`, even when it reduces wall-clock time. For an auxiliary task updated every (k) steps, record the selection schedule, loss reweighting, optimizer/EMA boundary, and work per logical update; do not call it a free systems speedup.

When two expensive auxiliary paths have the same expected frequency, prefer a deterministic mutually exclusive schedule when the scientific contract permits it. Prove the average update frequency and reweighting, then measure tail latency and compile/capture stability; independent Bernoulli masks can create avoidable overlap. An unselected task must skip its forward and derivative path, not merely mask a computed loss.

### Optimizer-aware thinning

The usual (1/p) loss reweighting preserves the first-moment gradient expectation, not Adam-like second moments or the resulting preconditioned update. Treat `p < 1` as a gradient-statistics change until proven otherwise: run a matched `p=1` control and compare optimizer second-moment state, preconditioned update norms, clipping fractions, late-window quality, and resume behavior. Do not call thinning objective-equivalent from unbiased gradients alone; if these statistics or quality move materially, record it as an `algorithmic_experiment` and keep the systems speed result separate.

### Gradient ownership and shared trunks

For sparse auxiliary tasks sharing a trunk, report per-task gradient norms and pairwise cosine/ownership for trunk, adapter, and head parameters before aggregating all auxiliary terms. An `auxiliary` total can hide cancellation or one sparse task dominating the shared trunk. A fresh conditioner attached to a pretrained trunk should be zero-initialized/gated or isolated behind a declared adapter when warm-start preservation is required; verify step-0 output closeness and the first-block gradient path. Freezing the trunk and training cached-feature heads can reduce recurring work, but it changes supervision/parameter ownership and needs a matched quality and time-to-quality experiment.

Use staged horizons before a long campaign: stop or revise at the shortest checkpoint that can distinguish regression from noise, then extend only candidates that pass validation Pareto and time-to-quality gates. A full horizon is not evidence when initialization, anchor provenance, supervision coverage, or objective coupling is unresolved.

## Scientific gradients and graph lifetime

Do not assume validation can use `inference_mode()`: force, stress, response, and higher-order quantities may require gradients. Audit `create_graph`, `retain_graph`, saved tensors, hooks, Python lists containing `grad_fn`, and validation outputs retained on GPU. A graph-lifetime failure is not allocator fragmentation. For activation checkpointing, record recomputation FLOPs, extra launches, live-activation peak, `use_reentrant`, `preserve_rng_state`, and the interaction with `autograd.grad`, accumulation, and DDP `no_sync()`.

## Grad transforms and optimizer-adjacent work

The scientific contract includes AMP unscale → gradient transforms/normalization → clipping → optimizer. Benchmark `foreach`/`fused` dispatch on the actual parameter/dtype/None-gradient/capture path; do not assume it is faster. Record optimizer kernel time, temporary memory, state dtype, parameter groups, EMA update time, SWA update time, scheduler cost, and clipping cost. Optimizer-in-backward is an opt-in memory route for cases without gradient accumulation, not a default recipe.

## Logging, validation, and checkpoints

Measure metric computation, all-reduce, D2H, serialization, logger flush/network, plotting, validation, and checkpoint staging independently. Aggregate on device and defer transfer only when the scientific monitoring cadence permits it. For DCP, record state materialization, D2H/pinned staging, bytes/rank, host/pinned memory, queue depth, wait time, write time, and load time; cap outstanding async checkpoints. Resume evidence includes optimizer/scheduler/GradScaler/RNG, pending gradients, sampler/dataloader cursor, shard position, epoch/order, augmentation RNG, EMA/SWA state, and whether compiler state is rebuilt.
