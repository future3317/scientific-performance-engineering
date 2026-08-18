# Repository-specific notes

These notes were prepared on 2026-07-27 from the public default branches. They are hypotheses and guardrails, not substitutes for reading the current checkout. Revalidate file contents and configuration before changing code.

## General-Equivariant-Covariance-Networks-for-Probabilistic-Structured-Prediction

Observed files:

- `README.md` blob `d79563c12f4284ae35f3d666cf906a1819a5f919`
- `scripts/train_dielectric.py` blob `0c0301b8ceadcc906dacc04b90b7e44e8269ecbf`

Existing performance contract to preserve:

- The project deliberately separates BF16 backbone execution from FP32 structured operator/NLL algebra and FP64 diagnostic materialization.
- Tensor-product backends and methods are explicit (`e3nn`, cuEquivariance naive/fused, optional compile); unavailable optimized paths fail rather than silently falling back.
- The repository already benchmarks tensor products on the actual shape/GPU and records output error plus forward/backward latency.
- Dielectric loading already exposes shard storage, shard cache size, workers, persistent workers, pinning, and prefetch.
- Algebra-preserving optimizations include reused degree normalization, native reductions, factorization reuse, and tree Schur elimination.

High-value profiling questions:

1. Verify the configured inference contract reaches every training/evaluation path and inspect the actual FP32/BF16 islands.
2. Profile OOF/pseudo-covariance sample lookup: the current loop moves `sample_index` to CPU and calls a scalar max check before indexing CPU caches. Determine whether this synchronizes the hot path and whether pinned/batched or device-resident lookup is feasible within memory and provenance constraints.
3. Measure `clip_grad_norm_`, progress logging, covariance statistics, and stage-specific frozen parameter behavior.
4. For `compile_tensor_products(dynamic=True)`, separate cold compile, graph breaks/recompiles, and steady state. Do not generalize a tensor-product microbenchmark to end-to-end training.
5. Keep cuEquivariance layout and fallback semantics explicit. A conversion in every layer may erase a fast kernel's gain.

Required acceptance includes strict checkpoint/model semantic compatibility, equivariance/regression tests, proper likelihood/calibration metrics, and the repository's hardware-specific benchmark reporting.

## gaugeflow

Observed files:

- `README.md` blob `690a920e3660cdc8799b974df600660999cc3b05`
- `docs/current_project_status_zh.md` blob `2cdd2517e86cee0d98a8b6cf80e9edd90a2b8b73`
- `pyproject.toml` blob `4278d33d38417334e0f973f21842dc68ed391736`

Existing performance/scientific contract to preserve:

- Production uses explicit Cartesian/equivariant geometry, with selected geometry-sensitive paths kept FP32 and BF16/FP32 output and gradient audits.
- Graph/edge reductions use target-contiguous linear-complexity segment reduction; runtime sorting and precision fallback are intentionally avoided.
- The project records graphs/s, memory, hardware identity, precision cosines, and scientific gates. Results from an RTX 4090 must not be relabeled as RTX 4060 Ti results.
- Faster reduced-NFE samplers have been rejected when held-out distributional non-inferiority failed. Latency alone cannot qualify a sampler.
- A prior no-grad versus accidentally retained-autograd comparison showed a large memory difference; inspect inference/evaluation graph retention.

High-value profiling questions:

1. Preserve the training–inference conditional contract. Do not “optimize” by corrupting observed side information or changing probability paths.
2. Profile graphs/s together with atoms/edges and time-regime mixture; batches are not necessarily equal cost.
3. Inspect per-edge message formation, segment reductions, dynamic neighbor/topology construction, and creation of Cartesian bases/moments.
4. Check whether fixed FP32 geometry islands dominate step time before proposing lower precision. Any new BF16 region needs output/gradient cosine and downstream gates.
5. For sampling, use common random numbers and frozen quality panels. A faster NFE/grid is accepted only after the registered distributional and structural guardrails.
6. Keep historical negative results and frozen protocols intact; do not rerun broad hyperparameter searches as a performance shortcut.

## PiezoJet

Observed file:

- `src/piezojet/pretraining/trainer.py` blob `2f9a2cbdedfc15a40ea09082cd9fb2cfd10959d9`

Existing performance-aware code to preserve:

- Precomputed graph data and a shard-local sampler already address repeated periodic graph construction and serialized-shard thrashing.
- Device transfer uses non-blocking mode on CUDA; DDP accumulation uses `no_sync()` and graph-count-correct normalization.
- Stage-specific scientific losses contain ragged tensors, eigensystems, optical projectors, and response algebra; naive batching can change semantics or consume excessive padding.

Immediate audit targets (confirm in current code):

1. `PretrainingTrainer` defines `_maybe_autocast()` and a `use_bf16` flag, but the inspected `_run_epoch()` path calls forward and loss without entering that helper. Verify actual dtypes with profiler/range assertions; if trainer-level autocast is intended, connect it with explicit FP32 islands and numerical tests.
2. Loader construction exposes `batch_size` and `num_workers`, pins memory on CUDA, but the inspected path does not expose `persistent_workers` or `prefetch_factor`. Add only after a worker/I/O sweep and custom-batch pinning verification.
3. DDP is constructed with `find_unused_parameters=True` because stages activate different heads. Measure its traversal overhead. Consider stage-specific wrapping or `static_graph` only if the used-parameter set and control flow are proven invariant for the complete stage.
4. Curvature and response losses contain per-graph Python loops, CUDA tensor-to-Python integer conversions, repeated basis/identity creation, per-graph transfers, and `torch.linalg.eigh`. Profile synchronization and kernel-launch overhead. Compare size bucketing, cached bases, padded batched `eigh`, or segmented formulations against the ragged reference.
5. Gradient normalization iterates over all parameters and launches `grad.div_` individually. Test grouped `torch._foreach_div_` by device/dtype while preserving `None` gradients and DDP semantics.
6. Validation for conservative E/F/S intentionally requires gradients. Do not wrap it in unconditional inference mode.

Required acceptance includes stage losses, conservative force/stress behavior, acoustic/optical constraints, DDP accumulation equivalence, resume determinism, split/manifest provenance, and checkpoint compatibility.
