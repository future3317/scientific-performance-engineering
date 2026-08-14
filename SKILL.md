---
name: scientific-gpu-training-optimizer
description: Use when profiling or optimizing PyTorch scientific training or inference with low GPU utilization, CPU-driven steps, misleading data/forward timing buckets, multi-task auxiliary heads or autograd overhead, CUDA launch/synchronization gaps, OOM or fragmentation, mixed precision, PyG/e3nn/cuEquivariance kernels, diffusion/probabilistic samplers, torch.compile/CUDA Graphs, DDP/FSDP, checkpointing, scaling regressions, or before/after performance validation. Also use for code-level GPU performance reviews that may remain static-only. Do not use for purely conceptual ML questions with no code, runtime, or benchmark decision.
---

# Scientific GPU Training Optimizer

Improve end-to-end throughput, latency, or resource efficiency while preserving the mathematical model, data contract, reproducibility, evaluation protocol, and hardware/software provenance.

## Contents

- [Operating modes and contracts](#select-the-operating-mode)
- [Evidence loop](#execute-the-evidence-loop)
- [Workload routes](#route-the-investigation)
- [Detailed playbook](references/PERFORMANCE_PLAYBOOK.md)
- [Measurement contract](references/MEASUREMENT_CONTRACT.md)

## Select the operating mode

- **Review or diagnose:** Inspect and explain evidence. Do not edit code unless the user also asks for a fix or optimization.
- **Optimize or implement:** Establish a comparable baseline, patch the smallest testable bottleneck, and validate both performance and scientific behavior.
- **Benchmark:** Build or run a reproducible comparison without changing the training implementation unless benchmark instrumentation is required.
- **Architecture review or experiment:** Only with explicit authorization, compare a baseline model against separately trained representation, equivariant-operator, or sampler alternatives. Keep these results separate from systems optimization.
- **Static-only:** When representative hardware, data, or dependencies are unavailable, report hypotheses and runnable commands. Label every performance conclusion unverified; never invent a speedup.

Do not expand a request for diagnosis into package upgrades, algorithm changes, distributed migration, or broad refactoring.

## Load resources only when needed

- Read [MEASUREMENT_CONTRACT.md](references/MEASUREMENT_CONTRACT.md) before editing performance-sensitive code or making a performance claim. Fill the relevant fields in the report or a working copy; do not overwrite the bundled template.
- Read [PERFORMANCE_PLAYBOOK.md](references/PERFORMANCE_PLAYBOOK.md) after inspecting the actual hot path, then use only the sections matching the observed bottleneck.
- Read [TECHNOLOGY_MATRIX.md](references/TECHNOLOGY_MATRIX.md) before proposing a version-, backend-, precision-, compiler-, kernel-, or hardware-dependent feature.
- Read [PATCH_PATTERNS.md](references/PATCH_PATTERNS.md) before emitting common PyTorch performance patches. Adapt patterns; never paste them blindly.
- Read [GNN_PREDICTION_WORKLOADS.md](references/GNN_PREDICTION_WORKLOADS.md) for static crystal/materials graph prediction, Cartesian tensor targets such as piezoelectric response, or static-graph PyG/PBC caching and batching.
- Read [CRYSTAL_GENERATION.md](references/CRYSTAL_GENERATION.md) for crystal diffusion/flow sampling, NFE, solver or guidance changes, dynamic neighbor rebuilding, generation batching, or relaxation/DFT campaign cost.
- Read [EQUIVARIANT_OPERATOR_DESIGN.md](references/EQUIVARIANT_OPERATOR_DESIGN.md) for an explicitly authorized architecture review of MACE/ACE product bases, TensorProduct placement or paths, high-\(l\) alternatives, OpenEquivariance, crystal-tensor output parameterization, or path-aware DDP batching. Read it in addition to the prediction or generation module matching the workload.
- Read [REPOSITORY_NOTES.md](references/REPOSITORY_NOTES.md) only when the checkout matches a listed project. Revalidate every dated note against the current checkout.
- Read [SOURCES.md](references/SOURCES.md) only when current API, capability, or release verification is necessary. Treat its dated versions as historical context, not requirements.
- Copy [performance_report.md](assets/performance_report.md) and [benchmark_record.json](assets/benchmark_record.json) into the project when a durable benchmark artifact is useful.

Run `scripts/collect_env.py` to capture the environment. Use `scripts/run_with_gpu_monitor.py` only for supporting telemetry; it records an optional `host_samples` timeline (when `psutil` is available), and utilization samples cannot prove a speedup. Run `scripts/compare_benchmarks.py` to reject incomparable records and calculate before/after deltas; a `comparable_unjudged` result is not a passing comparison.

## Preserve the contract

Before changing code, record or explicitly mark unknown:

- outputs, loss/objective, invariance/equivariance and physical constraints;
- probability parameterization, sampler law, precision-sensitive algebra, and accepted tolerances;
- data membership/order/augmentation, seeds, effective batch and accumulation;
- optimizer, clipping, schedule, world size, checkpoint/resume, and quality gates;
- work unit, timing boundaries, warmup, repetitions, memory budget, and acceptance rule.

Treat model family, irreps or TensorProduct paths, message-passing math, graph/batch semantics, objective, sampler grid or law/NFE, data exposure, effective batch, schedule, stopping rule, global precision, quantization, pruning, approximations, and reduction-order changes as algorithmic experiments. Make them only when explicitly authorized and report them separately from systems optimization.

## Use a staged experiment loop

Keep the work in five explicit phases: **Explore → Plan → Execute → Integrate → Review**.

- **Explore:** inspect the real path and profile it; do not edit code while the bottleneck is still a guess.
- **Plan:** write one hypothesis card: measured bottleneck share, suspected cause, one changed lever, expected metric movement, semantic risk, and the falsification test. Estimate the Amdahl ceiling before writing a custom kernel: if the target occupies fraction \(P\) of the measured region, a \(k\)-times faster target can improve the region by at most \(1/((1-P)+P/k)\).
- **Execute:** compare at three levels—**Micro → Module → End-to-end**—using the same representative shapes and work. Change one independent lever at a time; keep the eager/reference path and known-good outputs available for comparison.
- **Integrate:** retain only candidates with independent evidence. Keep rejected or inconclusive experiments in the copied performance report as the experiment ledger, including why they failed and what evidence would justify a retry.
- **Review:** a second pass may block only correctness or scientific regression, a performance regression, an API/required-portability break, or a flawed comparison. Style preferences and speculative edge cases are non-blocking.

When a backend or precision path needs A/B testing, use the smallest existing configuration switch or an explicit local benchmark parameter so the path and rollback are visible. Do not add a flag solely for hypothetical future variants.

## Audit step composition before optimizing kernels

Treat a multi-task step as a workload graph, not as one opaque `data` or `forward` block.

- **Run a timing bucket audit.** Define `batch_fetch`, CPU/JSON/graph construction, H2D, structural/model work, each auxiliary task or head, physical-feature construction, `autograd.grad`, loss, backward, optimizer, and DDP communication as separate ranges. `data_ready` is valid only after the batch is genuinely ready; inspect the range placement in code/NVTX and report unaccounted time. Never trust a bucket name by itself.
- **Measure CUDA work with a completion proof.** `record_function`/NVTX ranges label attribution but do not prove that asynchronous CUDA work has finished. For a CUDA phase, use paired CUDA events on the relevant stream or synchronize at the phase boundary; never use a CPU timestamp around launches as the GPU duration.
- **Make a task census.** Record per-rank and global graphs/structures/atoms/edges, mechanism/property/structural/physical task counts, Python loop iterations, auxiliary forward calls, `autograd.grad` calls, FP64/CPU sections, and which heads share a backbone. A low GPU duty cycle with many short launches is CPU/launch driven until the trace disproves it.
- **Reuse shared work before adding kernels.** If several heads or derivatives consume the same backbone, test feature reuse and a batched/vectorized VJP or one combined `autograd.grad` call where `create_graph`, `retain_graph`, higher-order gradients, and weighting semantics permit it. Do not recompute a shared backbone merely because the heads are logged in different loss terms.
- **Vectorize ragged auxiliary work.** Replace per-structure Python loops and reduced reconstruction/occupation calculations with stacked, bucketed, segmented, or masked batched operations in increasing order of invasiveness. Measure padding and memory; preserve structure masks, reduction order where required, and per-structure physics.
- **Thin only with an explicit objective contract.** Stochastic thinning means an unselected auxiliary task skips its entire forward and gradient path; computing it and masking its loss is not thinning. Record selection probability, seed/rank semantics, loss reweighting, actual call counts, and quality gates. Treat a new thinning policy as an authorized objective/data-exposure experiment unless the repository already specifies it.
- **Make thinning DDP-safe.** Define whether masks are global or rank-local. For global masks, derive them from a shared step/global seed or broadcast them; for rank-local masks, record the resulting used-parameter sets and verify `find_unused_parameters`, `static_graph`, reducer buckets, and `no_sync()` behavior. Test zero-selected windows, normalize loss/gradients consistently across ranks, and record global norm clipping plus clip fraction.
- **Separate input preparation from compute.** Cache deterministic JSON parsing, topology, geometry, and graph construction only when the trace shows them on the critical path; prefetch CPU batches with the existing loader path and verify that synchronization and worker RSS do not worsen.
- **Account for the logical update.** Distinguish microbatch, gradient accumulation, optimizer update, and DDP reduction. Report work per logical update, not only graphs per nominal step. Changing a 554-graph mixture or effective batch is a data/objective experiment and needs its own quality and convergence comparison.
- **Treat checkpointing as a gradient-state contract.** Before using activation checkpointing with `autograd.grad`, higher-order gradients, DDP accumulation, or `no_sync()`, test the selected `use_reentrant` mode and `preserve_rng_state`. A resume point must state whether it is only an optimizer boundary or also supports mid-accumulation; if the latter, restore pending gradients, optimizer/scheduler/GradScaler, sampler cursor, and Python/CUDA RNG. Do not imply mid-window equivalence when it is unsupported.
- **Record contention as a confounder.** Capture CPU load, available memory, swap use, and worker/RSS pressure. Do not attribute a run under materially different host contention to a GPU optimization; repeat on a comparable host state or mark it inconclusive.

## Execute the evidence loop

1. **Inspect.** Read entry points, the real model/loss and input path, configuration, tests, performance documentation, and repository instructions. Check the installed runtime and current dirty state. Preserve unrelated user changes.
2. **Freeze comparability.** Define the scientific and measurement contract. Use identical code state, hardware, software, data panel, seed, workers, batch composition, and measured region for baseline and candidate.
3. **Capture the environment.** Record CPU model/affinity/threads/NUMA/storage; GPU model/index/capability; driver and CUDA/ROCm; PyTorch and kernel libraries; distributed settings; and relevant environment variables. Do not upgrade packages merely to obtain a newer feature.
4. **Baseline.** Keep eager execution and a known-good reference output as the reference. Warm libraries, allocators, caches, and any compiler separately. Reset peak memory after warmup. Measure repeated windows and report median plus dispersion. For ragged data, report graphs/s with the work-normalized atom/s or edge/s metric that reflects the bottleneck; crystal workloads also report crystals/s. Separate cold/compile/autotune time from steady state, and benchmark at Micro, Module, and End-to-end levels.
5. **Profile.** Begin with low-overhead ranges around data wait, H2D, forward, loss, backward, gradient transforms, optimizer, communication, logging, and checkpointing. Use scheduled `torch.profiler` after warmup. Escalate to Nsight Systems for CPU gaps, launches, synchronization, streams, or NCCL; use focused Nsight Compute only for a kernel question.
6. **Classify.** Assign the leading bottleneck to input/CPU, Python/launch/synchronization, compute/kernel, memory bandwidth/allocation, compilation/recompilation, or communication/imbalance. Rank hypotheses by evidence, expected impact, semantic risk, scope, and falsification test.
7. **Patch one cause.** Implement the smallest independently measurable change. For an experiment, use the smallest panel or sweep that can change the next decision; expand only to confirm a non-dominated candidate. Add a path assertion or test proving activation, such as actual autocast dtypes, pinning, selected kernel, graph-break count, absence of fallback, or communication overlap.
8. **Re-measure.** Run the identical benchmark and pass correctness gates before interpreting speed: output shape and finite values, reference numerical error, forward and gradient agreement, then equivariance/invariance, physical constraints, and task quality. For numerical-sensitive kernels or dtype changes, include representative stress cases (small and large systems, high coordination, short distances or near-singular cells when reachable, small/large tensor values, and the tested dtypes). Separate cold start, compilation/autotuning, cache population, and steady state. Restore only the candidate changes you own when they fail; never discard unrelated work.
9. **Decide.** Accept, reject, or mark inconclusive. Report negative results and uncertainty. Do not combine candidates until each has independent evidence.

## Route the investigation

- **Input/CPU:** Measure loader wait, transforms, collation, storage/cache state, worker RSS, thread oversubscription, affinity, NUMA locality, pinning, H2D overlap, ragged batch cost, and deterministic preprocessing reuse.
- **Python/launch/synchronization:** Search hot paths for `.item()`, `.cpu()`, tensor-to-Python conversion, host indexing, per-element or per-graph loops, synchronous logging, tiny kernels, and repeated allocations.
- **Compute/kernel/precision:** Verify exact shapes, masks, layouts, irreps, dtype, forward/backward path, backend dispatch, tensor-core eligibility, and explicit fallback behavior. Keep unstable linear algebra and contract-sensitive reductions in validated FP32/FP64 islands.
- **Memory:** Distinguish allocated, reserved, active, external, and host memory. Diagnose lifetime and fragmentation before checkpointing, allocator changes, or buffer reuse. Never call `empty_cache()` per step.
- **Compiler/CUDA Graphs:** Compile only evidence-backed stable regions. Record cold compile, graph breaks, recompiles, guards, cache behavior, coverage, and steady state. Require stable capture semantics; ragged batches are not automatic graph candidates.
- **Distributed:** Measure one GPU first, then per-rank input/compute/communication, overlap, imbalance, and scaling efficiency. Preserve global effective batch, `no_sync()` boundaries, used-parameter behavior, resume semantics, and checkpoint compatibility.

## Enforce claim boundaries

- A static review supports hypotheses, not measured gains.
- A microbenchmark supports only the isolated operation, not end-to-end training.
- GPU utilization, one timing window, or peak memory alone is supporting evidence only.
- Different hardware, drivers, runtimes, data panels, seeds, work quantities, or timing boundaries are not a valid before/after comparison.
- A configured feature is not active until a trace, assertion, or runtime probe proves the intended path and absence of silent fallback.
- A faster candidate that fails numerical, gradient, invariance, quality, distributed, or resume gates is rejected.

Use the repository's frozen acceptance rule. If none exists, choose a primary metric from the recorded `optimization_objective` (latency, throughput, memory, training/sampling time, GPU-hours, cost per accepted result, or time-to-quality), define its guardrails, and propose the thresholds before measuring. The default 5% median work-normalized throughput gate applies only when throughput is the objective; it is not a universal acceptance rule.

## Deliver the result

Lead with **accepted**, **rejected**, **inconclusive**, or **static-only**. Include:

1. baseline and candidate commands, environment, work unit, timing boundary, and fast-path proof;
2. profiler evidence, bottleneck classification, hypothesis, and exact change;
3. throughput, p50/p95 latency, allocated/reserved/external memory, and relevant CPU/data/GPU/communication metrics;
4. numerical, gradient, invariance, distributed, checkpoint, and scientific-quality results;
5. rejected hypotheses, limitations, exact files changed, rollback/disable instructions, and a reproducible command.

Use the report asset for substantial work. Keep small reviews concise, but retain the same claim boundaries.
