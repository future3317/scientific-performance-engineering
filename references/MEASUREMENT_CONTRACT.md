# Measurement contract

Fill this before modifying performance-sensitive code.

## Identity

- Repository / base revision / dirty state:
- Benchmark harness hash / candidate patch hash / declared change set:
- Entry command and config:
- Hardware and device index:
- Selected GPU index/UUID and logical-to-physical mapping:
- Driver / CUDA / PyTorch / dependent kernel libraries:
- Data path, storage type, split, manifest/hash:
- Host contention snapshot: load average / available memory / swap / worker RSS:
- Environment privacy mode: redacted by default; sensitive host metadata explicitly enabled?

## Runtime preflight

- Compatibility status and rejected combinations (compile/checkpoint/DDP/FSDP/CUDA Graphs/custom op/higher-order autograd):
- Runtime topology: model construction order, rank/device mapping, NUMA/affinity, PCIe/NVLink/NIC, shared memory, threads, allocator:

## Fixed scientific contract

- Model/output family:
- Initialization checkpoint / stage / EMA provenance:
- Anchor or proximal reference: frozen-copy identity, parameter scope, coefficient, and matched-ablation status:
- Loss and component weights:
- Constraint stack: shared-LR scaling / gradient projection / norm caps / task weights:
- Invariance/equivariance/physical constraints:
- Precision policy and FP32/FP64 islands:
- Effective batch and accumulation:
- Optimizer/LR/scheduler/clipping:
- Stochastic thinning: enabled? selection probability, global/rank-local mask, seed/broadcast, loss reweighting, zero-selected behavior, used-parameter/`find_unused_parameters`/`static_graph` contract, global-norm clip fraction:
- Auxiliary cadence: per-task update interval, mutually exclusive schedule/overlap, expected-frequency proof, optimizer/EMA boundary:
- Activation checkpoint/autograd contract: `use_reentrant`, `preserve_rng_state`, higher-order/`autograd.grad` support, DDP `no_sync()`/accumulation support:
- Data membership/order/augmentation:
- Supervision coverage: eligible rows versus packed rows by task/family, leakage policy, fixed-condition/composition-shuffle controls:
- Seed(s), checkpoint/resume semantics and supported boundary (optimizer-only or mid-accumulation):
- Validation/sampling quality gates:
- Numerical tolerances:

## Work definition

- Optimization objective: latency / throughput / memory / training time / sampling time / GPU-hours / cost per accepted result:
- Unit: graphs / atoms / edges / samples / transitions:
- Microbatch and logical batch:
- Logical update definition and accumulation window:
- Task composition: structures / graphs / atoms / edges / mechanism / property / structural / physical counts:
- Steps included:
- Includes data loading? transfer? optimizer? logging? checkpoint? validation?:
- Warmup steps / measured steps / repetitions:
- Staged horizon / early-stop checkpoints / validation Pareto rule:
- Benchmark levels completed: micro / module / end-to-end:
- Timing bucket definition and unaccounted step time:
- CUDA timing proof: paired events/explicit synchronization and stream:
- Timing buckets: each bucket clock/stream/completion proof/p50_ms; bucket sum versus end-to-end step; unaccounted ratio:
- Logical-update DAG: fetch → preprocessing → transfer → forward → derivatives → backward → transforms → communication → update → lifecycle work:
- Synchronization census: event / count / required-removable-amortizable-overlappable / evidence:
- Cache contract and state: key components, cold/warm/disabled, hit/miss evidence:
- H2D proof: pinned state, non-blocking flag, copy stream, source lifetime, consumer dependency, overlap timeline:

## Baseline metrics

- Throughput:
- Raw independent run/window metrics and randomized/A-B run order:
- Median / IQR / MAD / bootstrap confidence interval / noise floor:
- Step time p50 / p95:
- Peak allocated / reserved / external memory:
- Data wait / H2D / forward / loss / backward / optimizer:
- Optimizer/EMA/SWA/scheduler/clipping/metrics/validation/checkpoint time and temporary/state/pinned memory:
- Steady-state train-step versus cadence-amortized throughput and time-to-quality:
- Mechanism / property / structural / physical / `autograd.grad` bucket times:
- Auxiliary forward calls / `autograd.grad` calls / skipped-task calls:
- GPU busy/SM/memory indicators (supporting evidence only):
- DDP communication / overlap / scaling efficiency:
- Validation and numerical checks:

## Candidate acceptance

- Hypothesis card: measured bottleneck, one changed lever, expected movement, risk, falsification test:
- Reference implementation/output retained at:
- Primary objective metric and threshold (objective-specific):
- Required quality gates (recorded in `acceptance.required_quality_gates`, never CLI-only):
- Guardrails: p95 latency / memory / host contention / quality:
- Quality/non-inferiority gate:
- Required tests:
- Rollback condition:

## Experiment ledger

- Status: accepted / rejected / inconclusive / algorithmic_experiment
- Candidate levers changed in this entry:
- Cold/compile/autotune versus steady-state result:
- Correctness order: shape/finite → numerical → gradient → physics/equivariance → task quality:
- Numerical divergence localization: eager/reference versus candidate dispatched op, first divergent op, dtype/layout/backend, focused reproduction:
- Resume cursor: sampler/dataloader/shard/order/augmentation RNG and EMA/SWA/compiler rebuild semantics:
- Rejected reason or retry evidence:
