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

## Fixed scientific contract

- Model/output family:
- Loss and component weights:
- Invariance/equivariance/physical constraints:
- Precision policy and FP32/FP64 islands:
- Effective batch and accumulation:
- Optimizer/LR/scheduler/clipping:
- Stochastic thinning: enabled? selection probability, global/rank-local mask, seed/broadcast, loss reweighting, zero-selected behavior, used-parameter/`find_unused_parameters`/`static_graph` contract, global-norm clip fraction:
- Activation checkpoint/autograd contract: `use_reentrant`, `preserve_rng_state`, higher-order/`autograd.grad` support, DDP `no_sync()`/accumulation support:
- Data membership/order/augmentation:
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
- Benchmark levels completed: micro / module / end-to-end:
- Timing bucket definition and unaccounted step time:
- CUDA timing proof: paired events/explicit synchronization and stream:
- Timing buckets: each bucket clock/stream/completion proof/p50_ms; bucket sum versus end-to-end step; unaccounted ratio:

## Baseline metrics

- Throughput:
- Raw independent run/window metrics and randomized/A-B run order:
- Median / IQR / MAD / bootstrap confidence interval / noise floor:
- Step time p50 / p95:
- Peak allocated / reserved / external memory:
- Data wait / H2D / forward / loss / backward / optimizer:
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
- Rejected reason or retry evidence:
