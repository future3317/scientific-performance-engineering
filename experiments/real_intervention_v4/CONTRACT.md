# v4 measurement and evidence contract

## Identity

- Primary study: `EVOSPE_REAL_INTERVENTION_V4_CHANNELS_LAST_20261020_REFERENCE_WRITEFIX`.
- Intervention: `CHANNELS_LAST_EAGER_LAYOUT_V1`.
- S0: eager, evaluation-mode FP32 forward in the model's original layout.
- S1: same construction, weights, input, precision, and eager forward; the
  channels-last conversion/preparation cost is included in the cold task.
- `torch.compile`, AMP, and TF32 are not primary-study levers. Compile/cache
  amortization has a separate study identity and cannot substitute for S1.
- v3 manifests, pair receipts, sealed/final outcomes, and UNKNOWN arms are
  read-only. The v3 failure audit is a new derived table, never a rewrite of
  historical evidence.

## Population and split

The full registered model/source roster is mapped to independent architecture
lineages and split before qualification. The frozen algorithm and seed live in
`config/SPLIT_FREEZE.json`. Qualification failures, unsupported operators,
and unavailable sources remain in the registered split denominator. Batch,
shape, horizon, seeds, and repeat sessions are within-family contexts, not new
independent families. No timing-based family exclusion or re-splitting is
allowed.

The primary confirmatory minima are 23 registered and eligible families: 8
development, 10 calibration, and 5 sealed final. The taskbook permits
baseline-only feasibility to register these gates before any treatment-effect
observation. If the real source universe or eligible roster does not satisfy
them, the report remains exploratory/underpowered and the final stays sealed.
The target confidence-interval precision is reported as a target, not assumed
to have been achieved.

## Paired execution and validity

Each planned pair contains two AB/BA sessions, with each session's S0 and S1
executed on the same pinned physical GPU UUID. A missing or mismatched UUID,
runtime identity, input digest, weight digest, source commit, or configuration
digest invalidates comparability. Qualification records model-level numerical
correctness and actual CUDA kernel-path exposure separately from timed work.
Profiling is not included in timing.

Every arm has a unique cold-cache scope and one worker. Its host-supervised
deadline covers cache setup, worker execution, and bounded cleanup. Intent,
started, receipt, and terminal records define the lifecycle. A started arm
without a verified terminal receipt is UNKNOWN, remains in the denominator,
and is not automatically replayed. Infrastructure failures, candidate
correctness failures, baseline failures, noise, and unknown outcomes remain
separate categories. Known numerical harm cannot be averaged away by speed.

## Statistical and ACRE boundary

The primary estimand is the family-equal mean of context-mean
`log(T_S0/T_S1)`. Session/context repeats are averaged within family before
cross-family intervals. Report raw seconds and log effects, bootstrap
uncertainty, positive/negative/zero families, missingness bounds, and
leave-one-family-out sensitivity.

Development may propose an ACRE rule from frozen pre-execution features.
Calibration evidence is independent and must pass the canonical ACRE
promotion, governance, support-regime, and scope-binding gates before routing
can be considered fitted. The canonical `p_min=0.8` and statistical budget are
not weakened. Baseline-only feasibility records the number of independent
families required by that gate. Without a canonical certificate, both ACRE
policies are `ACRE_NOT_FITTED`; abstention is not efficacy.

An ACRE certificate is necessary for ACRE deployment/comparison, but it is not
a requirement for the primary channels-last final effect or the frozen simple
comparators. The primary final opens only after its own family minima,
terminal development/calibration pairs, shakedown, and pre-final policy freeze
pass. Final outcomes are never read by development, calibration, or policy
training.

## Resource and evidence boundary

Execution is only supported on `dbcloud`, from the canonical repository, with
`/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python`. One selected
GPU, one worker, one CPU thread, 900-second arm deadline, bounded cleanup, a
48-hour study wall cap, and a 50-GiB results-root cap are frozen in the config.
The report distinguishes host wall time, child CPU time, sampled GPU activity,
storage, and unknown resource totals; missing telemetry is never filled with
zero.

The bundle includes required derived records plus raw attempt receipts and
lifecycle events with unique archive paths and member hashes. It excludes
cache trees, weights, and secrets. The source commit, environment, selected
GPU UUID, frozen manifest, and final exit record remain explicit.
