# EvoSPE V6 prospective systems pilot

The server `dbcloud`, canonical repository, and `equivcompiler` interpreter are
the only runtime. V3--V5 results and source identities remain historical evidence.
V6 does not resume a V5 identity or reinterpret a failed V5 session as valid.

V5 source `d8980253a91a52fa9491fa613eb4c20a1d16e804` samples NVML **before**
reading the ready file. All six attribution-failure receipts have ready times
104--130ms after that sample's start. V6 reads readiness first and samples afterward.
CUDA execution proof and foreign co-tenancy visibility remain separate fields.
An empty process table does not prove an uncontended GPU. Without lab-wide
reservation, comparable observations retain `EXPOSURE_UNCERTAIN` and are analyzed
only as a prospective exploratory pilot.

V5 already divided its ten-call manual timer by ten. The historical400x mismatch
is not evidence of missing normalization. The V6 reproducer retains call-level
times, compiler/guard counters, model attention-cache state and complete benchmark
raw times/counts; explicit warm-up and equivalent invocation contexts precede a
steady cross-check. Cold prefill cost remains in the deployment ledger.

The main study measures compile/cache amortization on the V5 correctness-qualified
conditional population. Family allocation uses only eligibility and prior exposure,
with deterministic6/6/4 for16 eligible families. The independent secondary BF16
layout study uses6/6/5 for17. Both arms retain the V5 numerical tolerance; no outcome
sign gates fixed-intervention final. Every new arm rechecks weights/inputs/reference
and numerical outputs. Family, not horizon/session/reuse scenario, is independent.

Deployment costs include process startup, model/input construction, cache-copy and
verification, H forwards, numerical/steady probes and result publication. They are
explicitly instrumented deployment costs, not bare GPU kernel time. Each cache
prefill is paid once, each reload executes in a new process with a private copy,
and persistent hits must be observed. R1 andR10 use actual jobs; R100 is projection.
Cold, persistent, in-worker job and steady measurements are distinct estimands.

`NEVER`, `ALWAYS`, `H_ONLY`, a simple stump, and `ACRE_SYS_RESEARCH` freeze before
final. The latter is an uncertified research ridge router with calibration-only
threshold selection and OOD fallback NEVER. The original ACRE certificate retains
p_min=.8, delta_mix=.0125 and the necessary42 calibration-group lower bound; its
infeasibility never substitutes for the separate pilot final decision.

One entry point, after the current-source test receipt and real platform validation:

```bash
cd /home/workspace/lrh/CODE/scientific-performance-engineering
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python -u \
  -m experiments.real_intervention_v6.run_all \
  --config experiments/real_intervention_v6/config/V6_MAIN_COMPILE.json \
  --tests-receipt <current-source-validation-root>/CPU_TESTS.json \
  --results-root /home/workspace/lrh/RESULTS/evospe-v6-compile-main-20261023 \
  --resume-safe
```

The full stage order is incident audit -> safe idle admission -> real CUDA sentinel,
prefill timer diagnosis and CPU/CUDA units -> cohort/manifest freeze -> actual CUDA
fault cleanup ->12 shakedown slots (minimum11 comparable, no known unsafe result)
-> development -> calibration -> policies/predictions -> pilot final -> report/bundle.
Observed foreign processes exclude an attempt. Identities with intent/started and
no terminal are permanent UNKNOWN and are never rerun. Same code/config/env/device
is required on resume. A study has24h wall,600s/arm,1800s idle wait, one GPU worker,
one native CPU thread and50GiB storage cap. No provider or paid GPU service is used.

Recompute the frozen family analysis on the server:

```bash
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python -m \
  experiments.real_intervention_v6.report --root <V6-study-root>
```

The terminal export includes full denominators, separate actual/projection rows,
policy comparisons and routing cost, missingness bounds, raw attempt index, source,
and a member-verified sanitized bundle. No weight or compiler-cache payload enters
Git or the export. Historical V5 readiness remains in the paper; V6 effects can be
added only after legitimate prospective final analysis.

## 2026-10-23 terminal disposition

The frozen `e9a261cff27acf7e1c9c90da2c0087dde6d894eb` campaign closed at
11:27:46 UTC: primary B exhausted its one-hour idle admission; secondary A was
`NOT_RUN_RESOURCE_CONSTRAINT`. All 121 six-GPU scans (726 device observations)
showed foreign compute. No CUDA worker, engineering slot, formal pair, cohort
manifest, policy freeze or final release was reached. This is resource NO-GO,
not a measured negative effect or GPU acceptance. The 16/17-family allocations
remain design-only; no retrospective freeze is asserted.

The source-level stale pre-ready attribution defect is established. The actual
Attention CPU probe supports early cache-state recompilation, but the historical
GPU prefill discrepancy remains unresolved. Targeted V6/V5 CPU tests passed 100;
the expanded older supervisor run retained five wall-ceiling failures.

Original receipts remain under `/home/workspace/lrh/RESULTS/evospe-v6-20261023`.
The independent audited closeout is
`/home/workspace/lrh/RESULTS/evospe-v6-20261023-closeout`, delivered to
`E:\DATA\EvoSPE_v6_20261023`. Further execution needs a permitted GPU window,
not dollar approval; this terminal campaign must not recursively restart.
