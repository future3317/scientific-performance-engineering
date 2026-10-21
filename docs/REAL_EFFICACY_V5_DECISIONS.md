# EvoSPE V5 independent systems studies

The 2026-10-21 user-approved research reset supersedes the V4 continuation
plan. V3/V4 roots, labels, final locks and frozen core ACRE defaults are
read-only. This study does not change hosted/provider or historical C/D claims.

The new implementation lives in `experiments/real_intervention_v5`. Study A
compares contiguous against channels-last with BF16 autocast in both arms;
layout preparation is included in the primary job time. Study B separately
compares eager, empty-cache cold compile, and a new process loading artifacts
from a measured prefill process. Prefill cost is charged at R=1/10/100.
Neither random-input study measures scientific task accuracy.

The mathematical certificate audit imports the original StatisticalBudget and
ReplaySequentialCertificate. At p_min=.8 and delta_mix=.0125, the all-success
mixture needs 36 independent successes. The historical support-holdout path
requires another support family and five non-support families: at least 42
calibration lineages. These are necessary conditions, never certification.
Repeated contexts and sessions do not add independent families. No thresholds
in `core/acre` are changed. The registered calibration split is below this
lower bound; ORIGINAL_ACRE therefore remains NOT_FITTED.

Lineage splitting precedes qualification. ResNet/ResNeXt/wide/SE wrappers,
RegNet X/Y variants, and ViT training variants are not extra families. The
published design basis for every nominal lineage is retained in the registry;
architectural relatedness remains a generalization limitation. Aliases cannot
replace their frozen representative after qualification. The correctness
contract requires exact A/A repeatability, matching tree/shape/dtype, finite
outputs and frozen BF16 S0/S1 tolerances (rtol=.02, atol=.002, relative L2<=.02). FP32 deviation
is separately reported, not represented as accuracy.

One GPU worker with one CPU thread runs at a time. Each complete AB/BA window
holds a logical flock reservation, waits for no external compute processes,
memory <=512 MiB and zero device utilization, and records NVML capability and
process trees. No GPU mode, driver, shared package or other user's process is
changed. An explicit GO/ACK handshake holds each CUDA owner alive across the
measurement samples. Missing process attribution is distinct from confirmed
competition. A session has a hard parent deadline and process-group cleanup;
started identities never receive another attempt. An independent host watchdog
also bounds a blocked monitoring call; an emergency timeout ends the controller
with an UNKNOWN/NOT_STARTED receipt, rather than continuing on a broken platform.

Engineering qualification requires >=11 genuinely matched and correct pairs
out of 12, plus controlled timeout, process interruption and same-card
contention faults. Classification tests do not substitute for this GPU gate.
Actual stuck GPU-kernel recovery is not claimed by process fault injection.

Each study has a 24-hour wall cap, a one-hour initial resource wait, a 50 GiB
storage cap, and no implicit retry. Primary independent final release needs
at least five qualified lineages in each frozen split, five observed valid
development and calibration lineages, closed receipts and policy predictions
frozen before final access. It does not depend on effect sign or ACRE success.
At small n, results are explicitly underpowered independent estimates. Power
sensitivity is calculated before GPU work; it is not a promise of 80% power.

Entry point on the canonical clean server in equivcompiler:

```bash
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python -u \
  -m experiments.real_intervention_v5.campaign \
  --results-parent /home/workspace/lrh/RESULTS/evospe-v5-20261021
```

The terminal report exposes registered, baseline-valid, candidate-correct,
paired-valid and missing denominators; dev/cal/final effects; unchanged
certificate feasibility; and CPU/wall/storage and GPU accounting limits.
The exported evidence bundle contains sanitized receipts, unique members,
and SHA256/size closure, excluding weights and compiler cache binaries.

The first preparation identity (`63fa92c`, root `evospe-v5-20261021`) started
zero model arms because this server's NVML Python binding lacks the optional
per-process utilization function. Those receipts and bundles are retained.
The OWNERATTRIBUTION identities record that optional capability explicitly as
NOT_SUPPORTED while still requiring complete running-process tables, UUID,
device memory/utilization and the GO/ACK owner lifecycle. They keep the GPU
selected before any model timing (GPU-520e1aa6-5eae-4984-b76a-724aecac1aae).
No missing required attribution is reclassified as matched.
