# Real fixed-intervention study

`EVOSPE_RT_V3_20261018` measures cold-inclusive inference jobs on real
torchvision and pinned TorchBench implementations. Randomly initialized
weights and materialized inputs establish numerical/system behavior, not
dataset accuracy or training time-to-quality.

## Contents

- [Observed interrupted execution](#observed-interrupted-execution)
- [Completed authorized GPU continuation](#completed-authorized-gpu-continuation)

Run on dbcloud in the dedicated worktree with the equivcompiler interpreter:

```bash
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python -u \
  -m experiments.real_intervention_v3.run_all \
  --study-config experiments/real_intervention_v3/config/STUDY_V3_MANIFEST.json \
  --results-root /home/workspace/lrh/RESULTS/evospe-rt-v3-20261018 \
  --mode full --resume-safe
```

The runner reads old loader tracebacks, requires three independent real
Inductor canaries, admits models using eager only, freezes architecture-family
splits, and measures development/calibration pairs. The predeclared fallback
is channels-last if the compiler platform cannot pass its canary. There is
one justified compiler-worker isolation attempt; shared packages/drivers are
never changed. Candidate timing does not select the fallback or model pool.

Both arms reconstruct identical seeded weights and inputs, use independent
processes and cache directories, execute exactly H primary forwards, and
compare against the saved admission reference. Cold cost includes conversion
or compilation and first execution. Ten additional steady calls are reported
separately. Sessions alternate arm order; all horizons/repeats remain in the
same architecture-family statistical cluster.

The final release requires at least 18 admitted independent families, five
valid independent calibration families, valid development effects, and a
frozen policy. Otherwise the runner completes exploratory analysis with
`UNDERPOWERED_NOT_FINAL` and preserves final candidate outcomes unopened.
Failures remain in the reliability denominator. Unknown subprocess attempts
stop resume rather than automatically executing again.

All raw specifications, start intents, results, full stdout/stderr and GPU
telemetry live under the result root. Caches and reference tensors remain
server-side; the evidence bundle includes unique recomputable records and an
index. No provider/API request is involved. Runtime caps are one GPU, one
worker, one native thread per process, 900 seconds per arm, 48 hours wall,
and 50 GiB within the new result root.

The first v3 attempt exposed a runner error when the disk monitor traversed a
Triton temporary directory concurrently with its removal. Its first S1 was
interrupted before a pair completed. `directory_bytes` now tolerates those
normal cache publications. A new execution root may use `--prior-results` to
retain the identical frozen cohort and completed qualification evidence after
this monitor-only correction. The loader checks that the model, numerical,
admission and canary code is unchanged. A started unknown arm stays unresolved
and is never executed again; other prespecified, unstarted arms continue. The
prior root is preserved and every inherited receipt retains its original
source identity.

## Observed interrupted execution

Measurement source `9d5a9ebe568ad3824882937a9ae078f61910954d` ran in
`evospe-rt-v3-20261018-monitor-corrected`. A server reboot interrupted the
study; its launcher exit code was not persisted. The last DLRM S1 has an
intent but no terminal receipt and must not be retried. Its original exit
remains unknown; the later authorized continuation is summarized below.

The frozen design contains 108 pairs across 18 families (10 development,
5 calibration, 3 final). There are 86 closed pairs: 16 valid, 52 unresolved,
11 numerical mismatches, 6 compile-execution errors, and 1 baseline failure.
The six compile errors concern cache permissions/missing files, not proven
model incompatibility. Original labels/receipts remain unchanged; derived
analysis separates infrastructure failures from candidate harm.

All 16 valid cold-inclusive effects are negative. They cover only three
development and two calibration families. This selected valid subset does
not establish universal compile harm. Three valid calibration families are
missing from the required five, final outcomes remain unopened, and policy
fitting was not reached. The scientific decision is
`UNDERPOWERED_NOT_FINAL`; neither learned ACRE efficacy nor threshold
superiority has been established. Never/Always scores are descriptive
controls on partial observed splits, not independent final validation.

The measurement revision's full CPU-only suite completed with 788 passed,
8 warnings, exit 0 (1227.61 seconds). Subsequent zero-GPU scoring corrections
keep unknown deployed harm/utility unknown and label the family-mean
per-pair seconds CI separately from total net seconds. Their revision and
targeted test receipt are recorded in the closeout bundle, separately from
the measurement source. Unknown attempt resource costs remain unknown;
observed parent session windows are not actual GPU busy time.

## Completed authorized GPU continuation

The continuation `evospe-rt-v3-20261019-gpu-resume` measured source
`49ee7d8293beaef8a4a67944eaaa236cba1c69e5` and exited 0. Its complete
CPU-only suite finished with 796 passed, 8 warnings, exit 0 (694.80 seconds).
GitHub Actions run 39143772397 also succeeded. Later analysis revisions are
recorded separately in `CLOSEOUT.json`; they do not change measurement code,
raw labels, the frozen manifest, or policy predictions.

Of 108 planned pairs, 90 development/calibration pair records closed and
18 final pairs remained unstarted. There are 19 valid pairs, 11 numerical
violations, 6 cache infrastructure errors, 1 baseline failure, and 53
unresolved pairs. Two interrupted candidate attempts remain unknown and
were never retried; they overlap the unresolved count, not extra samples.
The 19 valid effects span six families (3 development, 3 calibration), all
cold-negative. Calibration remains two valid families short of five:
`UNDERPOWERED_NOT_FINAL`, with no independent final efficacy claim.

The valid-subset context-median S0/S1 speedup is 0.0312933; the family-equal
mean log benefit is -3.599646 (95% family-bootstrap CI [-3.98265, -3.23516],
seed 101, 10,000 draws). This selected subset does not imply universal
compiler harm. The primary timer includes compilation and H forwards but
excludes model/input construction, process startup, hashing, correctness
checks and teardown: it is not full end-to-end job time. Separate steady
latencies exclude compilation and must not replace the cold objective.

Six historical PermissionError/FileNotFoundError cache failures retain their
raw labels; cause-aware analysis treats them as unknown infrastructure
outcomes rather than established model harm. Always has unknown full net
utility/risk; descriptive observed counts and bounds are retained. The
frozen threshold and supervised policies abstain. Both ACRE variants are
`ACRE_NOT_FITTED`, not demonstrations of learned gate efficacy. CIs for net
seconds describe a family-equal per-pair mean, not the total net sum.

The old GPU supplied 16 valid pairs; GPU3 supplied three additional valid
DLRM pairs after three independent real compiler canaries passed. Reports
stratify by observed physical UUID; same model and GPU type do not make
these independent model families. Before future authorized GPU launches,
scan all GPUs, prefer idle GPU3 then an idle same-model card, fix the chosen
UUID for the execution, and never migrate within an S0/S1 pair or kill
another user's process.

All prior roots are preserved. Complete CPU cost for the two unknown arms
and actual GPU busy seconds remain unknown, not zero. The historical
5619-second parent window exceeded the declared 900-second arm cap; its
cause is unestablished and the cap cannot be claimed satisfied. No provider
calls, new measurements, policy refitting, or additional models are part of
the zero-GPU closeout. The top-level report and hash-closed evidence bundle
are delivered under `E:/DATA/evospe-rt-v3-20261019-gpu-resume`.
