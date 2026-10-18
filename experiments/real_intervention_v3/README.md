# Real fixed-intervention study

`EVOSPE_RT_V3_20261018` measures cold-inclusive inference jobs on real
torchvision and pinned TorchBench implementations. Randomly initialized
weights and materialized inputs establish numerical/system behavior, not
dataset accuracy or training time-to-quality.

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
