# Project execution environment

## Server-only execution rule

The server is the only supported runtime for this project. Run every project
test, validator, CLI command, benchmark, calibration, qualification, timing
probe, GPU job, and evidence-producing command on `dbcloud` inside the
`equivcompiler` Conda environment. Do not run these commands from the local
Windows checkout, local Python, or any local Conda environment.

All benchmark, calibration, qualification, GPU, and evidence-producing runs
for this repository must execute on the server through `ssh dbcloud`, from
`/home/workspace/lrh/CODE/scientific-performance-engineering`, using the
project environment:

```bash
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python
```

Local access is limited to editing, reading, and static inspection. Local
outputs must never be used as runtime evidence or mixed into server results.
Before every server run, verify the server repository, commit, clean worktree,
interpreter path, and GPU/resource contract. Do not use Conda `base`.

The canonical server workflow is:

```bash
ssh dbcloud
cd /home/workspace/lrh/CODE/scientific-performance-engineering
git pull --rebase
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python -m pytest -q
```

Qualification and official calibration must use the declared server resource
contract and the frozen `SHARED_GPU_MATCHED_SESSION_V1` policy. Shared compute
is allowed with observed co-tenancy telemetry; selected GPU UUID/runtime
identity, session/provenance completeness, executor lifecycle, timeout, OOM,
and paired-cell completeness remain hard validity invariants.

## Public treatment revision boundary

The active experimental revision is `public-acre-treatment-20260915-v1`.
Its provider-backed artifacts are revisioned separately from historical formal
and provider-free evidence. The completed public fixed-skill audit covers four
SkillGen-compatible benchmarks under one pinned DeepSeek configuration; it is an
external boundary diagnostic, not SkillGen-generation efficacy. The completed
24-task ScienceWorld A/C/D run is a bounded treatment-calibration result, not a
sealed cross-benchmark or multi-model efficacy study.

Do not overwrite historical result roots or reuse public outcomes under a new
claim boundary. Any additional provider experiment must create a new revision,
record the exact model/benchmark/task-order configuration and provider-call
ledger before analysis, and preserve observation completeness and paired
evaluation semantics.
