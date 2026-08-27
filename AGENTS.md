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
