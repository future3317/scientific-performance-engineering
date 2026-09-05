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

## Current repository and evidence boundary

The canonical benchmark branch is server-side `main`; the evidence freeze is
anchored by the cross-repository tag `EVOSPE_EVIDENCE_FREEZE_20260905`.
Documentation-only commits after that tag do not change frozen evidence. The authoritative mapping of
benchmark, controller, paper, historical-formal, provider-free evidence, and
paper-PDF identities is
`docs/EVOSPE_EVIDENCE_FREEZE_20260905.json` on the server. Its manifest SHA256
is recorded in `docs/experiment_registry.md` and
`docs/paper_claim_matrix.md`.

The current freeze supports provider-free mechanism/resource claims only:

```text
HOSTED_EFFICACY_STATUS = UNSUPPORTED
HOSTED_FEASIBILITY_STATUS = INSUFFICIENT
SEALED_C_D_STATUS = NOT_RUN
RESULTS_CLAIMED = FALSE
```

Do not start provider, hosted, calibration, C/D, formal, sealed, or unblinded
runs from this checkout. The historical formal revision and all server
`RESULTS` are immutable evidence; do not overwrite, move, reinterpret, or
delete them. Any future experiment requires a new benchmark/controller
revision and a new evidence identity before execution.

## Local checkout hygiene

The Windows checkout is for editing and static inspection only. Preserve
uncommitted user changes. Personal analysis material belongs under the local
`local_analysis/` directory and is excluded through `.git/info/exclude`; it is
private working material and must not be staged, pushed, or mixed into server
evidence. Do not change the shared `.gitignore` merely to hide local files.
