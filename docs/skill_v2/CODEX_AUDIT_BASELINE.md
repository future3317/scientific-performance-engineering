# Research implementation baseline — 2026-10-15

Source plan: `E:/Downloads/EvoSPE_Codex_Research_Action_Plan_20261015.md`, read in full.

The actual local and canonical server code both started at
`49b267e45222d49639636bba324dd9e80a3d1255`. Local branch was
`codex/hosted-pilot-public-contract-eligibility`; server HEAD was detached.
New development branch: `codex/skill-evolution-study`, from the same revision.
Server `git pull --rebase origin codex/hosted-pilot-public-contract-eligibility`
reported up to date. No original changes were discarded.

Local edits predating this task: AGENTS.md; three campaign/formal/method tests;
budget-aware completion report; pilot surface script. Untracked boundary analysis,
LTT, upstream/SkillsBench helpers, tests, September design documents and `.tmp/`
also predate this work. These are excluded from this task's commits. Server
canonical checkout was clean. The separate dirty `external-real-prospective`
worktree remains untouched.

## Environment and evidence

Every runtime command uses `ssh dbcloud`, canonical code directory
`/home/workspace/lrh/CODE/scientific-performance-engineering`, interpreter
`/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python`.
Unit tests and read-only audit use CPU only, `CUDA_VISIBLE_DEVICES=''`, one
thread. No GPU is allocated to these commands. Six shared 4090 GPUs were visible;
GPU 1 was busy during baseline inspection. No provider credentials were read.

Current checkout lacks `docs/experiment_registry.md`, `docs/paper_claim_matrix.md`
and `docs/EVOSPE_EVIDENCE_FREEZE_20260905.json`, locally and on server. The exact
files exist in server Git history at `6282c0d`. They are historical references,
not reconstructed current files. The freeze tag exists. Its manifest reports
benchmark c5eb494d, controller 097a36a0, paper cafee6a4 and historical formal
dce4d194. See the historical registry for the original manifest SHA256
`b978c74f52b0f10a1b595d84c26c84f5ba5ab8278c161fea20d7bf80ad404b0c`.

The real historical campaign is under
`/home/workspace/lrh/RESULTS/formal-sealed-dce4d19-controller-db0d25d-runner-db0d25d-executordigestfix-20260901`.
The archived paper audit is `E:/PAPER/EvoSPE/results/formal_failure_audit.json`.
Both are read-only inputs. New audit/study outputs use a new root outside them.

## Compatibility assessment

The requested existing ACRE, governance, state store, mutation journal, ALFWorld
runner/synthesis, and formal campaign modules exist. Production routing already
requires canonical stable state; the offline predicate-fit discrepancy is not
evidence of a production promotion bypass. Preserve legacy runner and results.
New skill structures adapt to RuleSpec and existing store/journal rather than
creating a second canonical registry. Default CEGIS behavior remains compatible;
new bounded search must report incomplete enumeration explicitly.

The plan's proposed filenames are adjusted to a single `core/skills/` package
and a single `experiments/skill_evolution/` entry point. New work is experimental
code, not evidence of efficacy. The archived paper is unchanged.

## New execution boundary

User authorized implementation and experiments. The plan additionally requires
fresh revisions, preregistration and an approved resource/budget contract before
paid or sealed execution. Provider-free audit, mock tests and synthetic study
can proceed now. Real rollout spending limits are not supplied by the plan;
prepare a concrete bounded phase-I configuration before requesting that missing
budget. Never launch an old formal/sealed campaign.
