# EvoSPE Phase2 execution report

Source: `EvoSPE_Codex_Phase2_Real_Efficacy_Taskbook_20261016.md`, especially
sections 0 and 12. This is a new study, not a revision of historical results.

## 1. Repository / commit / evidence identity

Baseline: `1796fa36360cf0fbe0a175caec3b4868d4b55e2f`, branch
`codex/skill-evolution-study`, matching local/server/origin on entry.
The canonical server checkout was clean; pre-existing local modified and
untracked files were preserved and excluded from these changes.

Runtime: `/home/workspace/lrh/CODE/scientific-performance-engineering`,
`/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python`, via `dbcloud`.
CPU only, CUDA hidden; OMP/MKL/OpenBLAS/NumExpr/Inductor/SPE intra-op/inter-op
thread counts all one. No GPU job or credential access is needed here.

New evidence root:
`/home/workspace/lrh/RESULTS/skill-evolution-phase2-20261016`.
The historical formal root and October 15 evidence are read-only inputs.

## 2. Implementation by work package

Implementation and verification are in progress. Final accepted paths, code
revision and server logs will be recorded here after execution, not inferred
from a subagent's implementation report.

## 3. Real Agent viability

`NOT_RUN: APPROVAL_REQUIRED`. No new provider calls are authorized by the
taskbook. Native teacher diagnostics, if successful, are environment/scorer
controls only. They cannot establish autonomous Agent viability.

Stage B is stateful/no-Skill first; H1 legacy-current vs stateful remains
untested. Zero real wins stops paid Skill optimization. One win remains
insufficient to start C. Two different template-group wins plus independent
failure evidence permits requesting C approval, not automatic execution.

## 4. Model-driven Skill evolution

`REAL_OPTIMIZATION_NOT_RUN`. No real proposal, refinement or paired inner/outer
rollout has been executed in Phase2. Code/mocks cannot supply these outcomes.
The model-driven path must preserve source task/template/trace identities,
frozen candidates, independent outer tasks and rejected versions.

## 5. Fixed-Skill gate and held-out

`NOT_RUN`. No new SkillBank has real-effect qualification. No final held-out
outcome has been opened. Content improvement and routing improvement are
different contrasts; all fixed-content gates must share the same matched
base/Skill outcomes. Zero coverage means zero incremental success and
undefined conditional harm, not zero conditional risk.

## 6. Total cost and budgets

Phase2 new provider calls: **0**. Input/completion/reasoning/cache usage from
provider experiments: **not measured**, not fabricated zeros. Provider spend
incurred by this phase: none. This does not estimate Codex/subagent costs.

Approval has not been supplied for an exact run/manifest with a hard USD cap.
The old two connectivity calls do not extend authorization. The completion
cap is not calibrated by those calls. Cache savings are not assumed.
The disabled approval example is not permission to spend; prices must be
attributed to a checked official snapshot before live execution.

## 7. Test integrity and scientific claim scope

The entry full-suite run is pinned to `1796fa3`; raw command, environment,
stdout/stderr, start/end and exit code are under `baseline-tests/`.
Result: **497 passed, 8 warnings, 681.60 seconds, exit 0**. The separately
saved exit code is authoritative. This verifies the entry baseline, not the
new Phase2 changes. The previous 473-test result remains specific to
`f63ec84`. No synthetic qualification campaign is rerun.

No new Agent-success, Skill-benefit or ACRE-efficacy claim is supported yet.
The manuscript and its numerical evidence are unchanged.

## 8. Artifacts and no-overwrite proof

Final log references, immutable source comparison, split/teacher artifacts
and local review copies are pending execution. No current run is called
passed, no unknown is converted to failure, and no engineering completion is
substituted for a real efficacy result.
