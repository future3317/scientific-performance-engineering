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
Runtime implementation checkpoint: `d03cf90`. A newer documentation commit
does not relabel any test as having run on that commit.

## 2. Implementation by work package

| Package | Delivered implementation | Execution status |
|---|---|---|
| A | `split_manifest.py`, extended `readiness.py`, `native_verifier.py`; semantic-cluster separation, exposure ledger, official scorer | Server-tested; actual partition and native controls produced |
| B | Stateful runner, `real_study.py`, `ApprovedProviderBudget`; visible state/native grammar and scope/expiry/call/token/USD/wall guards | READY_FOR_APPROVAL; real model episodes NOT_RUN |
| C | `core/skills/proposal.py`, contract/evolution extensions, `real_learning.py`; JSON proposal/patch, matched inner/outer, rejected proposals and outer rollback | Interfaces and regression tested; real optimization NOT_RUN |
| D | `core/skills/pack.py`, `evaluation.py`; standard pack/readback, canonical routing decisions, paired analysis and CSV/JSON/report output | Offline contracts tested; real content/gating benefit NOT_RUN |

Three coherent Luna packages covered A, C/pack and D. Main-agent work covered
experimental judgment, budget/approval, Agent integration, cross-package review
and central server acceptance. No new user-owned task, PR, registry, benchmark
or provider framework was created.

Actual split: **1,383 semantic groups**, **462 train_dev / 461 train_inner /
460 train_outer**. The historical **88 template directories** and **3 old
native-probe episodes** are recorded/excluded. Unknown historical or selected
task identities: **0**. No final outcome/expert plan was opened. Final held-out
IDs are not selected yet, rather than invented.

Grouping uses train-only typed target relations, floor plan and layout, not
task-description wording or seed identity. These fields do not enter policy
prompts. The first actual pass exposed a loader bug: nine demonstrations had
`traj_data.json` but no compiled game. Metadata recovery must not require a
compiled game. The corrected CLI resolves them; the blocked artifact is kept.

## 3. Real Agent viability

`NOT_RUN: APPROVAL_REQUIRED`. No new provider calls are authorized by the
taskbook. Native teacher diagnostics, if successful, are environment/scorer
controls only. They cannot establish autonomous Agent viability.

Three already-exposed train tasks completed with the official handcoded teacher:
**9 / 22 / 12 steps**, all `won=True`, provider calls **0**. See
`teacher-47482ef-{0,1,2}/results.json`. The SSH shell wrapper returned 1 after
the three native commands because of a trailing Windows CR; this is not
presented as an exit-0 run. The official successful outcome receipts remain
environment controls. Later shell commands use explicit exit/status capture.

Saved native observations exposed actual Agent bugs before paid work:
`heat some mug` became target `some mug`, and native `move ... to ...` plus
`to the ...` feedback were unrecognized. These are fixed against public grammar
regressions. Inventory/transformation facts require visible acknowledgment of
the goal object. Private reasoning/expert-plan fields never become actions.

Proposed B: three independent train_inner source tasks, clean/heat/cool, no
Skill, unchanged 60-action horizon, at most 180 requests. None started. Source
tasks can never serve as their own validation pairs. This is engineering
viability, not an efficacy sample-size design.

Stage B is stateful/no-Skill first; H1 legacy-current vs stateful remains
untested. Zero real wins stops paid Skill optimization. One win remains
insufficient to start C. Two different template-group wins plus independent
failure evidence permits requesting C approval, not automatic execution.

## 4. Model-driven Skill evolution

`REAL_OPTIMIZATION_NOT_RUN`. No real proposal, refinement or paired inner/outer
rollout has been executed in Phase2. Code/mocks cannot supply these outcomes.
The model-driven path must preserve source task/template/trace identities,
frozen candidates, independent outer tasks and rejected versions.

The initial/patch APIs use the same metered final-content client. The real CLI
currently executes one bounded patch against a supplied parent, then fresh
inner pairs and, only after inner acceptance, outer pairs. It does not collect
a usable same-family contrast automatically when B lacks one. That next source
plan needs actual B evidence and separate approval, not fabricated data.

Both raw arm trajectories are persisted. Source digests, native task paths,
semantic groups and fixed Agent/configuration identities are checked. Outer
rejection is journaled and retires the candidate; outer undecided stays
undecided. Both stages must accept before review/export; no automatic canonical
promotion occurs. B-only approval cannot authorize a Skill proposal.

## 5. Fixed-Skill gate and held-out

`NOT_RUN`. No new SkillBank has real-effect qualification. No final held-out
outcome has been opened. Content improvement and routing improvement are
different contrasts; all fixed-content gates must share the same matched
base/Skill outcomes. Zero coverage means zero incremental success and
undefined conditional harm, not zero conditional risk.

The evaluator separates content-on/off from gate/global effects, preserves raw
four-way counts/missingness, and reports planned bounds and exploratory
semantic-cluster uncertainty where declared sample conditions hold. Shared
controls are neither new independent samples nor repeated costs. An abstainer
equals base and an always-reuser equals global even when that shared outcome
is missing: those contrast zeros are identities, not imputed task outcomes.
Actual unknown outcomes stay unknown. ACRE still uses `AcreEngine.route()` and
its original canonical/governance gates.

The pack provides standard name/description frontmatter, typed sidecars and
readback; no scripts are exported. Real pack loading, behavioral Skill use and
held-out benefit have not been demonstrated.

## 6. Total cost and budgets

Phase2 new provider calls: **0**. Input/completion/reasoning/cache usage from
provider experiments: **not measured**, not fabricated zeros. Provider spend
incurred by this phase: none. This does not estimate Codex/subagent costs.

Approval has not been supplied for an exact run/manifest with a hard USD cap.
The old two connectivity calls do not extend authorization. The completion
cap is not calibrated by those calls. Cache savings are not assumed.
The disabled approval example is not permission to spend; prices must be
attributed to a checked official snapshot before live execution.

An actual public pricing snapshot was captured server-side at
`2026-10-15T19:05:06+00:00`: official peak USD/million rates are cache hit
**0.006**, cache miss **0.3**, completion including reasoning **1.2**.
`pricing-source.html`, `pricing-table.json`, `pricing_snapshot.json` retain
source/extraction. Estimates are not provider bills or promises of future rates.

Proposed B envelope: 32,768 input allowance and 8,192 completion cap per request,
180 calls: **5,898,240 input + 1,474,560 completion tokens**, worst estimate
**USD 3.538944** with no cache discount. Suggested ceiling **USD 4**, B only,
zero retries, one fixed fresh live directory. The 8,192 cap is an uncalibrated
pilot rejection bound, not a scientific default. Input reservation uses declared
UTF-8/chat-framing assumptions, not an exact provider tokenizer. Actual usage
violations/unknown billing stop further requests. C/D or higher caps need new
approval; the disabled example is not that approval.

## 7. Test integrity and scientific claim scope

The entry full-suite run is pinned to `1796fa3`; raw command, environment,
stdout/stderr, start/end and exit code are under `baseline-tests/`.
Result: **497 passed, 8 warnings, 681.60 seconds, exit 0**. The separately
saved exit code is authoritative. This verifies the entry baseline, not the
new Phase2 changes. The previous 473-test result remains specific to
`f63ec84`. No synthetic qualification campaign is rerun.

| Revision | Actual server result | Log directory |
|---|---|---|
| `bcae5cd` | 122 passed / 3 failed, exit 1 | `targeted-bcae5cd` |
| `47482ef` | 125 passed, exit 0 | `targeted-47482ef` |
| `47482ef` | 563 passed / 2 failed, 8 warnings, 735.70 s, exit 1 | `full-suite-47482ef` |
| `54ccec5` | 166 passed, 1 warning, 34.12 s, exit 0 | `targeted-54ccec5` |
| `54ccec5` | **570 passed, 8 warnings, 675.93 s, exit 0** | `full-suite-54ccec5` |
| `2f9d548` | 171 passed / 3 failed, 1 warning, 34.49 s, exit 1 | `targeted-2f9d548` |
| `d03cf90` | **175 passed, 1 warning, 35.34 s, exit 0** | `targeted-d03cf90` |

Batches overlap and are not added. The full suite applies to `54ccec5`, not
the subsequent native-grammar/outer-journal/contrast-identity changes. Those
received the final 175-test affected/adjacent-path batch under taskbook 7.1.
No current-HEAD full-suite claim is made.

The two full-order failures were worker fixtures assuming signal-handler
readiness after 0.2 s and inheriting the capture pipe. Explicit ready-message
synchronization/separate child pipe replace the timing assumption. Production
cleanup, isolation, evidence persistence, task budgets and scientific gates
were unchanged; original assertions remain. Both tests and the full suite pass
at `54ccec5`. The later native-acknowledgment failures exposed the destination
article bug and were fixed, not waived.

No new Agent-success, Skill-benefit or ACRE-efficacy claim is supported yet.
The manuscript and its numerical evidence are unchanged.

## 8. Artifacts and no-overwrite proof

`frozen-source-check.json`: **499 historical files, zero mismatches**.
Historical campaign identity remains
`b67df06db177b7042e14931e293d3e5ea6db0cc84228876e465d8923770536f8`.
The historical cross-repository manifest still matches
`b978c74f52b0f10a1b595d84c26c84f5ba5ab8278c161fea20d7bf80ad404b0c`.

Actual split: `data-54ccec5/split_manifest.json`, identity
`adbf45195039868fafd1d9a568b5097a681263567f3f2b868ca4ee897745337c`;
its `SPLIT_OVERLAP_REPORT.md` records exposure/grouping. Earlier failed and
blocked outputs remain development records, not replacement evidence.

The active proposal is generated under `approval/` after final documentation
sync, binding the exact current revision. `prepared-54ccec5/` is an older,
non-active preparation. Local mirror:
`E:/DATA/EvoSPE/skill-evolution-phase2-20261016/`.

**Scientific status:** B **READY_FOR_APPROVAL**; real autonomous Agent viability,
automatic Skill benefit, independent content efficacy and real ACRE efficacy
all **NOT_RUN / UNSUPPORTED**. The manuscript/statistics remain unchanged.
No long provider/GPU job or new monitor was started.
