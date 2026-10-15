# Skill evolution qualification protocol

## Status and boundary

This package contains an executable, provider-free synthetic fixture and a
proposal for a future train-only viability phase. The fixture is not empirical
evidence. `real_phase_i.status` remains `NOT_RUN` until executable readiness,
server resource checks, a new evidence identity, and explicit study approval
are complete. This package does not read credentials, call a provider, start a
benchmark environment, or write into historical result roots.

## Synthetic qualification fixture

Run the mechanism fixture on the supported server, in `equivcompiler`, with a
new output path (the parent directory must already be a fresh results root):

```bash
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python -m experiments.skill_evolution --fixture --output /home/workspace/lrh/RESULTS/skill-evolution-fixture-NEW/report.json
```

The default fixture uses fixed seeds `17, 41, 83`, regimes `broad_benefit`,
`selective_transfer`, and `null_effect`, 80 independent task-family clusters
per seed/regime, and four matched task pairs per cluster. It compares `base`,
`global_reuse`, and a predeclared synthetic `context_gate`. Each cluster has two
supported and two unsupported context slots. The context gate is supplied the
synthetic support stratum, so it is an oracle-mechanism ceiling, not a learned
applicability model or a real ACRE evaluation.

Named hypotheses are frozen in `study.py`:

- H1_BROAD: global reuse improves paired success over base in broad-benefit
  conditions.
- H2_SELECTIVE: the predeclared context gate reduces negative transfer versus
  global reuse while retaining benefit on supported contexts.
- H3_NULL: the null-effect fixture does not systematically produce a positive
  paired effect; the interval is reported without forcing a pass decision.

The statistical unit is the independent task-family cluster. Outcomes are
paired within task and arm; reported contrasts are cluster means of within-task
treatment-minus-base differences with a normal 95% interval across clusters.
Seeds and regimes are reported separately; no pooling across regimes is used as
a confirmatory estimator. The power table is a normal-approximation planning
table over the frozen cluster/effect grids and a declared discordant-pair-rate
assumption. It is not observed power and does not authorize a sample-size
change after seeing outcomes.

The fixture pre-reserves all 2,880 matched task pairs (three planning units per
pair: one outcome for each of base, global reuse, and context gate), covering
all 720 independent-cluster blocks.
If the task budget cannot reserve the complete set, the report says
`NOT_LAUNCHED_INCOMPLETE_PAIR_RESERVATION` and contains no simulated outcomes.
Evolution-search budget is independent and cannot consume task reservations.
The three-unit estimate is a fixture planning unit, not tokens, wall time, GPU
time, or money. Fixture execution records pair completion but measured runtime
and token usage remain explicitly unknown.

## Paper-driven real-study order

The manuscript's present question is when a fixed reusable intervention needs
a context gate. Its controlled selective-support result and frozen 80-pair
LogicBench broad-support result already exist. New ALFWorld work must address
the missing native-agent selective-support evidence, not silently turn this
paper into a Skill generator or agent-memory comparison. No new result is
asserted here. Historical outputs remain read-only.

The former 720-call proposal was 6 tasks x 2 agents x 60 actions: a worst-case
engineering allowance for legacy/stateful comparison, not 720 independent
observations or a power calculation. That comparison is removed from the
default path. Restore it only for a concrete unresolved harness diagnosis or
a separately approved agent-improvement claim, not as a prerequisite to ACRE.

| Order | Question and design | Provider work | Role in the paper |
| --- | --- | --- | --- |
| Existing | Controlled paired support, counterexample dose, sensitivity; frozen LogicBench boundary | Reuse existing evidence, no reruns | Current core evidence |
| Readiness | Three stateful train episodes, one clean/heat/cool, at most 60 actions each | At most 180 calls, not a target | Engineering and cost diagnostic only |
| Development | Fixed train-derived skills; base/skill pairs on disjoint train contexts | One base rollout per unique task/seed/config and one rollout per task/skill treatment | Admission, headroom, representation and cost estimates, not final efficacy |
| Frozen evaluation | Disjoint native held-out tasks, fixed agent/skills/features/policies | Shared paired base/skill outcomes, not new rollouts per gate | One targeted external treatment study if prerequisites hold |
| Conditional extension | Skill-content editing or learned regime selection | Separate independent validation and budget | Separate estimands; not required to complete this manuscript |

## Proposed real Phase I: three-episode readiness diagnostic

`PHASE_I_CONFIG` in `study.py` proposes one train episode from each of `clean`,
`heat`, and `cool`, using only the stateful policy. The maximum remains 60
actions per episode: reducing it merely to lower cost would change the task
horizon. A finished episode stops immediately. The 180-call ceiling is not a
spending authorization and the diagnostic cannot establish a success rate or
stateful-vs-legacy advantage. No replacement episodes are selected because of
poor outcomes. Previously used episodes are development-only.

DeepSeek Flash thinking is enabled. The current 8,192 completion-token cap
(reasoning plus final answer) is uncalibrated, not a scientific default justified
by two trivial connectivity requests. Before episode execution, fix and enforce
a paid diagnostic budget; token/money limits are currently unset, which blocks
launch rather than meaning unlimited use. Readiness must record actual prompt,
cache-hit/miss, reasoning and answer/completion tokens, finish reasons, invalid
actions, native steps, verifier reach, official success and terminal reason.
Unavailable counters remain unknown. Examine navigation, state transformation
and failure recovery separately when those decisions actually occur; three
episodes need not cover all of them. A cap-induced incomplete answer is not
evidence that a larger cap improves task utility.

The three-episode diagnostic is not a sample-size calibration experiment. If
interaction fails or all episodes fail, diagnose once without launching the
next phase; do not automatically expand tasks or tune until success. If it
works, it only establishes that train-side development is executable.

Before any real stage: verify the supported server revision, clean worktree,
interpreter and CPU/resource contract; fix the exact train manifest, prompts,
model/thinking settings, horizons, no-retry behavior, and a fresh output root.
The current config is a proposal and remains `NOT_RUN`.

Executable preparation is `python -m experiments.skill_evolution.readiness
--data-root <official-data> --history <all-known-train-receipts> --output
<new-plan.json>`. It selects three compiled train tasks after excluding whole
historically used template directories, without importing a provider. Run
`python -m experiments.skill_evolution.native_probe --data-root <official-data>
--plan <new-plan.json> --output <fresh-directory>` to check these exact tasks
with admissible commands and zero model calls. This use burns the selected
tasks for final evidence; they are readiness/development-only thereafter.

The stateful runner CLI requires `--plan`, `--max-calls` and
`--max-total-tokens`. It enforces per-request reservations and records actual
usage before parsing the provider answer. Unknown usage freezes further
requests. This implements token/call accounting, not approval or monetary
enforcement. No paid command is authorized by these preparation steps.

## Train-side development and final-design freeze

Use the existing fixed-procedure synthesis path, not a new Skill-generation
framework. If the clean/heat/cool skill cohort is newly synthesized, use the
preselected train demonstrations and one synthesis request per family; record
all three, including unhelpful skills. Do not regenerate until a positive skill
appears. Keep synthesis demonstrations, support-fitting contexts, calibration
contexts and final tasks disjoint. Old studied tasks cannot become clean final
evidence. Any skill selection is train-side and explicit; the held-out claim
concerns the selected fixed intervention, not average skill-generation efficacy.

Plan development as a fixed balanced task batch chosen before outcomes, with
per-skill and source/cross-family reporting. Its size and spending ceiling are
not yet approved; do not replace the discarded 720-call allowance with another
arbitrary campaign number. Freeze the batch size, splits and stopping rule
before its calls. Treat the current typed pre-execution vocabulary as the
initial representation; do not expose hidden task plans or evaluation labels.

For every task/skill pair, report repair (0,1), regression (1,0), both correct
(1,1), both wrong (0,0), and official task success. Admission, average utility,
and support learning are separate questions. On binary paired outcomes:

- realized oracle-over-base headroom = repairs;
- realized oracle-over-global headroom = regressions;
- global-minus-base gain = repairs minus regressions.

The first does not measure the gain available to gating over global reuse.
These are realized paired-outcome ceilings, not a deployable or noise-free
oracle for expected treatment effect. Repeated runs of one task are not new
independent tasks. A broad/no-harm skill may support global reuse but does not
by itself provide an informative test of selective gating. If training shows
no meaningful interventions, insufficient negative transfer, or no observable
boundary, report that development result and stop this external study; do not
search new benchmarks or alter the final cohort to manufacture heterogeneity.

Freeze only five deployable policies: global reuse, task-type semantic gate,
nearest evidence, CatBoost, and Full ACRE. The common base is the fallback and
reference; a pairwise oracle is a scoring-only ceiling. Learn each skill's
support separately, with any pooling explicitly declared rather than silently
fitting one cross-skill coarse rule. Train-only validation determines features,
thresholds and any admitted skill; all policies use the same visible features
and the same held-out task pool. Hyperparameter search costs belong to total
cost. Do not add extra baseline families or a joint evolution ablation now.

Before final outcomes, choose the task count from a declared minimum detectable
paired success difference, train-side discordance and power/precision target,
not from an API-call quota or observed final significance. As an illustrative
independent-task approximation, two-sided alpha .05, power .80, discordance
.20 and a 10-point difference require about 157 tasks using
`(1.96 + .84)^2 * .20 / .10^2`; five points need about 628. These are not fixed
sample sizes: gate-vs-global discordance can differ, clustering increases the
requirement, and rare-harm precision or multiple primary contrasts may dominate.
Choose one primary contrast (Full ACRE vs global reuse); other policy contrasts
are secondary with multiplicity handling declared. Derive the final design
before outcomes, with a fixed task batch and no unregistered efficacy looks.
If affordable sample size cannot support that claim, label a feasibility study
instead of treating a tiny cohort as confirmatory efficacy.

## Shared outcome evaluation and cost

For a fixed agent and skill, every frozen gate uses the same task-level
potential-outcome draws: `Y_pi = g_pi(x)*Y1 + (1-g_pi(x))*Y0`. Serialize decisions
using pre-execution context before joining outcomes. Base outcomes may be
shared only for identical task, environment seed, agent/model and decoding
configuration; never reuse them across a changed agent or tuned configuration.
Treatment outcomes remain skill-version-specific. Report this as a frozen
paired policy evaluation, not prospective evidence that a live controller
actually skipped execution. No answer memoization across stochastic replicates.

With N tasks and K skills sharing one base rollout per task, outcome collection
needs N*(K+1) episodes, not 2*N*K per policy and certainly not another set per
gate. If each skill uses disjoint tasks, it instead needs twice the number of
task/skill pairs. With a 60-action horizon these are worst-case call ceilings;
actual calls are the sum of executed decisions, plus synthesis/editing calls.
Baseline breadth should mostly cost local evaluation, not more provider runs.

The primary outcome is official success and the primary contrast is paired
incremental success of Full ACRE over global reuse. Report success, coverage,
harm/deployed, repair recall and incremental success over base for every policy;
also report repair/regression counts and uncertainty at the declared independent
task/template unit. Three task families alone are too few clusters for a
credible cluster bootstrap; identify independent templates or restrict inference
to the finite cohort. Zero deployment implies zero incremental success and
undefined harm/deployed, not zero conditional risk. Do not score missing
executions as task failures or drop them silently; report ITT bounds where needed.

Keep prompt/skill prefixes byte-stable ahead of variable state without padding
the prompt to chase cache hits. Provider caching changes billed input costs,
not total context tokens or reasoning costs. The connectivity probe observed
zero hits; no savings are assumed. Persist measured usage and price schedules
with currency/date; compute monetary estimates only from actual applicable
rates. A per-call output cap is not a whole-episode budget. Before launching a
fixed paid batch, reserve completion capacity for the scheduled paired units
and enforce total calls, all input/completion tokens, money and wall limits.
Do not let an unpaired prefix of an underfunded campaign become efficacy evidence.

The fixture mode invokes no provider, ALFWorld environment, or real evidence
path. Neither synthetic mode can mark Phase I ready or change its `NOT_RUN`
status.

## Learned ACRE qualification mode

The separate `--qualification` mode calls the current
`StatisticalCEGIS.synthesize`, `classify_support_regime`,
`decision_from_synthesis`, and `decision_from_regime` APIs. It generates noisy
train-only paired groups on frozen seeds, fits CEGIS and the support regime,
then freezes both decisions before simulating outcomes on disjoint evaluation
seeds. Context strata and the allowed feature vocabulary are declared before
simulation; evaluation outcomes and labels are never passed to synthesis or
classification.

The five evaluation arms are base, global reuse, operational old selective
CEGIS, operational new regime-aware ACRE, and synthetic oracle applicability.
The oracle is an upper-bound reference only. The output reports the learned
CEGIS status, support-regime classification, activation, paired effects,
negative transfer, and independent-cluster intervals by generating regime.
The learned regime arm passes through `AcreEngine.route` with one frozen
synthetic canonical `RuleSpec`/`RuleState`; the scope's
`versions.skill_digest` is the same semantic digest that the engine derives for
that rule. This synthetic canonical fixture exercises the production routing
gate only. It does not imply canonical promotion of any real Skill or rule.

The frozen training seeds are `101, 211`; independent evaluation seeds are
`307, 401`. Each seed/regime has 60 groups per stratum and 1,024 paired binary
repetitions per group. This yields 120 independent groups per stratum for each
regime, enough for the predeclared classifier harm bound in the broad positive
control; a separate 80-repetition development fixture is expected to remain
`UNCERTAIN`. CEGIS uses the categorical `stratum` feature, a fixed
two-context decision lattice, and a certificate bound to `SUPPORT_SCOPE`. The
family alpha is 0.05: 0.025 is allocated to simultaneous two-sided training
group measurement intervals using Bonferroni across all 720 predeclared
training groups (0.025/720 per group), 0.0125 to support-regime classification,
and 0.0125 to CEGIS certification. This is the frozen synthetic confidence
allocation; its planned Hoeffding radius is
`sqrt(2*log(2/(0.025/720))/1024) ≈ 0.147`, within the frozen 0.15 target.
It is not a claim that the synthetic group-bound model establishes real-world
coverage. The support classifier also uses minimum 40 independent
groups, net-benefit threshold 0, and maximum harm probability 0.10. The named
hypotheses H4_BROAD_GLOBAL, H5_SELECTIVE_LEARNED, and H6_NULL_ABSTENTION are
emitted in the report. Comparisons remain synthetic qualification checks, not
confirmatory claims.

The qualification mode also stresses predicate cardinalities 8, 16, 32, and
64 using the real CEGIS search with `max_candidates=1000` and a 2 second wall
cap. Each row distinguishes exact from truncated search and records measured
elapsed wall seconds, Python traced peak allocation bytes, and serialized
result bytes. Truncation stays visible; it is never represented as an
identified boundary. These are synthetic host diagnostics, not performance
evidence for real workloads.

Suggested server command:

```bash
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python -m experiments.skill_evolution --qualification --output /home/workspace/lrh/RESULTS/skill-evolution-qualification-NEW/report.json
```

Qualification requires a fresh output path. Before calling the study routine or
generating any synthetic outcomes, the CLI writes the complete frozen config to
the sibling `report.preregistration.json` with exclusive-create semantics. The
report is then written exclusively as `report.json`. Create and verify a fresh
output directory first; if either file already exists, choose a new results
root rather than replacing it.
All 720 independent evaluation-group blocks are reserved for all five arms
before any synthetic evaluation outcome is generated. The separate CEGIS
search budget is frozen at 7,000 candidate evaluations total (1,000 per each
of three training fits and four stress calls); actual examined candidates are
charged to that ledger. Candidate evaluations and simulated arm-task outcomes
remain separate units.

## Future phases and estimands

Only after readiness and train-side development may the targeted external
fixed-Skill routing evaluation proceed. Skill-content optimization and the
joint system are separate, optional studies rather than manuscript prerequisites.
Keep model,
harness, task split, Skill body, and verifier fixed within each causal
contrast. The same Skill version must be used across global, semantic,
original ACRE, and regime-aware routing comparisons. Skill generation,
retrieval, and routing costs belong to total cost, alongside task execution.

For any future paired study, report the predeclared paired task-utility
difference; repairs and negative transfer; deployed-case harm and all-case
harm with their distinct denominators; activation, verifier reach, API-contract
failures, and completion under budget. Compute uncertainty across independent
task-family/template clusters, not repeated rollouts treated as independent
tasks. Freeze alpha allocation and any adaptive look schedule before outcomes.
Report intention-to-treat, execution-valid, and per-protocol populations, and
bound missing pairs instead of silently dropping them. A zero denominator is
`null`/undefined, not zero.

Report fixed task-execution-budget and fixed total-compute views. Separate
provider calls, input/output tokens, wall seconds, GPU seconds, verification
cost, retries, and concurrency/blocking. For every metric, serialize a value
only when measured and otherwise retain `null` with an explicit unknown or
not-measured status. Equal task execution is not equal total compute.

## Power table interpretation

The fixture includes two-sided alpha 0.05 planning approximations for paired
effects 0.05, 0.10, and 0.20; 20, 40, 80, and 160 independent clusters; and a
discordant-pair-rate assumption of 0.25. The approximation is intended to
show how sample size and effect assumptions relate. Real viability and final
power analysis require empirical baseline/disagreement estimates from an
approved training/calibration phase; no power value here is a real-study
qualification result.
