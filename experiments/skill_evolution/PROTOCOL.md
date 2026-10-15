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

## Proposed real Phase I: agent viability only

Phase I proposes six matched train-only task/game pairs, two each from the
`clean`, `heat`, and `cool` families, comparing `legacy_stateless` with
`stateful` on DeepSeek Flash with thinking disabled. It is matched by task/game
and environment seed. Pair order is randomized or counterbalanced with
the same schedule for both policies. The verifier and success definition,
maximum action budget, tool/API contract, model configuration, prompts, and
environment version are frozen before the first call. This comparison
estimates an agent/harness change; it is not evidence for Skill generation or
ACRE.

The proposed config is serialized in `PHASE_I_CONFIG` in `study.py`. The
60-step ceiling follows the current ALFWorld runner contract. The proposed
upper bounds are 720 provider calls total (12 episodes × 60 steps), 2,000,000
tokens total, and zero retries. These bounds, six pairs, the fixed verifier,
and thinking-disabled model setting still require budget approval and a
server-side resource contract; none authorizes or triggers execution.

Readiness is currently `NOT_RUN`. Before a real run, the operator must show:

1. A server-side repository/commit/clean-worktree/interpreter check and the
   declared resource contract, under the `AGENTS.md` server-only rule.
2. A new benchmark/controller revision, distinct evidence identity, approved
   train-only task manifest, and a fresh results root.
3. Executable stateful and legacy policies with a shared, deterministic
   verifier; matched pair IDs and a complete reservation for every pair.
4. A fixed provider/model configuration with thinking disabled, a call/token
   cap, provider retry accounting, and explicit cost fields. Unknown tokens,
   wall time, or failed-call usage remain `unknown`; they are never imputed as
   zero.
5. A predeclared viability decision based on verifier reach and nonzero
   success events. If both policies remain at a success floor, stop and
   diagnose the harness before any Skill-effect study.

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

Only after Phase I viability, later phases may separately study fixed-agent
Skill acquisition, frozen-Skill routing, and the joint system. Keep model,
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
