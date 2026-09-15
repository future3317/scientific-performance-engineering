# ScienceWorld upstream-intervention stop report

This report closes the U1–U5 diagnostic stage. It does not authorize ACRE
acquisition, a clean final test, C/D, or a SkillGen efficacy claim.

## U1: frozen 30-pair forensic boundary

The frozen train-side run contains 3 candidates and 30 verification pairs. The
persisted summaries establish 30/30 execution-valid pairs and 0/30
skill-success pairs. They do not contain per-pair native scores, action traces,
completion states, valid/invalid actions, prompt/token receipts, or skill
injection receipts. The resulting diagnosis is therefore
`underdetermined_from_frozen_artifact`.

The immediate root cause that is established is telemetry loss at the
verification boundary: aggregate `VerificationReport` objects were persisted,
but the paired trajectories were not. Categories A–F in the upstream plan
cannot be separated from the frozen artifact without a new provider run. The
diagnostic artifact records this explicitly rather than imputing missing scores.

## U2: native oracle sanity

Ten deterministic train-side ScienceWorld gold-path cases were executed with
the official simulator and evaluator. All ten completed and reached score
100, with `provider_calls=0` and `gate_passed=true`. This validates native
reset, variation selection, gold-path execution, score progression, and the
official evaluator for this sanity slice.

## U3: protocol audit

The released SkillGen repository was audited at commit
`816c91f458ddb87be32657c882cb0431b9cfea01`. Its ScienceWorld path uses the
upstream label/subgoal wrapper, gold-path-enabled environment, prompt example,
trajectory sampling, graph/action-contribution extraction, retrieval, and
skill-conditioned inference. The frozen project run used a provider-generated
single skill text and the project executor, without those upstream data and
extraction artifacts. It must not be described as a faithful SkillGen
reproduction. See `docs/skillgen_protocol_comparison.md`.

## U4–U5: upstream-style native feasibility audit

Qwen2.5-7B-Instruct could not be downloaded on the server because the
Hugging Face connection was unavailable. A bounded provider fallback therefore
used `deepseek-v4-flash` while preserving the upstream ScienceWorld prompt,
simplifications, train-side graph construction, and skill-conditioned native
execution structure. The audit used 4 rows from upstream `data/sc/train_data_0`
for graph construction and 4 disjoint rows from `data/sc/test_data_0` for the
diagnostic comparison, with an 8-step cap and 96 provider calls.

| Metric | Base | Graph-skill policy |
|---|---:|---:|
| mean native score | 7.25 | 7.25 |
| progress rate | 0.0725 | 0.0725 |
| success rate | 0 | 0 |
| grounding rate | 0.46875 | 0.40625 |
| AUPC | 0.064444 | 0.064444 |

The paired score summary is 4/4 ties, mean and median score delta 0, with no
completion repair and no completion regression. Thus the audit demonstrates a
valid native upstream-style execution path and nonzero partial scores, but not
a positive SkillGen intervention effect. It is not an exact Qwen reproduction
and it is not evidence for ACRE efficacy.

## Stop decision

No frozen intervention currently satisfies the conditions required to enter
ACRE: a positive train-side native effect and evidence of context-dependent
utility are not established. The correct state remains:

```text
UPSTREAM_INTERVENTION_STATUS = INSUFFICIENT
ACRE_STARTED = FALSE
FINAL_TEST_STARTED = FALSE
SKILLGEN_EFFICACY_CLAIM = FALSE
ACRE_TREATMENT_EFFICACY_CLAIM = FALSE
```

The next method change, if pursued, requires a new train-side protocol revision
with per-turn telemetry persisted from the first provider call. It must not
reuse the frozen 0/30 aggregate run as if it contained continuous outcomes.
