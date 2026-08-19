# Experience-driven Skill evolution

> Statistical authority: `references/STATISTICAL_PROTOCOL.md`. This document
> describes lifecycle and provenance, not a competing confidence definition.

This repository treats self-evolution as a governed maintenance workflow, not runtime self-editing.

## Contents

- [Roles and authority](#roles-and-authority)
- [Lifecycle](#lifecycle)
- [Capture gate](#capture-gate)
- [Rule OS implementation contract](#rule-os-implementation-contract)
- [Promotion and maintenance gates](#promotion-and-maintenance-gates)

## Roles and authority

- **Practitioner/collector:** uses the Skill and may write a structured record to `experience/inbox/`. It must not edit `SKILL.md`, canonical references, a rule registry, or promotion status.
- **Maintainer/curator:** reviews inbox records, removes project-specific noise, compares positive and negative cases, and proposes candidate rules under `evolution/candidates/`.
- **Reviewer:** checks the candidate against replay/regression cases, conflicts, applicability boundaries, and scientific invariants. Only an explicit reviewed change may update canonical documentation or registry entries.
- **Janitor:** may merge, specialize, supersede, archive, or retire rules when evidence shows duplication, conflict, or loss of utility.

An experience record is evidence, not a rule. `status` is limited to `inbox`, `case`, or `archived`; `canonical` is intentionally not a valid experience status.

## Lifecycle

```text
observation -> experience/inbox -> reviewed case -> candidate insight
           -> replay/regression -> canonical rule or archive/retire
```

Do not promote a single successful run. A candidate must state its trigger, required evidence, scope, counterexamples, risk, source cases, and regression cases. P0 scientific invariants, comparison semantics, acceptance semantics, and correctness policies never auto-promote. Human review is required before canonical changes.

## Capture gate

Capture only a reusable surprise or boundary: a falsified hypothesis, rule failure, better diagnostic order, silent fallback or hidden synchronization, repeated failure, applicability boundary, negative result, or previously unclassified bottleneck. If an existing rule already explains an ordinary result, do not create a record.

Each record must preserve the observed symptom, workload, evidence, every material attempt, measured reason for rejection, replacement evidence, conditional lesson, scope, collector confidence, and structured artifact provenance (`path`, `sha256`, `artifact_type`, `producer`, `benchmark_id`). Use `assets/experience_record.json` as the template and validate/capture with:

```powershell
python scripts/validate_experience.py experience/inbox/EXP-YYYY-MM-0001.json
python scripts/capture_experience.py prepared_record.json
```

The validator is a contract check, not a promotion tool. It never edits records and never changes runtime Skill behavior.

## Hook boundary

If a host Codex setup adds a `Stop` hook, it may remind the practitioner to prepare a record and invoke `capture_experience.py`. The hook must not parse an entire transcript, infer a lesson without an explicit record, edit canonical files, or promote a rule. `SessionEnd` is suitable only as a short cleanup/reminder path, not as the consolidation engine.

## Candidate and promotion contract

Use `assets/rule_candidate.json` as the rule-card shape and `assets/rule_regression_case.json` as the replay-case shape. A candidate must link existing reviewed source `EXP-*` cases, keep held-out `admission_cases` disjoint from post-promotion `regression_cases`, and preserve regression lineage that does not reuse source evidence. A card becomes canonical only when `scripts/run_rule_replay.py` has produced an existing manifest whose command, case-bundle digest, harness revision, machine-readable outcome, result digest, and attestation match the card; its paired intervention clears the utility lower-bound and scientific gates; its Beta posterior satisfies `P(p > p_min) > 1 - delta`; every referenced regression case has `status=pass`; and `promotion.human_review`, `review_commit`, `reviewer`, `reviewed_at`, and `review_diff_hash` are present. `registry/rules.json` is an index of canonical cards, not a second copy of their text. Run the read-only audit before a review:

```powershell
python scripts/validate_evolution.py .
```

The audit does not promote, retire, merge, or rewrite anything. Those are explicit maintenance edits reviewed as normal Git changes. Failed replay or conflict checks leave the card under `evolution/candidates/` or move it to `evolution/conflicts/`; retirement requires a reason and remains visible under `evolution/retired/`.

## Three measurable extensions

1. **Counterfactual Rule Utility (CRU):** `run_rule_replay.py` compares paired `do(rule=on)` and `do(rule=off)` outcomes on the same held-out contexts, reports the paired effect and lower confidence bound, and refuses admission when quality/scientific gates fail. This is causal attribution, not usage correlation.
2. **Mixture diagnostic:** replay records the implemented inverted
   Beta--Binomial mixture e-process as a promotion-probability diagnostic; it is
   not a Bayesian posterior or a routing utility bound.
3. **Evidence rate-distortion maintenance:** `score_rule_library.py` scores active-rule description length, measured utility distortion, and conflict cost. It emits merge/retire/specialize recommendations but never mutates the library; a maintainer reviews the resulting Git diff.

Usage telemetry is a separate record (`retrieved_rule_ids`, `triggered_rule_ids`, `followed_rule_ids`, `overridden_rule_ids`, and outcome). It closes the retrieval/use/utility loop without treating retrieval frequency as evidence of benefit.

### Runnable maintenance commands

Prepare a paired replay bundle (the same held-out context measured with the rule on and off):

```json
{"rule_id":"PERF-SYNC-004","epsilon":0.05,"p_min":0.8,"delta":0.05,
 "cases":[{"case_id":"REG-HELDOUT-001","paired_replay":true,
            "same_fixture_id":"fixture-001","utility_on":0.17,"utility_off":0.12,
            "scientific_ok":true,"quality_ok":true}]}
```

Then run `python scripts/run_rule_replay.py replay_input.json replay_manifest.json`. Point a candidate's `promotion.replay_manifest` at that manifest; `validate_evolution.py` recomputes the result and manifest digests. Capture usage with `python scripts/capture_rule_usage.py usage_record.json`. For maintenance review, pass a JSON list of cards and a reference/library utility mapping to `python scripts/score_rule_library.py cards.json --utility utility.json --output maintenance.json`.

## Repository boundary

Existing `references/` remain the canonical knowledge source for domain detail. Rule cards and the registry provide auditable promotion metadata; they do not duplicate reference prose or automatically enter the runtime prompt. Retrieval remains progressive: route by comparison class/evidence/domain, then load only the relevant reference or reviewed rule card.
# Rule OS implementation contract

The repository implements the lifecycle as a typed, auditable Rule OS. The
canonical objects are `RuleSpec` (immutable meaning), `EvidenceEvent`
(append-only paired evidence), and `RuleState` (materialized confidence,
retrieval, and drift state) in `core/models.py`. JSON Schema files in `assets/`
are generated by `scripts/generate_rule_schemas.py`; do not hand-edit a second
semantic validator.

Applicability uses the structured predicate DSL in `core/predicates.py`.
`core/retriever.py` first routes by domain and predicate, then greedily selects
non-conflicting rules under a token budget. Retrieval telemetry is not utility.

## Promotion and maintenance gates

`run_rule_replay.py` materializes paired on/off EvidenceEvents and records the
same revision/seed-family context, propensity, artifact references, and result
digest. Its promotion gate uses a time-uniform Hoeffding confidence sequence
with a summable alpha schedule, so a maintainer may inspect the stream after
any number of events without changing the nominal coverage. The legacy
beta-binomial value remains diagnostic only.

P0/P1 rules remain human-reviewed. P2/P3 bounded-auto promotion additionally
requires held-out regression cases, at least two provenance independence
groups, source-diversity checks, and an explicit policy version. A literal
user-controlled trigger is quarantined rather than promoted. `RuleState` drift
states are `stable`, `suspected_drift`, `stale`, and `revalidating`; drift
starts a new replay and never mutates or deletes the historical spec.

Library maintenance is counterfactual: rate counts the selected spec and
retrieval cost, distortion is measured by leave-one-rule-out utility, and
conflict cost is charged for active conflict edges. Zero distortion is not a
retirement proof.
