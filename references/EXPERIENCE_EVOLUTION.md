# Experience-driven Skill evolution

This repository treats self-evolution as a governed maintenance workflow, not runtime self-editing.

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

Each record must preserve the observed symptom, workload, evidence, every material attempt, measured reason for rejection, replacement evidence, conditional lesson, scope, confidence, and relative artifact references. Use `assets/experience_record.json` as the template and validate/capture with:

```powershell
python scripts/validate_experience.py experience/inbox/EXP-YYYY-MM-0001.json
python scripts/capture_experience.py prepared_record.json
```

The validator is a contract check, not a promotion tool. It never edits records and never changes runtime Skill behavior.

## Hook boundary

If a host Codex setup adds a `Stop` hook, it may remind the practitioner to prepare a record and invoke `capture_experience.py`. The hook must not parse an entire transcript, infer a lesson without an explicit record, edit canonical files, or promote a rule. `SessionEnd` is suitable only as a short cleanup/reminder path, not as the consolidation engine.

## Candidate and promotion contract

Use `assets/rule_candidate.json` as the rule-card shape and `assets/rule_regression_case.json` as the replay-case shape. A candidate must link source `EXP-*` cases, state required evidence and counterexamples, and list planned `REG-*` regression cases. A card becomes canonical only when `promotion.replay_status=passed`, `promotion.replay_evidence` is present, `validated_cases` and `regression_cases` are non-empty, every referenced regression case has `status=pass`, and `promotion.human_review=true`; P0 invariants still require the same human gate. `registry/rules.json` is an index of canonical cards, not a second copy of their text. Run the read-only audit before a review:

```powershell
python scripts/validate_evolution.py .
```

The audit does not promote, retire, merge, or rewrite anything. Those are explicit maintenance edits reviewed as normal Git changes. Failed replay or conflict checks leave the card under `evolution/candidates/` or move it to `evolution/conflicts/`; retirement requires a reason and remains visible under `evolution/retired/`.

## Repository boundary

Existing `references/` remain the canonical knowledge source for domain detail. Rule cards and the registry provide auditable promotion metadata; they do not duplicate reference prose or automatically enter the runtime prompt. Retrieval remains progressive: route by comparison class/evidence/domain, then load only the relevant reference or reviewed rule card.
