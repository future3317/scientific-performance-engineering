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

Each record must preserve the observed symptom, workload, evidence, every material attempt, measured reason for rejection, replacement evidence, conditional lesson, scope, confidence, and relative artifact references. Use `assets/experience_record.json` as the template and validate with:

```powershell
python scripts/validate_experience.py experience/inbox/EXP-YYYY-MM-0001.json
```

The validator is a contract check, not a promotion tool. It never edits records and never changes runtime Skill behavior.

## Repository boundary

Existing `references/` remain the canonical knowledge source during this first phase. Do not duplicate them into a parallel `rules/` tree. Candidate and registry structures may be introduced by a later reviewed maintenance change after replay coverage exists.
