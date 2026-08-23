# Staged calibration probes (2026-08-19)

These files are real, server-side probes from commit `580e1d6`, run in the
`equivcompiler` environment on the otherwise idle GPU 2. They are calibration
evidence only; they do not amend `pilot_calibration.json`, open the approval
gate, or support a formal efficacy claim.

The raw JSON is kept beside the compact `summary.json` so reviewers can inspect
the exact harness output. The probes used the normal noise-control artifact and
the normal verifier/episode path. The SciML probe is blocked by observed
same-host noise. The H2D replacement completed with valid correctness and
scientific gates, but its noise floor remains material and its activation record
is not sufficient for a headline claim. The evolution probe demonstrates D
governance/poison handling; its utility result is not an efficacy measurement.

No task threshold, scientific gate, ACRE rule, or formal slot was changed by
these probes.
