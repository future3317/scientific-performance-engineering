# ACRE interaction pilot

This directory contains the isolated synthetic experiments for ACRE stages 3--5:

- `factorial_bench.py` recovers the four 2x2 interaction classes and checks the bounded confidence-interval coverage contract.
- `acquisition_bench.py` compares random, uncertainty-only, and decision-aware query selection by cost to a fixed edge-identification error.
- `router_bench.py` compares governed routing, CEGIS specialization, causal interaction evidence, and the combined router.

The pilots use deterministic fixtures and are validated by `scripts/validate_acre.py`. Their summaries are diagnostic evidence for the ACRE layer only; they are not formal v1.0-20/v1.0-50 benchmark tasks and are not included in headline aggregation.
