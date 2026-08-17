# CORE-COMPILE-DYNAMIC-11 — tensor-only dynamic-shape specialization

The baseline forward contains no Python scalar branch. It is compiled with
`dynamic=False` and receives the registered variable batch-shape schedule, so
first-seen shapes are specialized independently. The reference intervention
uses `dynamic=True` without changing the workload or work-unit contract.

The primary metric is `schedule_wall_ms`, including first compilation and the
full registered shape schedule; post-cycle step latency is retained only as a
secondary diagnostic.
