# CORE-COMPILE-RECOMPILE-04 — graph break plus cold-shape recompilation

The baseline combines an `.item()`-dependent graph break with
`dynamic=False` and eight registered batch shapes. The reference intervention
removes the graph break and buckets those shapes into one fixed graph.

The primary metric is `schedule_wall_ms`: the end-to-end registered schedule
includes the first encounter with every shape and the subsequent work needed to
amortize cold compilation. Steady-state step latency is reported only as a
secondary diagnostic.
