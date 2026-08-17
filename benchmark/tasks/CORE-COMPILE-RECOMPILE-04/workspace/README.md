# CORE-COMPILE-RECOMPILE-04 — graph break plus cold-shape recompilation

The baseline combines an `.item()`-dependent graph break with
`dynamic=False` and eight registered batch shapes. The reference intervention
removes the graph break and buckets those shapes into one fixed graph.

The primary metric is `cold_shape_schedule_ms`: the first encounter with every
registered shape is measured rather than hidden in warmup. Steady-state step
latency is reported only as a secondary diagnostic.
