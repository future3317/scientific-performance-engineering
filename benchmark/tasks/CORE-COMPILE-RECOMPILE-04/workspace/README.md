# CORE-COMPILE-RECOMPILE-04 — graph break plus cold-shape recompilation

The baseline contains an `.item()`-dependent graph break inside a compiled
region with `dynamic=False` and one fixed batch shape. The reference intervention
removes only that graph break; model, optimizer, shape, and schedule stay fixed.

The primary metric is `schedule_wall_ms`: the end-to-end registered schedule
includes the first encounter with every shape and the subsequent work needed to
amortize cold compilation. Steady-state step latency is reported only as a
secondary diagnostic.
