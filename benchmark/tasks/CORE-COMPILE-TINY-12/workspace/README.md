# CORE-COMPILE-TINY-12 — tiny short-lived compile counterexample

This task is deliberately a counterexample. The one-block eager workload has
only two batch shapes and eight measured steps. Compiling it is a tempting
intervention, but cold compile/specialization cost should dominate the short
schedule. The correct governed action is to abstain and keep the eager path.

The harness reports `cold_shape_schedule_ms`, including the first encounter
with the registered shapes. Any steady-state timing is secondary evidence and
cannot hide the compile startup cost.
