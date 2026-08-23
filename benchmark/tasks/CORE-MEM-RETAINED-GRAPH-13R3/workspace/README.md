# CORE-MEM-RETAINED-GRAPH-13R3 — long-horizon retained autograd history

The workspace baseline intentionally retains each segmented activation with
its live autograd history and uses redundant `retain_graph=True`. This is the executable workload, not a compile
recompilation fixture. The task uses 12 residual segments, 50% checkpointing,
and a 240-iteration measured horizon so the cross-step retention is observable.

The paired oracle keeps the same model and optimizer work but clears the
history before each forward and stores detached activation diagnostics only.
Correctness is checked against a live fp64 recomputation on fresh inputs.

The required diagnosis is `memory_pressure`, supported by the benchmark-owned
retained-graph activation trace.
