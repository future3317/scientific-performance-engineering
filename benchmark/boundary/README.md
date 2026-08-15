# ACRE-v0 BoundaryBench

This pilot is separate from SPE-EvoBench. It tests deterministic predicate
synthesis after certified counterexamples in two parameterized families:

- `graph_cache_geometry_motion`: graph-cache applicability contracts shrink to
  a low-geometry-motion region.
- `compile_horizon`: compilation applicability contracts shrink to a sufficiently
  long logical horizon.

Each family has disjoint `representative_pool`, `query_pool`, and
`sealed_test_pool` records. Query outcomes may falsify a parent predicate but
never enter the sealed pool. A child predicate is accepted only when it covers
all positive representative anchors and rejects every certified counterexample.
Uncertain effects are retained as unresolved evidence and cannot shrink the
version space.

Run:

```powershell
& 'D:\Anaconda\envs\EGNN\python.exe' benchmark/boundary/test_cegis.py
& 'D:\Anaconda\envs\EGNN\python.exe' scripts/validate_acre.py
```
