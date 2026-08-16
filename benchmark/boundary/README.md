# ACRE-v0 BoundaryBench

BoundaryBench reuses the canonical generators in `benchmark/families/`. It
tests deterministic predicate synthesis after certified counterexamples across
all canonical families. The five original families remain the first calibration
set; the remaining families use the same FamilySpec contract.

The historical names below remain calibration aliases for the original ACRE-v0
tests; they do not define a second workload source:

- `graph_cache_geometry_motion`: graph-cache applicability contracts shrink to
  a low-geometry-motion region.
- `compile_horizon`: compilation applicability contracts shrink to a sufficiently
  long logical horizon.

Each canonical family has disjoint `representative_pool`, `active_query_pool`,
and `sealed_boundary_pool` records. Query outcomes may falsify a parent
predicate but never enter the sealed pool. A child predicate is retained only
when it covers all positive representative anchors and rejects every certified
counterexample.
Uncertain effects are retained as unresolved evidence and cannot shrink the
version space. Results are reported as `consistent`, `identified`, or
`underidentified`; sealed error is an offline score and is never an acquisition
stopping signal.

Ownership is explicit: `core/acre/predicates.py` and `core/acre/cegis.py`
define predicate semantics and synthesis, while this package only constructs
the hidden pools and evaluates the returned predicate through `evaluator.py`.
Boundary code must not define a second synthesis or promotion algorithm.

Run:

```powershell
& 'D:\Anaconda\envs\EGNN\python.exe' benchmark/boundary/test_cegis.py
& 'D:\Anaconda\envs\EGNN\python.exe' scripts/validate_acre.py
```
