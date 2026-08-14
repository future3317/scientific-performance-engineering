# Static GNN prediction workloads

Use this reference for static crystal/materials graph prediction, including property or Cartesian tensor targets. Read `PERFORMANCE_PLAYBOOK.md` for shared measurement and runtime rules. For an authorized equivariant representation/operator change, also read [EQUIVARIANT_OPERATOR_DESIGN.md](EQUIVARIANT_OPERATOR_DESIGN.md).

## 1. Fix the work unit before comparing

Record total crystals, atoms, and edges in every timing window plus their per-batch distribution. Report crystals/s, atoms/s, and edges/s. Use the metric that tracks the limiting work as the primary acceptance metric: edges/s for edge aggregation/tensor-product work, atoms/s for node-dominant work, and crystals/s for end-to-end property inference. Graphs/s alone is not comparable when graph sizes differ.

Keep the same graph/atom/edge totals and batch-size distribution for a systems comparison. Size bucketing is a candidate only when it preserves the data contract. `DynamicBatchSampler` or a changed number of graphs per optimizer step changes effective-batch semantics; treat it as an authorized training experiment.

## 2. Route static graph hot paths by trace evidence

- For scatter/gather kernels with high memory traffic or atomic activity, prioritize fewer temporary edge tensors, fused message-and-aggregate/SpMM when mathematically equivalent, cached immutable topology, and regular batches. Do not infer a compute bottleneck from low SM utilization.
- Precompute edge index/order, PBC shifts, CSR/CSC pointers, species indices, offsets, neighbor lists, or geometry bases only when invariant for the sample/configuration and supported augmentation. Preserve existing batch ownership and transfer path first; make the cache device-resident or introduce a new transfer path only if the trace proves the existing path became the next bottleneck. Dynamic coordinates, strain, or learned geometry make the affected value non-static.
- For CPU-to-GPU gaps, first remove CUDA scalar extraction, tensor-to-Python branches, per-graph loops, repeated device/dtype conversion, and repeated basis/mask allocation. Then consider a stable compiled subregion.
- Treat padding as an experiment: compare padding waste, peak memory, and end-to-end work-normalized throughput against the ragged reference.

## 3. Preserve tensor physics

For a raw rank-three Cartesian target \(e_{ijk}\), validate the project-specific transformation convention under a rotation \(R\):

\[
e'_{ijk}=R_{ia}R_{jb}R_{kc}e_{abc}.
\]

Also test the repository's permutation, translation, periodic-image, point/space-group, and component-symmetry conventions. For piezoelectric response, retain applicable strain-index symmetry and zero-response constraints; do not assume a generic rank-three test fully specifies the label convention. Compare candidate versus baseline component errors, loss terms, gradient norms/cosines, finite values, and physical downstream metrics. Use [materials_gnn_checks.py](../assets/materials_gnn_checks.py) for work-rate and raw Cartesian rank-three helpers; adapt the model invocation and project-specific gates at the call site.

## 4. Prediction acceptance

Accept only when the comparable steady-state run meets the general performance rule and all graph/tensor/scientific gates. Record compiler warmup and profiling overhead separately. A faster microkernel, lower memory use, or higher GPU utilization alone is not an accepted optimization.
