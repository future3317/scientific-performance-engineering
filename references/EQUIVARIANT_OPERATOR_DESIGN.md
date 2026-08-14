# Equivariant operator design

Use this reference only for an explicitly authorized architecture review or experiment. Its central order is: reduce the count and path cost of expensive equivariant operations, then optimize their placement/backend, then apply generic runtime tuning. Read `GNN_PREDICTION_WORKLOADS.md` for static graph/tensor gates or `CRYSTAL_GENERATION.md` for denoiser/sampling gates.

## 1. Freeze the experiment contract

Changing body order, message depth, irreps, TensorProduct instructions, `l_max`, graph/batch semantics, a canonicalization map, a Cartesian representation, point-group output basis, solver, or NFE changes the model or training contract. Establish a reference implementation and train each candidate separately. Report both fixed-budget quality and time/memory to reach a fixed quality target; do not present the resulting gain as a systems-only speedup.

For a kernel/backend comparison, instead keep irreps, instructions, normalization, layout, parameters, forward/backward work, data panel, and effective batch fixed. Compare output/gradient parity and prove selected forward and backward paths rather than a configured option.

## 2. Profile equivariant operations independently

Record forward and backward time, workspace, layout conversions, and dispatch for spherical harmonics, edge TensorProduct, `ChannelwiseTensorProduct`, `FullyConnectedTensorProduct`, `SymmetricContraction`/product basis, `Linear`/`IndexedLinear`, aggregation, and graph construction. In MACE-like models, treat `SymmetricContraction` as its own hotspot, not as a small TensorProduct variant.

For ragged crystals, also record edge count and allowed edge-TP paths plus node count and symmetric-contraction/product-basis paths. A useful DDP packing proxy is

\[
C_g\approx \alpha N_g+\beta E_g+\gamma E_gP_{\rm edgeTP}+\delta N_gP_{\rm SC}.
\]

Fit or validate this proxy against measured per-graph cost before using it to balance ranks. Do not change data membership, optimizer-step graph count, or global effective batch merely to make a balance chart look better.

## 3. Choose operator placement before a backend

| Observed condition | Authorized candidate | Required comparison |
|---|---|---|
| Edge TP dominates and \(E\gg N\) | Node-side product basis/symmetric contraction; body-order/depth trade | Same cutoff/receptive-field requirement, quality versus wall-clock, edge and node workspace |
| Full channel mixing dominates TP | `Linear → channelwise/depthwise TP → Linear` | Exact irreps/paths or retrained ablation; parameters, memory, tensor-core linear share, quality |
| \(|E|\times F\) message is materialized | Indexed/fused TP plus gather/scatter or `message_and_aggregate` | No edge-message workspace, exact aggregation semantics, forward/backward timing |
| High \(l\) TP dominates | SO(2)-reduced, Gaunt, VSTP, or SpinGTP route | Expressivity/parity path coverage, output/gradient/equivariance gates, actual end-to-end gain |

Higher body order can reduce the required message-passing depth, but it does not automatically preserve spatial receptive field. Test a small body-order/depth grid against the baseline rather than assuming “body order up, depth down” wins.

Do not use a fully connected TensorProduct as the default architecture when channelwise/depthwise coupling plus equivariant linear mixing expresses the intended paths. The dense linear stages may map better to GPU GEMMs, but this remains a model experiment when it changes coupling/mixing.

## 4. Select an equivariant kernel/backend at the real signature

For MACE-like work, separately benchmark `ChannelwiseTensorProduct`, `SymmetricContraction`, `Linear`, `IndexedLinear`, spherical harmonics, and indexed/fused convolution on the actual irreps, multiplicities, edge counts, dtype, layout, and backward workload.

- **cuEquivariance:** Test it as a backend pilot for a strict one-to-one O(3)/STF primitive mapping. Keep the equivariant path in `ir_mul` layout (`[2l+1, multiplicity]`) where supported; do not alternate e3nn and cuEquivariance layouts inside every layer. Account for every transpose, version/architecture restriction, fallback, and forward/backward/equivariance gate in the end-to-end benchmark. If the primitive cannot be mapped exactly, do not rewrite the scientific architecture just to adopt the library.
- **OpenEquivariance:** For general CG TensorProducts, compare it with the e3nn reference and cuEquivariance when the installed NequIP/runtime exposes the modifier and the exact path is supported. It is a candidate, not a required dependency.
- **Streaming/fused TP:** Do not materialize an \(|E|\times F\) message only because the reference code does. Prefer an indexed/fused or streaming formulation when it preserves aggregation semantics and has a proven backward path. Treat Sobek-style generated streaming kernels as experimental.

Never select a backend from a paper’s isolated speedup. Require kernel dispatch, no fallback, forward/backward correctness, layout cost, and end-to-end work-normalized throughput.

## 5. High-angular-degree alternatives have expressivity gates

At modest \(l_{\max}\), first exhaust optimized standard CG paths. When high \(l\) TensorProducts dominate, test the following only as separate architecture experiments:

- SO(2)-reduced convolution for a reduced high-degree formulation;
- Gaunt/Fourier TensorProducts when the required coupling and parity paths are present;
- VSTP or SpinGTP when high \(l\) and complete CG/parity expressivity are both required.

Do not treat a scalar Gaunt TensorProduct as a lossless drop-in for full CG TensorProduct. For piezoelectric, chiral, non-centrosymmetric, or other parity-sensitive tasks, specifically verify antisymmetric/parity-odd paths and the project’s rotation/tensor gates. Lowering `l_max`, pruning paths, or changing irreps is never lossless by default; retrain and compare scientific quality.

## 6. Crystal tensor targets: reduce representation only with a proven contract

A rank-three piezoelectric tensor with strain-index symmetry has 18 Cartesian degrees of freedom and can decompose into \(2\mathcal H^{(1)}\oplus\mathcal H^{(2)}\oplus\mathcal H^{(3)}\). The output rank does not require broad \(l=3\) features in every layer. Ablate tapered multiplicities (for example \(m_0>m_1>m_2>m_3\)), a late narrow high-order head, or an irreducible/Cartesian-moment readout against a full-equivariant reference.

When reliable point-group operations and label conventions are available, predict coefficients in a precomputed allowed basis \(T=B_Gc\) instead of unconstrained components. Include crystal symmetry, tensor convention, and forbidden-component checks. Do not derive or short-circuit zero response from an assumed symmetry convention that the dataset has not established.

For global crystal-tensor tasks without force, stress, or other coordinate/lattice derivatives, compare three retrained alternatives when authorized: full equivariance, a scalar/low-order backbone with a Cartesian or irreducible output head, and canonicalization plus a scalar backbone. Canonicalization is valid only if the frame is well-defined and stable over the actual symmetry/cell distribution, and the output is transformed back correctly. If no loss or evaluation term differentiates through positions or lattice, avoid creating coordinate/lattice autograd graphs; use `torch.inference_mode()` for eligible inference paths.

## 7. Couple graph, compile, and generation choices to their real regime

For static property graphs, precompute only topology and metadata invariant under the supported data contract: edge index/order, PBC shifts, CSR/CSC pointers, species indices, and compatible basis metadata. Dynamic generation coordinates/cells invalidate distances, edge vectors, and spherical harmonics; cache only timestep-independent values.

For ragged training, first test regional `torch.compile(..., dynamic=True)` and move data-dependent shape changes out of the compiled region. For generation, repeated denoiser calls under one shape bucket can justify `torch.compile`, `reduce-overhead`, and CUDA Graph candidates because the compile cost amortizes over actual NFE. Treat changed NFE, solver, flow matching, distillation, and one/few-step methods as separate algorithmic experiments.

Increase generation batch size only through a measured memory/latency sweep. Disable complete denoising-trajectory recording when it is not an output requirement; it can dominate host serialization and I/O. Cache conditioning, masks, batch indices, and species metadata only when they are invariant across NFE.

## 8. Priority and acceptance

| Priority | Use after evidence or authorization |
|---|---|
| P0 | Operation-level profiling, correct layout/backend dispatch, fused/indexed aggregation, BF16 with FP32 islands, atoms/edges/path-aware measurement |
| P1 | Channelwise/depthwise or product-basis architecture, OpenEquivariance, output-basis reduction, canonicalization/Cartesian alternatives, compile buckets, cost-aware DDP packing |
| P2 | Streaming TP, Gaunt/VSTP/SpinGTP, equivariant FP8/INT8, structural irrep pruning, one/few-step manifold generators |

An accepted architecture candidate preserves the relevant tensor/equivariance/physical/quality gates and wins on the declared fixed-budget or fixed-quality measure. A local kernel speedup, lower parameter count, or nominal complexity reduction alone is inconclusive.
