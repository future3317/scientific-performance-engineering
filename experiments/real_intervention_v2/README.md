# Real fixed-intervention study (v2)

This channel evaluates the fixed inference intervention
`torch.compile(..., backend="inductor", mode="default", dynamic=None)` against
eager execution. It does not claim that EvoSPE invented or generated this
optimization.

## Current run: A/A validity stop

The server run `evospe-rt-v2-real-compile-support-20261018-r1` used TorchBench
commit `61227cc119913f01e78e8e115a68208c4ca0541b`. Ten lineages were admitted
and 66 contexts / 132 paired sessions were scheduled. Before S1, five
pre-registered eager-vs-eager controls were completed: one passed output
correctness and four failed with `numerical_mismatch:output`. The A/A gate
therefore stopped the run. No S0/S1 session started, no policy was frozen, and
no final outcome was accessed. This is a validity stop, not an intervention
efficacy result.

The finalized result is in
`/home/workspace/lrh/RESULTS/evospe-rt-v2/20261018-one-shot-r3` on `dbcloud`.
It records zero provider calls, 214.516 GPU seconds, 78.992 CPU seconds,
223.553 wall seconds, 931,208,260 downloaded bytes, and 923,315,109 bytes in
the experiment directory. The sanitized evidence bundle contains the
manifest, aggregate outputs, and A/A receipts; raw server receipts remain on
the server. No efficacy or routing claim is supported by this run.

Do not resume this frozen cohort, relax the correctness threshold, or replace
models based on the observed A/A results. Any follow-up must establish a new
pre-registered study identity and pass correctness controls before measuring
S1.

## Frozen intervention contract

- `S0`: eager inference.
- `S1`: `torch.compile(module, backend="inductor", mode="default", dynamic=None)`.
- Primary performance metric, if a future valid run is authorized: total job
  wall-clock time including cold compilation.
- Correctness, A/A validity, and the declared resource contract are gates;
  failed controls are not performance observations.
- Policy comparisons, including the canonical `AcreEngine.route`, require
  valid paired outcomes. The pairwise oracle is post-hoc scoring only.
