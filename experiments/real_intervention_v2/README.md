# Real fixed-intervention study (v2)

The intervention is fixed eager inference versus
torch.compile(..., backend="inductor", mode="default", dynamic=None). It is
an evaluation of an existing PyTorch optimization, not an EvoSPE-generated
optimization or a scientific-training result.

## Latest measured cohort

Run: evospe-rt-v2-real-compile-support-20261018-deterministic-metadata-b1-r2

- TorchBench: 61227cc119913f01e78e8e115a68208c4ca0541b
- Execution source: dd2939ea295ed86f0c1c0ade2ccaa5fb57bbda4a
- Cohort: all six pinned TorchBench models declaring eval_deterministic: true; batch size 1 only. Admission retained four lineages and recorded two unavailable candidates without replacement.
- Results: 12 contexts, 24/24 paired sessions, 4 lineages (2 development, 1 calibration, 1 final); underpowered against the 18-lineage final-claim minimum.
- A/A: 2/2 controls passed. Eight of 24 main sessions were paired-valid; 16 were invalid/unresolved.
- Development: 3 valid contexts, all negative for compile; mean log benefit −0.559, lineage-bootstrap 95% CI [−0.967, −0.355]. Calibration had zero valid contexts. The single final lineage (MAML) had no valid arm outcomes.
- Compile failures were dominated by Triton cuda_utils.so load failures; MAML also failed the eager inference path with a missing autograd graph. ACRE abstained because no certified support rule was available.
- Provider calls: 0. GPU 1,011.424 s; CPU 789.419 s; wall 1,597.672 s; downloaded/cache bytes 561,498,722; result-root bytes 659,244,320.

This is exploratory TorchBench runtime evidence with an underpowered cohort.
It does not support an ACRE superiority, broad generalization, automatic Skill
generation, or scientific-dataset time-to-quality claim. The earlier
20261018-one-shot-r3 cohort remains preserved as an A/A validity stop; it
started no S0/S1 sessions. Do not resume either cohort, relax correctness, or
replace models based on their observed outcomes. Any follow-up requires a new
study identity and a predeclared correctness/admission plan.

Full receipts and the sanitized evidence bundle are in
/home/workspace/lrh/RESULTS/evospe-rt-v2/20261018-one-shot-deterministic-b1-r2
on dbcloud; the raw receipts remain server-side.
