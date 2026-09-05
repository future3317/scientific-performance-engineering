# EVOSPE revision history

This is a compact release-boundary record. It distinguishes code revisions
from experiment identities and keeps historical evidence immutable.

| Revision | Role | Status |
|---|---|---|
| `dce4d194354b998d94ac9eb5b02a3d3d804828c9` | Historical formal benchmark | Immutable; 228-cell observation-boundary case only |
| `59fe1a5bd4af06d368eecb1ffd6e50df161fb384` | Resource-aware provider-free method revision | Current mechanism evidence |
| `7995dac98d4107e8048075b8d805e300a574cc64` | Seeded reserve-stress runner | Current provider-free stress evidence |
| `f104f2d59701ffeeef622099f4fca33c9fc4c561` | Hosted public-contract eligibility revision | Current hosted feasibility boundary |
| `d72219a490a4c2e4c0bb6ea1fe5bb40be20468c2` | Hosted controller no-thinking/telemetry revision | Current hosted feasibility boundary |
| `EVOSPE_EVIDENCE_FREEZE_20260905` | Cross-repository evidence freeze | Benchmark tag `6282c0d77b453a28eef887f1a59a2b51fa5cf5ab`; controller tag `097a36a0c49812ba20d626310040f7e84f2192b3`; paper tag `cafee6a406f2d87d949a1b3c0cd4bedabdbfc1af` |

The hosted v3 protocol digest is
`1de6c4fd08e1a0c4324d3d4b14c974690688c18ff637f6a416c9c85b5cbf0346` and its
recovery-manifest digest is
`e3150492080b6f03c497dd1bc44865b4bf095f130e261bda460d931ed45f0c89`.

The repository merge target is `main`; feature branches are integration
vehicles, not additional scientific revisions.

## Frozen claim boundary

The authoritative mapping is
`docs/EVOSPE_EVIDENCE_FREEZE_20260905.json` (manifest SHA256
`b978c74f52b0f10a1b595d84c26c84f5ba5ab8278c161fea20d7bf80ad404b0c`). The
freeze supports provider-free mechanism/resource evidence only:

```text
HOSTED_EFFICACY_STATUS = UNSUPPORTED
HOSTED_FEASIBILITY_STATUS = INSUFFICIENT
SEALED_C_D_STATUS = NOT_RUN
RESULTS_CLAIMED = FALSE
```

The historical formal revision `dce4d194354b998d94ac9eb5b02a3d3d804828c9`
remains immutable and is used only as the treatment-coupled observation-boundary
case. The server `RESULTS` tree is evidence storage, not a cleanup target.
