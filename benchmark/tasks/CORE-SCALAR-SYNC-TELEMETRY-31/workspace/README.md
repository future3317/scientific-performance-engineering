# CORE-SCALAR-SYNC-TELEMETRY-31

This CUDA training-loop anchor emits a fixed 16-value telemetry vector once per
optimizer step. The baseline reads each value with `.item()`, while the oracle
transfers the complete device vector once. Model updates, metric definitions,
vector width, logging cadence, and work units must remain unchanged.
