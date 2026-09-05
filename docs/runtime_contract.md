# Hosted controller runtime contract

This document records the model-facing and executor-facing contract implemented
by `benchmark/formal/host_agent.py`. It is an execution contract, not an
efficacy claim.

## Fixed model-facing configuration

- Provider: DeepSeek-compatible `/chat/completions` endpoint.
- Model identity: `deepseek-v4-flash` in the registered hosted configuration.
- Thinking/reasoning: disabled (`thinking.type=disabled`, reasoning effort none).
- Temperature: `0`; `top_p=1`.
- Per-cell limits: input 65,536; output 16,384; maximum request output 4,096;
  context 32,768; 24 tool calls; 16 turns; 900 seconds wall time.
- Tool allowlist: `read_file`, `write_file`, and `run_command`.
- Tool arguments must decode to a JSON object; malformed or non-object arguments
  become an agent-visible tool error and do not mutate the scientific contract.
- Transient provider failures may use the registered two-attempt retry with a
  one-second delay; non-transient errors stop the cell.

## Worker and information boundary

The trusted host owns provider calls. The executor starts a network-disabled
worker namespace. The worker can read only public task, retrieved-context,
skill-view, context-state, and solution mounts; it can write only allowlisted
solution files and run commands in the solution workspace. Hidden oracle,
hidden verifier, anti-cheat details, sealed schedule, and repository root are
not exposed through the controller tool server.

Every completed or outcome-failure cell receives an executor canary and a
receipt with usage, model, fixed-prefix digest, tool-schema digest, tool trace,
runtime attestation, and outcome. A partial controller telemetry record is
separate diagnostic evidence and never substitutes for an executor receipt.

## Near-budget finalization

Before each provider request the controller estimates the next serialized
message size. At the last turn, or when the next request would cross the
65,536-input-token threshold after preserving a 4,096-token finalization
reserve, it sends a fixed finalize instruction before allowing another
exploratory request. Telemetry records the trigger point, remaining reserve,
candidate presence, public-preflight status, and submit status. The policy is
condition-symmetric and does not inspect hidden verifier state.

## Failure and evidence semantics

Input/output/tool/turn/wall limits are recorded as controller outcomes. They do
not create a fake worker result or efficacy observation. `BudgetOutcome` paths
persist partial turn traces and candidate state when configured. Infrastructure,
executor, sandbox, and protocol failures remain distinct from agent correctness
failures and remain invalid for scientific analysis.

The controller does not change tasks, prompts, budgets, scientific gates, or
condition semantics. Any such change requires a new benchmark/controller
revision and a new evidence identity.
