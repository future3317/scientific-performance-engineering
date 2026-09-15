# ALFWorld runner bug root cause

## Reproduction

The saved smoke receipt contains `Action: go to fridge 1` on a turn whose
admissible commands contain `go to fridge 1`. A zero-provider regression now
drives that response through the same `DeepSeekClient.choose_action` response
extraction, `run_episode`, action parser, admissibility check, and
`env.step([action])` path.

The replay prints and asserts:

- provider raw response fields;
- the response field selected by the online runner;
- parser input and parsed action;
- `repr(parsed_action)`;
- `admissible_commands[0]`;
- the exact argument passed to `env.step`.

The replay selects `message.content`, parses `go to fridge 1`, confirms exact
membership in the admissible command list, and passes `['go to fridge 1']` to
the environment.

## Root cause

The old receipt path collapsed `message.content` and
`message.reasoning_content` into one `raw_model_output` field. It did not
persist the provider fields, the selected response field, or the parser input.
Consequently the original in-memory extraction decision could not be audited
from the failed receipt. The saved response alone was insufficient to locate
the loss boundary.

The official ALFWorld batch environment was separately tested with a real
train game. A legal command taken from `info["admissible_commands"][0]` was
passed as `env.step([action])`; the step count advanced and the observation
changed. The batch argument shape is therefore not the loss boundary.

## Fixed path

The runner now persists `provider_raw`, `parser_input`, and
`response_field_used`, checks `parsed_action in admissible_commands`
explicitly, and records the exact argument passed to `env.step`. DeepSeek is
called as `deepseek-flash` with thinking disabled.

The existing 8-episode receipts remain immutable diagnostics. No skill
synthesis, viability experiment, ACRE evidence collection, or held-out
evaluation was started from them.

## Fixed two-episode smoke

At runner commit `c40f3721cff75c5380e4eb8727deed2bb2a0745a`, the smoke used
model `deepseek-flash` with thinking disabled. The two-episode receipt is
stored at `results/external_real/alfworld_smoke_2/` on the server.

- provider output parsed: **2/2 episodes**;
- invalid actions: **0**;
- official environment steps: **120 total**;
- complete trajectories persisted: **2/2**;
- verified task success: **0/2** (not a runner pass criterion).

Each persisted turn includes the raw provider fields, selected response field,
parser input, parsed action, admissible commands, and the exact action argument
passed to the batched environment.
