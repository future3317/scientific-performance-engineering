# DeepSeek Host Controller Contract Design

## Goal

Provide a real, reproducible OpenAI-compatible DeepSeek host controller for EVOSPE readiness without changing the frozen benchmark `main` revision `dce4d194354b998d94ac9eb5b02a3d3d804828c9` or invalidating its census and qualification evidence.

The controller is developed and committed in the isolated branch `codex/host-controller-contract`, rooted at `dce4d19`. The canonical benchmark checkout remains unchanged and invokes the controller by its separate committed path and commit identity.

## Existing Boundary

`benchmark/formal/host_agent.py` already implements the trusted-host/provider boundary and a network-free tool server. The missing contract is enforceable per-trial token partitioning: the current controller requests up to 8,192 output tokens on each of 16 turns while the campaign default allows only 12,000 total input plus output tokens.

The implementation updates only the controller branch. It does not change tasks, workloads, oracles, calibration thresholds, repetitions, measurement time budgets, resource-domain semantics, or the frozen evidence directories.

## Token and Thinking Policy

The controller keeps thinking enabled with medium reasoning effort. These tasks require diagnosis, source editing, testing, and preservation of scientific constraints; disabling reasoning would change the intended agent capability. High reasoning effort is not used because it increases hidden reasoning output without a demonstrated readiness benefit.

Per trial limits are:

- cumulative provider input: 65,536 tokens;
- cumulative provider output, including reasoning tokens: 16,384 tokens;
- maximum output requested from one API call: 4,096 tokens;
- declared context budget: 32,768 tokens;
- tool calls: 24;
- provider turns: 12;
- wall time: 900 seconds;
- one provider request timeout: 120 seconds;
- transient provider attempts: 2, with one-second fixed backoff.

The campaign's combined token budget is 81,920 tokens, exactly the sum of the input and output partitions. The controller reads `SPE_BUDGET_JSON`, rejects a launch whose combined budget disagrees with the partitions, decreases the per-call output request to the remaining output allowance, and stops with an outcome failure when provider-observed usage reaches a partition limit. It never retries a budget outcome.

The measured frozen efficacy subset has a maximum initial prompt of approximately 4,258 tokens. Its largest public task tree is 30.5 KB, approximately 7,600 additional tokens if fully read, and one command result is capped at 20,000 characters, approximately 5,000 tokens. The selected limits allow several inspect/edit/test turns while bounding cumulative repeated-context billing.

## Components

1. `benchmark/formal/host_agent.py`
   - parses the frozen controller limits from environment and `SPE_BUDGET_JSON`;
   - constructs provider payloads with thinking enabled, medium effort, and the remaining output allowance;
   - accounts provider-observed input, output, tool calls, and wall time after every turn;
   - writes authoritative usage and executor receipts on both success and controlled budget exhaustion;
   - keeps the existing three-tool allowlist and network-free tool execution boundary.

2. `benchmark/tests/test_host_agent_contract.py`
   - uses a local fake OpenAI-compatible HTTP server;
   - proves the first test fails before implementation;
   - covers payload policy, output allowance reduction, input/output exhaustion, tool-call accounting, receipt generation, and absence of secret persistence;
   - performs no external provider request.

3. `scripts/materialize_host_controller_contract.py`
   - builds the formal agent configuration from the committed controller, provider model-list digest, runtime manifest digest, pricing revision, and fixed policy;
   - validates the result with the existing formal attestation validator;
   - writes one canonical `host_controller_contract.json`.

4. Runtime and launch artifacts
   - a runtime manifest records the controller commit, Python executable and version, installed package versions required by the controller, operating-system identity, and controller source digest;
   - its canonical JSON digest populates the legacy-required `container_digest` field;
   - `runtime_attestation_kind=trusted_host_conda_prefix_manifest` and `container_digest_semantics=runtime_manifest_digest` make clear that no OCI container is claimed;
   - the launch plan passes provider credentials from the local `.env` over SSH standard input into process-scoped environment variables. API keys are never written to disk or included in receipts.

The readiness contract records `pricing_revision=unpriced-provider-probe-2026-08-30`. This is an explicit statement that Q3 readiness is token-accounted but carries no monetary-cost claim. It must be replaced by a cited provider price-sheet revision before any formal sealed campaign can be authorized.

## Data Flow

The canonical `dce4d19` campaign driver creates a public worker root and invokes the committed controller command. The controller reads only the public worker bundle, calls the configured OpenAI-compatible DeepSeek endpoint, and forwards model tool requests to the network-free `ReferenceExecutor` tool server. The tool server alone reads permitted public mounts, writes allowed solution files, and runs allowed workspace commands. Provider usage is accumulated by the controller and written to the campaign receipt. The provider never receives sealed efficacy outside a separately authorized formal campaign.

## Failure Semantics

- Authentication, malformed provider responses, unsupported thinking/tool payloads, and exhausted retry attempts are controller failures. They do not trigger repeated campaign attempts.
- Input, output, tool-call, turn, or wall-time exhaustion is an agent outcome failure with complete usage evidence, not infrastructure-invalid evidence.
- Tool server or executor receipt failures remain infrastructure-invalid under the existing campaign policy.
- A provider response that consumes reasoning tokens but produces no final content is valid only when it contains tool calls; otherwise the controller reports an incomplete agent outcome.
- Secrets and raw reasoning content are never stored in contract or attestation artifacts.

## Validation Sequence

1. Run the new focused tests and observe the expected missing-contract failure.
2. Implement the smallest controller changes that pass those tests.
3. Run the focused host-controller and formal provenance tests.
4. Run the complete static/formal test slice required by the repository.
5. Commit and push `codex/host-controller-contract` without moving canonical `main`.
6. Materialize and validate the runtime manifest and host-controller contract.
7. Run one non-sealed OpenAI-compatible provider/tool-call probe using ephemeral credentials.
8. Rebuild the non-agent launch bundle with the controller contract attached.

No Q3, RC8, sealed A/B/C/D, or sealed35 execution is authorized by this design.
