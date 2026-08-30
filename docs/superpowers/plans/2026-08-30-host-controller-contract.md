# DeepSeek Host Controller Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce the approved DeepSeek host-controller budget partitions, preserve receipts on controlled exhaustion, and materialize a reproducible trusted-host contract without changing frozen benchmark evidence.

**Architecture:** Extend the existing trusted-host controller with one immutable limits object parsed from `SPE_BUDGET_JSON`; the provider loop accounts authoritative provider usage and routes every exit through one receipt path. A standalone materializer derives a runtime manifest and validated agent configuration from the committed controller.

**Tech Stack:** Python 3.11 standard library, pytest, `benchmark.formal.reference_executor`, `benchmark.formal.attest`, and `benchmark.formal.budget`.

**Spec:** `docs/superpowers/specs/2026-08-30-host-controller-contract-design.md`

## Global Constraints

- Work only in `/home/workspace/lrh/MAINTENANCE/evospe-host-controller` on `codex/host-controller-contract`; canonical `main` stays at `dce4d194354b998d94ac9eb5b02a3d3d804828c9`.
- Run project commands on `dbcloud` using `/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python`.
- Do not change scientific inputs, gates, time budgets, resource semantics, or frozen evidence.
- Thinking is enabled with medium effort. Limits: input 65,536; output including reasoning 16,384; one response 4,096; context 32,768; tools 24; turns 12; wall 900s; request 120s; attempts 2 with fixed 1s backoff.
- Require campaign budget `tokens=81920`, `tool_calls=24`, `wall_time_s=900`, `context_tokens=32768`.
- Budget exhaustion is a non-retried outcome failure. Never persist credentials or raw reasoning.
- Use `runtime_attestation_kind=trusted_host_conda_prefix_manifest`, `container_digest_semantics=runtime_manifest_digest`, and `pricing_revision=unpriced-provider-probe-2026-08-30`.
- Do not run Q3, RC8, sealed A/B/C/D, or sealed35.

---

### Task 1: Frozen Limits and Agent Attestation

**Files:**
- Create: `benchmark/tests/test_host_agent_contract.py`
- Modify: `benchmark/formal/host_agent.py`

**Interfaces:**
- Consumes: exact `SPE_BUDGET_JSON` campaign object.
- Produces: `ControllerLimits`; `parse_controller_limits(raw: str) -> ControllerLimits`; updated `build_agent_attestation(...) -> dict[str, Any]`.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations
import json
import pytest
from benchmark.formal import host_agent

FROZEN_BUDGET = {"tokens": 81_920, "tool_calls": 24, "wall_time_s": 900, "context_tokens": 32_768}

def test_parse_controller_limits_accepts_exact_budget() -> None:
    value = host_agent.parse_controller_limits(json.dumps(FROZEN_BUDGET))
    assert (value.input_tokens, value.output_tokens, value.max_output_tokens) == (65_536, 16_384, 4_096)
    assert (value.context_tokens, value.tool_calls, value.turns) == (32_768, 24, 12)
    assert (value.wall_time_s, value.api_timeout_s) == (900, 120)

@pytest.mark.parametrize(("field", "value"), [("tokens", 81_919), ("tool_calls", 25), ("wall_time_s", 901), ("context_tokens", 4_096)])
def test_parse_controller_limits_rejects_contract_drift(field: str, value: int) -> None:
    budget = dict(FROZEN_BUDGET); budget[field] = value
    with pytest.raises(ValueError, match=field):
        host_agent.parse_controller_limits(json.dumps(budget))

def test_agent_attestation_records_policy_and_runtime_semantics() -> None:
    value = host_agent.build_agent_attestation(model_list_digest="a"*64, agent_code_commit="b"*40, container_digest="c"*64, pricing_revision="unpriced-provider-probe-2026-08-30")
    assert value["thinking"] == "enabled" and value["reasoning_effort"] == "medium"
    assert value["max_output_tokens"] == 4_096
    assert value["input_token_limit"] == 65_536 and value["output_token_limit"] == 16_384
    assert value["turn_limit"] == 12 and value["timeout_s"] == 900
    assert value["runtime_attestation_kind"] == "trusted_host_conda_prefix_manifest"
    assert value["container_digest_semantics"] == "runtime_manifest_digest"
```

- [ ] **Step 2: Verify RED**

Run: `/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python -m pytest benchmark/tests/test_host_agent_contract.py -q`

Expected: missing `parse_controller_limits`, and old `8192`/`high` attestation assertions fail.

- [ ] **Step 3: Implement the minimum limits contract**

```python
@dataclass(frozen=True)
class ControllerLimits:
    input_tokens: int = 65_536
    output_tokens: int = 16_384
    max_output_tokens: int = 4_096
    context_tokens: int = 32_768
    tool_calls: int = 24
    turns: int = 12
    wall_time_s: float = 900.0
    api_timeout_s: float = 120.0

def parse_controller_limits(raw: str) -> ControllerLimits:
    value = json.loads(raw)
    if not isinstance(value, dict): raise ValueError("SPE_BUDGET_JSON must be an object")
    limits = ControllerLimits()
    expected = {"tokens": limits.input_tokens + limits.output_tokens, "tool_calls": limits.tool_calls, "wall_time_s": limits.wall_time_s, "context_tokens": limits.context_tokens}
    for key, required in expected.items():
        if value.get(key) != required: raise ValueError(f"SPE_BUDGET_JSON {key} must equal {required}")
    return limits
```

Import `dataclass`; update the existing attestation with the tested values while retaining provider/model/digest/retry/decoding fields.

- [ ] **Step 4: Verify GREEN and commit**

Run the Task 1 command; expected all Task 1 tests pass. Then:

```bash
git add benchmark/formal/host_agent.py benchmark/tests/test_host_agent_contract.py
git commit -m "feat: freeze host controller limits"
```

### Task 2: Usage Accounting and Controlled Outcome Receipts

**Files:**
- Modify: `benchmark/tests/test_host_agent_contract.py`
- Modify: `benchmark/formal/host_agent.py`

**Interfaces:**
- Produces: `UsageState`; `BudgetOutcome(reason: str)`; `_remaining_output_tokens(limits, usage) -> int`; receipt `host_controller.outcome`.

- [ ] **Step 1: Write failing accounting tests**

```python
def test_remaining_output_allowance_shrinks() -> None:
    assert host_agent._remaining_output_tokens(host_agent.ControllerLimits(), host_agent.UsageState(output_tokens=15_900)) == 484

def test_provider_input_partition_is_outcome_failure() -> None:
    usage = host_agent.UsageState(input_tokens=65_000)
    with pytest.raises(host_agent.BudgetOutcome, match="input_token_limit"):
        usage.add_provider_usage({"prompt_tokens": 600, "completion_tokens": 1}, host_agent.ControllerLimits())
    assert (usage.input_tokens, usage.output_tokens) == (65_600, 1)

def test_provider_output_partition_counts_reasoning_output() -> None:
    usage = host_agent.UsageState(output_tokens=16_300)
    with pytest.raises(host_agent.BudgetOutcome, match="output_token_limit"):
        usage.add_provider_usage({"prompt_tokens": 10, "completion_tokens": 100}, host_agent.ControllerLimits())
    assert usage.output_tokens == 16_400

def test_tool_limit_blocks_extra_execution() -> None:
    usage = host_agent.UsageState(tool_calls=24)
    with pytest.raises(host_agent.BudgetOutcome, match="tool_call_limit"):
        usage.reserve_tool_call(host_agent.ControllerLimits())
    assert usage.tool_calls == 24
```

- [ ] **Step 2: Verify RED**

Run the focused test file; expected missing accounting types/functions.

- [ ] **Step 3: Implement accounting**

`BudgetOutcome(RuntimeError)` stores `reason`. `UsageState.add_provider_usage` adds provider-observed prompt/input and completion/output counts before raising at `>=` either partition. `reserve_tool_call` raises before call 25. `_remaining_output_tokens` returns `min(4096, 16384-output_tokens)` and raises when no allowance remains.

- [ ] **Step 4: Verify GREEN**

Run focused tests; expected all current tests pass.

- [ ] **Step 5: Add failing fake-provider integration tests**

Use a test-local `ThreadingHTTPServer` with queued OpenAI-compatible responses and request capture. Exercise `run()` through a temporary public worker and test substitute for namespace execution. Assert the first request has thinking enabled, medium effort, and `max_tokens=4096`; a later request with 16,000 observed output requests 384; a no-content/no-tools response is `incomplete_agent_outcome`; 12 turns become `turn_limit`; pre-request wall exhaustion becomes `wall_time_limit`; call 25 is not executed; a transient API failure is attempted exactly twice; a `BudgetOutcome` is never retried. Assert controlled exits still write provider-observed usage and a valid executor receipt, and serialized receipt/result files contain neither API key nor `reasoning_content`.

- [ ] **Step 6: Verify integration RED**

Run focused tests; expected failures against the old 16-turn, 8192-token, high-effort loop and missing receipt-on-outcome path.

- [ ] **Step 7: Implement one final receipt path**

Parse limits before process launch. Check wall/output allowance before each request. Use remaining output in payload, 12 turns, 120s timeout, enabled thinking, medium effort. Account usage immediately after each response, keep reasoning only in in-memory provider messages, reserve each tool before execution, and classify empty final/no-tools as an incomplete outcome. Always shut down the tool server and run the existing receipt canary. Write usage, `usage_meter_source=trusted-host-controller-provider-observed`, sanitized controller metadata/outcome, `agent_usage.json`, and `worker_result.json` before returning nonzero for controlled outcomes. Re-raise provider/executor failures after receipt handling.

- [ ] **Step 8: Verify GREEN and compatibility**

Run:

```bash
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python -m pytest benchmark/tests/test_host_agent_contract.py benchmark/tests/test_formal_freeze_contract.py -q
```

Expected: zero failures and stable prefix/tool schema tests remain green. Commit:

```bash
git add benchmark/formal/host_agent.py benchmark/tests/test_host_agent_contract.py
git commit -m "feat: enforce host controller budget partitions"
```

### Task 3: Runtime and Contract Materializer

**Files:**
- Create: `scripts/materialize_host_controller_contract.py`
- Modify: `benchmark/tests/test_host_agent_contract.py`

**Interfaces:**
- CLI consumes `--controller-root`, `--provider-probe`, `--out`.
- Produces `runtime_manifest.json`, `host_controller_contract.json`; functions `canonical_digest`, `build_runtime_manifest`, `build_contract`, and `validate_contract`.

- [ ] **Step 1: Write failing materializer tests**

Create a temporary clean one-commit controller Git root and sanitized probe fixture. Assert deterministic output, Python executable/version, OS identity, controller commit/source digest, `container_digest == canonical_digest(runtime)`, required runtime semantics/unpriced revision, campaign budget 81,920/24/900/32,768, medium effort, and `validate_contract(contract) == []`. Assert credential-looking probe keys and malformed provider model-list digests are rejected.

- [ ] **Step 2: Verify RED**

Run focused tests; expected import failure because the script is absent.

- [ ] **Step 3: Implement the materializer**

Use standard library only. Require a clean committed controller root. Record `sys.executable`, `platform.python_version()`, `platform.platform()`, commit, source digest, and versions of imported non-stdlib packages. Write sorted canonical JSON with newline and derive its SHA-256 digest. Build attestation via `host_agent.build_agent_attestation`, add campaign/runtime semantics, then embed `agent_config` in a smallest valid public-dev `SPE-EvoBench-v1.0-50` experiment and call `attest.validate_experiment`. Refuse existing output files and never copy arbitrary provider-probe fields.

- [ ] **Step 4: Verify GREEN and commit**

Run focused tests; expected zero failures. Then:

```bash
git add scripts/materialize_host_controller_contract.py benchmark/tests/test_host_agent_contract.py
git commit -m "feat: materialize trusted host controller contract"
```

### Task 4: Validation and Non-Sealed Evidence

**Files:**
- Generate only under `/home/workspace/lrh/RESULTS/shared-session-qualification-dce4d19-gpu5-20260830-clean-window/qualification/host-controller-contract/`.

**Interfaces:**
- Produces validated artifacts, one public provider/tool receipt, and a non-sealed launch binding referencing exact controller commit/digests.

- [ ] **Step 1: Run affected validation**

```bash
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python -m pytest benchmark/tests/test_host_agent_contract.py benchmark/tests/test_formal_freeze_contract.py benchmark/tests/test_formal.py benchmark/tests/test_formal_materializer.py benchmark/tests/test_formal_treatment.py -q
```

Any failure blocks materialization and gets a new RED→GREEN cycle.

- [ ] **Step 2: Verify identities**

Confirm controller branch is clean/committed and canonical main remains exact `dce4d194354b998d94ac9eb5b02a3d3d804828c9`.

- [ ] **Step 3: Materialize**

```bash
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python scripts/materialize_host_controller_contract.py --controller-root /home/workspace/lrh/MAINTENANCE/evospe-host-controller --provider-probe /home/workspace/lrh/RESULTS/shared-session-qualification-dce4d19-gpu5-20260830-clean-window/qualification/provider_probe.json --out /home/workspace/lrh/RESULTS/shared-session-qualification-dce4d19-gpu5-20260830-clean-window/qualification/host-controller-contract
```

Expected: exactly two validated canonical JSON artifacts.

- [ ] **Step 4: Run one public provider/tool-call probe**

Stream local `.env` provider values over SSH stdin into process environment without printing or writing secrets. Use only a synthetic public worker requiring one allowlisted tool and the exact frozen budget JSON. Verify tool trace, provider usage, valid receipt, and absence of reasoning/credentials. Never point at sealed or efficacy inputs.

- [ ] **Step 5: Bind launch plan and attempt one push**

Attach controller commit/command, runtime and contract digests, provider/model identity, and unpriced revision to the existing no-agent bundle while preserving `results_claimed=false`. Attempt one normal `git push -u origin codex/host-controller-contract`; on the known network failure, preserve local commits and do not alter remotes/retry repeatedly.

- [ ] **Step 6: Final review**

Confirm tests, branch cleanliness, canonical identity, contract validation, sanitized receipt, and unchanged evidence. Report Q3/RC8/sealed as not started and require a separate explicit decision before Q3.
