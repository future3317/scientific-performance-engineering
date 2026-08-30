from __future__ import annotations

import json

import pytest

from benchmark.formal import host_agent


FROZEN_BUDGET = {
    "tokens": 81_920,
    "tool_calls": 24,
    "wall_time_s": 900,
    "context_tokens": 32_768,
}


def test_parse_controller_limits_accepts_exact_budget() -> None:
    value = host_agent.parse_controller_limits(json.dumps(FROZEN_BUDGET))

    assert (value.input_tokens, value.output_tokens, value.max_output_tokens) == (65_536, 16_384, 4_096)
    assert (value.context_tokens, value.tool_calls, value.turns) == (32_768, 24, 12)
    assert (value.wall_time_s, value.api_timeout_s) == (900, 120)


@pytest.mark.parametrize(
    ("field", "value"),
    [("tokens", 81_919), ("tool_calls", 25), ("wall_time_s", 901), ("context_tokens", 4_096)],
)
def test_parse_controller_limits_rejects_contract_drift(field: str, value: int) -> None:
    budget = dict(FROZEN_BUDGET)
    budget[field] = value

    with pytest.raises(ValueError, match=field):
        host_agent.parse_controller_limits(json.dumps(budget))


def test_agent_attestation_records_policy_and_runtime_semantics() -> None:
    value = host_agent.build_agent_attestation(
        model_list_digest="a" * 64,
        agent_code_commit="b" * 40,
        container_digest="c" * 64,
        pricing_revision="unpriced-provider-probe-2026-08-30",
    )

    assert value["thinking"] == "enabled"
    assert value["reasoning_effort"] == "medium"
    assert value["max_output_tokens"] == 4_096
    assert value["input_token_limit"] == 65_536
    assert value["output_token_limit"] == 16_384
    assert value["turn_limit"] == 12
    assert value["timeout_s"] == 900
    assert value["runtime_attestation_kind"] == "trusted_host_conda_prefix_manifest"
