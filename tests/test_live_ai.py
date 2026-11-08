"""Live smoke tests against the real Claude API (skipped without an API key, e.g. in CI).

They spend a few cents: run with ``pytest -m live`` after changing a tool schema or prompt.
"""

import pytest

from gpl.agent import analyst as agent

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not agent.has_credentials(), reason="no ANTHROPIC_API_KEY"),
]


@pytest.fixture(scope="module")
def setup():
    from gpl.data import load_universe

    cfg = load_universe()
    return agent.make_client(), cfg["agent"]["model"], cfg


def test_analyst_tools_compile_with_a_large_universe(setup, ctx) -> None:
    from dataclasses import replace

    from gpl.agent.tools import tool_definitions
    from gpl.data import load_investable

    client, model, _ = setup
    big = replace(ctx, universe=list(load_investable()))  # 62 tickers in every enum
    tools = tool_definitions(big)
    assert all(t.get("strict") for t in tools)
    reply = client.messages.create(
        model=model, max_tokens=16, tools=tools, messages=[{"role": "user", "content": "hi"}]
    )
    assert reply.stop_reason in ("end_turn", "max_tokens", "tool_use")
