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


def test_strict_tool_sets_compile_with_the_full_universe(setup) -> None:
    """propose_spec and the analyst's tools stay strict; check they fit the grammar limit."""
    from gpl.agent.spec import spec_tool
    from gpl.data import load_investable

    client, model, cfg = setup
    tickers = list(load_investable())  # 62 securities: the largest realistic tool enums
    tools = [spec_tool(tickers, list(cfg["groups"]))]
    reply = client.messages.create(
        model=model, max_tokens=16, tools=tools, messages=[{"role": "user", "content": "hi"}]
    )
    assert reply.stop_reason in ("end_turn", "max_tokens", "tool_use")


def test_spec_from_goals(setup) -> None:
    from gpl.agent.spec import propose_spec

    client, model, _ = setup
    proposal = propose_spec(
        client, model, "make it safer, at least 20% GCC", {"SPY": "Global equity", "KSA": "GCC"}
    )
    assert proposal.draft["goal"] == "reduce_risk"


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
