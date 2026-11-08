"""Live smoke tests against the real Claude API (skipped without an API key, e.g. in CI).

They spend a few cents: run with ``pytest -m live`` after changing a tool schema or prompt.
"""

import pytest

from gpl.agent import analyst as agent
from gpl.client import ClientProfile

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not agent.has_credentials(), reason="no ANTHROPIC_API_KEY"),
]


@pytest.fixture(scope="module")
def setup():
    from gpl.data import load_universe

    cfg = load_universe()
    return agent.make_client(), cfg["agent"]["model"], cfg


SHORT = "Risky investor aiming for high returns in Europe"
LONG = (
    "Meeting notes, 6 Nov. Client: Ms Layla Haddad, 41, Lebanese national living in Dubai "
    "(DIFC booking), UAE tax resident, no US ties. Married, two children aged 9 and 12. "
    "Senior engineer at a utility, salaried; plans to retire at 60. Portfolio USD 1.2 million "
    "from salary savings and an inheritance; adds about 60k a year. Has invested for 12 years "
    "in equities and funds. Wants growth over 20 years, at least 15% in the Gulf, no tobacco "
    "or weapons, and can't accept losing more than 30%. Needs USD 150k in Sep 2028 for school "
    "fees. Reports in USD."
)


def test_short_note_fills_the_form(setup) -> None:
    from gpl.agent.profile import apply_proposal, propose_profile

    client, model, _ = setup
    proposal = propose_profile(client, model, SHORT)
    after = apply_proposal(ClientProfile(answers={"tradeoff": 2}), proposal.draft)
    assert after.answers == {"tradeoff": 2}  # the questionnaire is never filled
    assert proposal.summary  # it recorded something, and said what
    assert proposal.numbers_not_in_notes == []  # no invented numbers


def test_long_note_fills_the_form(setup) -> None:
    from gpl.agent.profile import apply_proposal, propose_profile

    client, model, _ = setup
    proposal = propose_profile(client, model, LONG)
    after = apply_proposal(ClientProfile(answers={"tradeoff": 2}), proposal.draft)
    assert after.answers == {"tradeoff": 2}
    assert "Layla" in after.name and after.age == 41
    assert after.portfolio_value_usd == 1_200_000 and after.horizon_years == 20
    assert after.home_bias_gcc_pct == 15 and after.max_acceptable_drawdown_pct == 30
    assert after.liquidity_needs and after.liquidity_needs[0]["date"] == "2028-09"
    assert after.liquidity_needs[0]["amount_usd"] == 150_000
    assert after.dependants == 2 and sorted(after.dependant_ages) == [9, 12]
    assert after.booking_location == "DIFC (Dubai)"
    # "two children" is written in words, so 2 is flagged for the adviser to confirm: the
    # number check only recognises digits (a documented limitation)
    assert set(proposal.numbers_not_in_notes) <= {"2"}


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
