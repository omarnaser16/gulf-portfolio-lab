"""Tests for filling the client form from plain English (the model transcribes, never scores)."""

import pytest

from gpl.agent.profile import (
    apply_proposal,
    numbers_in_text,
    numbers_not_in_notes,
    profile_tool,
    propose_profile,
)
from gpl.client import ClientProfile
from test_agent import FakeClient, response, text, tool_use

NOTES = (
    "Client in Dubai, 45, portfolio of $500k, needs 50k in 12 months for a property deposit. "
    "Wants Sharia-compliant investments only, no tobacco, at least 20% in the Gulf. Retiring "
    "in 15 years; can't stomach losing more than 25%."
)
DRAFT = {
    "horizon_years": 15,
    "return_target_pct": None,
    "portfolio_value_usd": 500_000,
    "liquidity_need_usd": 50_000,
    "liquidity_months": 12,
    "residency": "UAE",
    "us_person": None,
    "classification": None,
    "sharia_only": True,
    "excluded_sectors": ["tobacco"],
    "home_bias_gcc_pct": 20,
    "max_volatility_pct": None,
    "max_drawdown_pct": 25,
    "goals": [
        {"description": "Property deposit", "amount_usd": 50_000, "multiple": None, "years": None}
    ],
    "unmapped_requests": [],
    "summary": "Dubai client, Sharia only, retiring in 15 years.",
}


def test_tool_has_no_scores_or_weights() -> None:
    props = profile_tool()["input_schema"]["properties"]
    assert not {"answers", "risk_score", "capacity", "willingness", "weights"} & set(props)


def test_numbers_with_suffixes() -> None:
    assert {500.0, 500_000.0, 1.5, 1_500_000.0} <= numbers_in_text("$500k and 1.5 million")
    assert 2.0 in numbers_in_text("end with 2x his money")
    assert 12_000_000.0 not in numbers_in_text("within 12 months")  # "m" of months isn't million
    assert numbers_not_in_notes(DRAFT, NOTES) == []
    assert numbers_not_in_notes({**DRAFT, "return_target_pct": 7}, NOTES) == ["7"]


def test_propose_and_apply_keep_the_questionnaire() -> None:
    client = FakeClient([response([tool_use("p1", "propose_profile", DRAFT)], "tool_use")])
    proposal = propose_profile(client, "m", NOTES)
    assert proposal.numbers_not_in_notes == [] and client.requests[0]["tools"][0]["strict"]
    before = ClientProfile(reference="X", answers={"tradeoff": 2})
    after = apply_proposal(before, proposal.draft)
    assert after.answers == {"tradeoff": 2}  # never filled by the model
    assert after.portfolio_value_usd == 500_000 and after.sharia_only
    assert after.excluded_sectors == ["Tobacco"]
    assert after.overrides == {"max_drawdown_pct": 25.0} and after.home_bias_gcc_pct == 20
    assert after.goals == [{"description": "Property deposit", "amount_usd": 50_000}]
    assert after.us_person is False  # null leaves the existing value


def test_no_tool_call_is_an_error() -> None:
    client = FakeClient([response([text("Sounds like a balanced investor.")], "end_turn")])
    with pytest.raises(ValueError, match="did not fill the form"):
        propose_profile(client, "m", NOTES)
