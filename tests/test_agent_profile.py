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
    "annual_income_need_usd": None,
    "name": "",
    "age": 45,
    "jurisdiction": "UAE onshore",
    "tax_notes": "",
    "liquidity_needs": [
        {
            "description": "Property deposit",
            "amount_usd": 50_000,
            "when": "in 12 months",
        }
    ],
    "region_minimums": [],
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


def _unions(schema) -> int:
    """Parameters with a type list or anyOf (the API allows at most 16 in a strict tool)."""
    if isinstance(schema, dict):
        own = int(isinstance(schema.get("type"), list) or "anyOf" in schema)
        return own + sum(_unions(v) for v in schema.values())
    if isinstance(schema, list):
        return sum(_unions(v) for v in schema)
    return 0


def test_tool_stays_within_the_api_union_limit() -> None:
    assert _unions(profile_tool()["input_schema"]) <= 16


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


# ------------------------------------------------------------------ the QA walkthrough client

QA_NOTES = (
    "Mr Khalid Al Mansoori, 58, booked in Abu Dhabi. USD 3m portfolio, draws 96k a year. "
    "Needs USD 600k in Oct 2028 for his daughter's wedding. Keep at least 10% in Saudi. "
    "UAE tax resident, no US ties; holds a UK property."
)
QA_DRAFT = {
    **{k: None for k in DRAFT},
    "name": "Khalid Al Mansoori",
    "age": 58,
    "jurisdiction": "ADGM (Abu Dhabi)",
    "portfolio_value_usd": 3_000_000,
    "annual_income_need_usd": 96_000,
    "tax_notes": "UAE tax resident, no US ties; holds a UK property.",
    "liquidity_needs": [
        {
            "description": "Daughter's wedding",
            "amount_usd": 600_000,
            "when": "2028-10",
        }
    ],
    "region_minimums": [{"region": "Saudi Arabia", "min_pct": 10}],
    "excluded_sectors": [],
    "goals": [],
    "unmapped_requests": [],
    "summary": "Abu Dhabi client, 58, USD 3m, USD 600k needed Oct 2028, Saudi >= 10%.",
}


def test_intake_dates_regulator_name_tax_and_regions() -> None:
    from datetime import date

    from gpl.client import build_ips
    from gpl.data import load_universe

    CFG = load_universe()

    client = FakeClient([response([tool_use("p1", "propose_profile", QA_DRAFT)], "tool_use")])
    proposal = propose_profile(client, "m", QA_NOTES)
    assert proposal.numbers_not_in_notes == [] and proposal.problems == []
    profile = apply_proposal(ClientProfile(), proposal.draft, today=date(2026, 10, 25))
    assert profile.name == "Khalid Al Mansoori" and profile.age == 58
    assert profile.regulator == "FSRA (ADGM)"  # Abu Dhabi -> ADGM -> FSRA, by rule
    assert profile.tax_notes.startswith("UAE tax resident")
    assert profile.liquidity_needs == [
        {"description": "Daughter's wedding", "amount_usd": 600_000.0, "date": "2028-10"}
    ]
    assert profile.region_minimums == {"Saudi Arabia": 10.0}
    assert profile.annual_income_need_usd == 96_000
    ips = build_ips(profile, CFG, today=date(2026, 10, 25))
    # Oct 2028 is 24 months away (about two years), not 12 months; it binds now (<= 36)
    assert [n["months"] for n in ips.limits["cash_needs_due"]] == [24]
    assert ips.limits["liquidity_reserve"] == pytest.approx(0.20)  # 600k of 3m
    assert ips.limits["region_minimums"] == {"Saudi Arabia": 0.10}
    assert "by Oct 2028 (24 months)" in ips.sections["Liquidity"]
    assert "Client age: 58." in ips.sections["Time horizon"]


def test_relative_needs_are_dated_by_code_and_missing_dates_flagged() -> None:
    from datetime import date

    from gpl.agent.profile import need_date

    today = date(2026, 10, 25)
    assert need_date({"when": "in 2 years"}, today) == "2028-10"
    assert need_date({"when": "in 18 months"}, today) == "2028-04"
    assert need_date({"when": "2028-10"}, today) == "2028-10"
    assert need_date({"when": "soon"}, today) is None and need_date({"when": ""}, today) is None
    undated = {
        **QA_DRAFT,
        "liquidity_needs": [
            {
                "description": "Car",
                "amount_usd": 80_000,
                "date": None,
                "in_years": None,
                "in_months": None,
            }
        ],
    }
    client = FakeClient([response([tool_use("p1", "propose_profile", undated)], "tool_use")])
    proposal = propose_profile(client, "m", QA_NOTES + " A car for 80k.")
    assert proposal.problems and "Car" in proposal.problems[0]
    # a date whose year isn't in the notes is flagged like any invented number
    wrong = {**QA_DRAFT, "liquidity_needs": [{**QA_DRAFT["liquidity_needs"][0], "when": "2029-10"}]}
    assert "2029" in numbers_not_in_notes(wrong, QA_NOTES)


def test_regulator_for_jurisdictions() -> None:
    from gpl.client import regulator_for

    assert regulator_for("ADGM (Abu Dhabi)") == "FSRA (ADGM)"
    assert regulator_for("DIFC (Dubai)") == "DFSA (DIFC)"
    assert regulator_for("UAE onshore") == "SCA (UAE)"
    assert regulator_for("Saudi Arabia") == "CMA (Saudi Arabia)"
    assert regulator_for("QFC (Qatar)") == "QFCRA (Qatar)"
    assert regulator_for(None) is None and regulator_for("Mars") is None


def test_new_gulf_jurisdictions_and_booking_location_recorded() -> None:
    from gpl.client import regulator_for

    assert regulator_for("Bahrain") == "CBB (Bahrain)"
    assert regulator_for("Kuwait") == "CMA (Kuwait)"
    assert regulator_for("Oman") == "FSA (Oman, formerly CMA)"
    after = apply_proposal(ClientProfile(), {"jurisdiction": "Bahrain"})
    assert after.booking_location == "Bahrain" and after.regulator == "CBB (Bahrain)"
