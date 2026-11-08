"""Tests for filling the client form from plain English (the model transcribes, never scores)."""

import pytest

from gpl.agent.profile import (
    apply_proposal,
    clean_draft,
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
    assert proposal.numbers_not_in_notes == [] and proposal.problems == []
    assert "strict" not in client.requests[0]["tools"][0]  # too large for strict mode
    before = ClientProfile(reference="X", answers={"tradeoff": 2})
    after = apply_proposal(before, proposal.draft)
    assert after.answers == {"tradeoff": 2}  # never filled by the model
    assert after.portfolio_value_usd == 500_000 and after.sharia_only
    assert after.excluded_sectors == ["Tobacco"]
    # "can't lose more than 25%" is the client's capacity for loss, not an adviser override
    assert after.overrides == {} and after.max_acceptable_drawdown_pct == 25.0
    assert after.home_bias_gcc_pct == 20
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


def test_fact_find_facts_are_recorded_and_sentinels_mean_not_stated() -> None:
    from gpl.agent.profile import numbers_not_in_notes

    facts = {
        "nationality": "Emirati",
        "tax_residencies": ["UAE"],
        "marital_status": "Married",
        "dependants": 2,
        "dependant_ages": [8, 12],
        "occupation": "Surgeon",
        "employer": "",
        "employment_type": "",
        "retirement_age": 0,  # not stated
        "reporting_currency": "AED",
        "source_of_wealth": "Medical practice",
        "source_of_funds": "",
        "primary_objective": "Growth",
        "annual_contribution_usd": 0,  # none stated
        "years_investing": -1,  # not stated
        "products_used": ["Equities"],
    }
    after = apply_proposal(ClientProfile(), {**DRAFT, **facts})
    assert after.nationality == "Emirati" and after.dependants == 2
    assert after.dependant_ages == [8, 12] and after.reporting_currency == "AED"
    assert after.primary_objective == "Growth" and after.products_used == ["Equities"]
    assert after.retirement_age is None and after.years_investing is None
    assert after.employment_type == "" and after.pep is None  # never filled by the model
    notes = (
        "Emirati surgeon, 45, married with 2 kids aged 8 and 12, $500k, 15 years, "
        "property deposit 50k in 12 months, 20% GCC, can't lose more than 25%."
    )
    assert numbers_not_in_notes({**DRAFT, **facts}, notes) == []
    flagged = numbers_not_in_notes({**DRAFT, **facts, "retirement_age": 60}, notes)
    assert flagged == ["60"]


# ------------------------------------------------------------------ checking the output in code


def test_clean_draft_keeps_valid_facts_unchanged() -> None:
    draft, problems = clean_draft(DRAFT)
    assert problems == []
    for key, value in DRAFT.items():
        assert draft[key] == value, key
    assert draft["retirement_age"] is None and draft["products_used"] == []  # filled in


def test_clean_draft_drops_scores_and_bad_values() -> None:
    raw = {
        **DRAFT,
        "risk_score": 5,  # the AI never scores: not a form field
        "answers": {"tradeoff": 4},
        "age": "forty-five",  # not a number
        "horizon_years": 250,  # out of range
        "portfolio_value_usd": "500,000",  # a numeric string is fine
        "jurisdiction": "Mars",
        "marital_status": "complicated",
        "us_person": "maybe",
        "sharia_only": "true",
        "products_used": ["Equities", "Crypto lottery"],
        "dependant_ages": [7, 200, "x"],
        "region_minimums": [{"region": "Atlantis", "min_pct": 10},
                            {"region": "Saudi Arabia", "min_pct": 150}],
        "liquidity_needs": [{"description": "Deposit", "amount_usd": -5, "when": "2027-01"},
                            "not a dict"],
        "retirement_age": 0,  # sentinel: not stated
        "years_investing": -1,
    }  # fmt: skip
    draft, problems = clean_draft(raw)
    text = " ".join(problems)
    assert "risk_score" not in draft and "answers" not in draft
    assert "answers, risk_score" in text and "never sets risk answers or scores" in text
    _, junk = clean_draft({**DRAFT, "empty_placeholder": ""})
    assert junk == ["Ignored fields the form doesn't have: empty_placeholder."]
    assert draft["age"] is None and "not a number" in text
    assert draft["horizon_years"] is None and "outside 0-100" in text
    assert draft["portfolio_value_usd"] == 500_000.0
    assert draft["jurisdiction"] is None and draft["marital_status"] == ""
    assert draft["us_person"] is None and draft["sharia_only"] is True
    assert draft["products_used"] == ["Equities"] and draft["dependant_ages"] == [7]
    assert draft["region_minimums"] == [] and draft["liquidity_needs"] == []
    assert draft["retirement_age"] is None and draft["years_investing"] is None
    # what's left is safe to apply, and the questionnaire is untouched
    after = apply_proposal(ClientProfile(answers={"tradeoff": 2}), draft)
    assert after.answers == {"tradeoff": 2} and after.portfolio_value_usd == 500_000.0


def test_clean_draft_survives_garbage() -> None:
    draft, problems = clean_draft("not even a dict")
    assert draft["name"] == "" and draft["goals"] == [] and problems == []


def test_propose_reports_what_was_dropped() -> None:
    raw = {**DRAFT, "risk_score": 9, "age": 300}
    client = FakeClient([response([tool_use("p1", "propose_profile", raw)], "tool_use")])
    proposal = propose_profile(client, "m", NOTES)
    assert proposal.draft["age"] is None and "risk_score" not in proposal.draft
    assert any("risk_score" in p for p in proposal.problems)


def test_truncated_reply_asks_for_shorter_notes() -> None:
    client = FakeClient([response([text("…")], "max_tokens")])
    with pytest.raises(ValueError, match="too long"):
        propose_profile(client, "m", NOTES)
