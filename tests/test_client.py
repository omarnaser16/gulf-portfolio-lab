"""Tests for the client profile, risk questionnaire, IPS and IPS -> optimiser limits."""

from dataclasses import replace
from datetime import date

import pytest

from gpl.client import (
    EXAMPLE_CLIENTS,
    QUESTIONS,
    ClientProfile,
    build_ips,
    cash_like_holdings,
    default_regulator,
    effective_regulator,
    horizon_score,
    ips_constraints,
    liquidity_need_score,
    month_after,
    month_text,
    months_until,
    parse_month,
    risk_profile,
    tighter,
)
from gpl.data import load_universe
from gpl.optimise import OptimisationSpec

CFG = load_universe()


def profile(**kw) -> ClientProfile:
    base = dict(
        reference="C-001",
        horizon_years=10,
        answers={
            "income_stability": 4,
            "wealth_share": 3,
            "dependence": 4,
            "liquidity_need": 4,
            "loss_reaction": 3,
            "tradeoff": 3,
            "experience": 4,
        },
    )
    return ClientProfile(**{**base, **kw})


# ------------------------------------------------------------------- questionnaire


def test_horizon_score_bands() -> None:
    assert [horizon_score(y) for y in (1, 3, 7, 12, 20)] == [1, 2, 3, 4, 5]


def test_risk_score_is_the_lower_of_capacity_and_willingness() -> None:
    # capacity: horizon 10y -> 4, income 4, wealth 3, dependence 4, liquidity 4: mean 3.8 -> 4
    # willingness: loss 3, trade-off 3, experience 4: mean 3.33 -> 3
    r = risk_profile(profile())
    assert (r.capacity, r.willingness, r.score) == (4, 3, 3)
    assert not r.mismatch
    assert "lower of capacity (4) and willingness (3)" in r.explanation


def test_half_rounds_up_and_mismatch_is_flagged() -> None:
    p = profile(
        horizon_years=20,
        answers={
            "income_stability": 5,
            "wealth_share": 5,
            "dependence": 5,
            "liquidity_need": 5,
            "loss_reaction": 1,
            "tradeoff": 1,
            "experience": 5,  # knowledge & experience: not part of willingness
        },
    )
    r = risk_profile(p)  # capacity 5; willingness mean (1 + 1) / 2 = 1
    assert (r.capacity, r.willingness, r.score) == (5, 1, 1) and r.mismatch
    p2 = profile(
        answers={**profile().answers, "loss_reaction": 4, "tradeoff": 3}
    )  # willingness mean 3.5 -> 4 (half rounds up)
    assert risk_profile(p2).willingness == 4
    p3 = profile(answers={**profile().answers, "loss_reaction": 3, "tradeoff": 3, "experience": 1})
    assert risk_profile(p3).willingness == 3  # experience doesn't move it
    assert "4 answers" in risk_profile(p3).explanation  # capacity: horizon + 4 answers


def test_experience_is_not_required_for_the_score() -> None:
    answers = {k: v for k, v in profile().answers.items() if k != "experience"}
    assert risk_profile(profile(answers=answers)).score is not None
    assert {q["id"] for q in QUESTIONS if q["dimension"] == "knowledge"} == {"experience"}


def test_liquidity_need_score_from_cash_needs() -> None:
    today = date(2026, 11, 1)
    need = {"description": "Deposit", "amount_usd": 50_000, "date": "2027-10"}
    p = profile(liquidity_needs=[need], portfolio_value_usd=500_000)
    assert liquidity_need_score(p, today) == (2, 0.10)  # 10% is not under 10% -> 2
    assert liquidity_need_score(replace(p, portfolio_value_usd=2_000_000), today)[0] == 4  # 2.5%
    assert liquidity_need_score(replace(p, portfolio_value_usd=600_000), today)[0] == 3  # 8.33%
    assert liquidity_need_score(replace(p, portfolio_value_usd=150_000), today)[0] == 1  # 33%
    assert liquidity_need_score(profile(), today) == (5, 0.0)  # no needs
    later = {**need, "date": "2029-06"}  # 31 months away: outside two years
    assert liquidity_need_score(replace(p, liquidity_needs=[later]), today) == (5, 0.0)
    assert liquidity_need_score(replace(p, portfolio_value_usd=None), today) == (None, None)
    # build_ips uses the derived answer, overriding a hand-picked one
    ips = build_ips(replace(p, answers={**p.answers, "liquidity_need": 5}), CFG, today)
    assert ips.client.answers["liquidity_need"] == 2


def test_regulator_defaults_from_booking_then_residency() -> None:
    assert default_regulator("ADGM (Abu Dhabi)", "UAE") == "FSRA (ADGM)"
    assert default_regulator("", "UAE") == "SCA (UAE)"  # onshore unless booked in a centre
    assert default_regulator(None, "Saudi Arabia") == "CMA (Saudi Arabia)"
    assert default_regulator(None, "Qatar") == "QFCRA (Qatar)"
    assert default_regulator("Qatar onshore", "Qatar") == "QFMA (Qatar)"
    assert default_regulator(None, "bahrain") == "CBB (Bahrain)"
    assert default_regulator(None, "Kuwait") == "CMA (Kuwait)"
    assert default_regulator(None, "Oman") == "FSA (Oman, formerly CMA)"
    assert default_regulator(None, "France") == "Other"
    assert effective_regulator(ClientProfile(residency="Kuwait")) == "CMA (Kuwait)"
    chosen = ClientProfile(residency="Kuwait", regulator="DFSA (DIFC)")
    assert effective_regulator(chosen) == "DFSA (DIFC)"  # the adviser's choice wins
    assert (
        "Regulator: CMA (Kuwait)."
        in build_ips(ClientProfile(residency="Kuwait"), CFG).sections["Legal and regulatory"]
    )


def test_adviser_override_needs_a_reason_and_flags_higher_scores() -> None:
    base = profile()  # calculated score 3
    assert risk_profile(base).override == "" and risk_profile(base).calculated == 3
    ignored = risk_profile(replace(base, override_score=2))
    assert (
        ignored.score == 3
        and ignored.override == "ignored"
        and "needs a reason" in (ignored.explanation)
    )
    lower = risk_profile(replace(base, override_score=2, override_reason="Client is nervous"))
    assert (lower.score, lower.calculated, lower.override) == (2, 3, "lower")
    assert "because Client is nervous." in lower.explanation
    higher = risk_profile(replace(base, override_score=4, override_reason="Large other assets"))
    assert higher.override == "higher" and "compliance review" in higher.explanation
    ips = build_ips(replace(base, override_score=2, override_reason="Nervous"), CFG)
    assert ips.limits["max_volatility"] == 0.09 and ips.limits["source"]["max_volatility"] == (
        "band 2"
    )


def test_capacity_for_loss_tightens_the_drawdown_limit() -> None:
    band = build_ips(profile(), CFG).limits  # band 3: 20% drawdown
    assert band["max_drawdown"] == 0.20
    tighter_loss = build_ips(profile(max_acceptable_drawdown_pct=15), CFG).limits
    assert tighter_loss["max_drawdown"] == 0.15
    assert tighter_loss["source"]["max_drawdown"] == "client capacity for loss"
    looser = build_ips(profile(max_acceptable_drawdown_pct=30), CFG).limits
    assert looser["max_drawdown"] == 0.20  # the band's limit is tighter: it stays
    adviser = profile(max_acceptable_drawdown_pct=15, overrides={"max_drawdown_pct": 25})
    assert build_ips(adviser, CFG).limits["max_drawdown"] == 0.25  # explicit override wins


def test_fact_find_fields_round_trip_and_age_from_dob() -> None:
    p = ClientProfile.from_dict(
        {
            "dob": "1980-11-02",
            "income": {"salary": "300000", "bonus": 0},
            "dependants": "2",
            "dependant_ages": ["8", 12],
            "pep": False,
            "max_acceptable_drawdown_pct": "25",
        },
        today=date(2026, 11, 1),
    )
    assert p.age == 45 and p.income == {"salary": 300_000.0} and p.dependants == 2
    assert p.dependant_ages == [8, 12] and p.max_acceptable_drawdown_pct == 25.0
    assert p.version == 2 and ClientProfile.from_dict(p.to_dict()) == p
    with pytest.raises(ValueError, match="date of birth"):
        ClientProfile.from_dict({"dob": "31/05/1980"})


def test_unanswered_questions_are_reported_not_guessed() -> None:
    r = risk_profile(profile(answers={"income_stability": 4}))
    assert r.score is None and "loss_reaction" in r.missing
    assert {q["id"] for q in QUESTIONS} >= {"income_stability", "loss_reaction"}


# ------------------------------------------------------------------------------ IPS


def test_ips_limits_from_band_and_overrides() -> None:
    ips = build_ips(profile(), CFG)  # band 3
    assert ips.limits["max_volatility"] == 0.12 and ips.limits["max_drawdown"] == 0.20
    assert ips.limits["max_weight"] == 0.30 and ips.limits["source"]["max_volatility"] == "band 3"
    over = build_ips(profile(overrides={"max_volatility_pct": 10, "max_weight_pct": 25}), CFG)
    assert over.limits["max_volatility"] == 0.10 and over.limits["max_weight"] == 0.25
    assert over.limits["source"]["max_volatility"] == "adviser override"
    assert list(ips.sections) == [
        "Return",
        "Risk",
        "Time horizon",
        "Taxes",
        "Liquidity",
        "Legal and regulatory",
        "Unique circumstances",
    ]


TODAY = date(2026, 10, 25)


def test_liquidity_reserve() -> None:
    needs = [
        {"description": "Fees", "amount_usd": 50_000, "date": "2027-10"},
        {"description": "Villa", "amount_usd": 600_000, "date": "2035-01"},  # beyond 36 months
    ]
    p = profile(portfolio_value_usd=500_000, liquidity_needs=needs, reserve_pct=5)
    ips = build_ips(p, CFG, today=TODAY)
    # 50,000 due within 36 months + 5% of 500,000 = 75,000 -> 15% of the portfolio
    assert ips.limits["liquidity_reserve_usd"] == 75_000
    assert ips.limits["liquidity_reserve"] == pytest.approx(0.15)
    assert [n["months"] for n in ips.limits["cash_needs_due"]] == [12]
    assert [n["months"] for n in ips.limits["cash_needs_later"]] == [99]
    text = ips.sections["Liquidity"]
    assert "Fees: USD 50,000 by Oct 2027 (12 months)" in text
    assert "Later needs" in text and "Villa: USD 600,000 by Jan 2035" in text


def test_dated_needs_months() -> None:
    # the QA case: a need in Oct 2028 is about two years away, not 12 months
    assert months_until("2028-10", TODAY) == 24
    assert months_until("2026-03", TODAY) == 0  # passed: due now
    assert month_after(TODAY, 14) == "2027-12" and month_text("2028-10") == "Oct 2028"
    with pytest.raises(ValueError):
        parse_month("October")


def test_old_profiles_with_an_undated_need_still_load() -> None:
    old = {"reference": "OLD", "liquidity_need_usd": 80_000, "liquidity_months": 6}
    p = ClientProfile.from_dict(old, today=TODAY)
    assert p.liquidity_needs == [
        {"description": "Cash need", "amount_usd": 80_000.0, "date": "2027-04"}
    ]
    with pytest.raises(ValueError, match="YYYY-MM"):
        ClientProfile.from_dict({"liquidity_needs": [{"amount_usd": 1, "date": "soon"}]})


def test_cash_like_holdings_rule() -> None:
    groups = {"BIL": "Bonds", "AGG": "Bonds", "MM": "Cash & short-term", "SPY": "Global equity"}
    vols = {"BIL": 0.002, "AGG": 0.055, "MM": 0.01, "SPY": 0.17, "SPSK": 0.057}
    tickers = ["BIL", "AGG", "MM", "SPY", "SPSK"]
    # AGG (a broad bond fund) is not cash-like; SPSK (sukuk) is, because it is on the list
    assert cash_like_holdings(tickers, groups, ["SPSK"], vols) == ["BIL", "MM", "SPSK"]


def test_cash_need_becomes_a_hard_optimiser_minimum() -> None:
    needs = [{"description": "Villa", "amount_usd": 600_000, "date": "2028-10"}]
    p = profile(portfolio_value_usd=2_000_000, liquidity_needs=needs)
    ips = build_ips(p, CFG, today=TODAY)
    holdings = ["SPY", "AGG", "BIL"]
    spec, notes = ips_constraints(ips, holdings, {}, {}, cash_like=["BIL"])
    assert spec.sets["Cash need"] == (frozenset({"BIL"}), pytest.approx(0.30), None)
    assert any("at least 30.00% (USD 600,000) in cash-like holdings (BIL)" in n for n in notes)
    spec, notes = ips_constraints(ips, holdings[:2], {}, {}, cash_like=[])
    assert spec.sets["Cash need"][0] == frozenset()  # infeasible: the optimiser says why
    assert any("can't be met" in n and "BIL" in n for n in notes)
    no_value = build_ips(profile(liquidity_needs=needs), CFG, today=TODAY)
    _, notes = ips_constraints(no_value, holdings, {}, {}, cash_like=["BIL"])
    assert any("enter the portfolio value" in n for n in notes)


def test_ips_without_a_risk_score_has_no_risk_limits() -> None:
    ips = build_ips(profile(answers={}), CFG)
    assert (
        ips.limits["max_volatility"] is None
        and "questionnaire is incomplete" in ips.sections["Risk"]
    )


# ------------------------------------------------------------- IPS -> optimiser limits

HOLDINGS = ["SPY", "AGG", "2222.SR", "MYSTERY"]
GROUPS = {"SPY": "Global equity", "AGG": "Bonds", "2222.SR": "GCC", "MYSTERY": "Other"}
ATTRS = {
    "SPY": {"sharia": "not_compliant"},
    "AGG": {"sharia": "not_compliant", "sector": "Bonds"},
    "2222.SR": {"sharia": "compliant", "sector": "Energy"},
}


def test_ips_constraints_exclusions_and_home_bias() -> None:
    p = profile(sharia_only=True, excluded_sectors=["Energy"], home_bias_gcc_pct=20)
    spec, notes = ips_constraints(build_ips(p, CFG), HOLDINGS, GROUPS, ATTRS)
    assert spec.max_weight == 0.30 and spec.max_volatility == 0.12
    assert spec.group_bounds == {"GCC": (0.20, None)}
    # not Sharia-compliant: SPY, AGG; excluded sector: 2222.SR (Energy) -> max 0
    assert spec.asset_bounds == {"SPY": (None, 0.0), "AGG": (None, 0.0), "2222.SR": (None, 0.0)}
    # unknown status is not excluded by the optimiser, but flagged for review
    assert any("MYSTERY" in n and "unknown" in n for n in notes)


def test_tighter_takes_the_stricter_limit() -> None:
    a = OptimisationSpec(
        max_weight=0.4,
        max_volatility=0.15,
        groups=GROUPS,
        group_bounds={"GCC": (0.1, 0.5)},
        asset_bounds={"SPY": (None, 0.3)},
    )
    b = OptimisationSpec(
        max_weight=0.25,
        max_volatility=None,
        groups=GROUPS,
        group_bounds={"GCC": (0.2, None)},
        asset_bounds={"AGG": (0.05, None)},
    )
    t = tighter(a, b)
    assert t.max_weight == 0.25 and t.max_volatility == 0.15
    assert t.group_bounds == {"GCC": (0.2, 0.5)}
    assert t.asset_bounds == {"SPY": (None, 0.3), "AGG": (0.05, None)}


# ------------------------------------------------------------------------- JSON


def test_json_round_trip_and_missing_keys() -> None:
    p = profile(
        goals=[{"description": "University", "amount_usd": 200_000, "years": 8}], sharia_only=True
    )
    assert ClientProfile.from_dict(p.to_dict()) == p
    old = {"reference": "OLD", "horizon_years": 5}  # an older file without newer keys
    q = ClientProfile.from_dict(old)
    assert q.reference == "OLD" and q.sharia_only is False and q.answers == {}
    with pytest.raises(ValueError, match="horizon_years"):
        ClientProfile.from_dict({"horizon_years": "soon"})


def test_examples_are_complete() -> None:
    for name, ex in EXAMPLE_CLIENTS.items():
        assert risk_profile(ClientProfile.from_dict(ex)).score is not None, name


def test_goal_and_unique_text() -> None:
    p = profile(
        goals=[
            {"description": "University", "amount_usd": 200_000, "years": 8},
            {"description": "Retire", "multiple": 1.5, "years": 25},
        ],
        sharia_only=True,
        home_bias_gcc_pct=20,
    )
    ips = build_ips(p, CFG)
    assert ips.sections["Return"] == (
        "Goals: University (USD 200,000 in 8 years); Retire (1.5x the starting value in 25 years)"
    )
    assert ips.sections["Unique circumstances"].startswith(
        "Sharia-compliant holdings only. At least 20% in GCC holdings (home-market bias)."
    )


def test_sector_exclusion_ignores_case_and_one_year_is_singular() -> None:
    p = profile(excluded_sectors=["energy"], goals=[{"description": "Deposit", "years": 1}])
    spec, _ = ips_constraints(build_ips(p, CFG), HOLDINGS, GROUPS, ATTRS)
    assert spec.asset_bounds == {"2222.SR": (None, 0.0)}  # sector "Energy"
    assert build_ips(p, CFG).sections["Return"] == "Goals: Deposit (in 1 year)"


def test_tighter_merges_sets() -> None:
    a = OptimisationSpec(sets={"Cash need": (frozenset({"BIL", "SHV"}), 0.2, None)})
    b = OptimisationSpec(
        sets={
            "Cash need": (frozenset({"BIL"}), 0.3, 0.9),
            "Saudi Arabia": (frozenset({"KSA"}), 0.1, None),
        }
    )
    sets = tighter(a, b).sets
    assert sets["Cash need"] == (frozenset({"BIL"}), 0.3, 0.9)
    assert sets["Saudi Arabia"] == (frozenset({"KSA"}), 0.1, None)


def test_region_minimum_becomes_an_optimiser_set() -> None:
    p = profile(region_minimums={"Saudi Arabia": 10.0, "Qatar": 5.0})
    ips = build_ips(p, CFG, today=TODAY)
    regions = {"2222.SR": "Saudi Arabia", "KSA": "Saudi Arabia", "SPY": "United States (listing)"}
    spec, notes = ips_constraints(ips, ["2222.SR", "KSA", "SPY"], {}, {}, regions=regions)
    assert spec.sets["Region Saudi Arabia"] == (frozenset({"2222.SR", "KSA"}), 0.10, None)
    assert spec.sets["Region Qatar"][0] == frozenset()  # infeasible, with a reason
    assert any("At least 5.00% in Qatar can't be met" in n for n in notes)
    assert "At least 10% in Saudi Arabia." in ips.sections["Unique circumstances"]
    from gpl.client import in_region

    assert in_region("United States (listing)", "United States")
    assert not in_region("United Kingdom", "United States")


def test_band_caps_follow_the_risk_score() -> None:
    caps = {band: CFG["risk_bands"][band]["max_weight"] for band in range(1, 6)}
    assert caps == {1: 0.20, 2: 0.25, 3: 0.30, 4: 0.35, 5: 0.40}  # cautious -> lower cap


def test_required_return_hand_value() -> None:
    from gpl.client import required_return

    r = required_return(96_000, 3_000_000, 600_000, 0.025)
    assert r.investable_usd == 2_400_000 and r.real == pytest.approx(0.04)
    assert r.nominal == pytest.approx(1.04 * 1.025 - 1)  # 6.60%
    assert f"{r.nominal:.2%}" == "6.60%" and "USD 2,400,000 investable" in r.text()
    assert required_return(0, 1_000_000, 0, 0.02).nominal == pytest.approx(0.02)  # inflation only
    with pytest.raises(ValueError, match="nothing is left to invest"):
        required_return(50_000, 500_000, 500_000, 0.02)


def test_ips_shows_the_required_return_and_flags_a_gap() -> None:
    needs = [{"description": "Wedding", "amount_usd": 600_000, "date": "2028-10"}]
    p = profile(
        portfolio_value_usd=3_000_000,
        liquidity_needs=needs,
        annual_income_need_usd=96_000,
        return_target_pct=5.0,
    )
    ips = build_ips(p, CFG, today=TODAY)  # inflation from config: 2.50%
    assert ips.limits["required_return"] == pytest.approx(0.066)
    text = ips.sections["Return"]
    assert "Required return 6.60% a year" in text and "above the 5.00% target" in text
    no_value = build_ips(profile(annual_income_need_usd=96_000), CFG, today=TODAY)
    assert "enter the portfolio value" in no_value.sections["Return"]
    assert build_ips(profile(), CFG).limits["required_return"] is None


def test_example_clients_have_complete_fact_finds_and_unchanged_limits() -> None:
    from gpl.factfind import consistency_warnings, ips_missing

    expected = {  # (score, max volatility, max drawdown, max weight): as before the fact-find
        "Dubai, moderate, 10 years": (3, 0.12, 0.20, 0.30),
        "Riyadh, conservative, 5 years, Sharia": (2, 0.09, 0.15, 0.25),
        "Doha, aggressive, 20 years": (5, 0.20, 0.40, 0.40),
    }
    for name, data in EXAMPLE_CLIENTS.items():
        p = ClientProfile.from_dict(data)
        assert ips_missing(p, date(2026, 11, 4)) == [], name
        assert consistency_warnings(p, date(2026, 11, 4)) == [], name
        lim = build_ips(p, CFG, date(2026, 11, 4)).limits
        got = (risk_profile(p).score, lim["max_volatility"], lim["max_drawdown"], lim["max_weight"])
        assert got == expected[name], name
