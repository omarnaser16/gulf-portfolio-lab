"""Tests for the client profile, risk questionnaire, IPS and IPS -> optimiser limits."""

from datetime import date

import pytest

from gpl.client import (
    EXAMPLE_CLIENTS,
    QUESTIONS,
    ClientProfile,
    build_ips,
    cash_like_holdings,
    horizon_score,
    ips_constraints,
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
            "tradeoff": 2,
            "experience": 1,
        },
    )
    r = risk_profile(p)  # capacity 5; willingness mean 1.33 -> 1
    assert (r.capacity, r.willingness, r.score) == (5, 1, 1) and r.mismatch
    p2 = profile(
        answers={**profile().answers, "loss_reaction": 4, "tradeoff": 3, "experience": 4}
    )  # willingness mean 3.67 -> 4
    assert risk_profile(p2).willingness == 4
    p3 = profile(
        answers={
            **profile().answers,
            "loss_reaction": 3,
            "tradeoff": 4,
            "experience": 3,
        }
    )  # 3.33 -> 3
    assert risk_profile(p3).willingness == 3


def test_unanswered_questions_are_reported_not_guessed() -> None:
    r = risk_profile(profile(answers={"income_stability": 4}))
    assert r.score is None and "loss_reaction" in r.missing
    assert {q["id"] for q in QUESTIONS} >= {"income_stability", "loss_reaction"}


# ------------------------------------------------------------------------------ IPS


def test_ips_limits_from_band_and_overrides() -> None:
    ips = build_ips(profile(), CFG)  # band 3
    assert ips.limits["max_volatility"] == 0.12 and ips.limits["max_drawdown"] == 0.20
    assert ips.limits["max_weight"] == 0.40 and ips.limits["source"]["max_volatility"] == "band 3"
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
    assert spec.max_weight == 0.40 and spec.max_volatility == 0.12
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
