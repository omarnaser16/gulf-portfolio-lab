"""Tests for the seeded demo client: reproducible and internally consistent."""

from datetime import date

import numpy as np
import pytest

from gpl.client import PEGGED, age_on, build_ips, excluded_holdings, in_region, months_until
from gpl.data import load_investable, load_universe
from gpl.demo import CAP, EQUITY_RANGE, LOSS, POOL, SHARIA_POOL, demo_client
from gpl.factfind import completeness, consistency_warnings, finances, ips_missing

TODAY = date(2026, 11, 4)
CFG = load_universe()
SECURITIES = load_investable()
SEEDS = range(200)
RANDOM_SEEDS = [int(s) for s in np.random.default_rng(2026).integers(0, 100_000, 50)]


def test_same_seed_same_client_different_seed_different_client() -> None:
    a, b = demo_client(7, TODAY), demo_client(7, TODAY)
    assert a.profile == b.profile and a.holdings == b.holdings
    assert demo_client(8, TODAY).profile != a.profile


@pytest.mark.parametrize("seed", SEEDS)
def test_demo_client_is_consistent(seed: int) -> None:
    d = demo_client(seed, TODAY)
    p = d.profile
    assert p.demo and p.reference == f"DEMO-{seed}"
    # complete fact-find, final IPS possible, no contradictions
    assert all(not c.missing for c in completeness(p, TODAY).values())
    assert ips_missing(p, TODAY) == []
    assert consistency_warnings(p, TODAY) == []
    # age, retirement and horizon
    assert age_on(p.dob, TODAY) == p.age
    if p.employment_type != "Retired":
        assert p.retirement_age > p.age
        assert p.horizon_years == min(max(p.retirement_age - p.age, 3), 30)
    assert all(g["years"] <= p.horizon_years for g in p.goals)
    assert all(months_until(n["date"], TODAY) > 0 for n in p.liquidity_needs)
    # balance sheet: portfolio <= investable <= net worth; expenses < income
    f = finances(p)
    assert p.portfolio_value_usd <= f.investable <= f.net_worth
    assert f.expenses < f.income
    # 5-10 real tickers whose USD amounts add up to the portfolio value
    tickers = [h["ticker"] for h in d.holdings]
    pool = {t for ts in (SHARIA_POOL if p.sharia_only else POOL).values() for t in ts}
    assert 5 <= len(tickers) <= 10 and len(set(tickers)) == len(tickers)
    assert set(tickers) <= pool
    assert sum(h["value"] for h in d.holdings) == pytest.approx(p.portfolio_value_usd)
    assert all(h["value"] > 0 for h in d.holdings)
    # the IPS builds and scores the client
    assert build_ips(p, CFG, TODAY).risk.score is not None


def test_demo_clients_vary() -> None:
    profiles = [demo_client(s, TODAY).profile for s in SEEDS]
    assert len({p.residency for p in profiles}) == 6
    assert {p.employment_type for p in profiles} >= {"Employed", "Business owner", "Retired"}
    assert any(p.sharia_only for p in profiles) and not all(p.sharia_only for p in profiles)
    scores = {build_ips(p, CFG, TODAY).risk.score for p in profiles}
    assert len(scores) >= 3


def test_demo_tables_match_the_config() -> None:
    bands = CFG["risk_bands"]
    assert CAP == {k: v["max_weight"] for k, v in bands.items()}
    assert LOSS == {k: round(v["max_drawdown"] * 100) for k, v in bands.items()}
    for pool in (POOL, SHARIA_POOL):
        for tickers in pool.values():
            assert all(t in SECURITIES for t in tickers)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_random_demo_client_is_realistic_and_fits_its_ips(seed: int) -> None:
    """Each demo portfolio passes the IPS rules that depend on weights alone (volatility and
    drawdown need prices, and are what Analyse and Suitability then measure)."""
    d = demo_client(seed, TODAY)
    p = d.profile
    ips = build_ips(p, CFG, TODAY)
    lim, score = ips.limits, ips.risk.score
    # consistent facts (the full set is test_demo_client_is_consistent)
    f = finances(p)
    assert age_on(p.dob, TODAY) == p.age and p.horizon_years > 0
    assert p.portfolio_value_usd <= f.investable <= f.net_worth
    assert all(months_until(n["date"], TODAY) > 0 for n in p.liquidity_needs)
    assert consistency_warnings(p, TODAY) == [] and ips_missing(p, TODAY) == []
    assert p.max_acceptable_drawdown_pct == lim["max_drawdown"] * 100  # loss fits the score
    # 5-10 real tickers with verified prices since at least 2020
    w = {h["ticker"]: h["value"] / p.portfolio_value_usd for h in d.holdings}
    assert 5 <= len(w) <= 10 and len(w) == len(d.holdings)
    for t in w:
        sec = SECURITIES[t]
        assert sec["verified"] and sec["history_from"] <= "2020-01-01", t
    # weights add up to 100%, none above the IPS cap
    assert sum(w.values()) == pytest.approx(1.0) and min(w.values()) > 0
    assert max(w.values()) <= lim["max_weight"] + 1e-9
    # the equity share fits the client's own risk score
    equity = sum(v for t, v in w.items() if SECURITIES[t]["asset_class"] == "Equity")
    lo, hi = EQUITY_RANGE[score]
    assert lo - 1e-9 <= equity <= hi + 1e-9, (score, equity)
    # GCC exposure for a GCC client, meeting any home-market and region minimums
    gcc_regions = {"Saudi Arabia", "UAE", "Qatar", "Kuwait", "Bahrain", "Oman"}
    gcc = sum(v for t, v in w.items() if SECURITIES[t]["region"] in gcc_regions)
    assert p.residency in gcc_regions and gcc >= 0.08 - 1e-9
    assert gcc >= (lim["home_bias_gcc"] or 0) - 1e-9
    for region, minimum in lim["region_minimums"].items():
        assert sum(v for t, v in w.items() if in_region(SECURITIES[t]["region"], region)) >= minimum
    # cash needs due within 36 months are held in cash-like funds
    cash_like = set(CFG["cash_like"]) | {h["ticker"] for h in d.holdings
                                         if h["group"] == "Cash & short-term"}  # fmt: skip
    held = sum(h["value"] for h in d.holdings if h["ticker"] in cash_like)
    assert held >= lim["liquidity_reserve_usd"] - 0.5
    # currencies not pegged to the dollar within the IPS limit
    unpegged = sum(
        v for t, v in w.items() if SECURITIES[t]["currency"].replace("KWF", "KWD") not in PEGGED
    )
    assert unpegged <= lim["max_non_pegged_currency"] + 1e-9
    # nothing excluded or unknown: Sharia-only clients hold screened compliant funds only
    excluded, review = excluded_holdings(ips, list(w), CFG["security_attributes"])
    assert excluded == {} and review == {}
    if p.sharia_only:
        assert all(SECURITIES[t]["sharia"] == "compliant" for t in w)
