"""Tests for the seeded demo client: reproducible and internally consistent."""

from datetime import date

import pytest

from gpl.client import age_on, build_ips, months_until
from gpl.data import load_universe
from gpl.demo import POOL, SHARIA_POOL, demo_client
from gpl.factfind import completeness, consistency_warnings, finances, ips_missing

TODAY = date(2026, 11, 4)
CFG = load_universe()
SEEDS = range(200)


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
    # 5-8 real tickers whose USD amounts add up to the portfolio value
    tickers = [h["ticker"] for h in d.holdings]
    pool = {t for ts in (SHARIA_POOL if p.sharia_only else POOL).values() for t in ts}
    assert 5 <= len(tickers) <= 8 and len(set(tickers)) == len(tickers)
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
