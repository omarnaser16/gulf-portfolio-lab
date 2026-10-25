"""Tests for the suitability checker: one rule at a time, with hand-checkable portfolios."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from gpl.client import ClientProfile, build_ips, month_after
from gpl.data import load_universe
from gpl.suitability import check_suitability, suitability_table

CFG = load_universe()
ANSWERS = {
    "income_stability": 4,
    "wealth_share": 3,
    "dependence": 4,
    "liquidity_need": 4,
    "loss_reaction": 3,
    "tradeoff": 3,
    "experience": 4,
}  # band 3: vol 12%, DD 20%
IDX = pd.bdate_range("2019-01-01", periods=1000)
RNG = np.random.default_rng(5)
RETURNS = pd.DataFrame(
    {
        "EQ": RNG.normal(0.0004, 0.012, len(IDX)),  # ~19% vol
        "BOND": RNG.normal(0.0001, 0.002, len(IDX)),  # ~3% vol
        "GCC": RNG.normal(0.0003, 0.010, len(IDX)),  # ~16% vol
        "UK": RNG.normal(0.0003, 0.010, len(IDX)),
    },
    index=IDX,
)
GROUPS = {"EQ": "Global equity", "BOND": "Bonds", "GCC": "GCC", "UK": "Global equity"}
CURRENCIES = {"EQ": "USD", "BOND": "USD", "GCC": "SAR", "UK": "GBp"}
ATTRS = {
    "EQ": {"sharia": "not_compliant", "sector": "Broad equity"},
    "BOND": {"sharia": "not_compliant", "sector": "Conventional bonds"},
    "GCC": {"sharia": "compliant", "sector": "Energy"},
}
EPISODES = [{"name": "Test crash", "start": "2020-02-19", "end": "2020-03-23"}]


def ips(need: float = 0, **kw):
    """IPS for a band-3 client; ``need``: USD cash need due in 12 months."""
    if need:
        kw["liquidity_needs"] = [
            {"description": "Fees", "amount_usd": need, "date": month_after(date.today(), 12)}
        ]
    return build_ips(ClientProfile(horizon_years=10, answers=ANSWERS, **kw), CFG)


def run(weights, client=None, value=None, cash_like=("BOND",)):
    checks = check_suitability(
        pd.Series(weights),
        RETURNS,
        client or ips(),
        ATTRS,
        CURRENCIES,
        GROUPS,
        periods_per_year=252,
        episodes=EPISODES,
        value=value,
        liquid_max_volatility=0.03,
        cash_like=cash_like,
    )
    return {c.rule: c for c in checks}


def test_volatility_and_drawdown_rules() -> None:
    risky = run({"EQ": 1.0})
    vol = float(RETURNS["EQ"].std() * np.sqrt(252))
    assert risky["Volatility"].status == "fail"
    assert risky["Volatility"].measured == f"{vol:.2%}" and risky["Volatility"].limit == "12.00%"
    assert f"is {vol:.2%}, above the 12.00% IPS limit (band 3" in risky["Volatility"].reason
    assert "Volatility over the full history (Jan 2019 - " in risky["Volatility"].reason
    calm = run({"BOND": 1.0})
    assert calm["Volatility"].status == "pass" and calm["Max drawdown"].status == "pass"


def test_concentration_and_home_bias() -> None:
    w = {"EQ": 0.5, "BOND": 0.3, "GCC": 0.2}
    checks = run(w, ips(home_bias_gcc_pct=25))
    assert checks["Concentration"].status == "fail"  # 50% > 40%
    assert (
        "largest holding, EQ, is 50.00% of the portfolio, above the 40.00% limit"
        in checks["Concentration"].reason
    )
    assert checks["Home-market bias"].status == "fail"  # 20% < 25%
    assert run(w, ips(home_bias_gcc_pct=20))["Home-market bias"].status == "pass"


def test_liquidity_reserve() -> None:
    client = ips(20_000, portfolio_value_usd=100_000)  # needs 20%
    assert run({"EQ": 0.4, "BOND": 0.3, "GCC": 0.3}, client, 100_000)["Liquidity"].status == "pass"
    tight = run({"EQ": 0.4, "BOND": 0.1, "GCC": 0.4, "UK": 0.1}, client, 100_000)["Liquidity"]
    assert tight.status == "fail" and "USD 10,000" in tight.reason and "USD 20,000" in tight.reason
    unknown_value = run({"EQ": 1.0}, ips(20_000))["Liquidity"]
    assert unknown_value.status == "review" and "portfolio value" in unknown_value.reason


def test_exclusions_fail_and_unknowns_need_review() -> None:
    sharia = ips(sharia_only=True)
    assert run({"GCC": 1.0}, sharia)["Exclusions"].status == "pass"
    bad = run({"GCC": 0.5, "BOND": 0.5}, sharia)["Exclusions"]
    assert bad.status == "fail" and "BOND (50.00%): not Sharia-compliant" in bad.reason
    unknown = run({"GCC": 0.5, "UK": 0.5}, sharia)["Exclusions"]
    assert unknown.status == "review" and "UK" in unknown.reason
    sector = run({"GCC": 1.0}, ips(excluded_sectors=["energy"]))["Exclusions"]
    assert sector.status == "fail" and "sector Energy is excluded" in sector.reason


def test_currency_rule() -> None:
    # UK trades in pence (GBp -> GBP), which is not pegged to the dollar
    ok = run({"EQ": 0.85, "UK": 0.15})["Currency"]
    assert ok.status == "pass" and ok.measured == "15.00%"
    over = run({"EQ": 0.7, "UK": 0.3})["Currency"]
    assert over.status == "fail" and "GBP" in over.reason


def test_stress_episode_and_missing_limits() -> None:
    checks = run({"EQ": 1.0})
    assert checks["Worst stress episode"].status in ("pass", "fail")
    assert "Test crash" in checks["Worst stress episode"].reason
    no_band = build_ips(ClientProfile(horizon_years=10), CFG)  # questionnaire not answered
    assert run({"EQ": 1.0}, no_band)["Volatility"].status == "review"


def test_table_compares_portfolios() -> None:
    table = suitability_table({"Current": run({"EQ": 1.0}), "Proposed": run({"BOND": 1.0})})
    assert list(table.columns) == ["Current", "Proposed"]
    assert table.loc["Volatility", "Current"].startswith("Fail")
    assert table.loc["Volatility", "Proposed"].startswith("Pass")


def test_breach_amounts_and_text() -> None:
    from gpl.suitability import breach_text, closeness, failures

    checks = run({"EQ": 1.0})
    vol = float(RETURNS["EQ"].std() * np.sqrt(252))
    v = checks["Volatility"]
    assert v.breach == pytest.approx(vol - 0.12) and v.scale == 0.12
    assert (
        breach_text(v)
        == f"Volatility: {vol:.2%}, {(vol - 0.12) * 100:.2f} pp beyond the 12.00% limit"
    )
    conc = checks["Concentration"]
    assert conc.breach == pytest.approx(0.60) and "60.00 pp beyond the 40.00% limit" in breach_text(
        conc
    )
    client = ips(20_000, portfolio_value_usd=100_000)
    liq = run({"EQ": 0.4, "BOND": 0.1, "GCC": 0.4, "UK": 0.1}, client, 100_000)["Liquidity"]
    assert liq.breach == pytest.approx(
        10_000
    ) and "USD 10,000 short of the USD 20,000" in breach_text(liq)
    home = run({"EQ": 0.5, "BOND": 0.3, "GCC": 0.2}, ips(home_bias_gcc_pct=25))["Home-market bias"]
    assert home.breach == pytest.approx(0.05) and breach_text(home).endswith(
        "5.00 pp below the 25.00% minimum"
    )
    passing = run({"BOND": 0.4, "EQ": 0.3, "GCC": 0.3})
    assert all(c.breach is None for c in passing.values() if c.status != "fail")
    n, total = closeness(list(checks.values()))
    assert n == len(failures(list(checks.values()))) and total > 0


def test_no_stress_episodes_configured_needs_review() -> None:
    checks = check_suitability(
        pd.Series({"BOND": 1.0}), RETURNS, ips(), ATTRS, CURRENCIES, GROUPS, 252, []
    )
    stress = next(c for c in checks if c.rule == "Worst stress episode")
    assert stress.status == "review"


def test_cash_like_holdings_hold_the_cash_need() -> None:
    client = ips(20_000, portfolio_value_usd=100_000)
    w = {"EQ": 0.4, "BOND": 0.3, "GCC": 0.3}
    # BOND (3.17% volatility) only counts when it is on the cash-like list or in the group
    assert run(w, client, 100_000)["Liquidity"].status == "pass"
    off_list = run(w, client, 100_000, cash_like=())["Liquidity"]
    assert off_list.status == "fail" and off_list.measured == "USD 0"
    regrouped = check_suitability(
        pd.Series(w),
        RETURNS,
        client,
        ATTRS,
        CURRENCIES,
        {**GROUPS, "BOND": "Cash & short-term"},
        252,
        EPISODES,
        100_000,
    )
    assert next(c for c in regrouped if c.rule == "Liquidity").status == "pass"
    due = month_after(date.today(), 12)
    assert f"by {date(int(due[:4]), int(due[5:]), 1):%b %Y}" in off_list.reason


def test_region_minimums_rule() -> None:
    from gpl.suitability import breach_text

    client = ips(region_minimums={"Saudi Arabia": 25.0})
    regions = {"GCC": "Saudi Arabia", "EQ": "United States (listing)", "BOND": "United States"}

    def check(weights, regions=regions):
        out = check_suitability(
            pd.Series(weights),
            RETURNS,
            client,
            ATTRS,
            CURRENCIES,
            GROUPS,
            252,
            EPISODES,
            regions=regions,
        )
        return next(c for c in out if c.rule == "Region minimums")

    short = check({"EQ": 0.5, "BOND": 0.3, "GCC": 0.2})
    assert short.status == "fail" and short.breach == pytest.approx(0.05)
    assert breach_text(short) == (
        "Region minimums: Saudi Arabia 20.00%, 5.00 pp below the 25.00% minimum"
    )
    assert check({"EQ": 0.5, "BOND": 0.2, "GCC": 0.3}).status == "pass"
    assert check({"EQ": 1.0}, regions=None).status == "review"


def test_checks_name_their_period_and_show_the_headline_period_too() -> None:
    from gpl.metrics import annualised_volatility, period_label

    headline = RETURNS.index[500:]
    out = check_suitability(
        pd.Series({"EQ": 1.0}),
        RETURNS,
        ips(),
        ATTRS,
        CURRENCIES,
        GROUPS,
        252,
        EPISODES,
        headline_dates=headline,
    )
    checks = {c.rule: c for c in out}
    assert checks["Volatility"].period == f"full history: {period_label(RETURNS.index)}"
    assert checks["Concentration"].period == "today's weights"
    short_vol = float(annualised_volatility(RETURNS["EQ"].loc[headline], 252))
    assert (
        f"Over the backtest period ({period_label(headline)}): {short_vol:.2%}."
        in checks["Volatility"].reason
    )


def test_curated_sukuk_fund_is_sharia_compliant() -> None:
    from gpl.client import excluded_holdings

    client = ips(sharia_only=True)
    excluded, review = excluded_holdings(
        client, ["SPSK", "SPUS", "AGG"], CFG["security_attributes"]
    )
    assert "SPSK" not in excluded and "SPSK" not in review and "SPUS" not in review
    assert excluded == {"AGG": "not Sharia-compliant"}
