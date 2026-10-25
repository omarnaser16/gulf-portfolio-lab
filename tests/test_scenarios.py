"""Tests for what-if weights, historical stress episodes and exposures (hand values)."""

import numpy as np
import pandas as pd
import pytest

from gpl.metrics import kpi_table, rolling_volatility
from gpl.scenarios import apply_what_if, classify_holdings, episode_performance, exposures

W = {"A": 0.5, "B": 0.3, "C": 0.2}


# ----------------------------------------------------------------------------- what-if


def test_what_if_scales_the_others_pro_rata() -> None:
    out = apply_what_if(W, {"A": 0.10})
    # A fixed at 10%; B and C share the other 90% in their 30:20 ratio -> 54% and 36%
    assert out.to_dict() == pytest.approx({"A": 0.10, "B": 0.54, "C": 0.36})
    assert out.sum() == pytest.approx(1.0)


def test_what_if_with_two_targets_and_a_new_asset() -> None:
    out = apply_what_if(W, {"A": 0.2, "GLD": 0.1})
    # B and C share 70% in a 30:20 ratio -> 42% and 28%; GLD is added at the end
    assert list(out.index) == ["A", "B", "C", "GLD"]
    assert out.to_dict() == pytest.approx({"A": 0.2, "B": 0.42, "C": 0.28, "GLD": 0.1})


def test_what_if_can_remove_a_holding() -> None:
    out = apply_what_if(W, {"C": 0.0})
    assert out.to_dict() == pytest.approx({"A": 0.625, "B": 0.375, "C": 0.0})


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        ({"A": 0.7, "B": 0.4}, "add up to 110.00%"),
        ({"A": -0.1}, "between 0% and 100%"),
        ({"A": 0.2, "B": 0.2, "C": 0.2}, "no other holdings"),
    ],
)
def test_what_if_rejects_impossible_targets(targets, message) -> None:
    with pytest.raises(ValueError, match=message):
        apply_what_if(W, targets)


# ------------------------------------------------------------------------ stress tests

EPISODES = [
    {"name": "Crash", "start": "2024-01-03", "end": "2024-01-05", "description": "test"},
    {"name": "Before data", "start": "2023-06-01", "end": "2023-06-30", "description": "test"},
]


def test_episode_performance_hand_values() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)  # Mon 1 Jan .. Mon 8 Jan
    returns = pd.DataFrame(
        {
            # inside the episode (after the 3rd, up to the 5th): -10% then +5%
            "Portfolio": [0.01, 0.02, 0.03, -0.10, 0.05, 0.01],
            "Benchmark": [0.0, 0.0, 0.0, -0.20, 0.0, 0.0],
        },
        index=idx,
    )
    out = episode_performance(returns, EPISODES).set_index(["episode", "series"])
    crash = out.loc[("Crash", "Portfolio")]
    assert crash["covered"] and crash["total_return"] == pytest.approx(0.9 * 1.05 - 1)  # -5.50%
    assert crash["max_drawdown"] == pytest.approx(-0.10) and crash["worst_day"] == -0.10
    assert out.loc[("Crash", "Benchmark"), "total_return"] == pytest.approx(-0.20)
    before = out.loc[("Before data", "Portfolio")]
    assert not before["covered"] and np.isnan(before["total_return"])


# --------------------------------------------------------------------------- exposures


def test_classify_and_exposures() -> None:
    meta = classify_holdings(
        ["SPY", "KSA", "2222.SR", "MYSTERY"],
        config_meta={
            "SPY": {"region": "United States", "asset_type": "Equity"},
            "KSA": {"region": "Saudi Arabia", "asset_type": "Equity"},
        },
        groups={"SPY": "Global equity", "KSA": "GCC", "2222.SR": "GCC", "MYSTERY": "Other"},
        currencies={"SPY": "USD", "KSA": "USD", "2222.SR": "SAR"},
    )
    assert meta["2222.SR"] == {"region": "Saudi Arabia", "asset_type": "Equity", "currency": "SAR"}
    assert meta["MYSTERY"]["region"] == "Other / unknown"
    assert meta["MYSTERY"]["currency"] == "Other / unknown"
    w = pd.Series({"SPY": 0.4, "KSA": 0.3, "2222.SR": 0.2, "MYSTERY": 0.1})
    exp = exposures(w, meta)
    assert exp["region"].to_dict() == pytest.approx(
        {"Saudi Arabia": 0.5, "United States": 0.4, "Other / unknown": 0.1}
    )
    assert exp["currency"].to_dict() == pytest.approx(
        {"USD": 0.7, "SAR": 0.2, "Other / unknown": 0.1}
    )
    assert all(s.sum() == pytest.approx(1.0) for s in exp.values())  # nothing dropped


# ------------------------------------------------------------- rolling volatility, KPIs


def test_rolling_volatility() -> None:
    r = pd.Series([0.01, -0.01] * 5)  # std of a +1%/-1% window of 4 = 0.011547
    rv = rolling_volatility(r, window=4, periods_per_year=252)
    assert len(rv) == 7  # the first 3 dates have no full window (documented, not hidden)
    assert rv.iloc[0] == pytest.approx(np.std([0.01, -0.01, 0.01, -0.01], ddof=1) * np.sqrt(252))


def test_kpi_table_deltas() -> None:
    idx = pd.bdate_range("2024-01-01", periods=252)
    port = pd.Series(1.10 ** (1 / 252) - 1, index=idx)  # 10% a year, no volatility
    bench = pd.Series(1.05 ** (1 / 252) - 1, index=idx)
    port.iloc[100] -= 0.02  # one bad day so volatility is defined
    bench.iloc[100] -= 0.01
    k = kpi_table(port, bench, rf=0.0)
    assert list(k.index) == ["cagr", "volatility", "sharpe", "max_drawdown"]
    assert k.loc["cagr", "delta"] == pytest.approx(
        k.loc["cagr", "portfolio"] - k.loc["cagr", "benchmark"]
    )
    assert k.loc["volatility", "delta"] > 0  # more volatile than the benchmark
