"""Tests for benchmark-relative and extra screen metrics, with hand-checkable values."""

import numpy as np
import pandas as pd
import pytest

from gpl.metrics import (
    alpha,
    benchmark_metrics,
    beta,
    calmar_ratio,
    correlation_clusters,
    down_capture,
    information_ratio,
    relative_drawdown,
    tracking_error,
    trailing_return,
    up_capture,
)

IDX = pd.bdate_range("2024-01-01", periods=4)
B = pd.Series([0.02, -0.01, 0.01, -0.03], index=IDX)  # benchmark
R = pd.Series([0.01, -0.005, 0.02, -0.03], index=IDX)  # portfolio


def test_beta_hand_values() -> None:
    assert beta(2 * B, B) == pytest.approx(2.0)
    assert beta(B, B) == pytest.approx(1.0)
    frame = pd.DataFrame({"double": 2 * B, "minus": -B})
    pd.testing.assert_series_equal(
        beta(frame, B), pd.Series({"double": 2.0, "minus": -1.0}), check_names=False
    )
    assert np.isnan(beta(R, pd.Series(0.001, index=IDX)))  # constant benchmark: undefined


def test_alpha_of_a_constant_daily_edge() -> None:
    rng = np.random.default_rng(0)
    b = pd.Series(rng.normal(0, 0.01, 500))
    r = b + 0.0001  # same risk, 1 bp a day better
    assert beta(r, b) == pytest.approx(1.0)
    assert alpha(r, b, rf=0.0) == pytest.approx(0.0001 * 252)  # 2.52% a year


def test_tracking_error_and_information_ratio() -> None:
    b = pd.Series([0.0, 0.0])
    r = pd.Series([0.01, -0.01])  # active returns +1%, -1%: std = 0.01 * sqrt(2) = 0.014142
    assert tracking_error(r, b) == pytest.approx(0.01 * np.sqrt(2) * np.sqrt(252))  # 22.45%
    assert information_ratio(r, b) == pytest.approx(0.0)
    r2 = pd.Series([0.02, 0.0])  # active mean 1% a day
    # active [2%, 0%]: mean 1%, std 0.01 * sqrt(2)
    assert information_ratio(r2, b) == pytest.approx(0.01 * 252 / (0.01 * np.sqrt(2 * 252)))
    assert np.isnan(information_ratio(b, b))  # no active risk


def test_capture_ratios_hand_values() -> None:
    # Up days (B > 0): B mean 1.50%, R mean 1.50% -> 100%. Down days: B -2.00%, R -1.75% -> 87.5%
    assert up_capture(R, B) == pytest.approx(1.0)
    assert down_capture(R, B) == pytest.approx(0.875)
    assert np.isnan(up_capture(R, pd.Series(-0.01, index=IDX)))  # no up days


def test_calmar_ratio() -> None:
    # One year of 252 returns: a -20% fall then recovery to +10% over the year
    r = pd.Series([0.0] * 252)
    r.iloc[0] = -0.20
    r.iloc[1] = 1.10 / 0.80 - 1  # wealth 0.8 -> 1.1
    assert calmar_ratio(r) == pytest.approx(0.10 / 0.20)
    assert np.isnan(calmar_ratio(pd.Series([0.01] * 10)))  # never fell: undefined


def test_trailing_return_one_and_three_years() -> None:
    daily = 1.10 ** (1 / 252) - 1  # 10% a year
    r = pd.Series([daily] * 756)
    assert trailing_return(r, 1) == pytest.approx(0.10)  # 100 -> 110 over the last 252
    assert trailing_return(r, 3) == pytest.approx(0.10)  # 100 -> 133.1, annualised
    assert np.isnan(trailing_return(r.iloc[:500], 3))  # not enough history
    frame = pd.DataFrame({"A": r, "B": [0.0] * 756})
    out = trailing_return(frame, 1)
    assert out["A"] == pytest.approx(0.10) and out["B"] == 0.0
    # Calendars with ~183 observations a year look back 183 observations per year
    assert trailing_return(pd.Series([1.1 ** (1 / 183) - 1] * 183), 1, 183) == pytest.approx(0.1)


def test_relative_drawdown() -> None:
    # Portfolio +10% then flat; benchmark flat then +10%: ratio 1.10 -> 1.00, a 9.09% lag
    r = pd.Series([0.10, 0.0])
    b = pd.Series([0.0, 0.10])
    rd = relative_drawdown(r, b)
    assert rd.iloc[0] == 0.0 and rd.iloc[1] == pytest.approx(1.0 / 1.1 - 1)


def test_correlation_clusters() -> None:
    rng = np.random.default_rng(1)
    common, other = rng.normal(size=1000), rng.normal(size=1000)
    frame = pd.DataFrame(
        {
            "SPY": common,
            "URTH": common + 0.2 * rng.normal(size=1000),  # rho ~0.98 with SPY
            "GLD": other,  # unrelated
            "AGG": rng.normal(size=1000),
        }
    )
    clusters = correlation_clusters(frame, threshold=0.70)
    assert [c["members"] for c in clusters] == [["SPY", "URTH"]]
    assert clusters[0]["average_correlation"] == pytest.approx(frame["SPY"].corr(frame["URTH"]))
    assert correlation_clusters(frame[["SPY"]]) == []
    assert correlation_clusters(frame, threshold=0.999) == []


def test_benchmark_metrics_bundle_and_cash() -> None:
    out = benchmark_metrics(R, B)
    assert out["beta"] == pytest.approx(beta(R, B)) and out["up_capture"] == pytest.approx(1.0)
    assert out["down_capture"] == pytest.approx(0.875) and out["start"] == IDX[0]
    cash = benchmark_metrics(R, pd.Series(0.0001, index=IDX))
    assert all(np.isnan(cash[k]) for k in ("beta", "alpha", "up_capture", "correlation"))
    assert cash["tracking_error"] > 0  # still defined: how far you stray from cash
    # a real T-bill series drifts a little each day; it still has no market risk
    drifting = pd.Series([0.0001, 0.00011, 0.000105, 0.0001], index=IDX)
    raw = benchmark_metrics(R, drifting)
    assert not np.isnan(raw["beta"])  # numerically defined, but meaningless
    cash = benchmark_metrics(R, drifting, market_risk=False)
    assert all(np.isnan(cash[k]) for k in ("beta", "alpha", "up_capture", "correlation"))
