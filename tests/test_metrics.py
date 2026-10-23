"""Tests for gpl.metrics. Expected values are worked out by hand (see conftest docstrings)."""

import math

import numpy as np
import pandas as pd
import pytest

from gpl.metrics import (
    annualised_return,
    annualised_volatility,
    cagr,
    drawdown_series,
    expected_shortfall,
    historical_var,
    max_drawdown,
    parametric_var,
    sharpe_ratio,
    sortino_ratio,
    summary_table,
)

SQRT_252 = math.sqrt(252)
STD_SMALL = math.sqrt(0.0013 / 3)  # sample std of returns_small = 0.0208167
DAILY_RF_5PCT = 1.05 ** (1 / 252) - 1  # 0.00019363


@pytest.fixture
def returns_frame(returns_small: pd.Series) -> pd.DataFrame:
    """Two columns: X (returns_small) and Y = X doubled."""
    return pd.DataFrame({"X": returns_small, "Y": 2 * returns_small})


# ---------------------------------------------------------------------- annualised_return


def test_annualised_return_series(returns_small: pd.Series) -> None:
    assert annualised_return(returns_small) == pytest.approx(0.005 * 252)  # 1.26


def test_annualised_return_frame(returns_frame: pd.DataFrame) -> None:
    out = annualised_return(returns_frame)
    assert isinstance(out, pd.Series)
    assert list(out.index) == ["X", "Y"]
    assert out["Y"] == pytest.approx(2 * 1.26)


# ---------------------------------------------------------------------------------- cagr


def test_cagr_series(returns_small: pd.Series) -> None:
    expected = 1.019494 ** (252 / 4) - 1  # compound growth over 4 days, scaled to 252
    assert cagr(returns_small) == pytest.approx(expected, rel=1e-6)


def test_cagr_one_year_equals_total_return() -> None:
    # 252 days of 0 except one +10% day: exactly one year, so CAGR = 10%
    r = pd.Series([0.0] * 251 + [0.10], index=pd.bdate_range("2024-01-01", periods=252))
    assert cagr(r) == pytest.approx(0.10)


def test_cagr_is_below_arithmetic_mean_when_volatile() -> None:
    # +10% then -10% loses money: the arithmetic mean is 0 but CAGR is negative
    r = pd.Series([0.10, -0.10] * 126, index=pd.bdate_range("2024-01-01", periods=252))
    assert annualised_return(r) == pytest.approx(0.0)
    assert cagr(r) == pytest.approx(0.99**126 - 1)
    assert cagr(r) < 0


def test_cagr_ignores_nans(returns_small: pd.Series) -> None:
    with_nan = returns_small.copy()
    with_nan.iloc[3] = np.nan  # the 0.00 day becomes missing: 3 observations left
    expected = 1.019494 ** (252 / 3) - 1
    assert cagr(with_nan) == pytest.approx(expected, rel=1e-6)


# -------------------------------------------------------------------- annualised_volatility


def test_annualised_volatility_series(returns_small: pd.Series) -> None:
    assert annualised_volatility(returns_small) == pytest.approx(STD_SMALL * SQRT_252)  # 0.3305


def test_annualised_volatility_scales_linearly(returns_frame: pd.DataFrame) -> None:
    out = annualised_volatility(returns_frame)
    assert out["Y"] == pytest.approx(2 * out["X"])


# ---------------------------------------------------------------------------------- sharpe


def test_sharpe_zero_rf(returns_small: pd.Series) -> None:
    expected = (0.005 * 252) / (STD_SMALL * SQRT_252)  # 3.813
    assert sharpe_ratio(returns_small) == pytest.approx(expected)


def test_sharpe_annual_float_rf(returns_small: pd.Series) -> None:
    expected = ((0.005 - DAILY_RF_5PCT) * 252) / (STD_SMALL * SQRT_252)  # 3.665
    assert sharpe_ratio(returns_small, rf=0.05) == pytest.approx(expected)


def test_sharpe_daily_series_rf_matches_float(returns_small: pd.Series) -> None:
    rf_daily = pd.Series(DAILY_RF_5PCT, index=returns_small.index)
    assert sharpe_ratio(returns_small, rf=rf_daily) == pytest.approx(
        sharpe_ratio(returns_small, rf=0.05)
    )


def test_sharpe_series_rf_is_aligned_on_dates(returns_small: pd.Series) -> None:
    # rf is missing the 3rd date: the previous day's rate is carried forward
    rf_daily = pd.Series(DAILY_RF_5PCT, index=returns_small.index).drop(returns_small.index[2])
    assert sharpe_ratio(returns_small, rf=rf_daily) == pytest.approx(
        sharpe_ratio(returns_small, rf=0.05)
    )


def test_sharpe_series_rf_with_frame(returns_frame: pd.DataFrame) -> None:
    rf_daily = pd.Series(DAILY_RF_5PCT, index=returns_frame.index)
    out = sharpe_ratio(returns_frame, rf=rf_daily)
    assert list(out.index) == ["X", "Y"]
    assert out["X"] == pytest.approx(sharpe_ratio(returns_frame["X"], rf=0.05))


def test_sharpe_zero_volatility_is_nan() -> None:
    flat = pd.Series([0.001] * 5, index=pd.bdate_range("2024-01-01", periods=5))
    assert math.isnan(sharpe_ratio(flat))


def test_sharpe_zero_volatility_column_is_nan(returns_small: pd.Series) -> None:
    df = pd.DataFrame({"X": returns_small, "FLAT": 0.001})
    out = sharpe_ratio(df)
    assert math.isnan(out["FLAT"])
    assert not math.isnan(out["X"])


# --------------------------------------------------------------------------------- sortino


def test_sortino_zero_rf(returns_small: pd.Series) -> None:
    downside = 0.01 * SQRT_252  # sqrt(0.02^2 / 4) annualised
    assert sortino_ratio(returns_small) == pytest.approx(1.26 / downside)  # 7.937


def test_sortino_above_sharpe_when_upside_is_bigger(returns_small: pd.Series) -> None:
    assert sortino_ratio(returns_small) > sharpe_ratio(returns_small)


def test_sortino_no_losses_is_nan() -> None:
    up = pd.Series([0.01, 0.02, 0.0, 0.03], index=pd.bdate_range("2024-01-01", periods=4))
    assert math.isnan(sortino_ratio(up))


def test_cagr_empty_is_nan() -> None:
    assert math.isnan(cagr(pd.Series([], dtype=float)))
    out = cagr(pd.DataFrame({"X": [0.01, 0.02], "EMPTY": [np.nan, np.nan]}))
    assert math.isnan(out["EMPTY"])


# ------------------------------------------------------------------------------ drawdowns


def test_drawdown_series_hand_values(returns_small: pd.Series) -> None:
    # wealth: 1.01, 0.9898, 1.019494, 1.019494  -> peak 1.01, 1.01, 1.019494, 1.019494
    dd = drawdown_series(returns_small)
    np.testing.assert_allclose(dd.to_numpy(), [0.0, -0.02, 0.0, 0.0], atol=1e-12)


def test_drawdown_counts_loss_on_day_one() -> None:
    # Wealth starts at 1 before the first return, so a first-day loss is a drawdown
    r = pd.Series([-0.10, 0.05], index=pd.bdate_range("2024-01-01", periods=2))
    np.testing.assert_allclose(drawdown_series(r).to_numpy(), [-0.10, 0.9 * 1.05 - 1])


def test_max_drawdown_peak_to_trough() -> None:
    # 1 -> 1.2 -> 0.9 -> 1.0: worst fall is 0.9 / 1.2 - 1 = -25%
    r = pd.Series(
        [0.20, 0.9 / 1.2 - 1, 1.0 / 0.9 - 1], index=pd.bdate_range("2024-01-01", periods=3)
    )
    assert max_drawdown(r) == pytest.approx(-0.25)


def test_max_drawdown_frame_and_no_loss(returns_frame: pd.DataFrame) -> None:
    out = max_drawdown(returns_frame)
    assert out["X"] == pytest.approx(-0.02)
    up = pd.Series([0.01, 0.02], index=pd.bdate_range("2024-01-01", periods=2))
    assert max_drawdown(up) == 0.0


# ------------------------------------------------------------------------------ VaR / ES


@pytest.fixture
def tail_returns() -> pd.Series:
    """21 returns: two losses (-10%, -5%) and nineteen +1% days.

    The 5% quantile sits at position 0.05 * (21 - 1) = 1 of the sorted list, i.e. -5%.
    """
    values = [0.01] * 10 + [-0.10] + [0.01] * 5 + [-0.05] + [0.01] * 4
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=21))


def test_historical_var_is_positive_loss(tail_returns: pd.Series) -> None:
    assert historical_var(tail_returns, level=0.95) == pytest.approx(0.05)


def test_expected_shortfall_averages_the_tail(tail_returns: pd.Series) -> None:
    # mean of returns at or below -5%: (-0.10 + -0.05) / 2
    assert expected_shortfall(tail_returns, level=0.95) == pytest.approx(0.075)


def test_es_at_least_var(tail_returns: pd.Series, returns_frame: pd.DataFrame) -> None:
    assert expected_shortfall(tail_returns) >= historical_var(tail_returns)
    assert (expected_shortfall(returns_frame) >= historical_var(returns_frame)).all()


def test_parametric_var_hand_value(returns_small: pd.Series) -> None:
    # -(mean + z * std) with z = norm.ppf(0.05) = -1.644854
    expected = -(0.005 - 1.6448536269514722 * STD_SMALL)  # 0.02924
    assert parametric_var(returns_small, level=0.95) == pytest.approx(expected)


def test_var_rises_with_confidence(tail_returns: pd.Series) -> None:
    assert parametric_var(tail_returns, 0.99) > parametric_var(tail_returns, 0.95)
    assert historical_var(tail_returns, 0.99) >= historical_var(tail_returns, 0.95)


def test_var_frame_returns_series(returns_frame: pd.DataFrame) -> None:
    for fn in (historical_var, parametric_var, expected_shortfall):
        out = fn(returns_frame)
        assert list(out.index) == ["X", "Y"]


# --------------------------------------------------------------------------- summary_table


def test_summary_table_layout(returns_frame: pd.DataFrame) -> None:
    table = summary_table(returns_frame, rf=0.05)
    assert list(table.index) == ["X", "Y"]
    assert list(table.columns) == [
        "ann_return",
        "cagr",
        "ann_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "hist_var",
        "param_var",
        "expected_shortfall",
        "start",
        "end",
        "n_obs",
    ]
    assert table.loc["X", "n_obs"] == 4
    assert table.loc["X", "start"] == pd.Timestamp("2024-01-02")


def test_summary_table_matches_individual_metrics(returns_frame: pd.DataFrame) -> None:
    table = summary_table(returns_frame, rf=0.05)
    assert table.loc["X", "sharpe"] == pytest.approx(sharpe_ratio(returns_frame["X"], rf=0.05))
    assert table.loc["Y", "max_drawdown"] == pytest.approx(max_drawdown(returns_frame["Y"]))
    assert table.loc["X", "cagr"] == pytest.approx(cagr(returns_frame["X"]))


def test_summary_table_accepts_series(returns_small: pd.Series) -> None:
    table = summary_table(returns_small)
    assert list(table.index) == ["X"]


def test_var_rejects_bad_level(returns_small: pd.Series) -> None:
    with pytest.raises(ValueError):
        historical_var(returns_small, level=95)
