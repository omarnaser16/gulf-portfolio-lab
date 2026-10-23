"""Tests for gpl.metrics. Expected values are worked out by hand (see conftest docstrings)."""

import math

import numpy as np
import pandas as pd
import pytest

from gpl.metrics import (
    annualised_return,
    annualised_volatility,
    cagr,
    sharpe_ratio,
    sortino_ratio,
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
