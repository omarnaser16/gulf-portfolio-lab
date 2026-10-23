"""Tests for benchmark data: fixed-weight wealth, putting it on the portfolio's dates, loading."""

import pandas as pd
import pytest
import yfinance as yf

import gpl.data
from gpl.data import benchmark_wealth, load_benchmark, returns_on_dates


def test_benchmark_wealth_is_daily_rebalanced() -> None:
    idx = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"S": [100, 110, 99], "B": [100, 100, 101]}, index=idx, dtype=float)
    w = benchmark_wealth(prices, {"S": 0.6, "B": 0.4})
    # day 1: 0.6*10% + 0.4*0% = 6%; day 2: 0.6*(-10%) + 0.4*1% = -5.6%
    assert list(w.round(6)) == [1.0, 1.06, round(1.06 * 0.944, 6)]


def test_returns_on_dates_handles_holidays_and_late_start() -> None:
    wealth = pd.Series(
        [1.0, 1.1, 1.21],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]),  # no trade on the 4th
    )
    portfolio_dates = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )
    r, report = returns_on_dates(wealth, portfolio_dates)
    assert list(r.index) == list(portfolio_dates[2:])  # starts after the benchmark's first date
    assert list(r.round(6)) == [0.1, 0.0, 0.1]  # holiday shows 0%, caught up next day
    assert report == {"stale_dates": 1, "start": pd.Timestamp("2024-01-02"), "dropped": 1}


def test_returns_on_dates_compounds_across_sparse_dates() -> None:
    # Portfolio on Mondays only: the benchmark return spans the whole week
    wealth = pd.Series(1.01 ** pd.RangeIndex(10), index=pd.bdate_range("2024-01-01", periods=10))
    r, _ = returns_on_dates(wealth, pd.to_datetime(["2024-01-01", "2024-01-08"]))
    assert r.iloc[0] == pytest.approx(1.01**5 - 1)


def test_load_benchmark_cash_and_ftse_in_pence(monkeypatch, tmp_path) -> None:
    idx = pd.bdate_range("2024-01-01", periods=5)
    rf = pd.Series(0.0002, index=idx)
    cash = load_benchmark(
        "Cash", {"cash": True, "represents": "T-bills"}, idx, rf, "2024-01-01", None, tmp_path
    )
    assert cash.is_cash and cash.returns.round(8).tolist() == [0.0002] * 4

    def fake_download(ticker, start, end, cache_dir=None):
        if ticker == "GBP=X":
            return pd.Series(0.8, index=idx)  # 0.8 GBP per USD
        return pd.Series([1000.0, 1010, 1020, 1030, 1040], index=idx)  # pence

    monkeypatch.setattr(gpl.data, "download_series", fake_download)
    monkeypatch.setattr(gpl.data, "get_currency", lambda t, cache_dir=None: ("GBp", "test"))
    ftse = load_benchmark(
        "FTSE 100", {"weights": {"ISF.L": 1.0}}, idx, rf, "2024-01-01", None, tmp_path
    )
    assert ftse.returns.iloc[0] == pytest.approx(0.01)  # 1000p -> 1010p at a fixed rate
    assert not ftse.notes


def test_load_benchmark_without_data_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(gpl.data, "download_series", lambda *a, **k: None)
    monkeypatch.setattr(yf, "Ticker", None)  # never reached
    idx = pd.bdate_range("2024-01-01", periods=3)
    with pytest.raises(ValueError, match="No data for the URTH benchmark"):
        load_benchmark(
            "URTH",
            {"weights": {"URTH": 1.0}},
            idx,
            pd.Series(0.0, index=idx),
            "2024",
            None,
            tmp_path,
        )
