"""Tests for loading any asset: currencies, pegs, sub-units, calendars and history checks.

No network: yfinance is replaced by a fake that serves per-ticker series on chosen calendars.
"""

import json

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

import gpl.data as data
from gpl.data import (
    PEGS,
    DataUnavailable,
    align_common_dates,
    clean_prices,
    get_currency,
    get_prices,
    load_assets,
    observations_per_year,
    to_usd,
    trading_weekdays,
)

MON_FRI = pd.bdate_range("2023-01-02", "2024-12-31")
SUN_THU = pd.DatetimeIndex(
    [d for d in pd.date_range("2023-01-01", "2024-12-31") if d.day_of_week in (6, 0, 1, 2, 3)]
)


def walk(index: pd.DatetimeIndex, seed: int, level: float = 100.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(level * np.cumprod(1 + rng.normal(0.0003, 0.01, len(index))), index=index)


class FakeYahoo:
    """yf.download stand-in serving given series; records which tickers were downloaded."""

    def __init__(self, series: dict[str, pd.Series], currencies: dict[str, str]) -> None:
        self.series, self.currencies, self.calls = series, currencies, []

    def download(self, tickers, start=None, end=None, **kwargs) -> pd.DataFrame:
        tickers = [tickers] if isinstance(tickers, str) else list(tickers)
        self.calls += tickers
        if not any(t in self.series for t in tickers):
            return pd.DataFrame()  # what yfinance returns for an unknown ticker
        frame = pd.DataFrame({t: self.series.get(t, pd.Series(dtype=float)) for t in tickers})
        if start is not None:
            frame = frame[frame.index >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame.index < pd.Timestamp(end)]
        if frame.dropna(how="all").empty:
            return pd.DataFrame()
        out = pd.concat({"Close": frame}, axis=1, names=["Price", "Ticker"])
        out.index.name = "Date"
        return out

    def ticker(self, symbol: str):
        if symbol not in self.currencies:
            raise KeyError(symbol)
        return type("T", (), {"fast_info": {"currency": self.currencies[symbol]}})()


@pytest.fixture
def yahoo(monkeypatch, tmp_path) -> FakeYahoo:
    fake = FakeYahoo(
        series={
            "SPY": walk(MON_FRI, 1, 400.0),
            "AGG": walk(MON_FRI, 2, 100.0),
            "2222.SR": walk(SUN_THU, 3, 30.0),
            "NBK.KW": walk(SUN_THU, 4, 900.0),
            "KWD=X": pd.Series(0.3082, index=MON_FRI),
        },
        currencies={"2222.SR": "SAR", "NBK.KW": "KWF", "KWD=X": "KWD"},
    )
    monkeypatch.setattr(yf, "download", fake.download)
    monkeypatch.setattr(yf, "Ticker", fake.ticker)
    monkeypatch.setattr(data, "DEFAULT_CACHE_DIR", tmp_path)
    monkeypatch.setattr(data, "RETRY_DELAYS", (0.0, 0.0))
    return fake


# ------------------------------------------------------------------------------ to_usd


def test_pegged_currency_hand_value() -> None:
    local = pd.DataFrame({"2222.SR": [37.5]}, index=pd.to_datetime(["2024-01-07"]))
    usd, notes = to_usd(local, {"2222.SR": "SAR"}, {})
    assert usd.iloc[0, 0] == pytest.approx(10.0)  # 37.50 SAR / 3.75
    assert notes["2222.SR"] == "SAR peg 3.75 per USD"


def test_subunit_and_market_fx_hand_value() -> None:
    # 851 fils = 0.851 KWD; KWD=X = 0.3082 KWD per USD -> 0.851 / 0.3082 = 2.7612 USD
    local = pd.DataFrame({"NBK.KW": [851.0]}, index=pd.to_datetime(["2024-01-04"]))
    fx = {"KWD": pd.Series([0.3082], index=pd.to_datetime(["2024-01-04"]))}
    usd, notes = to_usd(local, {"NBK.KW": "KWF"}, fx)
    assert usd.iloc[0, 0] == pytest.approx(0.851 / 0.3082)
    assert notes["NBK.KW"] == "KWD market rate (KWD=X) (KWF = KWD/1000)"


def test_sunday_price_uses_fridays_rate() -> None:
    fx = {
        "KWD": pd.Series([0.30, 0.31], index=pd.to_datetime(["2024-01-05", "2024-01-08"]))
    }  # Fri, Mon
    local = pd.DataFrame({"NBK.KW": [1000.0]}, index=pd.to_datetime(["2024-01-07"]))  # Sunday
    usd, _ = to_usd(local, {"NBK.KW": "KWF"}, fx)
    assert usd.iloc[0, 0] == pytest.approx(1.0 / 0.30)


def test_pence_and_usd() -> None:
    local = pd.DataFrame({"VOD.L": [122.7], "SPY": [500.0]}, index=pd.to_datetime(["2024-01-02"]))
    fx = {"GBP": pd.Series([0.8], index=local.index)}
    usd, notes = to_usd(local, {"VOD.L": "GBp", "SPY": "USD"}, fx)
    assert usd["VOD.L"].iloc[0] == pytest.approx(1.227 / 0.8)
    assert usd["SPY"].iloc[0] == 500.0 and notes["SPY"] == "USD"


def test_missing_fx_raises() -> None:
    local = pd.DataFrame({"X.DE": [10.0]}, index=pd.to_datetime(["2024-01-02"]))
    with pytest.raises(ValueError, match="EUR"):
        to_usd(local, {"X.DE": "EUR"}, {})


def test_all_pegs_are_local_per_usd() -> None:
    assert PEGS["SAR"] == 3.75 and PEGS["AED"] == 3.6725 and PEGS["BHD"] < 1


# -------------------------------------------------------------------------- calendars


def test_trading_weekdays() -> None:
    assert trading_weekdays(walk(MON_FRI, 0)) == frozenset({0, 1, 2, 3, 4})
    assert trading_weekdays(walk(SUN_THU, 0)) == frozenset({6, 0, 1, 2, 3})


def test_align_common_dates_keeps_mon_to_thu() -> None:
    prices = pd.DataFrame({"US": walk(MON_FRI, 0), "SA": walk(SUN_THU, 1)})
    aligned, report = align_common_dates(prices)
    assert set(aligned.index.day_of_week) == {0, 1, 2, 3}
    assert not aligned.isna().any().any()
    assert report["kept"] == len(aligned) and report["dropped"] > 0
    assert report["missing_by_asset"]["SA"] > 0 and report["missing_by_asset"]["US"] > 0


def test_observations_per_year() -> None:
    assert observations_per_year(MON_FRI) == pytest.approx(261, abs=2)  # weekdays incl. holidays
    mon_thu = MON_FRI[MON_FRI.day_of_week < 4]
    assert observations_per_year(mon_thu) == pytest.approx(209, abs=2)  # 4/5 of the year


# ---------------------------------------------------------------------------- currency


def test_currency_lookup_cached_and_suffix_fallback(yahoo, tmp_path) -> None:
    assert get_currency("SPY") == ("USD", "us listing")
    assert get_currency("2222.SR") == ("SAR", "yahoo")
    assert json.loads((tmp_path / "currencies.json").read_text())["2222.SR"] == "SAR"
    assert get_currency("UNKNOWN.QA") == ("QAR", "suffix")  # lookup fails -> exchange suffix


# --------------------------------------------------------------------------- load_assets


def test_load_assets_converts_aligns_and_reports(yahoo) -> None:
    out = load_assets(
        ["SPY", "2222.SR", "NBK.KW", "FAB.AE", "bad ticker!"], "2023-01-01", "2024-12-31"
    )
    assert out.invalid == {
        "FAB.AE": "No data on Yahoo Finance",
        "bad ticker!": "Not a valid ticker format",
    }
    assert list(out.prices.columns) == ["SPY", "2222.SR", "NBK.KW"]
    assert out.mixed_calendars and set(out.prices.index.day_of_week) <= {0, 1, 2, 3}
    assert out.periods_per_year == pytest.approx(209, abs=4)
    # Converted: 2222.SR in USD is the SAR walk / 3.75 on the kept dates
    sar = yahoo.series["2222.SR"].reindex(out.prices.index)
    np.testing.assert_allclose(out.prices["2222.SR"], sar / 3.75)
    h = out.history
    assert h.loc["NBK.KW", "conversion"].startswith("KWD market rate")
    assert h.loc["2222.SR", "trading_days"] == "Mon Tue Wed Thu Sun"
    assert out.price_as_of == out.prices.index[-1]


def test_same_calendar_matches_the_original_cleaning(yahoo) -> None:
    """Default US ETFs must give exactly the old pipeline's prices (README reproducibility)."""
    out = load_assets(["SPY", "AGG"], "2023-01-01", "2024-12-31")
    expected = clean_prices(get_prices(["SPY", "AGG"], "2023-01-01", "2024-12-31"))
    pd.testing.assert_frame_equal(out.prices, expected, check_freq=False, check_names=False)
    assert not out.mixed_calendars and out.periods_per_year == 252


def test_history_flags_short_assets(yahoo) -> None:
    yahoo.series["NEW"] = walk(MON_FRI[-100:], 9)
    out = load_assets(["SPY", "NEW"], "2023-01-01", "2024-12-31", lookback=252)
    assert bool(out.history.loc["NEW", "too_short"]) and bool(out.history.loc["NEW", "sets_start"])
    assert not out.history.loc["SPY", "too_short"]


def test_downloads_are_cached_per_ticker(yahoo) -> None:
    load_assets(["SPY", "AGG"], "2023-01-01", "2024-12-31")
    load_assets(["SPY", "2222.SR"], "2023-01-01", "2024-12-31")
    assert yahoo.calls.count("SPY") == 1  # reused across different combinations


def test_blocked_download_raises_after_retries(yahoo, monkeypatch) -> None:
    monkeypatch.setattr(data, "_looks_blocked", lambda t: True)
    with pytest.raises(DataUnavailable):
        load_assets(["NOPE"], "2023-01-01", "2024-12-31")
    assert yahoo.calls.count("NOPE") == 3  # first try + 2 retries


def test_nothing_valid_returns_empty(yahoo) -> None:
    out = load_assets(["NOPE"], "2023-01-01", "2024-12-31")
    assert out.prices.empty and out.invalid == {"NOPE": "No data on Yahoo Finance"}
