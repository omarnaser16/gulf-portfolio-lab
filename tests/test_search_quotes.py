"""Tests for security search, live quotes and correlation to the rest of the portfolio.

No network: yf.Search and yf.Ticker are replaced by fakes with Yahoo-shaped responses.
"""

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

from gpl.data import latest_quote, search_securities, usd_value
from gpl.metrics import correlation_to_rest


class FakeSearch:
    def __init__(self, query, max_results=8, news_count=8, **kwargs):
        self.quotes = [
            {
                "symbol": "2222.SR",
                "longname": "Saudi Arabian Oil Company",
                "exchDisp": "Saudi",
                "quoteType": "EQUITY",
            },
            {
                "symbol": "XAAPL=F",
                "shortname": "Apple futures",
                "exchange": "CME",
                "quoteType": "FUTURE",
            },
            {"symbol": "SPY", "shortname": "SPDR S&P 500", "exchange": "PCX", "quoteType": "ETF"},
        ]


INFO = {
    "2222.SR": {
        "regularMarketPrice": 25.4,
        "currency": "SAR",
        "regularMarketTime": 1792671587,
        "exchangeTimezoneName": "Asia/Riyadh",
        "longName": "Saudi Arabian Oil Company",
        "fullExchangeName": "Saudi",
        "quoteType": "EQUITY",
    },
    "NBK.KW": {
        "regularMarketPrice": 825.0,
        "currency": "KWF",
        "regularMarketTime": 1792663800,
        "exchangeTimezoneName": "Asia/Kuwait",
        "longName": "National Bank of Kuwait",
        "fullExchangeName": "Kuwait",
        "quoteType": "EQUITY",
    },
    "KWD=X": {"regularMarketPrice": 0.3082, "currency": "KWD", "regularMarketTime": 1792663800},
}


class FakeTicker:
    def __init__(self, symbol):
        if symbol not in INFO:
            raise KeyError(symbol)
        self.info = INFO[symbol]


@pytest.fixture(autouse=True)
def fake_yahoo(monkeypatch):
    monkeypatch.setattr(yf, "Search", FakeSearch)
    monkeypatch.setattr(yf, "Ticker", FakeTicker)


def test_search_keeps_securities_and_drops_futures() -> None:
    results = search_securities("aramco")
    assert [r["symbol"] for r in results] == ["2222.SR", "SPY"]
    assert results[0] == {
        "symbol": "2222.SR",
        "name": "Saudi Arabian Oil Company",
        "exchange": "Saudi",
        "type": "Stock",
    }
    assert search_securities("a") == []  # too short to search


def test_search_never_raises(monkeypatch) -> None:
    def broken(*a, **k):
        raise ConnectionError("offline")

    monkeypatch.setattr(yf, "Search", broken)
    assert search_securities("emaar") == []


def test_quote_pegged_currency_and_market_time() -> None:
    q = latest_quote("2222.SR")
    assert q.price == 25.4 and q.price_usd == pytest.approx(25.4 / 3.75)
    assert q.as_of.endswith("+03") and q.name == "Saudi Arabian Oil Company"


def test_quote_subunit_with_market_fx() -> None:
    q = latest_quote("NBK.KW")  # 825 fils = 0.825 KWD; KWD=X 0.3082 KWD per USD
    assert q.price_usd == pytest.approx(0.825 / 0.3082)


def test_unknown_quote_is_none() -> None:
    assert latest_quote("NOPE") is None


def test_usd_value() -> None:
    assert usd_value(37.5, "SAR") == pytest.approx(10.0)
    assert usd_value(500.0, "USD") == 500.0
    with pytest.raises(ValueError):
        usd_value(10.0, "EUR")


# ------------------------------------------------------------------- correlation to the rest


def test_correlation_to_rest_two_assets_is_their_correlation() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    r = pd.DataFrame({"A": a, "B": 0.5 * a + rng.normal(size=300)})
    out = correlation_to_rest(r, pd.Series({"A": 0.5, "B": 0.5}))
    assert out["A"] == pytest.approx(r["A"].corr(r["B"]))
    assert out["B"] == pytest.approx(r["A"].corr(r["B"]))


def test_correlation_to_rest_uses_relative_weights_of_others() -> None:
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(size=(300, 3)), columns=["A", "B", "C"])
    out = correlation_to_rest(r, pd.Series({"A": 0.2, "B": 0.6, "C": 0.2}))
    rest_of_a = r["B"] * 0.75 + r["C"] * 0.25  # B and C at 0.6 : 0.2
    assert out["A"] == pytest.approx(r["A"].corr(rest_of_a))


def test_correlation_to_rest_single_holding_is_nan() -> None:
    r = pd.DataFrame({"A": [0.01, -0.02, 0.03]})
    assert np.isnan(correlation_to_rest(r, pd.Series({"A": 1.0}))["A"])
