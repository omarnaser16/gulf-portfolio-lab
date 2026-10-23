"""Headless test of the Streamlit dashboard: synthetic data, no network, temp cache."""

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
import yfinance as yf
from streamlit.testing.v1 import AppTest

import gpl.data

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def synthetic_download(tickers, start=None, end=None, **kwargs) -> pd.DataFrame:
    """yfinance-shaped random walks for any ticker; reproducible per ticker name."""
    if isinstance(tickers, str):
        tickers = tickers.split()
    idx = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
    close = {}
    for t in tickers:
        rng = np.random.default_rng(zlib.crc32(t.encode()))
        if t == "^IRX":
            close[t] = np.full(len(idx), 3.0)
        elif t == "^VIX":
            close[t] = (
                18 + 6 * np.abs(np.sin(np.arange(len(idx)) / 150)) + rng.normal(0, 1, len(idx))
            )
        else:
            vol = 0.008 + 0.012 * (np.sin(np.arange(len(idx)) / 200) > 0.8)  # occasional stress
            close[t] = 100 * np.cumprod(1 + rng.normal(0.0003, vol))
    frame = pd.DataFrame(close, index=idx)
    out = pd.concat({"Close": frame, "Open": frame}, axis=1, names=["Price", "Ticker"])
    out.index.name = "Date"
    return out


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppTest:
    monkeypatch.setattr(yf, "download", synthetic_download)
    monkeypatch.setattr(gpl.data, "DEFAULT_CACHE_DIR", tmp_path)
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def test_app_renders_every_tab(app: AppTest) -> None:
    assert not app.exception, app.exception
    assert [t.label for t in app.tabs] == [
        "Assets",
        "Frontier",
        "Backtest",
        "Monte Carlo",
        "Regime",
        "Analyst",
    ]
    assert len(app.dataframe) >= 4  # assets, frontier weights, strategies, regime folds
    assert any("P(stressed)" in m.label for m in app.metric)


def test_regime_toggle_adds_regime_strategies(app: AppTest) -> None:
    app.toggle(key="use_regime").set_value(True).run()
    assert not app.exception, app.exception
    names = list(app.selectbox(key="mc_strategy").options)
    assert "Equal weight + regime" in names


def test_infeasible_cap_shows_error(app: AppTest) -> None:
    app.slider(key="max_weight").set_value(0.10).run()  # 10% x 9 assets < 100%
    assert app.error
