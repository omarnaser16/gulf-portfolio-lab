"""Headless test of the Streamlit dashboard: synthetic data, no network, temp cache."""

import json
import re
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
    # metric tables (assets, strategies) are st.table; frontier weights and folds are dataframes
    assert len(app.table) == 2 and len(app.dataframe) == 2
    assert any("P(stressed)" in m.label for m in app.metric)


def test_regime_toggle_adds_regime_strategies(app: AppTest) -> None:
    app.toggle(key="use_regime").set_value(True).run()
    assert not app.exception, app.exception
    names = list(app.selectbox(key="mc_strategy").options)
    assert "Equal weight + regime" in names


def test_infeasible_cap_shows_error(app: AppTest) -> None:
    app.slider(key="max_weight").set_value(0.10).run()  # 10% x 9 assets < 100%
    assert app.error


def _frontier_weights(app: AppTest) -> pd.DataFrame:
    tables = [  # weights table: portfolios as rows, tickers as columns (no metric columns)
        df.value
        for df in app.dataframe
        if "Min variance" in df.value.index and "Sharpe" not in df.value.columns
    ]
    assert len(tables) == 1
    return tables[0]


@pytest.mark.parametrize("cap", [0.15, 0.30, 0.75])
def test_frontier_tab_follows_the_sidebar_cap(app: AppTest, cap: float) -> None:
    app.slider(key="max_weight").set_value(cap).run()
    assert not app.exception, app.exception
    weights = _frontier_weights(app)
    assert (weights.max(axis=1) <= cap + 1e-9).all(), weights.max(axis=1)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert any(f"{cap:.2%}" in c.value for c in app.caption)  # the cap in use is shown


def _metric_tables(app: AppTest) -> list[pd.DataFrame]:
    """Assets and backtest metric tables (static st.table, so headers can wrap)."""
    return [t.value for t in app.table if "Sharpe" in t.value.columns]


def test_metric_tables_are_readable(app: AppTest) -> None:
    assets_table, backtest_table = _metric_tables(app)
    for table in (assets_table, backtest_table):
        assert not {"start", "end", "n_obs", "sharpe", "ann_vol"} & set(table.columns)
        assert {"Return (ann.)", "CAGR", "Volatility (ann.)", "Max drawdown"} <= set(table.columns)
    assert "Turnover / yr" in backtest_table.columns
    assert list(backtest_table.columns).index("Turnover / yr") <= 6  # near the left, not cut off
    assert "Turnover / yr" not in assets_table.columns  # assets aren't traded


def test_benchmark_label_and_backtest_caption(app: AppTest) -> None:
    _, backtest_table = _metric_tables(app)
    assert "60/40 (SPY/AGG)" in backtest_table.index
    assert any("starts 252 days after the data" in c.value for c in app.caption)


def test_weights_chart_uses_readable_group_names(app: AppTest) -> None:
    names = set()
    for chart in app.get("plotly_chart"):
        names |= {t.get("name") for t in json.loads(chart.proto.spec)["data"]}
    assert {"GCC equity", "Global equity", "Bonds", "Gold"} <= names
    assert not {"gcc_equity", "real_assets"} & names


def test_every_percentage_metric_has_two_decimals(app: AppTest) -> None:
    percents = [m.value for m in app.metric if m.value.endswith("%")]
    assert len(percents) >= 6  # Monte Carlo and Regime headline numbers
    assert all(re.fullmatch(r"-?\d+\.\d{2}%", v) for v in percents), percents
    assert all("pp vs always-calm" in m.delta for m in app.metric if m.label == "Accuracy")


def test_cap_caption_uses_two_decimals(app: AppTest) -> None:
    assert any("at most **40.00%** per asset" in c.value for c in app.caption)


def test_benchmark_caption_uses_two_decimals(app: AppTest) -> None:
    text = " ".join(c.value for c in app.caption)
    assert "holds 60.00% SPY and 40.00% AGG; SPY holds 100.00%" in text
