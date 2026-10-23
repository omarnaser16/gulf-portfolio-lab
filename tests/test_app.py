"""Headless test of the Streamlit dashboard: synthetic data, no network, temp cache."""

import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
import yfinance as yf
from streamlit.testing.v1 import AppTest

import gpl.data
from gpl.agent import analyst as agent
from test_agent import FakeClient, response, text, tool_use

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
sys.path.insert(0, str(Path(APP).parent))  # the app's own modules (workstation, inputs)


SUN_THU = (".SR", ".QA", ".KW")  # Saudi, Qatari and Kuwaiti markets trade Sunday-Thursday
FAKE_CURRENCY = {".SR": "SAR", ".QA": "QAR", ".KW": "KWF", ".AE": "AED"}
FAKE_TODAY = pd.Timestamp("2026-10-22")


def synthetic_download(tickers, start=None, end=None, **kwargs) -> pd.DataFrame:
    """yfinance-shaped random walks for any ticker; reproducible per ticker name.

    Tickers starting with NOPE have no data (like a wrong symbol). GCC suffixes trade Sun-Thu.
    """
    if isinstance(tickers, str):
        tickers = tickers.split()
    last = pd.Timestamp(end) - pd.Timedelta(days=1) if end is not None else FAKE_TODAY
    close = {}
    for t in tickers:
        if t.startswith("NOPE"):
            continue
        if t.startswith("NEW"):  # a recent listing: only ~100 trading days of history
            idx = pd.bdate_range(last - pd.Timedelta(days=140), last)
        elif t.endswith(SUN_THU):
            idx = pd.DatetimeIndex(
                [d for d in pd.date_range(start, last) if d.day_of_week in (6, 0, 1, 2, 3)]
            )
        else:
            idx = pd.bdate_range(start, last)
        rng = np.random.default_rng(zlib.crc32(t.encode()))
        if t == "^IRX":
            values = np.full(len(idx), 3.0)
        elif t == "^VIX":
            values = 18 + 6 * np.abs(np.sin(np.arange(len(idx)) / 150)) + rng.normal(0, 1, len(idx))
        elif t.endswith("=X"):
            values = np.full(len(idx), 0.3082)
        else:
            vol = 0.008 + 0.012 * (np.sin(np.arange(len(idx)) / 200) > 0.8)  # occasional stress
            values = 100 * np.cumprod(1 + rng.normal(0.0003, vol))
        close[t] = pd.Series(values, index=idx)
    if not close:
        return pd.DataFrame()
    frame = pd.DataFrame(close)
    out = pd.concat({"Close": frame, "Open": frame}, axis=1, names=["Price", "Ticker"])
    out.index.name = "Date"
    return out


class FakeTicker:
    def __init__(self, symbol: str) -> None:
        suffix = "." + symbol.rsplit(".", 1)[-1]
        self.fast_info = {"currency": FAKE_CURRENCY.get(suffix, "USD")}


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppTest:
    monkeypatch.setattr(yf, "download", synthetic_download)
    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    monkeypatch.setattr(gpl.data, "DEFAULT_CACHE_DIR", tmp_path)
    monkeypatch.setattr(agent, "has_credentials", lambda: False)  # independent of a local .env
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def test_app_renders_every_tab(app: AppTest) -> None:
    assert not app.exception, app.exception
    sections = {"Holdings", "Screen", "Analyse & backtest", "Optimise"}  # inside Portfolio
    assert [t.label for t in app.tabs if t.label not in sections] == [
        "Portfolio",
        "Assets",
        "Frontier",
        "Backtest",
        "Monte Carlo",
        "Regime",
        "Analyst",
    ]
    # static tables: data notes, screen, my holdings, my comparison, assets, strategies,
    # regime folds; dataframes: the editable holdings table and the frontier weights
    assert len(app.table) == 7 and len(app.dataframe) == 2
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
    """Assets and Backtest-tab metric tables (static st.table, so headers can wrap)."""
    tables = [
        t.value
        for t in app.table
        if "Sharpe" in t.value.columns and "Correlation to rest" not in t.value.columns
    ]
    return [t for t in tables if "My portfolio" not in t.index][:2]


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
    assert any("starts 252 observations after the data" in c.value for c in app.caption)


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
    assert all(m.delta.endswith("pp vs calm") for m in app.metric if m.label == "Accuracy")


def test_cap_caption_uses_two_decimals(app: AppTest) -> None:
    assert any("at most **40.00%** per asset" in c.value for c in app.caption)


def test_benchmark_caption_uses_two_decimals(app: AppTest) -> None:
    text = " ".join(c.value for c in app.caption)
    assert "holds 60.00% SPY and 40.00% AGG; SPY holds 100.00%" in text


# ------------------------------------------------------------------------------ Monte Carlo


MC_LABELS = ("P(≥", "Median outcome", "Bad case", "P(lose money)", "Drawdown")


def _mc_metrics(app: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in app.metric if m.label.startswith(MC_LABELS)}


def _mc_units(app: AppTest) -> str:
    return next(m.value for m in app.markdown if m.value.startswith("**Over "))


def test_monte_carlo_defaults_to_block_bootstrap_in_nominal_usd(app: AppTest) -> None:
    assert app.radio(key="mc_method").value == "block_bootstrap"
    assert "crashes cluster" in app.radio(key="mc_method").help.lower()
    metrics = _mc_metrics(app)
    assert len(metrics) == 5
    assert "in nominal USD" in _mc_units(app)
    caption = " ".join(c.value for c in app.caption)
    assert "rebalance daily and ignore transaction costs" in caption
    assert "unlike the backtest" in caption


def test_real_terms_toggle_deflates_outcomes(app: AppTest) -> None:
    nominal = _mc_metrics(app)
    app.toggle(key="mc_real").set_value(True).run()
    assert not app.exception, app.exception
    real = _mc_metrics(app)
    assert "in real USD" in _mc_units(app)

    def median(metrics: dict[str, str]) -> float:
        return float(next(v for k, v in metrics.items() if k.startswith("Median"))[:-1])

    # 10 years at 2.5%: real median = nominal / 1.025**10 (up to the 2-decimal display)
    assert median(real) == pytest.approx(median(nominal) / 1.025**10, abs=0.01)
    assert any("2.50% inflation" in c.value for c in app.caption)


@pytest.mark.parametrize("method", ["bootstrap", "gbm"])
def test_other_methods_still_available(app: AppTest, method: str) -> None:
    app.radio(key="mc_method").set_value(method).run()
    assert not app.exception, app.exception
    assert len(_mc_metrics(app)) == 5


def test_drawdown_along_the_way_metric(app: AppTest) -> None:
    value = next(v for k, v in _mc_metrics(app).items() if k.startswith("Drawdown"))
    assert re.fullmatch(r"-\d+\.\d{2}%", value), value  # negative, two decimals
    app.radio(key="mc_method").set_value("bootstrap").run()
    daily = next(v for k, v in _mc_metrics(app).items() if k.startswith("Drawdown"))
    assert daily != value  # the method changes path roughness


# ----------------------------------------------------------------------------------- Regime

REGIME_LABELS = ["ROC AUC", "Precision", "Recall", "Accuracy", "Latest P(stressed)"]


def test_regime_headline_leads_with_auc_and_base_rate(app: AppTest) -> None:
    labels = [m.label for m in app.metric if m.label in REGIME_LABELS]
    assert labels == REGIME_LABELS  # AUC first, accuracy not the lead
    by_label = {m.label: m for m in app.metric}
    assert re.fullmatch(r"0\.\d{2}|1\.00", by_label["ROC AUC"].value)
    assert by_label["Precision"].delta.endswith("pp vs base")
    assert any("base rate (actual stressed days)" in m.value for m in app.markdown)


def test_regime_fold_table_has_rule_baseline_and_total(app: AppTest) -> None:
    folds = next(t.value for t in app.table if "Vol-rule precision" in t.value.columns)
    assert {"Avg P(stressed)", "Model ROC AUC", "Vol-rule recall", "Always-calm accuracy"} <= set(
        folds.columns
    )
    assert folds.index[-1] == "All folds" and len(folds) == 6


def test_regime_chart_is_monthly(app: AppTest) -> None:
    chart = next(
        json.loads(c.proto.spec)
        for c in app.get("plotly_chart")
        if "stressed month ahead" in json.loads(c.proto.spec)["layout"]["title"]["text"]
    )
    dates = pd.to_datetime(chart["data"][0]["x"])
    assert (dates.to_series().diff().dropna().dt.days >= 28).all()  # one point per month
    assert dates.is_month_end.all() or (dates.to_series().dt.day >= 25).all()


def test_regime_caption_states_history_and_relative_definition(app: AppTest) -> None:
    caption = next(c.value for c in app.caption if "regime model uses" in c.value)
    assert "SPY, AGG and VIX history from **Jan 2005**" in caption
    assert "'Stressed' is relative:" in caption


# ---------------------------------------------------------------------------------- Analyst


def test_analyst_without_key_explains_setup(app: AppTest) -> None:
    assert any("No API key found" in i.value for i in app.info)


@pytest.fixture
def analyst_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[AppTest, FakeClient]:
    client = FakeClient(
        [
            response([tool_use("t1", "get_universe", {})], "tool_use", (800, 40)),
            response(
                [text("## Profile\nModerate investor. CAGR 12.34% would be nice.")],
                "end_turn",
                (1200, 400),
            ),
            response([text("A more conservative choice is risk parity.")], "end_turn", (300, 60)),
        ]
    )
    monkeypatch.setattr(yf, "download", synthetic_download)
    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    monkeypatch.setattr(gpl.data, "DEFAULT_CACHE_DIR", tmp_path)
    monkeypatch.setattr(agent, "has_credentials", lambda: True)
    monkeypatch.setattr(agent, "make_client", lambda: client)
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at, client


def test_analyst_writes_an_audited_memo(analyst_app) -> None:
    at, client = analyst_app
    at.button(key="write_memo").click().run()
    assert not at.exception, at.exception
    assert any("Moderate investor" in m.value for m in at.markdown)
    assert any("not investment advice" in m.value.lower() for m in at.markdown)
    assert any("`12.34%`" in w.value for w in at.warning)  # invented number flagged
    assert any("1 tool calls" in c.value and "about $" in c.value for c in at.caption)
    assert any("Audit trail: 1 tool calls" in e.label for e in at.expander)
    request = client.requests[0]
    assert request["model"] == "claude-haiku-5-5"
    assert "Dubai" in request["messages"][0]["content"]


def test_analyst_follow_up_continues_the_conversation(analyst_app) -> None:
    at, client = analyst_app
    at.button(key="write_memo").click().run()
    at.chat_input(key="follow_up").set_value("What if I were more conservative?").run()
    assert not at.exception, at.exception
    assert any("risk parity" in m.value for m in at.markdown)
    roles = [m["role"] for m in client.requests[-1]["messages"]]
    assert roles[-1] == "user" and roles.count("user") == 3  # profile, tool results, follow-up


# --------------------------------------------------------------------- any asset, data period


def test_latest_data_is_the_default_with_as_of_line(app: AppTest) -> None:
    assert app.radio(key="data_period").value == "Latest data"
    assert any("Data as of 22 Oct 2026" in m.value for m in app.markdown)
    app.radio(key="data_period").set_value(app.radio(key="data_period").options[1]).run()
    assert not app.exception, app.exception
    assert any("Data as of 30 Sep 2026" in m.value and "README" in m.value for m in app.markdown)


def test_invalid_typed_ticker_gets_a_friendly_warning(app: AppTest) -> None:
    app.text_input(key="custom_tickers").set_value("NOPE.AE").run()
    assert not app.exception, app.exception
    warning = " ".join(w.value for w in app.warning)
    assert "NOPE.AE" in warning and "no data on yahoo finance" in warning
    assert "finance.yahoo.com" in warning


def test_gcc_preset_converts_and_aligns_calendars(app: AppTest) -> None:
    app.multiselect(key="presets").set_value(["GCC stocks"]).run()
    assert not app.exception, app.exception
    info = " ".join(i.value for i in app.info)
    assert "Sunday to Thursday" in info and "periods a year instead of 252" in info
    notes = next(t.value for t in app.table if "Converted to USD with" in t.value.columns)
    assert notes.loc["2222.SR", "Converted to USD with"] == "SAR peg 3.75 per USD"
    assert notes.loc["NBK.KW", "Converted to USD with"].startswith("KWD market rate")
    assert notes.loc["2222.SR", "Name"] == "Saudi Aramco"


def test_short_history_warning(app: AppTest) -> None:
    app.text_input(key="custom_tickers").set_value("NEWCO").run()
    assert not app.exception, app.exception
    assert any("NEWCO" in w.value and "trading days of history" in w.value for w in app.warning)
    # the common sample is now too short for a backtest: explained, not a crash
    assert any("It is limited by NEWCO" in w.value for w in app.warning)


def test_blocked_download_offers_retry(monkeypatch, tmp_path) -> None:
    def blocked(*args, **kwargs):
        raise gpl.data.DataUnavailable("rate limited")

    monkeypatch.setattr(gpl.data, "download_series", blocked)
    monkeypatch.setattr(gpl.data, "DEFAULT_CACHE_DIR", tmp_path)
    monkeypatch.setattr(agent, "has_credentials", lambda: False)
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert any("Yahoo Finance didn't respond" in e.value for e in at.error)
    assert at.button(key="retry").label == "Retry"


# ------------------------------------------------------------------------------ My portfolio


def _my_comparison(app: AppTest) -> pd.DataFrame:
    return next(t.value for t in app.table if "My portfolio" in t.value.index)


def test_my_portfolio_default_compares_with_strategies(app: AppTest) -> None:
    table = _my_comparison(app)
    assert {"My portfolio", "Min variance", "Risk parity", "60/40 (SPY/AGG)"} <= set(table.index)
    holdings = next(t.value for t in app.table if "Share of risk" in t.value.columns)
    assert list(holdings.index) == ["SPY", "AGG", "GLD", "KSA"]
    assert holdings.loc["SPY", "Weight"] == "50.00%"


def _set_holdings(app: AppTest, rows: list[dict], mode: str = "Weight (%)") -> None:
    app.session_state["ws_rows"] = rows
    app.session_state["ws_mode"] = mode
    app.session_state["ws_mode_prev"] = mode
    app.session_state["ws_version"] = app.session_state["ws_version"] + 1
    app.run()


def test_holdings_in_usd_outside_universe_and_mixed_calendar(app: AppTest) -> None:
    _set_holdings(
        app,
        [
            {"ticker": "SPY", "group": "Global equity", "value": 6000.0},
            {"ticker": "2222.SR", "group": "GCC", "value": 4000.0},
        ],
        mode="USD amount",
    )
    assert not app.exception, app.exception
    holdings = next(t.value for t in app.table if "Share of risk" in t.value.columns)
    assert holdings.loc["SPY", "Weight"] == "60.00%"
    assert holdings.loc["2222.SR", "Converted to USD with"] == "SAR peg 3.75 per USD"
    assert any("Sunday to Thursday" in i.value for i in app.info)  # holdings joined the dates
    assert "My portfolio" in _my_comparison(app).index


def test_holdings_report_bad_values_and_unknown_tickers(app: AppTest) -> None:
    _set_holdings(
        app,
        [
            {"ticker": "SPY", "group": "Global equity", "value": 60.0},
            {"ticker": "AGG", "group": "Bonds", "value": 0.0},
            {"ticker": "NOPE1", "group": "Other", "value": 40.0},
        ],
    )
    assert not app.exception, app.exception
    assert any("AGG: enter a positive weight" in w.value for w in app.warning)
    assert any("No price history for NOPE1" in e.value for e in app.error)
    assert not any(
        "My portfolio" in t.value.index for t in app.table if "Sharpe" in t.value.columns
    )


def test_switching_to_usd_keeps_the_weights(app: AppTest) -> None:
    app.radio(key="ws_mode").set_value("USD amount").run()
    assert not app.exception, app.exception
    values = {r["ticker"]: r["value"] for r in app.session_state["ws_rows"]}
    assert values == {"SPY": 5000.0, "AGG": 3000.0, "GLD": 1000.0, "KSA": 1000.0}


def test_import_list_replaces_holdings(app: AppTest) -> None:
    app.text_area(key="ws_import").set_value("SPY 70\nTLT 30")
    app.button(key="ws_import_go").click().run()
    assert not app.exception, app.exception
    rows = app.session_state["ws_rows"]
    assert [(r["ticker"], r["value"]) for r in rows] == [("SPY", 70.0), ("TLT", 30.0)]


def test_picking_a_search_result_adds_it(monkeypatch, tmp_path) -> None:
    import workstation  # noqa: PLC0415 - the app's module, importable once the path is set

    monkeypatch.setattr(yf, "download", synthetic_download)
    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    monkeypatch.setattr(gpl.data, "DEFAULT_CACHE_DIR", tmp_path)
    monkeypatch.setattr(agent, "has_credentials", lambda: False)
    monkeypatch.setattr(workstation, "st_searchbox", lambda *a, **k: "2222.SR")
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    rows = {r["ticker"]: r for r in at.session_state["ws_rows"]}
    assert rows["2222.SR"]["group"] == "GCC" and rows["2222.SR"]["value"] == 10.0
    assert len(rows) == 5  # added once, not again on the rerun


def test_screen_ranks_holdings(app: AppTest) -> None:
    screen = next(t.value for t in app.table if "Correlation to rest" in t.value.columns)
    assert set(screen.index) == {"SPY", "AGG", "GLD", "KSA"} and list(screen["Rank"]) == [
        1,
        2,
        3,
        4,
    ]
    app.selectbox(key="ws_rank_by").set_value("Volatility (ann.)").run()
    screen = next(t.value for t in app.table if "Correlation to rest" in t.value.columns)
    vols = [float(v.rstrip("%")) for v in screen["Volatility (ann.)"]]
    assert vols == sorted(vols)


def test_monte_carlo_offers_my_portfolio(app: AppTest) -> None:
    assert "My portfolio" in app.selectbox(key="mc_strategy").options
    app.selectbox(key="mc_strategy").set_value("My portfolio").run()
    assert not app.exception, app.exception
    assert any(
        "My portfolio: simulated growth" in json.loads(c.proto.spec)["layout"]["title"]["text"]
        for c in app.get("plotly_chart")
    )


def test_analyst_writes_a_memo_about_my_portfolio(analyst_app) -> None:
    at, client = analyst_app
    at.button(key="write_portfolio_memo").click().run()
    assert not at.exception, at.exception
    question = client.requests[0]["messages"][0]["content"]
    assert question.startswith("Please review my portfolio (SPY 50.00%, AGG 30.00%")
    tools = {t["name"]: t for t in client.requests[0]["tools"]}
    assert (
        "My portfolio"
        in tools["backtest_strategies"]["input_schema"]["properties"]["strategies"]["items"]["enum"]
    )
