"""Headless test of the Streamlit dashboard: synthetic data, no network, temp cache."""

import json
import re
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

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
import workstation as ws  # noqa: E402

SUN_THU = (".SR", ".QA", ".KW")  # Saudi, Qatari and Kuwaiti markets trade Sunday-Thursday
FAKE_CURRENCY = {".SR": "SAR", ".QA": "QAR", ".KW": "KWF", ".AE": "AED", ".L": "GBp"}
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
            # Dubai and Abu Dhabi moved to Mon-Fri on 1 Jan 2022; the other GCC markets didn't
            switch = pd.Timestamp("2022-01-01") if t.endswith((".AE", ".AD")) else None
            idx = pd.DatetimeIndex(
                [
                    d
                    for d in pd.date_range(start, last)
                    if d.day_of_week
                    in ((0, 1, 2, 3, 4) if switch is not None and d >= switch else (6, 0, 1, 2, 3))
                ]
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
    labels = [t.label for t in app.tabs]
    assert [x for x in labels if x[0].isdigit()] == [
        "1 Client",
        "2 Portfolio",
        "3 Analyse",
        "4 Optimise",
        "5 Suitability",
        "6 Report & trades",
    ]
    assert {x for x in labels if not x[0].isdigit()} == {
        "Holdings",
        "Screen",
        "Performance & backtest",
        "Benchmark",
        "Assets",
        "Monte Carlo",
        "Regime",
        "Optimise",
        "Frontier",
        "Strategy backtests",
        "Trades",
        "Reports",
        "AI analyst",
    }
    # static tables: data notes, my holdings, my comparison, benchmark comparison, diagnosis
    # by holding, diagnosis vs benchmark, assets, strategies, regime folds, analyst holdings,
    # domicile check, Optimise's active limits (the rebalance drift table needs a portfolio
    # value: none by default);
    # dataframes: the
    # editable holdings table, the screen (wide, scrollable), the optimiser's group and asset
    # limits editors, the rebalance targets and the frontier weights
    assert len(app.table) == 12 and len(app.dataframe) == 6
    assert any("P(stressed)" in m.label for m in app.metric)


def test_regime_toggle_adds_regime_strategies(app: AppTest) -> None:
    app.toggle(key="use_regime").set_value(True).run()
    assert not app.exception, app.exception
    names = list(app.selectbox(key="mc_strategy").options)
    assert "Equal weight + regime" in names


def test_too_low_cap_explains_instead_of_stopping(app: AppTest) -> None:
    app.slider(key="max_weight").set_value(0.20).run()  # 20% x 4 holdings < 100%
    assert not app.exception, app.exception
    assert any("can't hold 4 holdings" in i.value for i in app.info)
    assert app.dataframe  # the Portfolio tab still works, so the cap or holdings can be fixed


def test_no_sidebar_settings_in_a_popover_and_light_charts(app: AppTest) -> None:
    assert len(app.sidebar.children) == 0  # nothing in the sidebar at all
    assert app.radio(key="data_period").label == "Data period"  # now in ⚙ Settings
    assert app.selectbox(key="ws_benchmark").label == "Benchmark"  # top bar
    charts = app.get("plotly_chart")
    assert charts and all(c.proto.theme == "" for c in charts)  # theme=None: our template
    assert all(json.loads(c.proto.spec)["layout"]["paper_bgcolor"] == "#ffffff" for c in charts)
    before = _metric_tables(app)[1].loc["Max Sharpe", "CAGR"]
    app.number_input[0].set_value(100.0).run()  # costs (bps) in the settings popover
    assert not app.exception, app.exception
    assert _metric_tables(app)[1].loc["Max Sharpe", "CAGR"] != before


def test_holdings_drive_the_tabs(app: AppTest) -> None:
    assert app.slider(key="max_weight").label == "Max weight per asset (optimiser limit)"
    assets_table = _metric_tables(app)[0]
    assert set(assets_table.index) == {"SPY", "AGG", "GLD", "KSA"}  # the default holdings
    _set_holdings(app, [{"ticker": "SPY", "group": "Global equity", "value": 100.0}])
    assert not app.exception, app.exception
    assert any("Add at least two holdings" in i.value for i in app.info)


def _frontier_weights(app: AppTest) -> pd.DataFrame:
    tables = [  # weights table: portfolios as rows, tickers as columns (no metric columns)
        df.value
        for df in app.dataframe
        if "Min variance" in df.value.index and "Sharpe" not in df.value.columns
    ]
    assert len(tables) == 1
    return tables[0]


@pytest.mark.parametrize("cap", [0.30, 0.50, 0.75])
def test_frontier_tab_follows_the_cap(app: AppTest, cap: float) -> None:
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
        if {"Sharpe", "Return (ann.)"} <= set(t.value.columns)
        and "My portfolio" not in t.value.index
    ]
    assert len(tables) == 2, [t.columns for t in tables]
    return tables


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
    at.selectbox(key="profile_example").set_value("Dubai, moderate, 10 years").run()
    at.button(key="write_portfolio_memo").click().run()
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
    at.button(key="write_portfolio_memo").click().run()
    at.chat_input(key="follow_up").set_value("What if I were more conservative?").run()
    assert not at.exception, at.exception
    assert any("risk parity" in m.value for m in at.markdown)
    roles = [m["role"] for m in client.requests[-1]["messages"]]
    assert roles[-1] == "user" and roles.count("user") == 3  # review, tool results, follow-up


# --------------------------------------------------------------------- any asset, data period


def test_latest_data_is_the_default_with_as_of_line(app: AppTest) -> None:
    assert app.radio(key="data_period").value == "Latest data"
    assert any("Data as of 22 Oct 2026" in m.value for m in app.markdown)
    app.radio(key="data_period").set_value(app.radio(key="data_period").options[1]).run()
    assert not app.exception, app.exception
    assert any("Data as of 30 Sep 2026" in m.value and "README" in m.value for m in app.markdown)


def test_invalid_holding_gets_a_friendly_warning(app: AppTest) -> None:
    _set_holdings(
        app,
        [
            {"ticker": "SPY", "group": "Global equity", "value": 50.0},
            {"ticker": "AGG", "group": "Bonds", "value": 30.0},
            {"ticker": "NOPE.AE", "group": "GCC", "value": 20.0},
        ],
    )
    assert not app.exception, app.exception
    warning = " ".join(w.value for w in app.warning)
    assert "NOPE.AE" in warning and "no data on yahoo finance" in warning
    assert "finance.yahoo.com" in warning


def test_gcc_preset_converts_and_aligns_calendars(app: AppTest) -> None:
    app.selectbox(key="ws_preset").set_value("GCC stocks")
    app.button(key="ws_preset_go").click().run()
    assert not app.exception, app.exception
    info = " ".join(i.value for i in app.info)
    assert "2222.SR trades Sun–Thu" in info and "periods a year instead of 252" in info
    notes = next(t.value for t in app.table if "Converted to USD with" in t.value.columns)
    assert notes.loc["2222.SR", "Converted to USD with"] == "SAR peg 3.75 per USD"
    assert notes.loc["NBK.KW", "Converted to USD with"].startswith("KWD market rate")
    assert notes.loc["2222.SR", "Name"] == "Saudi Aramco"


def test_short_history_warning(app: AppTest) -> None:
    _set_holdings(
        app,
        [
            {"ticker": "SPY", "group": "Global equity", "value": 50.0},
            {"ticker": "NEWCO", "group": "Other", "value": 50.0},
        ],
    )
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
    assert any("2222.SR trades Sun–Thu" in i.value for i in app.info)  # joined the dates
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
    app.number_input(key="pf_value_holdings_0").set_value(20_000.0).run()
    app.radio(key="ws_mode").set_value("USD amount").run()
    assert not app.exception, app.exception
    values = {r["ticker"]: r["value"] for r in app.session_state["ws_rows"]}
    assert values == {"SPY": 10000.0, "AGG": 6000.0, "GLD": 2000.0, "KSA": 2000.0}


def test_import_list_replaces_holdings(app: AppTest) -> None:
    app.text_area(key="ws_import").set_value("SPY 70\nTLT 30")
    app.button(key="ws_import_go").click().run()
    assert not app.exception, app.exception
    rows = app.session_state["ws_rows"]
    assert [(r["ticker"], r["value"]) for r in rows] == [("SPY", 70.0), ("TLT", 30.0)]


def test_equal_weight_and_normalise_buttons(app: AppTest) -> None:
    app.button(key="ws_equal").click().run()
    assert not app.exception, app.exception
    assert [r["value"] for r in app.session_state["ws_rows"]] == [25.0] * 4
    _set_holdings(
        app,
        [
            {"ticker": "SPY", "group": "Global equity", "value": 50.0},
            {"ticker": "AGG", "group": "Bonds", "value": 30.0},
            {"ticker": "GLD", "group": "Gold", "value": 30.0},
        ],
    )
    app.button(key="ws_normalise").click().run()
    values = [r["value"] for r in app.session_state["ws_rows"]]
    assert values == pytest.approx([45.4545, 27.2727, 27.2727], abs=1e-4)
    assert not any("scaled to 100.00%" in c.value for c in app.caption)
    app.number_input(key="pf_value_holdings_0").set_value(20_000.0).run()
    app.radio(key="ws_mode").set_value("USD amount").run()
    assert app.button(key="ws_normalise").disabled


def test_preset_replaces_holdings_with_equal_weights(app: AppTest) -> None:
    app.selectbox(key="ws_preset").set_value("GCC stocks")
    app.button(key="ws_preset_go").click().run()
    assert not app.exception, app.exception
    rows = app.session_state["ws_rows"]
    assert [r["ticker"] for r in rows][:2] == ["2222.SR", "1120.SR"]
    assert all(r["value"] == pytest.approx(100 / 6, abs=1e-3) for r in rows)  # 4 decimals


def test_picking_a_search_result_adds_it(monkeypatch, tmp_path) -> None:
    import workstation  # noqa: PLC0415 - the app's module, importable once the path is set

    monkeypatch.setattr(yf, "download", synthetic_download)
    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    monkeypatch.setattr(gpl.data, "DEFAULT_CACHE_DIR", tmp_path)
    monkeypatch.setattr(agent, "has_credentials", lambda: False)
    monkeypatch.setattr(workstation, "st_searchbox", fake_searchbox("2222.SR"))
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    rows = {r["ticker"]: r for r in at.session_state["ws_rows"]}
    assert rows["2222.SR"]["group"] == "GCC" and rows["2222.SR"]["value"] == 10.0
    assert len(rows) == 5  # added once, not again on the rerun
    # removing the stock just added keeps it removed (the searchbox still holds the pick)
    _set_holdings(at, [r for r in at.session_state["ws_rows"] if r["ticker"] != "2222.SR"])
    at.run()
    assert "2222.SR" not in {r["ticker"] for r in at.session_state["ws_rows"]}


def fake_searchbox(pick: str):
    """Like streamlit_searchbox: one submit calls ``submit_function`` once, then the component
    keeps returning its last result on every later run."""

    def searchbox(*args, key="searchbox", submit_function=None, **kwargs):
        state = st.session_state.setdefault(key, {"result": None, "submitted": False})
        if not state["submitted"]:
            state["submitted"], state["result"] = True, pick
            submit_function(pick)
        return state["result"]

    return searchbox


def _screen(app: AppTest) -> pd.DataFrame:
    return next(d.value for d in app.dataframe if "Rank" in d.value.columns)


def test_screen_ranks_holdings(app: AppTest) -> None:
    screen = _screen(app)
    assert set(screen.index) == {"SPY", "AGG", "GLD", "KSA"}
    assert list(screen["Rank"]) == [1, 2, 3, 4] and screen.columns[2] == "Sharpe ▼"
    app.selectbox(key="ws_rank_by").set_value("Volatility (ann.)").run()
    vols = [float(v.rstrip("%")) for v in _screen(app)["Volatility (ann.) ▼"]]
    assert vols == sorted(vols)
    for by in ws.SCREEN_COLUMNS:
        app.selectbox(key="ws_rank_by").set_value(by).run()
        assert not app.exception, (by, app.exception)
    # Beta follows the chosen benchmark; against cash it is undefined and shown as '–'
    app.selectbox(key="ws_benchmark").set_value("Cash (13-week T-bills)").run()
    assert set(_screen(app)["Beta to benchmark"]) == {"–"}


def test_benchmark_tab_compares_with_the_chosen_benchmark(app: AppTest) -> None:
    table = next(t.value for t in app.table if "Information ratio" in t.value.index)
    assert table.loc["Beta", "Benchmark"] == "–" and table.loc["Beta", "Your portfolio"] != "–"
    assert set(table.columns) == {"Your portfolio", "Benchmark"}
    titles = [json.loads(c.proto.spec)["layout"]["title"]["text"] for c in app.get("plotly_chart")]
    assert any(t.startswith("Relative drawdown") for t in titles)
    app.selectbox(key="ws_benchmark").set_value("Cash (13-week T-bills)").run()
    assert not app.exception, app.exception
    table = next(t.value for t in app.table if "Information ratio" in t.value.index)
    assert table.loc["Beta", "Your portfolio"].startswith("n/a")
    app.selectbox(key="ws_benchmark").set_value("FTSE 100 (ISF.L)").run()  # pence -> USD
    assert not app.exception, app.exception
    assert any("Information ratio" in t.value.index for t in app.table)


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


# ---------------------------------------------------------------------------------- Optimise

GOALS_TEXT = "make it safer, at least 15% GCC, no single asset over 45%"


def _spec_call(**overrides) -> SimpleNamespace:
    draft = {
        "goal": "reduce_risk",
        "max_weight_pct": 45,
        "asset_limits": [],
        "group_limits": [{"group": "GCC", "min_pct": 15, "max_pct": None}],
        "max_volatility_pct": None,
        "min_return_pct": None,
        "unmapped_requests": [],
        "summary": "Reduce risk; at least 15% GCC; max 45% per asset.",
    }
    return tool_use("s1", "propose_spec", {**draft, **overrides})


@pytest.fixture
def optimise_app(monkeypatch, tmp_path):
    def make(script: list) -> tuple[AppTest, FakeClient]:
        client = FakeClient(script)
        monkeypatch.setattr(yf, "download", synthetic_download)
        monkeypatch.setattr(yf, "Ticker", FakeTicker)
        monkeypatch.setattr(gpl.data, "DEFAULT_CACHE_DIR", tmp_path)
        monkeypatch.setattr(agent, "has_credentials", lambda: True)
        monkeypatch.setattr(agent, "make_client", lambda: client)
        st.cache_data.clear()
        at = AppTest.from_file(APP, default_timeout=240)
        at.run()
        return at, client

    return make


def _candidates(at: AppTest) -> pd.DataFrame | None:
    return next((t.value for t in at.table if "Held up?" in t.value.columns), None)


def _latest_weights(at: AppTest) -> pd.DataFrame | None:
    return next(
        (t.value for t in at.table if "Current" in t.value.index and "KSA" in t.value.columns), None
    )


def test_diagnosis_is_shown_before_any_goal(optimise_app) -> None:
    at, client = optimise_app([])
    assert not at.exception, at.exception
    diag = next(
        t.value
        for t in at.table
        if "Share of risk" in t.value.columns and "Sharpe" in t.value.columns
    )
    assert list(diag.index) == ["SPY", "AGG", "GLD", "KSA"]
    assert any(c.startswith("Beta to") for c in diag.columns)
    assert any("Your portfolio vs" in m.value for m in at.markdown)
    assert _candidates(at) is None and client.requests == []  # nothing runs until asked


def test_manual_goal_runs_candidates_and_recommends_by_rule(optimise_app) -> None:
    at, client = optimise_app([])
    at.radio(key="ws_goal_0").set_value("reduce_risk").run()
    at.button(key="ws_run_opt").click().run()
    assert not at.exception, at.exception
    table = _candidates(at)
    names = {n.lstrip("★ ") for n in table.index}
    # identical methods are shown once, with a caption naming the one they match
    merged = {c.value.split(":")[0] for c in at.caption if "same weights and results as" in c.value}
    assert names | merged >= {
        "Current",
        "Benchmark",
        "Max Sharpe",
        "Min variance",
        "Risk parity",
        "Max diversification",
        "Target volatility",
    }
    starred = [n for n in table.index if n.startswith("★ ")]
    recommended = any(s.value.startswith("**Recommended method") for s in at.success)
    assert len(starred) == (1 if recommended else 0)
    if recommended:  # reduce risk: lower walk-forward volatility than current
        vol = lambda n: float(table.loc[n, "Volatility (ann.)"].rstrip("%"))  # noqa: E731
        assert vol(starred[0]) < vol("Current")
    assert any("Lowest walk-forward volatility" in c.value for c in at.caption)
    assert any("not a forecast or a promise" in i.value for i in at.info)
    weights = _latest_weights(at)
    assert all(
        float(v.rstrip("%")) <= 40.0 + 1e-6 for v in weights.drop("Current").to_numpy().ravel()
    )
    assert client.requests == []  # no AI involved on the manual path


def test_target_goal_needs_its_number(optimise_app) -> None:
    at, _ = optimise_app([])
    at.radio(key="ws_goal_0").set_value("target_return").run()
    assert any("Enter the target return" in w.value for w in at.warning)
    v = at.session_state["ws_draft_v"]  # a new goal redraws the form
    at.number_input(key=f"ws_target_ret_{v}").set_value(6.0).run()
    at.button(key="ws_run_opt").click().run()
    assert not at.exception, at.exception
    assert _candidates(at) is not None


def test_ai_fills_goal_and_limits_for_the_user_to_confirm(optimise_app) -> None:
    at, client = optimise_app([response([_spec_call(min_return_pct=7)], "tool_use")])
    at.text_area(key="ws_goals").set_value(GOALS_TEXT)
    at.button(key="ws_interpret").click().run()
    assert not at.exception, at.exception
    assert any("The AI understood" in i.value for i in at.info)
    assert any("not in your text" in w.value and "7%" in w.value for w in at.warning)
    assert at.slider(key="max_weight").value == pytest.approx(0.45)  # the AI's per-asset cap
    plan = next(m.value for m in at.markdown if m.value.startswith("**Goal:**"))
    assert "Reduce risk" in plan and _limits(at)["Group GCC"] == ("at least 15.00%", "Form")
    assert _candidates(at) is None  # nothing runs until the user presses Run
    at.button(key="ws_run_opt").click().run()
    weights = _latest_weights(at).drop("Current")
    assert (weights["KSA"].map(lambda v: float(v.rstrip("%"))) >= 15.0 - 1e-6).all()
    assert [r["tools"][0]["name"] for r in client.requests] == ["propose_spec"]


def test_impossible_limits_are_explained_without_weights(optimise_app) -> None:
    impossible = _spec_call(
        max_weight_pct=20, group_limits=[{"group": "GCC", "min_pct": 90, "max_pct": None}]
    )
    at, _ = optimise_app([response([impossible], "tool_use")])
    at.text_area(key="ws_goals").set_value("at least 90% GCC, no asset over 20%")
    at.button(key="ws_interpret").click().run()
    at.button(key="ws_run_opt").click().run()
    assert not at.exception, at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "can't meet your limits" in warnings and "Group 'GCC' needs at least 90.00%" in warnings
    assert any("Keep your current portfolio" in i.value for i in at.info)
    assert _latest_weights(at) is None  # no method ran, so no weights are shown


def test_memo_explains_the_stored_recommendation(optimise_app) -> None:
    at, client = optimise_app(
        [
            response([tool_use("t1", "get_optimisation_result", {})], "tool_use"),
            response([text("### Recommendation and why\nThe rule picked a method.")], "end_turn"),
        ]
    )
    at.button(key="ws_run_opt").click().run()
    at.button(key="ws_opt_memo").click().run()
    assert not at.exception, at.exception
    tools = [t["name"] for t in client.requests[0]["tools"]]
    assert {"get_optimisation_result", "get_portfolio_diagnosis"} <= set(tools)
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert '"recommendation"' in tool_result["content"]
    assert any("The rule picked a method" in m.value for m in at.markdown)


def test_interpret_with_empty_goals_explains(optimise_app) -> None:
    at, client = optimise_app([])
    at.button(key="ws_interpret").click().run()
    assert any("Describe your goals first" in w.value for w in at.warning)
    assert client.requests == []


def test_changing_the_goal_hides_stale_results(optimise_app) -> None:
    at, _ = optimise_app([])
    at.button(key="ws_run_opt").click().run()
    assert _candidates(at) is not None
    at.radio(key="ws_goal_0").set_value("reduce_risk").run()
    assert _candidates(at) is None
    assert any("run the candidate methods again" in c.value for c in at.caption)


def test_rank_screen_puts_missing_values_last() -> None:
    table = pd.DataFrame(
        {col: [0.1, 0.2, 0.3] for col, _ in ws.SCREEN_COLUMNS.values()}
        | {"weight": [0.2, 0.3, 0.5]},
        index=["A", "B", "C"],
    )
    table.loc["C", "ret_3y"] = float("nan")  # too little history
    shown = ws.rank_screen(table, "3Y return (ann.)")
    assert list(shown.index) == ["B", "A", "C"] and shown.loc["C", "3Y return (ann.) ▼"] == "–"
    shown = ws.rank_screen(table, "Volatility (ann.)")  # lower is better
    assert list(shown.index) == ["A", "B", "C"] and shown.loc["A", "Sharpe"] == "0.10"


def test_kpi_cards_with_deltas_vs_the_benchmark(app: AppTest) -> None:
    labels = [m.label for m in app.metric]
    assert labels[:5] == [
        "Portfolio value",
        "Return (CAGR)",
        "Volatility",
        "Sharpe",
        "Max drawdown",
    ]
    assert app.metric[0].value == "–"
    assert app.metric[1].delta.endswith(" pp") and app.metric[1].value.endswith("%")
    assert any("deltas against 60/40 (SPY/AGG)" in c.value for c in app.caption)
    app.number_input(key="pf_value_holdings_0").set_value(10_000.0).run()  # entered once
    assert app.metric[0].value == "$10.0k" and app.metric[0].help == "USD 10,000.00"
    app.selectbox(key="ws_benchmark").set_value("Cash (13-week T-bills)").run()
    assert app.metric[3].delta in (None, "")  # no Sharpe delta against cash


# ------------------------------------------------------------- Analyst: charts, what-if, stress


@pytest.fixture
def chart_app(monkeypatch, tmp_path):
    def make(script: list) -> tuple[AppTest, FakeClient]:
        client = FakeClient(script)
        monkeypatch.setattr(yf, "download", synthetic_download)
        monkeypatch.setattr(yf, "Ticker", FakeTicker)
        monkeypatch.setattr(gpl.data, "DEFAULT_CACHE_DIR", tmp_path)
        monkeypatch.setattr(agent, "has_credentials", lambda: True)
        monkeypatch.setattr(agent, "make_client", lambda: client)
        st.cache_data.clear()
        at = AppTest.from_file(APP, default_timeout=240)
        at.run()
        return at, client

    return make


def _titles(at: AppTest) -> list[str]:
    return [json.loads(c.proto.spec)["layout"]["title"]["text"] for c in at.get("plotly_chart")]


def test_analyst_page_shows_the_portfolio_first(app: AppTest) -> None:
    holdings = next(
        t.value
        for t in app.table
        if "Share of risk" in t.value.columns
        and "Volatility (ann.)" in t.value.columns
        and "Name" in t.value.columns
    )
    assert list(holdings.index) == ["SPY", "AGG", "GLD", "KSA"]
    assert "Allocation by holding" in _titles(app)


def test_memo_sections_are_paired_with_charts_built_by_code(chart_app) -> None:
    at, client = chart_app(
        [
            response(
                [
                    tool_use("c1", "show_chart", {"chart": "money_vs_risk", "section": "Risk"}),
                    tool_use("c2", "show_chart", {"chart": "stress_test", "section": "Stress"}),
                    tool_use(
                        "c3", "show_chart", {"chart": "correlation_heatmap", "section": "Nowhere"}
                    ),
                ],
                "tool_use",
            ),
            response([text("### Risk\nSPY dominates.\n\n### Stress\nCOVID hurt.")], "end_turn"),
        ]
    )
    at.button(key="write_portfolio_memo").click().run()
    assert not at.exception, at.exception
    titles = _titles(at)
    risk_i = titles.index("Share of money vs share of risk (current)")
    assert titles[risk_i + 1] == "Historical stress tests"  # in the memo's order
    assert titles[risk_i + 2] == "Correlation of daily returns"  # unmatched: at the end
    md = [m.value for m in at.markdown]
    assert any(m.startswith("### Risk") for m in md) and any(m.startswith("### Stress") for m in md)


def test_what_if_follow_up_updates_the_charts(chart_app) -> None:
    at, client = chart_app(
        [
            response([text("### Summary\nA balanced portfolio.")], "end_turn"),
            response(
                [
                    tool_use(
                        "w1",
                        "what_if_portfolio",
                        {"changes": [{"ticker": "SPY", "weight_pct": 30}]},
                    ),
                    tool_use(
                        "c1",
                        "show_chart",
                        {"chart": "weights_current_vs_proposed", "section": "What changes"},
                    ),
                ],
                "tool_use",
            ),
            response([text("### What changes\nLess SPY.")], "end_turn"),
        ]
    )
    at.button(key="write_portfolio_memo").click().run()
    at.chat_input(key="follow_up").set_value("What if I cut SPY to 30%?").run()
    assert not at.exception, at.exception
    specs = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
    weights_fig = next(
        f for f in specs if f["layout"]["title"]["text"] == "Current vs proposed weights"
    )
    what_if = next(t for t in weights_fig["data"] if t["name"] == "What-if")
    values = dict(zip(what_if["y"], _decode(what_if["x"]), strict=True))
    assert values["SPY"] == pytest.approx(0.30)
    assert values["AGG"] == pytest.approx(0.42)  # 30/50 of the remaining 70%


def _decode(values) -> list[float]:
    """Plotly sends numeric arrays as base64 typed arrays ({"dtype", "bdata"})."""
    import base64

    if isinstance(values, dict):
        return list(np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"]))
    return list(values)


def test_stress_test_button(chart_app) -> None:
    at, client = chart_app(
        [
            response([tool_use("s1", "stress_test", {})], "tool_use"),
            response([text("### Stress\nSee the episodes.")], "end_turn"),
        ]
    )
    at.button(key="stress_memo").click().run()
    assert not at.exception, at.exception
    assert client.requests[0]["messages"][0]["content"].startswith("Stress test my portfolio")
    result = client.requests[1]["messages"][-1]["content"][0]["content"]
    assert "COVID crash" in result and "2020-02-19 to 2020-03-23" in result


def test_memo_sections_pairing_and_disclaimer_last() -> None:
    memo = "Intro.\n### Risk\nSPY dominates.\n### Outlook\nCalm.\n\n---\n*Not investment advice.*"
    out = ws.memo_sections(memo, [("risk", "A"), ("**Outlook**", "B"), ("Elsewhere", "C")])
    assert [charts for _, charts in out] == [[], ["A"], ["B"], ["C"], []]
    assert out[0][0] == "Intro." and out[-1][0].strip().startswith("---")  # disclaimer last
    assert ws.memo_sections("No headings.", [("x", "A")]) == [("No headings.", []), ("", ["A"])]


# ---------------------------------------------------------------------------- Client and IPS


def test_client_tab_starts_empty_and_loads_an_example(app: AppTest) -> None:
    assert any("Load an example" in i.value for i in app.info)
    app.selectbox(key="client_example").set_value("Riyadh, conservative, 5 years, Sharia")
    app.button(key="client_load_example").click().run()
    assert not app.exception, app.exception
    scores = {m.label: m.value for m in app.metric}
    assert scores["Risk score (lower of the two)"] == "2 / 5"
    ips = next(m.value for m in app.markdown if m.value.startswith("**Return.**"))
    assert "**Risk.** Risk score 2 of 5" in ips and "volatility at most 9.00% a year" in ips
    assert "**Unique circumstances.** Sharia-compliant holdings only." in ips
    assert any(d.label == "Download IPS (PDF)" for d in app.get("download_button"))


def test_final_ips_blocked_until_complete_draft_always_available(app: AppTest) -> None:
    app.button(key="client_new").click().run()
    assert not app.exception, app.exception
    buttons = {d.label: d for d in app.get("download_button")}
    assert buttons["Download IPS (PDF)"].proto.disabled
    assert not buttons["Download DRAFT IPS (PDF)"].proto.disabled
    assert any("final IPS can't be issued yet" in w.value for w in app.warning)
    # the blank client lives in the UAE: regulator SCA by default, not "Other"
    assert app.selectbox(key="cl_reg_1").value == "SCA (UAE)"
    app.selectbox(key="cl_book_1").set_value("ADGM (Abu Dhabi)").run()
    assert app.selectbox(key="cl_reg_1").value == "FSRA (ADGM)"


def test_domicile_asks_for_the_value_when_missing(app: AppTest) -> None:
    box = app.number_input(key="pf_value_suitability_0")  # the Suitability step's domicile check
    assert box.label.startswith("Portfolio value (USD) for the estate-tax")
    box.set_value(2_000_000.0).run()
    assert not app.exception, app.exception
    assert app.session_state["pf_value"] == 2_000_000.0
    keys = [n.key for n in app.number_input]
    assert "pf_value_suitability_1" not in keys  # the value is known now: no box


def test_ips_limits_flow_into_optimise(app: AppTest) -> None:
    app.selectbox(key="client_example").set_value("Riyadh, conservative, 5 years, Sharia")
    app.button(key="client_load_example").click().run()
    plan = next(m.value for m in app.markdown if m.value.startswith("**Goal:**"))
    # band 2: volatility at most 9%; home bias 30% GCC; SPY and AGG not Sharia-compliant
    limits = _limits(app)
    assert limits["Volatility cap (a year)"] == ("9.00%", "Client IPS")
    assert limits["Group GCC"] == ("at least 30.00%", "Client IPS")
    assert limits["Holding SPY"] == ("at most 0.00%", "Client IPS") and "Client IPS applied" in plan
    assert any("GLD" in c.value and "unknown" in c.value for c in app.caption)
    app.toggle(key="ws_use_ips").set_value(False).run()
    assert "Volatility cap (a year)" not in _limits(app)


def test_profile_json_round_trip_in_the_app(app: AppTest) -> None:
    from gpl.client import EXAMPLE_CLIENTS, ClientProfile

    app.button(key="client_new").click().run()
    assert app.text_input(key="cl_ref_1").value == ""
    saved = next(d for d in app.get("download_button") if d.label == "Save profile (JSON)")
    assert saved.proto.label == "Save profile (JSON)"
    # loading a profile through the uploader callback path
    import client_ui

    profile = ClientProfile.from_dict(EXAMPLE_CLIENTS["Doha, aggressive, 20 years"])
    app.session_state["client"] = profile.to_dict()
    app.session_state["client_v"] = 5
    app.run()
    assert app.text_input(key="cl_ref_5").value == "EX-DOH"
    assert client_ui.CLIENT == "client"


def test_suitability_panels_need_a_client_then_show_reasons(app: AppTest) -> None:
    assert any("Load or create a client on step 1 (Client)" in i.value for i in app.info)
    app.selectbox(key="client_example").set_value("Riyadh, conservative, 5 years, Sharia")
    app.button(key="client_load_example").click().run()
    assert not app.exception, app.exception
    table = next(t.value for t in app.table if "Exclusions" in t.value.index)
    assert table.loc["Exclusions", "Current"].startswith("❌ Fail")  # SPY, AGG not compliant
    assert table.loc["Concentration", "Current"].startswith("❌ Fail")  # SPY 50% > 40%
    reasons = " ".join(m.value for m in app.markdown)
    assert "SPY (50.00%): not Sharia-compliant" in reasons
    assert "**Liquidity.**" in reasons and "USD 100,000" in reasons  # the client's need


def test_optimise_results_include_suitability(optimise_app) -> None:
    at, _ = optimise_app([])
    at.selectbox(key="client_example").set_value("Dubai, moderate, 10 years")
    at.button(key="client_load_example").click().run()
    at.button(key="ws_run_opt").click().run()
    assert not at.exception, at.exception
    tables = [
        t.value for t in at.table if "Volatility" in t.value.index and "Current" in t.value.columns
    ]
    assert tables and all(v.split(" ")[0] in ("✅", "❌", "⚠️") for v in tables[0]["Current"])


def test_optimise_never_recommends_a_portfolio_that_fails_the_ips(optimise_app) -> None:
    at, _ = optimise_app([])
    # Riyadh: Sharia-only excludes SPY and AGG, GCC at least 30%, band 2 (9% volatility)
    at.selectbox(key="client_example").set_value("Riyadh, conservative, 5 years, Sharia")
    at.button(key="client_load_example").click().run()
    at.button(key="ws_run_opt").click().run()
    assert not at.exception, at.exception
    table = _candidates(at)
    assert "Passes IPS?" in table.columns
    assert table.loc["Current", "Passes IPS?"].startswith("no: ")  # current fails too
    starred = [n for n in table.index if n.startswith("★ ")]
    for name in starred:  # anything starred passed every rule
        assert table.loc[name, "Passes IPS?"] == "yes"
    if not any(s.value.startswith("**Recommended method") for s in at.success):
        warning = next(w.value for w in at.warning if "No method meets the client's IPS" in w.value)
        assert "pp beyond" in warning or "pp below" in warning or "held (" in warning
        assert "current portfolio fails the IPS too" in warning
        assert not any(i.value.startswith("**Keep your current") for i in at.info)


def test_cash_need_is_a_hard_optimiser_limit(app: AppTest) -> None:
    # Dubai: USD 50,000 due Oct 2027 on USD 500,000 -> at least 10% in cash-like holdings
    app.selectbox(key="client_example").set_value("Dubai, moderate, 10 years")
    app.button(key="client_load_example").click().run()
    notes = " ".join(c.value for c in app.caption)
    assert "can't be met: no holding is cash, short-term bonds or sukuk" in notes  # AGG isn't
    app.button(key="ws_add_cash").click().run()
    assert not app.exception, app.exception
    rows = {r["ticker"]: r["group"] for r in app.session_state[ws.ROWS]}
    assert rows["BIL"] == "Cash & short-term"
    assert _limits(app)["Cash need (BIL)"] == ("at least 10.00%", "Client IPS")


def test_client_report_button_builds_a_pdf(app: AppTest) -> None:
    app.selectbox(key="client_example").set_value("Dubai, moderate, 10 years")
    app.button(key="client_load_example").click().run()
    app.button(key="prepare_report").click().run()
    assert not app.exception, app.exception
    pdf, name = app.session_state["client_report"]
    assert pdf.startswith(b"%PDF") and name.startswith("Client_report_EX-DXB_")
    assert any(d.label == "Download client report" for d in app.get("download_button"))


# ------------------------------------------------------------------------------- Rebalance


def test_rebalance_no_drift_by_default(app: AppTest) -> None:
    # no portfolio value yet (no $10,000 default): Rebalance asks for it
    assert any("Enter the portfolio value" in i.value for i in app.info)
    app.number_input(key="pf_value_holdings_0").set_value(250_000.0).run()
    assert not app.exception, app.exception
    drift = next(t.value for t in app.table if "Outside band" in t.value.columns)
    assert set(drift["Outside band"]) == {"no"}
    assert any("no rebalance needed" in s.value for s in app.success)


def test_rebalance_trade_list_against_proposed_target(app: AppTest) -> None:
    _set_holdings(
        app,
        [
            {"ticker": "SPY", "group": "Global equity", "value": 66_000.0},
            {"ticker": "AGG", "group": "Bonds", "value": 34_000.0},
        ],
        mode="USD amount",
    )
    app.session_state["ctx"].proposed = {
        "name": "Target 60/40",
        "weights": {"SPY": 0.6, "AGG": 0.4},
        "source": "test",
    }
    app.run()
    assert not app.exception, app.exception
    drift = next(t.value for t in app.table if "Outside band" in t.value.columns)
    assert drift.loc["SPY", "Drift"] == "+6.00 pp" and drift.loc["SPY", "Outside band"] == "⚠️ yes"
    trades = next(t.value for t in app.table if "Est. cost (USD)" in t.value.columns)
    assert trades.loc["SPY", "Action"] == "Sell" and trades.loc["AGG", "Action"] == "Buy"
    assert trades.loc["SPY", "Amount (USD)"] == "6,007.20"  # cash-neutral after 10 bps costs
    assert any(d.label == "Download trade list (CSV)" for d in app.get("download_button"))
    assert app.session_state["ctx"].rebalance["totals"]["costs_usd"] == "12.00"


def test_domicile_check_on_analyst_and_client_tabs(app: AppTest) -> None:
    table = next(t.value for t in app.table if "Irish-domiciled alternative" in t.value.columns)
    assert table.loc["SPY", "US-situs"] == "yes"
    assert table.loc["SPY", "Irish-domiciled alternative"].startswith("CSPX.L")
    assert any("Information, not tax advice" in i.value for i in app.info)
    assert any("Enter the portfolio value" in c.value for c in app.caption)
    app.number_input(key="pf_value_holdings_0").set_value(10_000.0).run()  # below the exemption
    assert any("estate tax if the client died today: USD 0 " in c.value for c in app.caption)


# ------------------------------------------------------------------- one portfolio value


def _metric(app: AppTest, label: str) -> str:
    return next(m.value for m in app.metric if m.label == label)


def test_one_portfolio_value_everywhere(app: AppTest) -> None:
    # no default value; switching to USD needs one
    assert _metric(app, "Portfolio value") == "–"
    app.radio(key="ws_mode").set_value("USD amount").run()
    assert app.radio(key="ws_mode").value == "Weight (%)"
    assert any("Enter the portfolio value first" in w.value for w in app.warning)
    # a client's value fills the single value; every copy of the input shows it
    app.selectbox(key="client_example").set_value("Riyadh, conservative, 5 years, Sharia")
    app.button(key="client_load_example").click().run()
    assert not app.exception, app.exception
    v = app.session_state[ws.PF_VERSION]
    for key in ("pf_value_holdings", "pf_value_client", "pf_value_rebalance"):
        assert app.number_input(key=f"{key}_{v}").value == 1_000_000.0
    assert _metric(app, "Portfolio value") == "$1.00m"
    reasons = " ".join(m.value for m in app.markdown)
    assert "the client needs USD 100,000 held now" in reasons  # suitability uses the value
    # switching to USD keeps the weights and turns them into amounts of that value
    app.radio(key="ws_mode").set_value("USD amount").run()
    amounts = {r["ticker"]: r["value"] for r in app.session_state[ws.ROWS]}
    assert amounts == {"SPY": 500_000.0, "AGG": 300_000.0, "GLD": 100_000.0, "KSA": 100_000.0}
    # editing the value on the Client tab changes it everywhere
    app.radio(key="ws_mode").set_value("Weight (%)").run()
    v = app.session_state[ws.PF_VERSION]
    app.number_input(key=f"pf_value_client_{v}").set_value(750_000.0).run()
    assert _metric(app, "Portfolio value") == "$750.0k"
    assert app.session_state["ctx"].portfolio_value == 750_000.0


def test_import_as_usd_amounts_keeps_the_amounts(app: AppTest) -> None:
    app.text_area(key="ws_import").set_value("SPY 120,000\nAGG 80000\nGLD 50,000")
    app.radio(key="ws_import_kind").set_value("Amounts (USD)")
    app.button(key="ws_import_go").click().run()
    assert not app.exception, app.exception
    assert app.radio(key="ws_mode").value == "USD amount"
    amounts = {r["ticker"]: r["value"] for r in app.session_state[ws.ROWS]}
    assert amounts == {"SPY": 120_000.0, "AGG": 80_000.0, "GLD": 50_000.0}
    assert _metric(app, "Portfolio value") == "$250.0k"


def test_no_hidden_default_portfolio_value() -> None:
    import inspect

    import workstation

    assert not hasattr(workstation, "DEFAULT_TOTAL_USD")
    assert "10_000" not in inspect.getsource(workstation)


# ------------------------------------------------------------------- holdings grid stability


@pytest.fixture
def grid_state(monkeypatch):
    """workstation's callbacks against a plain dict standing in for st.session_state."""
    import types

    state = {
        ws.ROWS: [
            {"ticker": "SPY", "group": "Global equity", "value": 6000.0},
            {"ticker": "AGG", "group": "Bonds", "value": 3000.0},
            {"ticker": "GLD", "group": "Gold", "value": 1000.0},
        ],
        ws.MODE: ws.USD,
        ws.VERSION: 0,
    }
    monkeypatch.setattr(ws, "st", types.SimpleNamespace(session_state=state))
    return state


def _edit(state, edited_rows: dict) -> None:
    key = f"ws_table_{state[ws.VERSION]}"
    state[key] = {"edited_rows": edited_rows, "added_rows": [], "deleted_rows": []}
    ws._apply_edits(key, [r["ticker"] for r in state[ws.ROWS]], ws.USD)


def test_blank_cell_is_stored_once_and_does_not_loop(grid_state) -> None:
    _edit(grid_state, {1: {ws.USD: None}})
    # stored as None; the editor keeps its key (a redraw would drop an edit typed meanwhile)
    assert grid_state[ws.ROWS][1]["value"] is None and grid_state[ws.VERSION] == 0
    _edit(grid_state, {1: {ws.USD: float("nan")}})  # NaN is stored as None too, not NaN
    assert grid_state[ws.ROWS][1]["value"] is None
    assert ws.portfolio_value() == 7000.0  # the blank holding is left out


def test_several_edits_are_all_kept(grid_state) -> None:
    _edit(grid_state, {0: {ws.USD: 5000.0}, 2: {ws.USD: 2500.0, "Group": "Other"}})
    rows = {r["ticker"]: r for r in grid_state[ws.ROWS]}
    assert rows["SPY"]["value"] == 5000.0 and rows["GLD"]["value"] == 2500.0
    assert rows["GLD"]["group"] == "Other" and rows["AGG"]["value"] == 3000.0


def test_remove_works_after_adding_a_stock(grid_state) -> None:
    grid_state[ws.CASH_LIKE] = []
    ws.add_holding("2222.SR", {"exchange": "Saudi", "type": "Stock"}, {})
    assert [r["ticker"] for r in grid_state[ws.ROWS]][-1] == "2222.SR"
    version = grid_state[ws.VERSION]
    _edit(grid_state, {3: {"Remove": True}})  # the new stock
    assert grid_state[ws.VERSION] == version + 1  # rows shifted: the editor is redrawn
    _edit(grid_state, {0: {"Remove": True}})  # then another one
    assert [r["ticker"] for r in grid_state[ws.ROWS]] == ["AGG", "GLD"]


def test_blank_cell_in_the_app_warns_without_a_rerun_loop(app: AppTest) -> None:
    rows = [
        {"ticker": "SPY", "group": "Global equity", "value": 6000.0},
        {"ticker": "AGG", "group": "Bonds", "value": None},
    ]
    _set_holdings(app, rows, mode="USD amount")
    assert not app.exception, app.exception
    version = app.session_state[ws.VERSION]
    app.run()
    assert app.session_state[ws.VERSION] == version  # nothing changes by itself
    assert any("Enter a positive amount for AGG" in w.value for w in app.warning)


def test_workflow_stepper_header_next_and_ask_the_analyst(app: AppTest) -> None:
    html = " ".join(m.value for m in app.markdown)
    assert 'class="gpl-stepper"' in html and "no client loaded" in html
    assert "<b>Client</b> none loaded" in html and "<b>IPS</b> –" in html
    app.button(key="next_0").click().run()
    assert app.session_state["main_tab"] == "2 Portfolio"
    app.button(key="ask_analyst").click().run()
    assert app.session_state["main_tab"] == "6 Report & trades"
    assert app.session_state["report_tab"] == "AI analyst"
    app.selectbox(key="client_example").set_value("Dubai, moderate, 10 years")
    app.button(key="client_load_example").click().run()
    html = " ".join(m.value for m in app.markdown)
    assert "<b>Client</b> EX-DXB" in html and "<b>Risk score</b> 3 / 5" in html
    assert "<b>IPS</b> Draft" in html  # the example's fact-find is incomplete
    # the suitability step isn't marked done while today's weights fail (e.g. the 50% SPY)
    assert re.search(r"Suitability</b><small>Current: \d+ rules? fail", html)
    assert any(e.label == "Advanced settings" for e in app.expander)


def test_selected_tab_survives_a_holdings_change(app: AppTest) -> None:
    assert app.session_state["main_tab"] == "1 Client"
    app.session_state["main_tab"] = "6 Report & trades"
    app.session_state["report_tab"] = "Trades"
    app.run()
    _set_holdings(app, [{"ticker": "SPY", "group": "Global equity", "value": 100.0}])
    assert not app.exception, app.exception
    assert app.session_state["main_tab"] == "6 Report & trades"
    assert app.session_state["report_tab"] == "Trades"


def test_ai_intake_fills_the_client_tab_and_constrains_optimise(optimise_app) -> None:
    from test_agent_profile import QA_DRAFT, QA_NOTES

    at, client = optimise_app([response([tool_use("p1", "propose_profile", QA_DRAFT)], "tool_use")])
    at.button(key="client_new").click().run()
    at.text_area(key="client_notes").set_value(QA_NOTES)
    at.button(key="client_interpret").click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    v = at.session_state["client_v"]
    assert at.text_input(key=f"cl_name_{v}").value == "Khalid Al Mansoori"
    assert at.number_input(key=f"cl_age_{v}").value == 58
    assert at.selectbox(key=f"cl_reg_{v}").value == "FSRA (ADGM)"
    assert at.text_area(key=f"cl_tax_{v}").value.startswith("UAE tax resident")
    ips = next(m.value for m in at.markdown if m.value.startswith("**Return.**"))
    assert "USD 600,000 by Oct 2028" in ips and "At least 10% in Saudi Arabia." in ips
    # the client's value (USD 3m) is the portfolio value; the IPS limits reach Optimise
    assert at.session_state["ctx"].portfolio_value == 3_000_000.0
    limits = _limits(at)
    assert limits["Region Saudi Arabia (KSA)"][0] == "at least 10.00%"
    assert limits["Cash need (no holding)"][0] == "at least 20.00%"


def test_report_uses_the_optimiser_recommendation_not_a_chat_what_if(
    optimise_app, monkeypatch
) -> None:
    import gpl.report

    captured = {}
    real = gpl.report.client_report

    def capture(inputs, compress=True):
        captured["inputs"] = inputs
        return real(inputs, compress)

    monkeypatch.setattr(gpl.report, "client_report", capture)  # the app imports it per run
    at, _ = optimise_app([])
    at.selectbox(key="client_example").set_value("Doha, aggressive, 20 years")
    at.button(key="client_load_example").click().run()
    at.button(key="ws_run_opt").click().run()
    assert not at.exception, at.exception
    rec = at.session_state["ws_opt"]["report"]["recommendation"]
    # a later what-if in the chat replaces the Analyst's proposed portfolio...
    at.session_state["ctx"].proposed = {
        "name": "What-if: SPY 90%",
        "weights": {"SPY": 0.9, "AGG": 0.1},
        "source": "what-if",
    }
    at.text_input(key="rep_adviser").set_value("A. Adviser")
    at.text_input(key="rep_licence").set_value("QFCRA-42")
    at.button(key="prepare_report").click().run()
    assert not at.exception, at.exception
    inputs = captured["inputs"]
    # ...but the report shows the optimiser's own result, checked against the IPS
    keep = rec["choice"] == "Keep your current portfolio"
    assert inputs.recommendation["choice"] == (None if keep else rec["choice"])
    assert "What-if: SPY 90%" not in inputs.suitability
    assert inputs.adviser == "A. Adviser" and inputs.adviser_licence == "QFCRA-42"
    assert inputs.value == 250_000.0 and "Value (USD)" in inputs.holdings.columns
    if not keep and rec["status"] == "recommended":
        assert inputs.trades is not None and inputs.trades["rows"][0][0] == "Holding"


def test_headline_figures_and_suitability_name_their_periods(app: AppTest) -> None:
    caption = next(c.value for c in app.caption if c.value.startswith("**Backtest period "))
    assert "–" in caption and "deltas against" in caption
    app.selectbox(key="client_example").set_value("Dubai, moderate, 10 years")
    app.button(key="client_load_example").click().run()
    periods = next(c.value for c in app.caption if c.value.startswith("**Periods.**"))
    assert "full history" in periods and "backtest period" in periods
    assert any(
        m.value.startswith("**Your portfolio vs") and "backtest period" in m.value
        for m in app.markdown
    )


def _limits(at: AppTest) -> dict[str, tuple[str, str]]:
    """The Optimise 'Active limits' table: limit -> (value, source)."""
    table = next((t.value for t in at.table if list(t.value.columns) == ["Value", "From"]), None)
    if table is None:
        return {}
    return {name: (row["Value"], row["From"]) for name, row in table.iterrows()}


def test_changing_the_goal_resets_the_limits(app: AppTest) -> None:
    # a draft (as the AI writes it): target volatility 8% and a 30% GCC cap
    app.session_state["ws_draft"] = {
        **{"asset_limits": [], "min_return_pct": None, "unmapped_requests": [], "summary": ""},
        "goal": "target_volatility",
        "max_weight_pct": None,
        "max_volatility_pct": 8,
        "group_limits": [{"group": "GCC", "min_pct": None, "max_pct": 30}],
    }
    app.session_state["ws_draft_v"] = 7
    app.run()
    goal_text = next(m.value for m in app.markdown if m.value.startswith("**Goal:**"))
    assert "8.00%" in goal_text and _limits(app)["Group GCC"] == ("at most 30.00%", "Form")
    app.radio(key="ws_goal_7").set_value("max_risk_adjusted").run()
    assert not app.exception, app.exception
    limits = _limits(app)
    # the old target doesn't become an 8% volatility cap, and the old group cap is gone
    assert "Volatility cap (a year)" not in limits and "Group GCC" not in limits
    v = app.session_state["ws_draft_v"]
    assert app.number_input(key=f"ws_maxvol_{v}").value is None


def test_limit_boxes_clear_and_clear_all(app: AppTest) -> None:
    v = app.session_state["ws_draft_v"]
    app.number_input(key=f"ws_maxvol_{v}").set_value(12.0).run()
    assert _limits(app)["Volatility cap (a year)"] == ("12.00%", "Form")
    app.number_input(key=f"ws_maxvol_{v}").set_value(None).run()  # emptied: no cap
    assert "Volatility cap (a year)" not in _limits(app)
    app.number_input(key=f"ws_maxvol_{v}").set_value(12.0).run()
    app.button(key="ws_clear_limits").click().run()
    assert not app.exception, app.exception
    assert _limits(app) == {}  # the per-holding cap too
    assert app.slider(key="max_weight").value == 1.0
    assert any(c.value.startswith("No limits:") for c in app.caption)


def test_tickers_without_prices_are_rejected_not_added(app: AppTest) -> None:
    app.text_area(key="ws_import").set_value("SPY 60\nNOPE1 40")
    app.button(key="ws_import_go").click().run()
    assert not app.exception, app.exception
    assert [r["ticker"] for r in app.session_state[ws.ROWS]] == ["SPY"]
    assert app.session_state[ws.ROWS][0]["value"] == pytest.approx(100.0)  # rescaled
    assert any("NOPE1: no price data on Yahoo Finance" in e.value for e in app.error)


def test_a_new_listing_that_shortens_the_sample_can_be_excluded(app: AppTest) -> None:
    app.text_area(key="ws_import").set_value("SPY 50\nAGG 40\nNEWCO 10")
    app.button(key="ws_import_go").click().run()
    assert not app.exception, app.exception
    warning = next(w.value for w in app.warning if "shortens the common sample" in w.value)
    assert warning.startswith("**NEWCO** starts on")
    app.button(key="ws_exclude_NEWCO").click().run()
    assert [r["ticker"] for r in app.session_state[ws.ROWS]] == ["SPY", "AGG"]
    assert not any("shortens the common sample" in w.value for w in app.warning)
