"""Gulf Portfolio Lab dashboard.

Presentation only: every number on screen comes from a ``gpl`` function. The app picks
inputs, calls the engine, and draws the results. Run with::

    streamlit run app/streamlit_app.py
"""

import json
import os
from datetime import date
from html import escape

import anthropic
import client_ui
import layout
import numpy as np
import pandas as pd
import streamlit as st
import workstation as ws
from inputs import asset_groups, combine_tickers
from layout import APP_CSS

from formatting import money, pct, plot, show_metrics_table, style_columns, style_percent
from gpl import backtest as bt
from gpl import regime as rg
from gpl.agent import analyst as agent
from gpl.agent.api import friendly_error
from gpl.agent.charts import build_chart, chart_data
from gpl.agent.spec import (
    active_limits,
    describe_goal,
    spec_from_draft,
)
from gpl.agent.tools import (
    AnalystContext,
    ToolError,
    UniverseData,
    _ips_checks,
    diagnose_portfolio,
    run_build,
    run_candidates,
)
from gpl.client import (
    CLOSEST_MIX_LABEL,
    ClientProfile,
    build_ips,
    cash_like_holdings,
    ips_constraints,
    month_text,
    tighter,
)
from gpl.construct import (
    POLICY_BLOCKS,
    POLICY_NAME,
    BuildSettings,
    _groups,
    policy_benchmark,
    screen_universe,
)
from gpl.data import (
    AssetData,
    BenchmarkData,
    DataUnavailable,
    clean_prices,
    get_prices,
    get_risk_free,
    has_price_data,
    latest_quote,
    load_assets,
    load_benchmark,
    load_investable,
    load_universe,
    sample_limiters,
    search_securities,
    security_attributes,
    to_returns,
)
from gpl.domicile import domicile_report, format_domicile
from gpl.factfind import (
    SECTION_BY_KEY,
    completeness,
    ips_missing,
    overall_completeness,
    product_type,
)
from gpl.metrics import (
    drawdown_series,
    kpi_table,
    period_label,
    relative_drawdown,
    summary_table,
)
from gpl.montecarlo import (
    goal_probability,
    path_max_drawdowns,
    simulate_paths,
    terminal_stats,
    to_real,
)
from gpl.optimise import (
    OptimisationSpec,
    efficient_frontier,
    max_sharpe,
    min_variance,
    portfolio_performance,
    risk_contributions,
    risk_parity,
)
from gpl.plots import (
    SERIES_COLORS,
    drawdown_chart,
    equity_chart,
    fan_chart,
    frontier_chart,
    paired_bars_chart,
    regime_chart,
    risk_contribution_chart,
    risk_return_chart,
    weights_chart,
)
from gpl.recommend import GOALS, KEEP
from gpl.report import ReportInputs, client_report, recommendation_section, trade_section
from gpl.scenarios import (
    cash_need_check,
    classify_holdings,
    trade_list,
    trades_covering_cash_need,
)
from gpl.suitability import (
    breach_text,
    check_suitability,
    failures,
    rationale,
)
from gpl.summary import GLOSSARY, kpi_comparison, main_reason, result_summary, window_note

CFG = load_universe()
COST_BY_SUFFIX = CFG.get("trading_costs", {}).get("by_suffix", {})  # market rates, bps

# On Streamlit Community Cloud the API key comes from st.secrets; locally from .env.
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ.setdefault("ANTHROPIC_API_KEY", st.secrets["ANTHROPIC_API_KEY"])
except Exception:  # no secrets file locally
    pass
D, R, MC = CFG["defaults"], CFG["regime"], CFG["montecarlo"]
MC_METHODS = {
    "block_bootstrap": f"Block bootstrap ({MC['block_size']}-day)",
    "bootstrap": "Daily bootstrap",
    "gbm": "GBM (normal)",
}
MC_METHOD_HELP = (
    f"**Block bootstrap** (default) replays runs of {MC['block_size']} consecutive real trading "
    "days. Market crashes cluster: big losses tend to follow big losses (think February-March "
    "2020). Keeping whole months together keeps those losing streaks, so the worst months and "
    "drawdowns *along the way* are deeper. It also keeps the sharp rebounds that often followed, "
    "so end-of-horizon outcomes can look *better* than with daily draws.\n\n"
    "**Daily bootstrap** draws days one at a time. It keeps the size of extreme days but "
    "scatters them, losing both the streaks and the rebounds.\n\n"
    "**GBM** assumes normally distributed log returns: no fat tails and no clustering."
)
PRESETS = CFG["presets"]
HOLDING_PRESETS = {
    "README universe (the 9 default ETFs)": PRESETS["Default ETFs"],
    **{name: tickers for name, tickers in PRESETS.items() if name != "Default ETFs"},
}
NAMES = CFG.get("ticker_names", {})
LATEST, README_MODE = (
    "Latest data",
    f"Reproduce README results (to {date.fromisoformat(D['end']):%d %b %Y})",
)
STRATEGY_NAMES = ["Equal weight", "Min variance", "Max Sharpe", "Risk parity"]
SIXTY_FORTY = "60/40 (SPY/AGG)"
BENCHMARK_NAMES = [SIXTY_FORTY, "SPY"]
GROUP_LABELS = {
    "gcc_equity": "GCC equity",
    "global_equity": "Global equity",
    "bonds": "Bonds",
    "real_assets": "Gold",
}
ANALYSE_TABS = ["Performance", "Benchmark", "Monte Carlo", "Assets · analyst", "Regime · analyst"]
OPTIMISE_TABS = ["Optimise", "Frontier", "Strategy backtests"]


# ----------------------------------------------------------------------------- data layer
# Cached wrappers around gpl functions. Arguments are plain values so Streamlit can hash them.


CACHE_TTL = 6 * 3600  # "latest data" refreshes at most every 6 hours


@st.cache_data(show_spinner="Loading prices…", ttl=CACHE_TTL)
def load_data(
    tickers: tuple[str, ...], start: date, end: date | None, lookback: int, day: date | None
) -> AssetData:
    """Any Yahoo tickers in USD, calendars aligned. ``day`` keys the cache for latest data."""
    return load_assets(list(tickers), start, end, lookback=lookback, max_ffill=D["max_ffill"])


@st.cache_data(show_spinner="Loading the risk-free rate…", ttl=CACHE_TTL)
def load_rf(start: date, end: date | None, day: date | None) -> pd.Series:
    return get_risk_free(start, end)


@st.cache_data(show_spinner="Loading the benchmark…", ttl=CACHE_TTL)
def get_benchmark(
    name: str,
    choice: dict,
    _index: pd.DatetimeIndex,
    loaded: tuple[str, ...],
    start: date,
    end: date | None,
    day: date | None,
) -> BenchmarkData:
    """The chosen benchmark's returns on the portfolio's dates (see ``gpl.data``).

    ``_index`` isn't hashed; the loaded tickers and dates (which determine it) key the cache.
    """
    return load_benchmark(name, choice, _index, load_rf(start, end, day), start, end)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def price_data_ok(ticker: str) -> tuple[bool, str]:
    """``gpl.data.has_price_data``; if Yahoo is unreachable the ticker is allowed (and the
    data notes report it later)."""
    try:
        return has_price_data(ticker)
    except DataUnavailable:
        return True, ""


def render_sample_limiters() -> None:
    """Name the holding whose late start shortens the common sample, and offer to exclude it."""
    held = [t for t in data.history.index if t in my_weights]
    for lim in sample_limiters(data.history.loc[held]):
        t = lim["ticker"]
        st.warning(
            f"**{t}** starts on {lim['first_date']:%d %b %Y} and shortens the common sample by "
            f"{lim['gain_years']:.1f} years (without it, the sample would start on "
            f"{lim['start_without']:%d %b %Y}). Short samples make every estimate noisier."
        )
        st.button(
            f"Exclude {t}",
            key=f"ws_exclude_{t}",
            on_click=ws.remove_holding,
            args=(t,),
            help="Removes it from the holdings; add it back any time from the search box.",
        )


@st.cache_data(show_spinner=False, ttl=900)
def get_quote(ticker: str):
    """Latest price and market time (refreshed every 15 minutes)."""
    return latest_quote(ticker)


@st.cache_data(show_spinner=False, ttl=3600)
def find_securities(term: str) -> list[dict]:
    return search_securities(term)


@st.cache_data(show_spinner="Fitting the regime model month by month…", ttl=CACHE_TTL)
def load_regime(end: date | None, day: date | None) -> dict:
    px = clean_prices(get_prices([R["equity"], R["bond"]], R["start"], end))
    vix = get_prices(["^VIX"], R["start"], end)["^VIX"]
    features = rg.make_features(px, vix, equity=R["equity"], bond=R["bond"])
    labels = rg.make_labels(
        px[R["equity"]].pct_change().iloc[1:], R["horizon"], R["quantile"], R["window"]
    )
    p_stressed = rg.walk_forward_probabilities(
        features,
        labels,
        model=R["model"],
        refit=R["refit"],
        min_train=R["min_train"],
        horizon=R["horizon"],
    )
    vol_rule = rg.naive_stress_signal(features, R["quantile"], R["window"])
    report = rg.evaluate_model(
        features,
        labels,
        model=R["model"],
        horizon=R["horizon"],
        threshold=R["threshold"],
        rule=vol_rule,
    )
    return {"features": features, "labels": labels, "p_stressed": p_stressed, "report": report}


def _strategy(name: str, rf: pd.Series, max_weight: float, periods: int) -> bt.Strategy:
    match name:
        case "Equal weight":
            return bt.equal_weight
        case "Min variance":
            return bt.min_variance_strategy(max_weight=max_weight)
        case "Max Sharpe":
            return bt.max_sharpe_strategy(rf=rf, max_weight=max_weight, periods_per_year=periods)
        case "Risk parity":
            return bt.risk_parity_strategy(max_weight=max_weight)
        case "60/40 (SPY/AGG)":
            return bt.fixed_weights(CFG["benchmarks"]["sixty_forty"])
        case "SPY":
            return bt.fixed_weights(CFG["benchmarks"]["spy"])
    raise ValueError(name)


@st.cache_data(show_spinner="Running backtests…", ttl=CACHE_TTL)
def run_backtests(
    universe: tuple[str, ...],
    loaded: tuple[str, ...],
    start: date,
    end: date | None,
    day: date | None,
    names: tuple[str, ...],
    max_weight: float,
    cost_bps: float,
    lookback: int,
    regime_threshold: float | None,
) -> dict[str, bt.BacktestResult]:
    """Strategies run on the chosen universe; benchmarks on SPY/AGG; both on the same dates."""
    data = load_data(loaded, start, end, lookback, day)
    rf = load_rf(start, end, day)
    periods = data.periods_per_year
    strategies = {name: _strategy(name, rf, max_weight, periods) for name in names}
    if regime_threshold is not None:
        p = load_regime(end, day)["p_stressed"]
        defensive = bt.min_variance_strategy(max_weight=max_weight)
        for name in [n for n in names if n in STRATEGY_NAMES and n != "Min variance"]:
            strategies[f"{name} + regime"] = bt.regime_switch_strategy(
                strategies[name], defensive, p, regime_threshold
            )
    run = dict(
        rebalance=D["rebalance"],
        lookback=lookback,
        cost_bps=cost_bps,
        cost_by_suffix=COST_BY_SUFFIX,
    )
    universe_prices = data.prices[list(universe)]
    return {
        name: bt.run_backtest(data.prices if name in BENCHMARK_NAMES else universe_prices, s, **run)
        for name, s in strategies.items()
    }


@st.cache_data(show_spinner="Backtesting your portfolio…", ttl=CACHE_TTL)
def run_my_portfolio(
    weights: tuple[tuple[str, float], ...],
    loaded: tuple[str, ...],
    start: date,
    end: date | None,
    day: date | None,
    cost_bps: float,
    lookback: int,
) -> bt.BacktestResult:
    """The user's holdings, rebalanced monthly to target, on the same dates as the strategies."""
    data = load_data(loaded, start, end, lookback, day)
    return bt.run_backtest(
        data.prices,
        bt.fixed_weights(dict(weights)),
        rebalance=D["rebalance"],
        lookback=lookback,
        cost_bps=cost_bps,
        cost_by_suffix=COST_BY_SUFFIX,
    )


# -------------------------------------------------------------------------- presentation


def first_eight(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape[1] > len(SERIES_COLORS):
        st.caption(f"Charts show the first {len(SERIES_COLORS)} series; the table has all of them.")
    return frame.iloc[:, : len(SERIES_COLORS)]


st.set_page_config(
    page_title="Gulf Portfolio Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(APP_CSS, unsafe_allow_html=True)
header = st.container(key="gpl_header")
head_title, head_bench, head_demo, head_settings = header.columns(
    [4.4, 3.2, 2.0, 0.55], vertical_alignment="center"
)
head_chips = header.container()
with head_demo:
    st.button(
        "▶ Try a demo client",
        key="fill_demo",
        on_click=client_ui.fill_demo,
        width="stretch",
        help="Fills every field with a random, realistic GCC client and a 5-10 holding "
        "portfolio of real tickers (no AI), then opens step 1 (Client). Marked DEMO DATA "
        "everywhere; the seed in the header reproduces it (⚙ Settings › Demo seed).",
    )
head_title.markdown(
    '<p class="gpl-brandmark"><i><b style="height:6px"></b><b style="height:11px"></b>'
    '<b style="height:16px"></b></i>Gulf Portfolio Lab</p>',
    unsafe_allow_html=True,
)


def header_info() -> layout.HeaderInfo:
    """Client, risk score, value and IPS status for the header chips, from session state."""
    profile = client_ui.current_profile()
    ips = None if profile is None else build_ips(profile, CFG)
    value = ws.portfolio_value()
    return layout.HeaderInfo(
        client=(profile.name or profile.reference or "unnamed") if profile else None,
        risk_score=ips.risk.score if ips else None,
        value=f"{money(value)}" if value else "not set",
        ips_status="–" if profile is None else "Draft" if ips_missing(profile) else "Ready",
        demo=bool(profile and profile.demo),
        seed=st.session_state.get("demo_seed") if profile and profile.demo else None,
    )


# Drawn first, from state the buttons' callbacks have already set, so a click updates the
# chips, the highlighted step and the page in the same render; refreshed at the end of the run.
chips_slot = head_chips.empty()
layout.render_header(chips_slot, header_info())
step_slots = layout.render_stepper(
    st.session_state.get("step_statuses"), st.session_state.get(layout.STEP_KEY)
)
with head_settings.popover("", icon=":material/settings:", width="stretch", help="Settings"):
    period = st.radio(
        "Data period",
        [LATEST, README_MODE],
        key="data_period",
        help="Latest data refreshes daily. The README results use a fixed end date so they can "
        "be reproduced exactly.",
    )
    start = st.date_input("Start", date.fromisoformat(D["start"]), min_value=date(2005, 1, 1))
    if period == README_MODE:
        st.caption(
            "To reproduce the README tables, load the **README universe** preset on Portfolio › "
            "Holdings (Import a list or start from a preset)."
        )
    cost_bps = st.number_input(
        "Costs for US listings (bps on turnover)",
        0.0,
        100.0,
        float(D["cost_bps"]),
        5.0,
        help="Local exchanges cost more to trade; their rates are set in config "
        "(trading_costs) and listed under Holdings.",
    )
    demo_box = st.expander("Demo seed")
    demo_box.number_input(
        "Seed (blank: random)",
        0,
        99_999,
        None,
        1,
        key="demo_seed_input",
        placeholder="random",
        help="The same seed always fills the same demo client and portfolio.",
    )
    demo_box.button(
        "Load the demo client with this seed", key="fill_demo_seed", on_click=client_ui.fill_demo
    )
    advanced = st.expander("Advanced settings")
    lookback = advanced.select_slider(
        "Estimation window (observations)", [126, 252, 504], value=D["lookback"]
    )
    names = advanced.multiselect(
        "Strategies to show",
        STRATEGY_NAMES + BENCHMARK_NAMES,
        default=STRATEGY_NAMES + BENCHMARK_NAMES,
    )
    use_regime = advanced.toggle(
        "Add regime-aware versions",
        key="use_regime",
        value=False,
        help="Switch to min variance when the model forecasts a stressed month.",
    )
    threshold = advanced.slider(
        "De-risk when P(stressed) ≥", 0.2, 0.9, float(R["threshold"]), 0.05, disabled=not use_regime
    )

SCROLL_TOP_JS = """<script>
const doc = window.parent.document;
window.parent.scrollTo(0, 0);
doc.querySelectorAll('[data-testid="stMain"], [data-testid="stAppViewContainer"], section.main')
  .forEach(e => e.scrollTo({top: 0}));
</script>"""
if st.session_state.pop("demo_toast", False):
    st.toast(client_ui.DEMO_TOAST, icon="✅")
    st.iframe(SCROLL_TOP_JS, height=1)  # our own constant script: back to the top of step 1
end = None if period == LATEST else date.fromisoformat(D["end"])
# Set on step 4 (Optimise); read here so every tab uses the current value.
max_weight = float(st.session_state.get("max_weight", D["max_weight"]))
day = date.today() if end is None else None
ws.init_state(CFG["groups"], CFG.get("cash_like", []))
my_weights, holding_problems, holding_note = ws.current_weights()
# The holdings are the universe for every tab; SPY and AGG are always loaded for 60/40.
loaded = tuple(combine_tickers([list(my_weights)], ["SPY", "AGG"]))
try:
    data = load_data(loaded, start, end, lookback, day)
    rf = load_rf(start, end, day)
except DataUnavailable, ValueError:
    st.error(
        "Yahoo Finance didn't respond just now. This is usually a temporary limit on how often "
        "prices can be downloaded; nothing is wrong with your selection. Please try again in a "
        "minute."
    )
    if st.button("Retry", type="primary", key="retry"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

bad = {t: why for t, why in data.invalid.items() if t in my_weights or t in ("SPY", "AGG")}
if bad:
    st.warning(
        "Couldn't load "
        + "; ".join(f"**{t}** ({why.lower()})" for t, why in bad.items())
        + ". Check the symbol on finance.yahoo.com. Some exchanges, such as Abu Dhabi (ADX), "
        "are not covered by Yahoo Finance."
    )
universe = [t for t in my_weights if t in data.prices.columns]
CAP_SETTING = "Max weight per asset (optimiser limit)"
if len(universe) < 2:
    universe_issue = (
        "Every tab analyses your holdings. Add at least two holdings with price history on "
        "step 2 (Portfolio › Holdings) to compare strategies."
    )
elif max_weight * len(universe) < 1:
    universe_issue = (
        f"A {pct(max_weight)} cap can't hold {len(universe)} holdings fully invested. Raise "
        f"'{CAP_SETTING}' on step 4 (Optimise), or add holdings."
    )
else:
    universe_issue = None

tickers = tuple(universe)
periods = data.periods_per_year
all_returns = to_returns(data.prices)
prices = data.prices[universe]
returns = all_returns[universe]
groups = asset_groups(universe, CFG["groups"], GROUP_LABELS)
missing_holdings = [t for t in my_weights if t not in data.prices.columns]
portfolio_ok = bool(my_weights) and not missing_holdings
my_key = tuple(sorted(my_weights.items()))
DATA_AS_OF = f"Data as of {data.price_as_of:%d %b %Y}"
DATA_HELP = (
    f"Latest price {data.price_as_of:%d %b %Y}, loaded {data.loaded_at} · "
    f"{'latest data' if end is None else 'fixed sample used in the README'}. Sample: "
    f"{data.prices.index[0]:%d %b %Y} – {data.prices.index[-1]:%d %b %Y} ({len(returns):,} daily "
    f"returns from {len(returns) + 1:,} price dates, about {periods} a year). Every result "
    "depends on this period."
)


def render_data_notes(history: pd.DataFrame) -> None:
    shown = history.loc[
        [t for t in history.index if t in universe or t in my_weights or t in ("SPY", "AGG")]
    ]
    table = pd.DataFrame(
        {
            "Name": [NAMES.get(t, "") for t in shown.index],
            "First date": shown["first_date"].dt.strftime("%d %b %Y"),
            "Trading days": shown["observations"].map("{:,}".format),
            "Currency": shown["currency"],
            "Converted to USD with": shown["conversion"],
            "Trades on": shown["trading_days"],
        },
        index=shown.index,
    )
    st.table(table)


short = data.history[data.history["too_short"] & data.history.index.isin(universe)]


def render_data_warnings() -> None:
    """Short histories: visible on Portfolio (they limit every result)."""
    for t, row in short.iterrows():
        st.warning(
            f"**{t}** has only {row['observations']:,} trading days of history (from "
            f"{row['first_date']:%d %b %Y}), less than the {lookback}-observation estimation "
            "window plus a month. Strategies that use it start late and its estimates are noisy."
        )


def render_data_details() -> None:
    """Data notes (Details on Portfolio): dates, currencies, conversion and calendars."""
    with st.expander("Data notes: start dates, currencies and trading calendars"):
        st.caption(DATA_HELP)
        render_data_notes(data.history)
        if data.mixed_calendars:
            sun_thu = {
                t: data.history.loc[t, "trading_days"]
                for t in data.history.index
                if "Sun" in data.history.loc[t, "trading_days"]
            }
            st.info(
                "**Different trading calendars.** "
                + "; ".join(f"{t} trades {days}" for t, days in sun_thu.items())
                + ", while US-listed assets trade Mon–Fri. (Dubai and Abu Dhabi moved from a "
                "Sunday to a Monday week on 1 Jan 2022, but Yahoo has no Friday prices for them, "
                "so they still differ from US markets.) The analysis uses only the "
                f"{data.alignment['kept']:,} price dates ({data.alignment['kept'] - 1:,} returns) "
                f"on which every selected market traded (about {periods} a year), so some returns "
                f"span more than one day; figures are annualised with {periods} periods a year "
                "instead of 252."
            )
        setters = list(data.history.index[data.history["sets_start"]])
        st.caption(
            f"The common sample starts on {data.prices.index[0]:%d %b %Y}, when the most recent "
            f"asset ({', '.join(setters)}) begins."
        )


enough_history = len(returns) > lookback + 21
SHORT_SAMPLE = (
    f"The common sample has only {len(returns):,} observations, not enough for a "
    f"{lookback}-observation estimation window plus a month of results. It is limited by "
    f"{', '.join(data.history.index[data.history['sets_start']])}: remove it, add an earlier start "
    "date, or choose a shorter estimation window in ⚙ Settings."
)

A = CFG["agent"]
settings_key = (tickers, loaded, start, end, day, max_weight, cost_bps, lookback, my_key)


def shared_context() -> AnalystContext:
    """One agent context per settings, shared by the AI analyst and the optimiser."""
    if st.session_state.get("ctx_key") != settings_key:
        regime_data = load_regime(end, day)
        ctx = AnalystContext(
            prices=data.prices,
            rf=rf,
            groups=CFG["groups"],
            benchmarks=CFG["benchmarks"],
            regime_features=regime_data["features"],
            regime_labels=regime_data["labels"],
            settings={
                "max_weight": max_weight,
                "cost_bps": cost_bps,
                "cost_by_suffix": COST_BY_SUFFIX,
                "lookback": lookback,
                "periods_per_year": periods,
                "rebalance": D["rebalance"],
                "var_level": D["var_level"],
                "regime": R,
                "montecarlo": MC,
                "optimise_lab": CFG.get("optimise_lab", {}),
                "stress_episodes": CFG.get("stress_episodes", []),
                "exposure_meta": CFG.get("exposure_meta", {}),
                "domicile": CFG["domicile"],
                "ucits_equivalents": CFG.get("ucits_equivalents", {}),
                "cash_like": CFG.get("cash_like", []),
            },
            universe=universe,
            user_portfolio=my_weights if portfolio_ok else None,
            asset_notes=data.history,
        )
        ctx._cache["p_stressed"] = regime_data["p_stressed"]  # already computed
        st.session_state.ctx = ctx
        st.session_state.ctx_key = settings_key
        st.session_state.pop("ws_opt", None)  # an optimisation belongs to the old holdings
    ctx = st.session_state.ctx
    ctx.benchmark = bench  # follows the top-bar choice without rebuilding the context
    ctx.holding_groups = {t: g for t, g in ws.groups_of_holdings().items() if t in my_weights}
    ctx.ips = client_ips()
    ctx.attributes = get_attributes(tuple(data.prices.columns))
    ctx.portfolio_value = portfolio_value()
    ctx.product_types = holding_products(tuple(my_weights))  # knowledge & experience check
    uni = (st.session_state.get("ws_opt") or {}).get("universe")
    ctx.curated = uni  # Build / swap proposals: prices and tags for new securities
    return ctx


def ask_analyst(analyst: agent.Analyst, question: str) -> dict:
    """One question to the analyst; returns a chat turn (the reply, or a readable error)."""
    try:
        with st.spinner("The analyst is calling tools and writing…"):
            return {"role": "assistant", "reply": analyst.ask(question)}
    except anthropic.APIError as e:
        return {"role": "error", "text": friendly_error(e, "the analyst's answer")}


def render_reply(reply: agent.AnalystReply, key: str = "reply") -> None:
    """Memo sections, each followed by the charts the analyst attached to it (drawn by code
    from engine data), then the unverified-number warning, cost line and audit trail."""
    for i, (markdown, charts) in enumerate(ws.memo_sections(reply.text, reply.charts)):
        if markdown.strip():
            st.markdown(markdown)
        for j, data in enumerate(charts):
            built = build_chart(data)
            if isinstance(built, pd.DataFrame):
                st.markdown(f"**{data.title}**")
                st.table(built)
            else:
                plot(built, key=f"{key}_{i}_{j}")
            if data.notes:
                st.caption(" ".join(data.notes))
    if reply.unverified:
        st.warning(
            "These figures did not come from any tool result, so treat them with suspicion: "
            + ", ".join(f"`{n}`" for n in reply.unverified)
        )
    cost = agent.estimate_cost(reply, A["price_per_mtok"])
    st.caption(
        f"{len(reply.tool_calls)} tool calls · {reply.input_tokens:,} input + "
        f"{reply.output_tokens:,} output tokens · about ${cost:.4f}"
    )
    with st.expander(f"Audit trail: {len(reply.tool_calls)} tool calls"):
        for call in reply.tool_calls:
            st.markdown(
                f"**`{call.name}`**({json.dumps(call.input)})"
                + (" ⚠️ error" if call.is_error else "")
            )
            st.json(call.output, expanded=False)


BLANK_DRAFT = {
    "goal": "max_risk_adjusted",
    "max_weight_pct": None,
    "asset_limits": [],
    "group_limits": [],
    "max_volatility_pct": None,
    "min_return_pct": None,
    "unmapped_requests": [],
    "summary": "",
}
CANDIDATE_TABLE_COLUMNS = {
    "cagr": "CAGR",
    "annual_volatility": "Volatility (ann.)",
    "sharpe": "Sharpe",
    "max_drawdown": "Max drawdown",
    "turnover_per_year": "Turnover / yr",
    "in_sample_sharpe": "In-sample Sharpe",
    "held_up_out_of_sample": "Held up?",
    "months_limits_unmet_previous_weights_kept": "Months limits unmet (weights kept)",
    "excess_return_vs_benchmark": "Excess vs benchmark (a year)",
    "information_ratio": "Information ratio",
    "tracking_error": "Tracking error",
    "hit_rate_months": "Hit rate (months)",
    "worst_relative_drawdown": "Worst relative drawdown",
}
IMPROVE, BUILD = "Improve my current portfolio", "Build a portfolio for me"
UNIVERSE_TAGS = load_investable()
LAB = CFG.get("optimise_lab", {})


def universe_data(include_holdings: bool = True) -> UniverseData:
    """The curated universe screened by the client's IPS, loaded in USD on common dates.

    Securities with a short history are left out (10 years; 6 for a Sharia-only client, whose
    funds start in 2019). The holdings and the benchmark's funds are loaded alongside.
    """
    ips = client_ips()
    years = (
        LAB.get("build_min_history_years_sharia", 6)
        if ips is not None and ips.limits["sharia_only"]
        else LAB.get("build_min_history_years", 10)
    )
    screen = screen_universe(
        UNIVERSE_TAGS, ips, date.today(), years, CFG.get("security_attributes", {})
    )
    bench_funds = list(bench_choices.get(bench_name, {}).get("weights", {}))
    wanted = [*screen.eligible, *(list(my_weights) if include_holdings else []), *bench_funds]
    loaded_u = load_data(tuple(dict.fromkeys(wanted)), start, end, lookback, day)
    prices = loaded_u.prices
    tagged = [t for t in prices.columns if t in UNIVERSE_TAGS]
    return UniverseData(
        prices=prices,
        tags=UNIVERSE_TAGS,
        eligible=[t for t in screen.eligible if t in prices.columns],
        excluded={
            **screen.excluded,
            **{t: f"no prices: {why}" for t, why in loaded_u.invalid.items()},
        },
        groups=_groups(tagged, UNIVERSE_TAGS),
        currencies=loaded_u.history["currency"].to_dict(),
        periods_per_year=loaded_u.periods_per_year,
    )


def _limits_table(rows: list[dict], key_name: str, names: list[str]) -> pd.DataFrame:
    by_name = {r[key_name]: r for r in rows}
    return pd.DataFrame(
        {
            "Min %": [by_name.get(n, {}).get("min_pct") for n in names],
            "Max %": [by_name.get(n, {}).get("max_pct") for n in names],
        },
        index=pd.Index(names, name=key_name.capitalize()),
        dtype="float",
    )


def _limits_rows(table: pd.DataFrame, key_name: str) -> list[dict]:
    rows = []
    for name, r in table.iterrows():
        lo = None if pd.isna(r["Min %"]) else float(r["Min %"])
        hi = None if pd.isna(r["Max %"]) else float(r["Max %"])
        if lo is not None or hi is not None:
            rows.append({key_name: name, "min_pct": lo, "max_pct": hi})
    return rows


def _goal_changed(v: int) -> None:
    """Callback: a new goal starts from blank limits, so an old target (e.g. 8% target
    volatility) never carries over as a cap. The client's IPS limits stay (separate toggle)."""
    goal = st.session_state.get(f"ws_goal_{v}")
    st.session_state.ws_draft = {**BLANK_DRAFT, "goal": goal}
    st.session_state.ws_draft_v = time_key()


def _clear_limits() -> None:
    """Callback: clear every limit on the form, including the per-holding cap."""
    goal = st.session_state.get("ws_draft", {}).get("goal") or "max_risk_adjusted"
    st.session_state.ws_draft = {**BLANK_DRAFT, "goal": goal}
    st.session_state.ws_draft_v = time_key()
    st.session_state["max_weight"] = 1.0


def _limit_box(
    container, label: str, lo: float, hi: float, step: float, key: str, value, placeholder: str
):
    """A number box that can always be emptied (``value=None``; the draft value is seeded
    into session state once, when the form is drawn for this version)."""
    if key not in st.session_state:
        st.session_state[key] = None if value is None else float(value)
    return container.number_input(
        label, lo, hi, value=None, step=step, key=key, placeholder=placeholder
    )


def _diagnose(ctx: AnalystContext) -> dict:
    """The diagnosis (cached per data and benchmark); also handed to the AI memo via ``ctx``."""
    key = (settings_key, bench_name)
    if st.session_state.get("ws_diag_key") != key:
        st.session_state.ws_diag = diagnose_portfolio(ctx, my_weights)
        st.session_state.ws_diag_key = key
    ctx.diagnosis = st.session_state.ws_diag
    return ctx.diagnosis


def _diagnosis_section(ctx: AnalystContext) -> None:
    diag = _diagnose(ctx)
    holdings = pd.DataFrame(diag["holdings"]).T.rename(
        columns={
            "weight": "Weight",
            "share_of_risk": "Share of risk",
            "annual_volatility": "Volatility (ann.)",
            "sharpe": "Sharpe",
            "beta_to_benchmark": f"Beta to {bench_name}",
        }
    )
    holdings.insert(0, "Name", [holding_names.get(t, "") for t in holdings.index])
    st.table(holdings)
    w = pd.Series(my_weights)
    rc = risk_contributions(w, all_returns[w.index].cov())
    plot(
        paired_bars_chart(w, rc, ("Share of money", "Share of risk"), "Money vs risk"),
        width="stretch",
    )
    if diag["correlation_clusters"]:
        st.markdown(
            "**Holdings that move together** ("
            + diag["cluster_rule"]
            + "):\n"
            + "\n".join(
                f"- {', '.join(c['holdings'])}: average correlation {c['average_correlation']}, "
                f"{c['share_of_money']} of the money and {c['share_of_risk']} of the risk"
                for c in diag["correlation_clusters"]
            )
        )
    else:
        st.caption(f"No correlation clusters: no group of {diag['cluster_rule']}.")
    if "vs_benchmark" in diag:
        absolute = pd.DataFrame(diag["vs_benchmark"]["absolute"]).T.rename(
            columns={
                "cagr": "CAGR",
                "annual_volatility": "Volatility (ann.)",
                "sharpe": "Sharpe",
                "max_drawdown": "Max drawdown",
            }
        )
        rel = diag["vs_benchmark"]["relative"]
        st.markdown(f"**Your portfolio vs {bench_name}** ({rel['period']}, after costs)")
        st.table(absolute)
        st.caption(
            f"Beta {rel['beta']} · alpha {rel['alpha_annual_jensen']} · tracking error "
            f"{rel['tracking_error']} · information ratio {rel['information_ratio']} · up "
            f"capture {rel['up_capture']} · down capture {rel['down_capture']} · correlation "
            f"{rel['correlation']}. Definitions are on the Benchmark tab."
        )


def _goal_section(holding_groups: dict[str, str], ai_ok: bool) -> tuple | None:
    """Goal, target and limits chosen by the adviser; returns (goal, limits, target)."""
    st.markdown('<div class="gpl-section">Goal and limits</div>', unsafe_allow_html=True)
    draft = st.session_state.setdefault("ws_draft", initial_draft())
    v = st.session_state.setdefault("ws_draft_v", 0)

    def as_float(value):  # the model writes integers (25); number inputs need floats
        return None if value is None else float(value)

    goal = st.selectbox(
        "Goal",
        list(GOAL_CHOICES),
        format_func=lambda g: f"{GOAL_CHOICES[g][0]}: {GOAL_CHOICES[g][1]}",
        index=list(GOAL_CHOICES).index(draft.get("goal") or "max_risk_adjusted"),
        key=f"ws_goal_{v}",
        on_change=_goal_changed,
        args=(v,),
        help="It starts on the goal that fits the client's risk score (1-2 lower risk, 3 best "
        "risk-adjusted return, 4-5 beat the benchmark). Changing it clears the limits below "
        "(not the client's IPS limits).",
    )
    edited = {**draft, "goal": goal, "max_weight_pct": max_weight * 100}
    if goal == "target_return":
        edited["min_return_pct"] = _limit_box(
            st,
            "Target return (% a year)",
            -50.0,
            100.0,
            0.5,
            f"ws_target_ret_{v}",
            as_float(draft.get("min_return_pct")),
            "e.g. 8",
        )
    elif goal == "target_volatility":
        edited["max_volatility_pct"] = _limit_box(
            st,
            "Target volatility (% a year)",
            0.5,
            100.0,
            0.5,
            f"ws_target_vol_{v}",
            as_float(draft.get("max_volatility_pct")),
            "e.g. 10",
        )
    with st.expander("More limits (optional)"):
        n1, n2 = st.columns(2)
        if goal != "target_volatility":
            edited["max_volatility_pct"] = _limit_box(
                n1,
                "Volatility cap (% a year)",
                0.5,
                100.0,
                0.5,
                f"ws_maxvol_{v}",
                as_float(draft.get("max_volatility_pct")),
                "no cap",
            )
        if goal != "target_return":
            edited["min_return_pct"] = _limit_box(
                n2,
                "Return floor (% a year, historical)",
                -50.0,
                100.0,
                0.5,
                f"ws_minret_{v}",
                as_float(draft.get("min_return_pct")),
                "no floor",
            )
        g1, g2 = st.columns(2)
        g1.caption("Group limits (blank = none)")
        group_table = g1.data_editor(
            _limits_table(
                draft.get("group_limits", []), "group", sorted(set(holding_groups.values()))
            ),
            key=f"ws_groups_{v}",
            width="stretch",
        )
        g2.caption("Asset limits (blank = none)")
        asset_table = g2.data_editor(
            _limits_table(draft.get("asset_limits", []), "ticker", list(my_weights)),
            key=f"ws_assets_{v}",
            width="stretch",
        )
        st.button(
            "Clear all limits",
            key="ws_clear_limits",
            on_click=_clear_limits,
            help="Empties every box and table here and removes the per-holding cap. The "
            "client's IPS limits are controlled by the toggle below.",
        )
    edited["group_limits"] = _limits_rows(group_table, "group")
    edited["asset_limits"] = _limits_rows(asset_table, "ticker")
    try:
        goal, limits, target = spec_from_draft(edited, holding_groups)
    except ValueError as e:
        st.warning(str(e))
        return None
    form_limits, ips_spec = limits, None
    ips = client_ips()
    if ips is not None and st.toggle(
        "Use the client's IPS limits",
        value=True,
        key="ws_use_ips",
        help="Adds the IPS limits (volatility, max weight, GCC minimum, exclusions) to the ones "
        "above; where both set a limit, the stricter one applies.",
    ):
        ips_spec, notes = ips_constraints(
            ips,
            list(my_weights),
            holding_groups,
            get_attributes(tuple(my_weights)),
            cash_like_now(),
            holding_regions(),
        )
        limits = tighter(limits, ips_spec)
        for note in notes:
            st.caption("• " + note)
        if "Cash need" in ips_spec.sets and not ips_spec.sets["Cash need"][0]:
            st.button(
                "Add BIL (1-3 month T-bills) as a cash holding",
                key="ws_add_cash",
                on_click=ws.add_holding,
                args=("BIL", {"type": "ETF", "name": "SPDR 1-3 Month T-Bill ETF"}, CFG["groups"]),
                help="Adds BIL to the holdings (Cash & short-term group) so the optimiser can "
                "hold the cash need. Its weight is then set by the optimiser.",
            )
    st.markdown(
        f"**Goal:** {describe_goal(goal, target)}  \n**Benchmark:** {bench_name}"
        + (
            "  \n**Client IPS applied**"
            if ips is not None and st.session_state.get("ws_use_ips")
            else ""
        )
    )
    rows = active_limits(form_limits, ips_spec, limits)
    st.markdown("**Active limits for every method**")
    if rows:
        st.table(
            pd.DataFrame(rows)
            .rename(columns={"limit": "Limit", "value": "Value", "source": "From"})
            .set_index("Limit")
        )
    st.caption(
        ("No other limits. " if rows else "No limits: ")
        + "Every method is long-only and fully invested."
    )
    return goal, limits, target


ATTRIBUTES = CFG.get("security_attributes", {})


@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def get_attributes(tickers: tuple[str, ...]) -> dict[str, dict]:
    """Sharia status and sector: config first, then Yahoo's sector for stocks (cached)."""
    return security_attributes(list(tickers), ATTRIBUTES)


def holding_regions() -> dict[str, str]:
    """Each holding's region (``gpl.scenarios.classify_holdings``), for region minimums."""
    classes = classify_holdings(
        list(my_weights),
        CFG.get("exposure_meta", {}),
        ws.groups_of_holdings(),
        data.history["currency"].to_dict(),
    )
    return {t: c["region"] for t, c in classes.items()}


def cash_like_now() -> list[str]:
    """Holdings that count as cash-like for the client's cash needs (``gpl.client`` rule)."""
    held = [t for t in my_weights if t in all_returns.columns]
    vols = (all_returns[held].std() * np.sqrt(periods)).to_dict()
    return cash_like_holdings(held, ws.groups_of_holdings(), CFG.get("cash_like", []), vols)


def cash_need_for_trades() -> dict | None:
    """The client's cash need to check the trade list against (None without one)."""
    ips = client_ips()
    if ips is None or ips.limits["liquidity_reserve_usd"] <= 0:
        return None
    due = "; ".join(
        f"{money(n['amount_usd'])} by {month_text(n['date'])}" for n in ips.limits["cash_needs_due"]
    )
    if ips.client.reserve_pct:
        due = (due + ", plus " if due else "") + f"a {ips.client.reserve_pct:g}% reserve"
    return {
        "need_usd": ips.limits["liquidity_reserve_usd"],
        "due": due,
        "cash_like": cash_like_now(),
    }


def portfolio_value() -> float | None:
    """The one portfolio value (USD) used by every tab (see ``ws.portfolio_value``)."""
    return ws.portfolio_value()


def domicile_section() -> dict | None:
    """US-situs check for the current holdings (formatted); None without holdings."""
    if not portfolio_ok:
        return None
    profile = client_ui.current_profile()
    return format_domicile(
        domicile_report(
            pd.Series(my_weights), portfolio_value(), CFG, bool(profile and profile.us_person)
        ),
        {t: getattr(get_quote(t), "quote_type", "") for t in my_weights},
    )


def render_domicile(where: str = "domicile") -> None:
    """Fund domicile (GCC clients): collapsed, with its one-line summary as the label."""
    result = domicile_section()
    if result is None:
        st.info("Add holdings on step 2 (Portfolio) to check their domicile.")
        return
    first = result["summary"][0] if result["summary"] else result["label"]
    with st.expander(f"**Fund domicile (GCC clients)** &nbsp; :gray[{first}]"):
        st.info(result["label"])
        if portfolio_value() is None:
            ws.value_input(
                st, f"pf_value_{where}", "Portfolio value (USD) for the estate-tax estimate"
            )
        if result["rows"]:
            st.table(pd.DataFrame(result["rows"]).set_index("Ticker"))
        for line in result["summary"]:
            st.caption("• " + line)
    st.caption("Information, not tax advice.")


SUITABILITY_ICON = {"pass": "✅", "fail": "❌", "review": "⚠️"}


def suitability_results(portfolios: dict[str, pd.Series]) -> dict | None:
    """``check_suitability`` for each portfolio against the client's IPS (None: no client)."""
    ips = client_ips()
    if ips is None:
        return None
    portfolios = {n: w[w > 1e-9] for n, w in portfolios.items()}
    tickers = tuple(sorted(set().union(*[set(w.index) for w in portfolios.values()])))
    uni = (st.session_state.get("ws_opt") or {}).get("universe")
    if uni is not None and not set(tickers) <= set(all_returns.columns):
        # Build / swaps: securities not held today are checked on the universe's prices
        ctx = shared_context()
        return {name: _ips_checks(ctx, w, uni) for name, w in portfolios.items()}
    currencies = data.history["currency"].to_dict()
    regions = {
        t: c["region"]
        for t, c in classify_holdings(
            list(tickers), CFG.get("exposure_meta", {}), ws.groups_of_holdings(), currencies
        ).items()
    }
    return {
        name: check_suitability(
            w,
            all_returns,
            ips,
            get_attributes(tickers),
            currencies,
            ws.groups_of_holdings(),
            periods,
            CFG.get("stress_episodes", []),
            portfolio_value(),
            0.03,
            CFG.get("cash_like", []),
            regions,
            headline_dates(),
            holding_products(tickers),
        )
        for name, w in portfolios.items()
    }


def holding_products(tickers) -> dict[str, str]:
    """Ticker -> product type ('ETFs', 'Equities', 'Funds') from the Yahoo quote type."""
    out = {}
    for t in tickers:
        kind = product_type(getattr(get_quote(t), "quote_type", ""))
        if kind:
            out[t] = kind
    return out


def _save_record(key: str) -> None:
    client = st.session_state.get(client_ui.CLIENT)
    if client is not None:
        client["suitability_record"] = st.session_state.get(key, "")


def _reset_record() -> None:
    client = st.session_state.get(client_ui.CLIENT)
    if client is not None:
        client["suitability_record"] = ""
    st.session_state["suit_record_v"] = st.session_state.get("suit_record_v", 0) + 1


def generated_record(results: dict) -> tuple[str, str]:
    """(portfolio name, rule-generated suitability record) for the last portfolio checked."""
    name = list(results)[-1]
    reasons: list[str] = []
    opt = st.session_state.get("ws_opt")
    if opt and opt["report"]["recommendation"].get("choice") == name:
        reasons = list(opt["report"]["recommendation"]["reasons"])
    ips = client_ips()
    lines = rationale(ips, name, results[name], reasons)
    note = client_ui.closest_mix_note(ips)
    if note:  # the example's own portfolio is the closest mix, not a passing one
        lines.insert(1, f"{CLOSEST_MIX_LABEL} {note}")
    return name, "\n".join(f"- {line}" for line in lines)


def render_suitability_record(results: dict) -> None:
    """The suitability rationale: generated from rules and checks, editable by the adviser."""
    profile = client_ui.current_profile()
    name, generated = generated_record(results)
    st.markdown(
        '<div class="gpl-section">Suitability rationale</div>',
        unsafe_allow_html=True,
        help=f"For {name}. Generated from the IPS rules and the checks above (no AI); your "
        "edits are saved with the client profile and go into the IPS and the client report.",
        width="content",
    )
    edited = bool(profile.suitability_record) and profile.suitability_record != generated
    key = (
        f"suit_record_{st.session_state.get(client_ui.VERSION, 0)}_"
        f"{st.session_state.get('suit_record_v', 0)}"
    )
    st.text_area(
        "Rationale (edit before it goes in the IPS and the client report)",
        profile.suitability_record or generated,
        height=180,
        key=key,
        on_change=_save_record,
        args=(key,),
        label_visibility="collapsed",
    )
    c1, c2 = st.columns([2, 1], vertical_alignment="center")
    c1.caption(
        "Edited by the adviser."
        if edited
        else "Goes into the IPS and the client report (generated from the rules, no AI)."
    )
    c2.button(
        "Reset to generated text",
        key="suit_record_reset",
        on_click=_reset_record,
        width="stretch",
    )


def suitability_html(results: dict) -> str:
    """The rules table: rule and a one-line reason, limit, each portfolio's value, status.

    The last portfolio (the recommendation, when there is one) is tinted and sets the status.
    """
    names = list(results)
    last = names[-1]
    by_rule = {name: {c.rule: c for c in checks} for name, checks in results.items()}
    mark = {"pass": "✓", "fail": "✗", "review": "⚠"}
    tone = {"pass": "pass", "fail": "fail", "review": "warn"}
    word = {"pass": "Pass", "fail": "Fail", "review": "Review"}

    def short(text: str) -> str:
        first = text.split(". ")[0].rstrip(".") + "."
        return first if len(first) <= 110 else first[:107].rstrip() + "…"

    head = "".join(
        f'<th class="num{" rec" if n == last and len(names) > 1 else ""}">{escape(n)}</th>'
        for n in names
    )
    rows = []
    for c in results[last]:
        cells = []
        for n in names:
            ck = by_rule[n].get(c.rule)
            value = escape(ck.measured) if ck else "–"
            if n == last and len(names) > 1:
                cells.append(f'<td class="num rec">{value}</td>')
            else:
                sym = mark[ck.status] if ck else ""
                colour = {"pass": "var(--gpl-pass)", "fail": "var(--gpl-fail)"}.get(
                    ck.status if ck else "", "var(--gpl-warn)"
                )
                cells.append(
                    f'<td class="num">{value} <span style="color:{colour}">{sym}</span></td>'
                )
        rows.append(
            f"<tr><td><b>{escape(c.rule)}</b><small>{escape(short(c.reason))}</small></td>"
            f'<td class="num" style="text-align:left">{escape(c.limit)}</td>{"".join(cells)}'
            f'<td class="num">{layout.pill(word[c.status], tone[c.status])}</td></tr>'
        )
    return (
        f'<table class="gpl-table"><thead><tr><th>Rule</th><th>Limit</th>{head}'
        f'<th class="num">Status</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def render_suitability(portfolios: dict[str, pd.Series], key: str, record: bool = False) -> None:
    """Pass / fail / review per IPS rule for each portfolio, with the written reasons."""
    results = suitability_results(portfolios)
    if results is None:
        st.info("Load or create a client on step 1 (Client) to check suitability.")
        if record:  # the domicile check needs holdings, not a client
            with st.columns([1.75, 1])[1].container(border=True, key="domicile_card"):
                render_domicile(key)
        return
    ips = client_ips()
    last = list(results)[-1]
    st.session_state["suit_fails"] = (last, [c.rule for c in results[last] if c.status == "fail"])
    counts = {k: sum(c.status == k for c in results[last]) for k in ("pass", "fail", "review")}
    suit_counts.markdown(
        '<div style="display:flex;gap:6px;justify-content:flex-end">'
        + layout.pill(f"{counts['pass']} pass", "pass")
        + layout.pill(f"{counts['fail']} fail", "fail" if counts["fail"] else "")
        + layout.pill(f"{counts['review']} review", "warn" if counts["review"] else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    note = client_ui.closest_mix_note(ips)
    if note and record:
        fails = [breach_text(c) for c in results["Current"] if c.status == "fail"]
        st.warning(f"**{CLOSEST_MIX_LABEL}** {note}" + "".join(f"\n- {line}" for line in fails))
    with st.container(border=True, key=f"{key}_rules"):
        st.markdown(suitability_html(results), unsafe_allow_html=True)
    full = period_label(all_returns.index)
    dates = headline_dates()
    with st.expander(
        "**Reasons for each rule** &nbsp; :gray[The full sentence behind every pass, fail and "
        "review, and the periods used]"
    ):
        for name, checks in results.items():
            st.markdown(f"**{name}**")
            for c in checks:
                st.markdown(f"{SUITABILITY_ICON[c.status]} **{c.rule}.** {c.reason}")
        st.caption(
            f"**Periods.** Policy risk checks (volatility, drawdown) use the full history, {full}: "
            "longer than the headline cards and including earlier crises. "
            + (
                f"The headline cards cover the backtest period, {period_label(dates)}; the "
                "reasons give both. "
                if dates is not None
                else ""
            )
            + f"Client {ips.client.reference or '(no reference)'}. Today's weights; ⚠️ Review "
            "means the data can't decide and the adviser must check. Decision support for a "
            "licensed adviser."
        )
    if record:
        rec_col, dom_col = st.columns([1.75, 1], gap="medium")
        with rec_col.container(border=True, key="rationale_card"):
            render_suitability_record(results)
        with dom_col.container(border=True, key="domicile_card"):
            render_domicile(key)


def client_ips():
    """The current client's IPS, or None if no client profile is loaded."""
    profile = client_ui.current_profile()
    return None if profile is None else build_ips(profile, CFG)


GOAL_CHOICES = {  # goal -> (label, one-line explanation), in the dropdown's order
    "reduce_risk": ("Lower risk", "fall less in bad markets; the return may drop"),
    "reduce_risk_keep_return": (
        "Lower risk, keep today's return",
        "less volatility, giving up at most a quarter point of return a year",
    ),
    "max_risk_adjusted": ("Best risk-adjusted return", "the most return for each unit of risk"),
    "beat_benchmark_less_risk": (
        "Beat the benchmark",
        "beat the benchmark's return with less risk than it takes",
    ),
    "target_return": ("Reach a target return", "the least risk that still reached your target"),
    "target_volatility": ("Hit a target volatility", "the most return at the risk you set"),
}


def default_goal(risk_score: int | None) -> str:
    """The goal to start from: 1-2 lower risk, 3 best risk-adjusted, 4-5 beat the benchmark."""
    if risk_score is None:
        return "max_risk_adjusted"
    return "reduce_risk" if risk_score <= 2 else (
        "max_risk_adjusted" if risk_score == 3 else "beat_benchmark_less_risk"
    )  # fmt: skip


def initial_draft() -> dict:
    """Blank Optimise form, starting on the goal that fits the client's risk score; with a
    client return target, on 'Reach a target return'."""
    draft = dict(BLANK_DRAFT)
    profile = client_ui.current_profile()
    ips = client_ips()
    draft["goal"] = default_goal(ips.risk.score if ips is not None else None)
    if profile is not None and profile.return_target_pct is not None:
        draft.update(goal="target_return", min_return_pct=profile.return_target_pct)
    return draft


IMPROVE_MOVE = "improve_move"
OPTIMISE_EXPLAINED = (
    "We test five ways to re-weight your holdings, plus a few swaps from our fund list, month "
    "by month on past data after trading costs. We recommend a change only if it beats your "
    "current portfolio on your goal and fits the IPS; otherwise we tell you to keep it. Takes "
    "about a minute."
)
MEMO_QUESTION = (  # the only request the app sends to the AI: fixed text, no user input
    "Write a short plain-English memo (under 250 words) for the client explaining the Optimise "
    "recommendation: what the rule chose and why it fits the goal, the main trade-offs, and how "
    "much to trust historical, walk-forward results. Use only figures from the tools."
)
FULL_MOVE = "Full restructure to target"


def full_move_chosen(result: dict) -> bool:
    """Did the adviser choose the full restructure for the recommended method (Improve)?"""
    run = result["run"]
    return (
        str(st.session_state.get(IMPROVE_MOVE, "")).startswith(FULL_MOVE)
        and run.full_targets is not None
        and run.recommendation.choice in run.full_targets.index
    )


def optimise_weights(result: dict) -> pd.DataFrame:
    """Latest weights per method; the recommended row is its full target if the adviser chose
    the full restructure, else today's move within the turnover limit (the default)."""
    run = result["run"]
    if not full_move_chosen(result):
        return run.latest_weights
    latest = run.latest_weights.copy()
    choice = run.recommendation.choice
    latest.loc[choice] = run.full_targets.loc[choice].reindex(latest.columns).fillna(0.0)
    return latest


def move_options(report: dict, choice: str) -> list[str]:
    """The two moves the adviser chooses between, each with its turnover (and cost)."""
    lim = report["turnover_limited_move"][choice]
    full = report["full_restructure_to_target"][choice]
    meets = (
        "meets every weight and group limit"
        if lim["meets_weight_and_group_limits"] == "yes"
        else "still outside a weight or group limit"
    )
    cost = (
        f"USD {full['estimated_cost_usd']}"
        if "estimated_cost_usd" in full
        else f"{full['estimated_cost_pct_of_portfolio']} of the portfolio"
    )
    return [
        f"Closest compliant move within {report['max_turnover']} turnover: {lim['turnover']} "
        f"traded, {meets}",
        f"{FULL_MOVE}: {full['turnover']} turnover, estimated cost {cost}; IPS: "
        f"{full['passes_ips']}",
    ]


def _apply_improve_move() -> None:
    """Callback: the adviser's choice of move becomes the proposal (Analyst, Trades, report)."""
    result = st.session_state.get("ws_opt")
    if not result:
        return
    run = result["run"]
    choice = run.recommendation.choice
    w = optimise_weights(result).loc[choice]
    full = full_move_chosen(result)
    fails = (
        bool(failures(run.full_checks.get(choice, [])))
        if full
        else run.recommendation.status == "closest_fails"
    )
    label = "full restructure" if full else "within the turnover limit"
    shared_context().proposed = {
        "name": f"{choice} ({label}{'; fails the IPS' if fails else ''})",
        "weights": w[w > 0].to_dict(),
        "source": f"Optimise recommendation ({FULL_MOVE.lower()}, chosen by the adviser)"
        if full
        else "Optimise recommendation"
        if not fails
        else "Optimise (closest; fails the IPS)",
    }


def run_with_progress(title: str, work):
    """Run ``work(progress)`` inside a status box with live steps and a progress bar.

    ``progress(share, text)`` moves the bar and shows the step (e.g. "2/4 Building …").
    """
    with st.status(f"{title} This takes about a minute.", expanded=True) as box:
        bar = st.progress(0.0, text="Starting…")

        def progress(share: float, text: str) -> None:
            bar.progress(min(max(share, 0.0), 1.0), text=text)

        out = work(progress)
        bar.progress(1.0, text="Done.")
        box.update(label="Done: the summary is below.", state="complete", expanded=False)
    st.session_state["scroll_to_result"] = True
    return out


def proposal_label(choice: str, status: str) -> str:
    """How the proposed portfolio is named elsewhere (trades, report, Analyst page)."""
    word = {"closest_fails": "closest, fails the IPS", "next_best": "next best option"}
    return f"{choice} ({word.get(status, 'recommended')})"


SCROLL_JS = """<script>
const doc = window.parent.document;
let tries = 0;
const go = () => {
  const el = doc.querySelector('.st-key-result_summary');
  if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
  else if (tries++ < 40) { setTimeout(go, 150); }
};
go();
</script>"""


def _period_words(backtest_period: str) -> str:
    """'2022-01-04 to 2026-11-05' -> '4 Jan 2022 – 5 Nov 2026'."""
    start, _, end = backtest_period.partition(" to ")
    return period_label(pd.DatetimeIndex([start, end]))


def _summary_card(result: dict, latest: pd.DataFrame) -> None:
    """The four-part summary (did / found / recommend / changes), then current vs recommended."""
    report = result["report"]
    rec = report["recommendation"]
    choice = rec["choice"]
    proposed = latest.loc[choice] if choice in latest.index and choice != KEEP else None
    summary = result_summary(
        report,
        latest.loc["Current"][latest.loc["Current"] > 1e-9].to_dict(),
        None if proposed is None else proposed[proposed > 1e-9].to_dict(),
    )
    after = (
        " After today's move, the quarterly turnover limit applies: expected trading is the "
        "'Turnover / yr' figure under Full results."
        if report.get("move_from_current_turnover") and proposed is not None
        else ""
    )
    did = " ".join(summary.did)
    parts = [
        ("What we did", did, ""),
        ("What we found", summary.found, ""),
        ("What we recommend", summary.recommend, "rec"),
        ("What changes", summary.changes + after, ""),
    ]
    with st.container(key="result_summary_details"):
        st.markdown(
            '<div class="gpl-summary">'
            + "".join(
                f'<div class="{cls}"><h5>{escape(h)}</h5>{escape(text)}</div>'
                for h, text, cls in parts
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    shown = choice if proposed is not None else "Current"
    head, when = st.columns([1, 1], vertical_alignment="bottom")
    name = (
        f'Current vs <span style="color:var(--gpl-accent-btn)">{escape(choice)}</span>'
        if proposed is not None
        else "Current portfolio (kept)"
    )
    head.markdown(f'<div class="gpl-section">{name}</div>', unsafe_allow_html=True)
    layout.meta(
        f"Walk-forward after costs, {_period_words(report['backtest_period'])}",
        help="Each method is re-optimised every period on the data available then and held to "
        "the next, after trading costs. "
        + window_note(
            "build" if report.get("mode") == "build" else "improve",
            _period_words(report["backtest_period"]),
        ),
        container=when,
    )
    cards = []
    for k in kpi_comparison(report, shown):
        tone = "" if k["better"] is None else ("pass" if k["better"] else "fail")
        arrow = "" if k["better"] is None else ("↑ " if k["change"].startswith("+") else "↓ ")
        change = k["change"].lstrip("+-−")
        if k["label"] == "Max drawdown" and k["better"]:
            change += " shallower"
        cards.append(
            layout.kpi_card(
                k["label"],
                k["proposed"],
                delta="" if proposed is None else arrow + change,
                tone=tone,
                before="" if proposed is None else k["current"],
            )
        )
    layout.kpi_row(cards)


def _candidates_card(report: dict) -> None:
    """Every candidate as a row: Sharpe · volatility · drawdown and its status pill."""
    status = report.get("candidate_status", {})
    rows = report["walk_forward_after_costs"]
    rec = report["recommendation"]["choice"]
    head, tip = st.columns([3, 1.4], vertical_alignment="center")
    head.markdown('<div class="gpl-section">Candidates</div>', unsafe_allow_html=True)
    with tip.popover("How to read this", width="stretch"):
        for term, text in GLOSSARY.items():
            st.markdown(f"**{term}:** {text}")

    def figures(r: dict) -> str:
        return " · ".join(
            str(r.get(k, "–")) for k in ("sharpe", "annual_volatility", "max_drawdown")
        )

    html, refs = [], []
    for name, r in rows.items():
        if name not in status:
            continue
        label = status[name]["status"]
        if label.startswith("—"):  # reference rows: current portfolio, benchmark
            refs.append(f"{escape(name.lower() if name == 'Current' else name)} {figures(r)}")
            continue
        tone = "pass" if label.startswith("✓") else "fail" if label.startswith("✗") else ""
        cls = "gpl-row rec" if name == rec else "gpl-row"
        why = status[name]["why"]
        html.append(
            f'<div class="{cls}"><div><b>{escape(name)}</b><small>{figures(r)}</small>'
            f'<small class="gpl-why">{escape(why)}</small></div>{layout.pill(label, tone)}</div>'
        )
    st.markdown(
        '<div class="gpl-rows">' + "".join(html) + "</div>"
        f'<p class="gpl-meta" style="text-align:left;margin-top:10px">Sharpe · volatility · max '
        f"drawdown. Reference: {'; '.join(refs)}.</p>",
        unsafe_allow_html=True,
    )


def _status_table(report: dict) -> None:
    """Every candidate with its status, reason and figures (under Full results)."""
    status = report.get("candidate_status", {})
    if not status:
        return
    rows = report["walk_forward_after_costs"]
    table = pd.DataFrame(
        {
            name: {
                "Status": status[name]["status"],
                "Why": status[name]["why"],
                "Sharpe": rows[name].get("sharpe", "–"),
                "Volatility": rows[name].get("annual_volatility", "–"),
                "Max drawdown": rows[name].get("max_drawdown", "–"),
                "Turnover / yr": rows[name].get("turnover_per_year", "–"),
            }
            for name in rows
            if name in status
        }
    ).T
    st.markdown(f"**The candidates ({_period_words(report['backtest_period'])})**")
    st.table(table)


GROWTH_CAPTION = (
    "Lower-risk methods usually end lower in a strong bull market like 2017-2026: they're built "
    "to fall less in crashes, not to grow faster. Judge them on the risk-vs-return chart."
)


def _risk_return_points(report: dict) -> pd.DataFrame:
    """Volatility and CAGR (fractions) for every row of the walk-forward table."""

    def frac(text) -> float:
        return float(str(text).rstrip("%")) / 100

    rows = report["walk_forward_after_costs"]
    return pd.DataFrame(
        {
            name: {"volatility": frac(r["annual_volatility"]), "cagr": frac(r["cagr"])}
            for name, r in rows.items()
            if r.get("annual_volatility") not in (None, "–") and r.get("cagr") not in (None, "–")
        }
    ).T


def _answer_card(report: dict) -> None:
    """The result in one sentence, with the rule's main reason; "keep" is a normal answer."""
    rec = report["recommendation"]
    with st.container(border=True, key="result_summary"):
        st.markdown(
            f'<div class="gpl-answer">{escape(ws.answer_sentence(report))}</div>',
            unsafe_allow_html=True,
        )
        reasons = rec["reasons"] or [""]
        reason = main_reason(reasons) if rec["choice"] == KEEP else reasons[0]
        if reason:
            st.caption(reason)


def _results_section(result: dict, ai_ok: bool) -> None:
    report, run = result["report"], result["run"]
    rec = report["recommendation"]
    latest = optimise_weights(result)
    gate = report.get("suitability_gate")
    if rec["status"] == "closest_fails":
        closest = "your current portfolio" if rec["choice"] == KEEP else rec["choice"]
        failed = gate["failed_rules"]["Current" if rec["choice"] == KEEP else rec["choice"]]
        st.warning(
            f"**No method meets the client's IPS. The closest is {closest}**, which still "
            "fails:\n\n"
            + "\n".join(f"- {line}" for line in failed)
            + "\n\n"
            + "\n".join(f"- {r}" for r in rec["reasons"])
            + "\n\nNot suitable as it stands: change the holdings (e.g. add a lower-risk or "
            "cash-like holding) or review the IPS with the client."
        )
    _answer_card(report)
    if st.session_state.pop("scroll_to_result", False):
        st.iframe(SCROLL_JS, height=1)  # our own constant script, not user content
    build = report.get("mode") == "build"
    choice = rec["choice"]
    shown = choice if choice in run.equity else None
    rr_col, g_col = st.columns(2, gap="medium")
    with rr_col.container(border=True, key="risk_return_card"):
        st.markdown(
            '<div class="gpl-section">Risk vs return</div>',
            unsafe_allow_html=True,
            width="content",
            help="Each dot is one portfolio, tested month by month after costs: further left is "
            "calmer, higher up grew faster. The recommendation is pink.",
        )
        plot(risk_return_chart(_risk_return_points(report), highlight=shown), key="opt_rr")
    with g_col.container(border=True, key="opt_growth_card"):
        st.markdown('<div class="gpl-section">Growth of $1</div>', unsafe_allow_html=True)
        order = [n for n in ("Current", shown) if n and n in run.equity]
        order += [n for n in run.equity.columns if n not in order]
        equity = run.equity[order]
        plot(
            equity_chart(
                equity, "", reference="Benchmark",
                visible=[n for n in ("Current", shown, "Benchmark") if n],
            ),
            key="opt_growth",
        )  # fmt: skip
        st.caption(GROWTH_CAPTION)
    d_col, c_col = st.columns(2, gap="medium")
    with d_col.container(border=True, key="opt_drawdown_card"):
        st.markdown('<div class="gpl-section">Drawdown from peak</div>', unsafe_allow_html=True)
        cols = [n for n in ("Current", shown, "Benchmark") if n and n in run.equity]
        daily = run.equity[cols].pct_change().dropna()
        plot(drawdown_chart(drawdown_series(daily), "", reference="Benchmark"), key="opt_dd")
    with c_col.container(border=True, key="candidates_card"):
        _candidates_card(report)
    if not build and choice in report.get("full_restructure_to_target", {}):
        st.radio(
            "Today's move",
            move_options(report, choice),
            key=IMPROVE_MOVE,
            on_change=_apply_improve_move,
            horizontal=True,
            help=f"For {choice}, the adviser chooses. Default: keep the turnover limit "
            f"({report['max_turnover']} of the portfolio). The full restructure goes all the way "
            "to the method's target today; its turnover and estimated cost (at each market's "
            "rate) are shown. Your choice feeds the suitability checks, the trade list and the "
            "client report.",
        )
    if choice in latest.index and choice != KEEP:
        with st.container(border=True, key="weights_card"):
            pair = latest.loc[["Current", choice]]
            pair = pair.loc[:, (pair > 1e-9).any()]  # only what either portfolio holds
            st.markdown('<div class="gpl-section">Weights</div>', unsafe_allow_html=True)
            plot(
                paired_bars_chart(pair.loc["Current"], pair.loc[choice], ("Current", choice), ""),
                width="stretch",
            )
    if (
        gate
        and gate["current_passes_ips"] is False
        and rec["status"] != "closest_fails"
        and "Current" in gate["failed_rules"]
    ):
        st.caption(
            "Your current portfolio fails the IPS: "
            + "; ".join(gate["failed_rules"]["Current"])
            + "."
        )
    # one line on suitability (the table lives on step 5) and the AI memo
    line = st.columns([2.2, 1.1, 2.4, 3], vertical_alignment="center")
    if rec["choice"] != KEEP and rec["choice"] in latest.index:
        checks = suitability_results({rec["choice"]: latest.loc[rec["choice"]]})
        if checks:
            failing = failures(checks[rec["choice"]])
            n = len(checks[rec["choice"]])
            line[0].markdown(
                f'<span style="color:var(--gpl-pass);font-weight:600">✓ Passes all {n} IPS '
                "rules</span>"
                if not failing
                else f'<span style="color:var(--gpl-fail);font-weight:600">✗ Fails '
                f"{len(failing)} of {n} IPS rules</span>",
                unsafe_allow_html=True,
            )
    line[1].button(
        "See Suitability",
        key="opt_see_suitability",
        type="tertiary",
        on_click=layout.go_to,
        args=(layout.STEPS[4],),
    )
    memo_clicked = False
    if ai_ok:
        memo_clicked = line[2].button(
            "Explain this recommendation (AI)",
            key="ws_opt_memo",
            type="tertiary",
            help="A short plain-English memo written by the AI from the tool results (every "
            "number comes from a tool). It is added to the client report.",
        )
    else:
        line[2].caption(
            "An AI explanation is available when an API key is set (`ANTHROPIC_API_KEY`); the "
            "recommendation itself never depends on it."
        )
    st.caption(f"The rule for this goal: {rec['rule']}")

    # details, collapsed
    rows = pd.DataFrame(report["walk_forward_after_costs"]).T
    columns = dict(CANDIDATE_TABLE_COLUMNS)
    if gate:
        columns["passes_ips"] = "Passes IPS?"
    rows = rows.reindex(columns=list(columns)).rename(columns=columns)
    rows = rows.astype(object).fillna("–")  # columns a row doesn't have (e.g. Benchmark)
    mark = "≈ " if rec["status"] == "closest_fails" else "★ "  # ≈: closest, still fails
    rows.index = [(mark if n == rec["choice"] else "") + n for n in rows.index]
    with st.expander(
        "**Summary: what we did, found, recommend and what changes** &nbsp; :gray[The full "
        "story of this run, current against recommended]"
    ):
        _summary_card(result, latest)
    full = st.expander(
        "**Full results for analysts** &nbsp; :gray[Every walk-forward measure and today's "
        "weights for each candidate]"
    )
    with full:
        _status_table(report)
        st.markdown(f"**Walk-forward results after costs ({report['backtest_period']})**")
        st.table(rows)
        st.caption(
            (
                "Each candidate re-chooses its securities and weights every quarter from the data "
                "available then (risk from all history so far, returns from the last year), "
                "within the IPS and your limits, then holds them to the next quarter, after "
                "trading costs. "
                if build
                else "Each method is re-optimised every month on the trailing window only "
                "(Ledoit-Wolf covariance), within your limits, then held to the next month, after "
                "trading costs. "
            )
            + "'Held up?': it kept at least half of its in-sample Sharpe ratio out of sample. "
            "'Months limits unmet': months when the limits couldn't be met with the data then "
            "available, so the previous weights were kept (no limit was broken). Hit rate: the "
            "share of months it beat the benchmark."
            + (
                " The target-volatility method aims at "
                f"{report['volatility_target_for_target_volatility_candidate']} a year."
                if "volatility_target_for_target_volatility_candidate" in report
                else ""
            )
        )
        if len(latest) > 1:
            st.markdown("**Latest weights of each method** (what it would hold now)")
            st.table(latest.map(pct))
        for name, why in report.get("infeasible_candidates", {}).items():
            st.warning(f"**{name}** can't meet your limits, so it wasn't run: " + " ".join(why))
        for name, why in {
            **report.get("identical_methods_shown_once", {}),
            **report.get("same_latest_weights_different_history", {}),
        }.items():
            st.caption(f"{name}: {why}.")
        if report.get("mode") == "improve":
            st.caption(
                f"Turnover limit {report['max_turnover']} of the portfolio: each month's move, "
                "and today's suggested change from your current weights, stay within it. If "
                "today's portfolio breaks a weight or group limit, the move heads for the "
                "least-turnover portfolio that meets them and stops at the limit."
            )
    chosen = report.get("securities_chosen")
    if chosen:
        with st.expander(
            f"**Securities chosen and why** &nbsp; :gray[{rec['choice']}: name, asset class, "
            "region and the reason for each]"
        ):
            st.table(
                pd.DataFrame(chosen).T.rename(
                    columns={
                        "weight": "Weight",
                        "name": "Name",
                        "asset_class": "Asset class",
                        "region": "Region",
                        "domicile": "Domicile",
                        "sharia": "Sharia",
                        "why": "Why",
                    }
                )
            )
    swaps = report.get("swaps", [])
    if swaps:
        with st.expander(f"**Swap ideas** &nbsp; :gray[{swaps[0]['why'].split('. ')[0]}.]"):
            for swap in swaps:
                st.markdown(f"🔁 **Swap ({swap['kind']})**: {swap['why']}")
    st.session_state["opt_diag_slot"] = True  # the diagnosis expander follows (render_optimise)
    st.caption(report["caveats"])

    if memo_clicked:
        memo_analyst = agent.Analyst(
            shared_context(),
            agent.make_client(),
            model=A["model"],
            max_tokens=A["max_tokens"],
            effort=A["effort"],
            max_tool_rounds=A["max_tool_rounds"],
        )
        with st.spinner("Writing the explanation from the tool results…"):
            result["memo"] = ask_analyst(memo_analyst, MEMO_QUESTION)
    memo = result.get("memo")
    if memo:
        if memo["role"] == "error":
            st.warning(memo["text"])
        else:
            with st.container(border=True, key="memo_card"):
                st.markdown(
                    '<div class="gpl-section">AI explanation</div>',
                    unsafe_allow_html=True,
                    help="Written by the AI from the app's own tool results (every figure cites "
                    "its tool; figures no tool returned are flagged). Added to the client report.",
                    width="content",
                )
                render_reply(memo["reply"])


def render_build(ctx, ai_ok: bool) -> None:
    """ "Build a portfolio for me": screen the universe, build candidates, recommend by rule."""
    if bench is None or bench.is_cash:
        st.warning("Choose a market benchmark (not cash) in the header: Build aims to beat it.")
        return
    result = st.session_state.get("ws_opt")
    has = bool(result) and result["key"][0] == "build"
    bar = st.container()
    box = st.expander("Edit & re-run" if has else "Run settings", expanded=not has)
    box.__enter__()
    _cap_slider()
    if POLICY and bench_name != POLICY_NAME:
        st.info(
            f"**Suggested benchmark for Build: {POLICY_NAME}.** {POLICY['represents']} The "
            f"current benchmark, {bench_name}, may itself break the client's limits; it stays "
            "selectable in the header."
        )
        st.button(
            "Use the policy benchmark",
            key="use_policy_bench",
            on_click=lambda: st.session_state.update(ws_benchmark=POLICY_NAME),
        )
    elif not POLICY and POLICY_WHY:
        st.caption(f"No IPS-fitting policy benchmark for this client: {POLICY_WHY}")
    c1, c2, c3 = st.columns(3)
    te = c1.number_input(
        "Tracking-error limit (% a year)",
        0.5,
        20.0,
        float(LAB.get("tracking_error_max", 0.04) * 100),
        0.5,
        key="build_te",
        help="How far the portfolio may stray from the benchmark: the annualised volatility of "
        "its return minus the benchmark's.",
    )
    turn = c2.number_input(
        "Turnover limit per quarter (%)",
        5.0,
        200.0,
        float(LAB.get("build_max_turnover", 0.25) * 100),
        5.0,
        key="build_turnover",
        help="Buys plus sells at each quarterly rebalance, as a share of the portfolio, after "
        "the first build. 25% a quarter is at most 1.0x the portfolio traded a year.",
    )
    c3.caption(
        f"At most {pct(max_weight)} in one holding (above); the client's IPS limits "
        "apply on top. Tilts: low volatility and momentum are scored from prices; quality "
        "uses quality-factor ETFs."
    )
    lo, hi = LAB.get("build_holdings", [8, 15])
    settings = BuildSettings(
        tracking_error_max=te / 100,
        n_min=int(lo),
        n_max=int(hi),
        min_weight=float(LAB.get("build_min_weight", 0.02)),
        tilt_min=float(LAB.get("tilt_min", 0.25)),
        max_turnover=turn / 100,
    )
    limits = OptimisationSpec(max_weight=max_weight)
    profile = client_ui.current_profile()
    client_key = None if profile is None else json.dumps(profile.to_dict(), sort_keys=True)
    run_key = (
        "build", settings_key, bench_name, te, turn, max_weight, client_key, portfolio_value()
    )  # fmt: skip
    if st.button("Build the candidate portfolios", key="ws_run_build", type="primary"):
        uni = None

        def work(progress):
            nonlocal uni
            progress(0.02, f"1/4 Screening {len(UNIVERSE_TAGS)} securities against the IPS")
            uni = universe_data(True)
            progress(
                0.10,
                f"1/4 Screened {len(UNIVERSE_TAGS)} securities against the IPS: "
                f"{len(uni.eligible)} eligible",
            )
            return run_build(
                ctx, uni, limits, my_weights, settings,
                bench_choices.get(bench_name, {}).get("weights", {}), progress=progress,
            )  # fmt: skip

        try:
            report, run = run_with_progress("Building the candidate portfolios.", work)
        except ToolError as e:
            st.error(str(e))
            box.__exit__(None, None, None)
            return
        ctx.optimisation = report
        choice = run.recommendation.choice
        if choice in run.latest_weights.index:
            w = run.latest_weights.loc[choice]
            fails = run.recommendation.status == "closest_fails"
            ctx.proposed = {
                "name": proposal_label(choice, run.recommendation.status),
                "weights": w[w > 0].to_dict(),
                "source": "Build (closest; fails the IPS)" if fails else "Build recommendation",
            }
        st.session_state.ws_opt = {
            "report": report, "run": run, "memo": None, "key": run_key, "universe": uni,
        }  # fmt: skip
        st.rerun()  # draw the header, settings and result once, from the new result
    box.__exit__(None, None, None)
    result = st.session_state.get("ws_opt")
    if result and result["key"] == run_key:
        _run_with(bar, [
            ("Benchmark", bench_name), ("Tracking error", f"≤ {te:g}%"),
            ("Turnover", f"≤ {turn:g}% a quarter"), ("Cap", f"{pct(max_weight)} per holding"),
        ])  # fmt: skip
        _results_section(result, ai_ok)
        with st.expander(
            f"Universe after the IPS screen: {result['report']['universe']['eligible']} eligible"
        ):
            out = result["report"]["universe"]["excluded"]
            if out:
                st.table(pd.DataFrame({"Left out because": out}))
    elif result and result["key"][0] == "build":
        st.caption("The limits, client or benchmark changed: build the candidates again.")


def _cap_slider() -> None:
    # Default set through session state only: the AI's interpretation may also set it.
    st.session_state.setdefault("max_weight", float(D["max_weight"]))
    st.slider(
        CAP_SETTING,
        0.10,
        1.00,
        step=0.01,
        key="max_weight",
        help="The most any one holding may weigh in an optimised portfolio; 1.00 means no cap "
        "('Clear all limits' sets it). It also applies to the strategies on the Frontier, "
        "Backtest and Monte Carlo tabs.",
    )


def _run_with(bar, chips: list[tuple[str, str]]) -> None:
    """The settings of the run shown above the result (edit them in "Edit & re-run")."""
    with bar.container(border=True, key="run_with"):
        st.markdown(
            '<div class="gpl-runwith"><span style="color:var(--gpl-muted)">Run with</span>'
            + "".join(
                f'<span class="gpl-pill">{escape(k)}: <b style="color:var(--gpl-text)">'
                f"{escape(v)}</b></span>"
                for k, v in chips
            )
            + "</div>",
            unsafe_allow_html=True,
        )


def _diagnosis_details(ctx: AnalystContext) -> None:
    with st.expander(
        "**Diagnosis: where the risk comes from** &nbsp; :gray[Holdings that move together and "
        f"the current portfolio against {bench_name}]"
    ):
        _diagnosis_section(ctx)


def _optimise_guidance() -> str:
    """The one guidance line, from the latest result (or what to do before the first run)."""
    mode = st.session_state.get("opt_mode", IMPROVE)
    result = st.session_state.get("ws_opt")
    if not result or (result["key"][0] == "build") != (mode == BUILD):
        if mode == BUILD:
            return f"Build portfolios from a curated universe that aim to beat {bench_name}."
        return "Choose a goal, then press Optimise portfolio."
    rec = result["report"]["recommendation"]
    first = ((client_ui.current_profile() or ClientProfile()).name or "").split(" ")[0]
    whose = f"{first}'s" if first else "your"
    if rec["status"] == "closest_fails":
        return "No portfolio meets every IPS rule: the closest is shown with its breaches."
    if rec["choice"] == KEEP:
        return f"Keep {whose} current portfolio: no method did enough better under the rule."
    if rec["status"] == "next_best":
        return f"No build beat {bench_name} reliably; {rec['choice']} is the next best option."
    if mode == BUILD:
        return f"{rec['choice']} beat {bench_name} reliably after costs and fits the IPS."
    fits = " and fits the IPS" if client_ips() is not None else ""
    return f"{rec['choice']} beats {whose} current portfolio{fits}."


def _show_frontier() -> None:
    st.session_state["optimise_tab"] = "Frontier"


def _optimise_header() -> None:
    with optimise_head:
        right = layout.page_header(
            "Optimise",
            _optimise_guidance(),
            help="Improve: start from your holdings, re-weight within a turnover limit, then try "
            "a few explained swaps. Build: choose 8-15 securities from a curated universe that "
            f"aim to beat the benchmark on a risk-adjusted basis (from {len(UNIVERSE_TAGS)} "
            "liquid securities screened by the client's IPS). Every result is judged walk-"
            "forward after costs.",
            right_width=0.58,
        )
        m, link = right.columns([1.55, 1], vertical_alignment="center")
        m.radio(
            "What should Optimise do?",
            [IMPROVE, BUILD],
            key="opt_mode",
            horizontal=True,
            label_visibility="collapsed",
            format_func={IMPROVE: "Improve my portfolio", BUILD: "Build one for me"}.get,
        )
        link.button(
            "Frontier & strategy backtests", key="opt_frontier_link", type="tertiary",
            on_click=_show_frontier,
        )  # fmt: skip


def render_optimise() -> None:
    """Goal and limits -> candidate methods walk-forward -> recommendation -> memo."""
    _optimise_header()
    mode = st.session_state.get("opt_mode", IMPROVE)
    if not portfolio_ok or len(my_weights) < 2:
        _cap_slider()
        st.info("Add at least two holdings with price history on the Holdings tab.")
        return
    if not enough_history:
        _cap_slider()
        st.warning(SHORT_SAMPLE)
        return
    ctx = shared_context()
    ctx.benchmark = bench
    holding_groups = {t: g for t, g in ws.groups_of_holdings().items() if t in my_weights}
    ai_ok = agent.has_credentials()
    if mode == BUILD:
        render_build(ctx, ai_ok)
        return
    _diagnose(ctx)
    result = st.session_state.get("ws_opt")
    has = bool(result) and result["key"][0] != "build"
    bar = st.container()
    box = st.expander("Edit & re-run" if has else "Run settings", expanded=not has)
    with box:
        _cap_slider()
        chosen = _goal_section(holding_groups, ai_ok)
    if chosen is None:
        _diagnosis_details(ctx)
        return
    goal, limits, target = chosen
    if goal == "beat_benchmark_less_risk" and bench is None:
        st.warning("Choose a benchmark with data in the header.")
        return
    box.__enter__()
    adv = st.expander("Advanced settings")
    c1, c2, c3 = adv.columns(3)
    max_turnover = c1.number_input(
        "Max turnover (% of the portfolio)",
        5.0,
        200.0,
        float(LAB.get("max_turnover", 0.30) * 100),
        5.0,
        key="opt_turnover",
        help="Buys plus sells as a share of the portfolio: 30% = e.g. selling 15% of one "
        "holding and buying 15% of another. Applied to each month's move and to today's change.",
    )
    use_swaps = c2.toggle(
        "Also try swaps from our fund list",
        value=True,
        key="opt_swaps",
        help="Sell a near-duplicate holding, or add a missing bond / sukuk diversifier: rules "
        "applied each month to the data available then.",
    )
    max_swaps = c3.number_input(
        "Max swaps", 1, 5, int(LAB.get("max_swaps", 3)), 1, key="opt_max_swaps"
    )
    st.markdown(f'<p class="gpl-explain">{escape(OPTIMISE_EXPLAINED)}</p>', unsafe_allow_html=True)
    profile = client_ui.current_profile()
    client_key = None if profile is None else json.dumps(profile.to_dict(), sort_keys=True)
    run_key = (
        settings_key, bench_name, goal, target, repr(limits), client_key, portfolio_value(),
        max_turnover, use_swaps, max_swaps,
    )  # fmt: skip
    if st.container(key="cta_optimise").button(
        "Optimise portfolio", key="ws_run_opt", type="primary"
    ):
        st.session_state.pop(IMPROVE_MOVE, None)  # a new run starts on the default move
        uni = None

        def work(progress):
            nonlocal uni
            progress(0.02, "1/4 Loading prices for your holdings and the swap universe")
            uni = universe_data(True) if use_swaps else None
            return run_candidates(
                ctx,
                goal,
                limits,
                my_weights,
                target,
                max_turnover=max_turnover / 100,
                universe=uni,
                max_swaps=int(max_swaps),
                progress=progress,
            )

        report, run = run_with_progress("Testing the candidate methods.", work)
        ctx.optimisation = report
        choice = run.recommendation.choice
        if choice != KEEP:  # charts on the Analyst page compare current with this
            w = run.latest_weights.loc[choice]
            fails = run.recommendation.status == "closest_fails"
            ctx.proposed = {
                "name": proposal_label(choice, run.recommendation.status),
                "weights": w[w > 0].to_dict(),
                "source": "Optimise (closest; fails the IPS)"
                if fails
                else "Optimise recommendation",
            }
        st.session_state.ws_opt = {
            "report": report, "run": run, "memo": None, "key": run_key, "universe": uni,
        }  # fmt: skip
        st.rerun()  # draw the header, settings and result once, from the new result
    box.__exit__(None, None, None)
    result = st.session_state.get("ws_opt")
    if result and result["key"] == run_key:
        ips = client_ips()
        chips = [("Goal", GOALS[goal])]
        if ips is not None and st.session_state.get("ws_use_ips"):
            lim = ips.limits
            ips_bits = [f"≤ {pct(lim['max_weight'])} per holding"] if lim["max_weight"] else []
            if lim["max_volatility"]:
                ips_bits.append(f"vol ≤ {pct(lim['max_volatility'])}")
            if lim["home_bias_gcc"]:
                ips_bits.append(f"GCC ≥ {pct(lim['home_bias_gcc'])}")
            chips.append(("IPS limits", " · ".join(ips_bits) or "on"))
        chips.append(("Max turnover", f"{max_turnover:g}%"))
        chips.append(("Swaps", f"up to {int(max_swaps)}" if use_swaps else "off"))
        _run_with(bar, chips)
        _results_section(result, ai_ok)
    elif result and result["key"][0] != "build":
        st.caption("Your goal, limits or benchmark changed: press Optimise portfolio again.")
    _diagnosis_details(ctx)


def time_key() -> int:
    """A fresh widget-key suffix so a new draft resets the spec form."""
    st.session_state["ws_spec_counter"] = st.session_state.get("ws_spec_counter", 0) + 1
    return st.session_state["ws_spec_counter"]


def policy_benchmark_choice() -> tuple[dict | None, str]:
    """The client's policy benchmark (a simple mix that passes the IPS), or why there is none."""
    ips = client_ips()
    if ips is None:
        return None, ""
    blocks = POLICY_BLOCKS["sharia" if ips.limits["sharia_only"] else "conventional"]
    tickers = tuple(sorted({t for funds in blocks.values() for t in funds}))
    try:
        blocks_data = load_data(tickers, start, end, lookback, day)
    except (DataUnavailable, ValueError) as e:
        return None, f"its funds couldn't be loaded ({e})"
    policy = policy_benchmark(ips, to_returns(blocks_data.prices), blocks_data.periods_per_year)
    if not policy.passes:
        return None, policy.reasons[0]
    return {
        "weights": policy.weights,
        "represents": f"A simple mix that itself passes the client's IPS: {policy.description}. "
        + policy.reasons[0],
    }, ""


POLICY, POLICY_WHY = policy_benchmark_choice()
bench_choices = {
    **CFG["benchmark_choices"],
    **({POLICY_NAME: POLICY} if POLICY else {}),
    **{
        f"{t} (an asset in your data)": {
            "weights": {t: 1.0},
            "represents": f"{NAMES.get(t, t)}, converted to USD.",
        }
        for t in data.prices.columns
    },
}
if st.session_state.get("ws_benchmark") not in bench_choices:
    st.session_state.pop("ws_benchmark", None)  # e.g. the policy benchmark of another client
_bench_now = st.session_state.get("ws_benchmark") or next(iter(bench_choices))
bench_name = head_bench.selectbox(
    "Benchmark",
    list(bench_choices),
    label_visibility="collapsed",
    key="ws_benchmark",
    format_func=lambda n: f"Benchmark · {n}",
    help=f"{_bench_now}: {bench_choices[_bench_now]['represents']} What your portfolio is "
    "compared with across the dashboard: total-return ETF proxies, so dividends count as they "
    "do in your portfolio.",
)
try:
    bench = get_benchmark(
        bench_name, bench_choices[bench_name], all_returns.index, loaded, start, end, day
    )
except (ValueError, DataUnavailable) as e:
    bench = None
    st.warning(f"Couldn't load the benchmark: {e} Choose another one.")

tab_client, step_portfolio, step_analyse, step_optimise, step_suitability, step_report = st.tabs(
    layout.STEPS,
    key=layout.STEP_KEY,  # tracked, so a rerun keeps the step and "Next" can move to one
    on_change="rerun",
)
with step_analyse:
    analyse_head = st.container()  # page header, filled once the backtest period is known
    kpi_slot = st.container()  # filled once the KPI helpers are defined below
    sec_analyse, sec_bench, tab_mc, tab_assets, tab_regime = st.tabs(
        ANALYSE_TABS, key="analyse_tab", on_change="rerun"
    )
with step_optimise:
    optimise_head = st.container()  # page header with the mode switch (render_optimise)
    sec_opt, tab_frontier, tab_backtest = st.tabs(
        OPTIMISE_TABS, key="optimise_tab", on_change="rerun"
    )
with step_report:
    _proposed = (st.session_state.get("ctx") and st.session_state.ctx.proposed) or None
    layout.page_header(
        "Report & trades",
        (
            f"Check the trades to reach {_proposed['name']}, then download the client report."
            if _proposed and st.session_state.get("ws_opt")
            else "Run Optimise first to get trades; the client report is available now."
        ),
        help="Decision support for a licensed adviser: check taxes and liquidity before trading.",
    )
    sec_rebal = st.container()

with tab_client:
    client_ui.render_client_tab(
        CFG,
        list(my_weights),
        ws.groups_of_holdings(),
        get_attributes(tuple(my_weights)),
        cash_like_now() if portfolio_ok else [],
        holding_regions() if portfolio_ok else {},
        holding_products(tuple(my_weights)),
        data_as_of=DATA_AS_OF,
        data_help=DATA_HELP,
    )

with tab_assets:
    st.markdown('<div class="gpl-section">Risk and return by asset</div>', unsafe_allow_html=True)
    if universe_issue:
        st.info(universe_issue)
    else:
        show_metrics_table(
            summary_table(
                returns, rf=rf, level=D["var_level"], periods_per_year=periods
            ).sort_values("sharpe", ascending=False),
            D["var_level"],
        )
        st.caption(
            f"Full sample above. Annualised over {periods} observations a year. VaR and expected "
            "shortfall: one observation (usually 1 day), "
            f"{D['var_level']:.0%} confidence, shown as positive losses."
        )


def _optimise_nav(here: str) -> None:
    """Back to Optimise, and a switch between Frontier and Strategy backtests."""
    back, f, b, _ = st.columns([1.4, 0.9, 1.4, 3], vertical_alignment="center")
    back.button(
        "← Back to Optimise",
        key=f"opt_back_{here}",
        type="tertiary",
        on_click=lambda: st.session_state.update(optimise_tab="Optimise"),
    )
    for col, name in ((f, "Frontier"), (b, "Strategy backtests")):
        col.button(
            name,
            key=f"opt_nav_{here}_{name}",
            type="primary" if name == here else "secondary",
            width="stretch",
            on_click=lambda n=name: st.session_state.update(optimise_tab=n),
        )


with tab_frontier:
    _optimise_nav("Frontier")
    st.markdown(
        '<div class="gpl-section">Efficient frontier (full sample, in hindsight)</div>',
        unsafe_allow_html=True,
    )
    if universe_issue:
        st.info(universe_issue)
    else:
        st.caption(
            "Uses the whole sample's means and covariances, so nobody could have held these at "
            "the time. The backtest tab is the honest test."
        )
        st.caption(
            f"Every portfolio here is long-only with at most **{pct(max_weight)}** per asset "
            f"('{CAP_SETTING}' on step 4 (Optimise))."
        )
        mu, cov = returns.mean() * periods, returns.cov() * periods
        rf_annual = float(
            (1 + rf.loc[returns.index[0] :]).prod() ** (252 / len(rf.loc[returns.index[0] :])) - 1
        )
        portfolios = {
            "Min variance": min_variance(cov, max_weight),
            "Max Sharpe": max_sharpe(mu, cov, rf_annual, max_weight),
            "Risk parity": risk_parity(cov, max_weight),
            "Equal weight": pd.Series(1 / len(universe), index=universe),
        }
        points = {}
        for name, w in portfolios.items():
            ret, vol = portfolio_performance(w, mu, cov)
            points[name] = (vol, ret)
        plot(frontier_chart(efficient_frontier(mu, cov, 40, max_weight), points), width="stretch")
        st.dataframe(style_percent(pd.DataFrame(portfolios).T), width="stretch")
with tab_backtest:
    _optimise_nav("Strategy backtests")
    if universe_issue:
        st.info(universe_issue)
    else:
        if not names:
            st.info("Pick at least one strategy in ⚙ Settings.")
        elif not enough_history:
            st.warning(SHORT_SAMPLE)
        else:
            results = run_backtests(
                tickers,
                loaded,
                start,
                end,
                day,
                tuple(names),
                max_weight,
                cost_bps,
                lookback,
                threshold if use_regime else None,
            )
            strat_returns = pd.DataFrame({n: r.returns for n, r in results.items()})
            table = summary_table(
                strat_returns, rf=rf, level=D["var_level"], periods_per_year=periods
            )
            table["annual_turnover"] = pd.Series(
                {n: r.turnover.iloc[1:].mean() * 12 for n, r in results.items()}
            )
            first_day = next(iter(results.values())).equity.index[0]
            st.subheader("Strategies after costs")
            st.caption(
                f"Monthly rebalancing on a trailing {lookback}-observation window; new weights "
                f"apply from the next observation; {cost_bps:g} bps per unit of turnover."
            )
            show_metrics_table(table.sort_values("sharpe", ascending=False), D["var_level"])
            st.caption(
                f"**Backtest period: {first_day:%d %b %Y} – {prices.index[-1]:%d %b %Y}.** Each "
                f"strategy needs {lookback} observations of history before its first rebalance, "
                f"so the backtest starts {lookback} observations after the data does. Figures "
                "(even for SPY) therefore differ from the Assets tab, which uses the full sample."
            )
            st.caption(
                f"Strategies choose among your holdings, at most **{pct(max_weight)}** each "
                f"('{CAP_SETTING}' on step 4 (Optimise)). Benchmarks are "
                f"fixed by definition: {SIXTY_FORTY} holds "
                + " and ".join(f"{pct(w)} {t}" for t, w in CFG["benchmarks"]["sixty_forty"].items())
                + f"; SPY holds {pct(CFG['benchmarks']['spy']['SPY'])}."
            )
            equity = first_eight(pd.DataFrame({n: r.equity for n, r in results.items()}))
            plot(equity_chart(equity, reference=BENCHMARK_NAMES), width="stretch")
            plot(
                drawdown_chart(
                    drawdown_series(strat_returns[equity.columns]), reference=BENCHMARK_NAMES
                ),
                width="stretch",
            )
            chosen = st.selectbox("Weights over time for", list(results))
            plot(
                weights_chart(
                    results[chosen].weights,
                    groups=groups,
                    title=f"{chosen}: target weights by group",
                ),
                width="stretch",
            )


def portfolio_kpis() -> pd.DataFrame | None:
    """KPI table (``gpl.metrics.kpi_table``): current weights rebalanced monthly after costs vs
    the benchmark, on the dates both cover. None if the portfolio or benchmark isn't usable."""
    if not portfolio_ok or not enough_history or bench is None:
        return None
    mine = run_my_portfolio(my_key, loaded, start, end, day, cost_bps, lookback)
    return kpi_table(mine.returns.iloc[1:], bench.returns, rf, periods)


def headline_dates() -> pd.Index | None:
    """Dates of the headline figures: the backtest period of today's weights."""
    if not portfolio_ok or not enough_history:
        return None
    return run_my_portfolio(my_key, loaded, start, end, day, cost_bps, lookback).returns.index[1:]


def render_portfolio_kpis() -> None:
    kpi = portfolio_kpis()
    ws.render_kpis(kpi, ws.portfolio_value_text(), bench_name, bench is not None and bench.is_cash)


with analyse_head:
    _kpi = portfolio_kpis()
    _meta = layout.page_header(
        "Analyse the current portfolio",
        f"How today's weights did after costs, compared with {bench_name}.",
        help="Return, volatility, Sharpe and max drawdown: your current weights rebalanced "
        f"monthly after costs; the coloured differences are against {bench_name}.",
    )
    if _kpi is not None:
        layout.meta(
            f"Backtest {_kpi.attrs['period']}",
            help=window_note("analyse", _kpi.attrs["period"]),
            container=_meta,
        )
with kpi_slot:
    render_portfolio_kpis()

with step_portfolio:
    _who = client_ui.current_profile()
    _first = ((_who.name if _who else "") or "the client").split(" ")[0]
    port_meta = layout.page_header(
        "Current portfolio",
        f"Add what {_first} holds today. Weights update as you type.",
        help="Prices are the latest quotes from Yahoo Finance (refreshed every 15 minutes), in "
        "each security's currency and in USD. The analysis uses daily history in USD.",
    ).empty()
    sec_hold, sec_screen = st.tabs(["Holdings", "Screen"], key="portfolio_tab", on_change="rerun")
    holding_names = {t: getattr(get_quote(t), "name", None) or NAMES.get(t, "") for t in my_weights}

    with sec_hold:
        render_data_warnings()
        hold_col, alloc_col = st.columns([1.8, 1], gap="medium")
        with hold_col.container(border=True, key="holdings_card"):
            prices_as_of = ws.render_holdings(
                find_securities, get_quote, CFG["groups"], NAMES, HOLDING_PRESETS, price_data_ok
            )
        if prices_as_of:
            first_time = prices_as_of.split(" (")[0]
            layout.meta(f"Prices as of {first_time}", help=f"Prices as of: {prices_as_of}",
                        container=port_meta)  # fmt: skip
        with alloc_col.container(border=True, key="allocation_card"):
            if my_weights:
                ws.render_allocation(pd.Series(my_weights), ws.groups_of_holdings())
            else:
                st.caption("Add holdings to see the allocation.")
        render_sample_limiters()
        for problem in holding_problems:
            st.warning(problem)
        if holding_note:
            st.caption(holding_note)
        if missing_holdings:
            st.error(
                "No price history for " + ", ".join(missing_holdings) + ". Remove these "
                "holdings or check the symbols (see the warning at the top of the page)."
            )
        render_data_details()
        rates = bt.cost_rates(list(my_weights), cost_bps, COST_BY_SUFFIX)
        local = rates[rates != cost_bps]
        if len(local):
            with st.expander("Trading costs by market"):
                st.caption(
                    "Trading costs used in backtests and trades: "
                    + ", ".join(f"{t} {r:g} bps" for t, r in local.items())
                    + f"; US listings {cost_bps:g} bps (⚙ Settings). Local-market rates are "
                    "judgement estimates (config trading_costs)."
                )

    with sec_screen:
        if not portfolio_ok:
            st.info("Add holdings with price history on the Holdings tab.")
        elif len(my_weights) < 2:
            st.info("Add at least two holdings to compare them.")
        else:
            ws.render_screen(
                ws.screen_table(
                    all_returns,
                    pd.Series(my_weights),
                    rf,
                    D["var_level"],
                    periods,
                    None if bench is None or bench.is_cash else bench.returns,
                ),
                holding_names,
                bench_name,
            )
            st.caption(
                f"Over the full common sample ({all_returns.index[0]:%d %b %Y} – "
                f"{all_returns.index[-1]:%d %b %Y}), annualised with {periods} observations a year."
            )

    with sec_analyse:
        if not portfolio_ok:
            st.info("Add holdings with price history on the Holdings tab.")
        elif not enough_history:
            st.warning(SHORT_SAMPLE)
        else:
            w = pd.Series(my_weights)
            cov = all_returns[w.index].cov() * periods
            rc = risk_contributions(w, cov)
            mine = run_my_portfolio(my_key, loaded, start, end, day, cost_bps, lookback)
            risk_col, growth_col = st.columns(2, gap="medium")
            with risk_col.container(border=True, key="risk_share_card"):
                st.markdown(
                    '<div class="gpl-section">Share of money vs share of risk</div>',
                    unsafe_allow_html=True,
                    help="Share of risk is each holding's contribution to the portfolio's "
                    "volatility (full sample). A holding whose share of risk is much larger than "
                    "its weight dominates how the portfolio moves.",
                    width="content",
                )
                st.markdown(
                    f"{layout.pill('KEY FINDING', 'accent')} {ws.key_finding(w, rc)}",
                    unsafe_allow_html=True,
                )
                plot(risk_contribution_chart(w, rc, title=""), width="stretch")
            with growth_col.container(border=True, key="growth_card"):
                st.markdown('<div class="gpl-section">Growth of $1</div>', unsafe_allow_html=True)
                if bench is not None:
                    port_r = mine.returns.iloc[1:]
                    both = port_r.index.intersection(bench.returns.index)
                    first = (client_ui.current_profile() or ClientProfile()).name.split(" ")[0]
                    mine_label = f"{first}'s portfolio" if first else "Your portfolio"
                    growth = pd.DataFrame(
                        {
                            mine_label: (1 + port_r.loc[both]).cumprod(),
                            bench_name: (1 + bench.returns.loc[both]).cumprod(),
                        }
                    )
                    plot(
                        equity_chart(growth, title="", reference=bench_name),
                        width="stretch",
                        key="perf_growth_chart",
                    )
                else:
                    st.caption("Choose a benchmark with data in the header.")
            others = run_backtests(
                tickers,
                loaded,
                start,
                end,
                day,
                tuple(([] if universe_issue else STRATEGY_NAMES) + [SIXTY_FORTY]),
                max_weight,
                cost_bps,
                lookback,
                None,
            )
            results_mine = {"My portfolio": mine, **others}
            rets = pd.DataFrame({n: r.returns for n, r in results_mine.items()})
            table = summary_table(rets, rf=rf, level=D["var_level"], periods_per_year=periods)
            table["annual_turnover"] = pd.Series(
                {n: r.turnover.iloc[1:].mean() * 12 for n, r in results_mine.items()}
            )
            with st.expander(
                "**Backtest after costs, compared with the strategies and 60/40** &nbsp; "
                ":gray[Return, volatility, Sharpe, Sortino, drawdown, turnover, VaR and expected "
                "shortfall]"
            ):
                show_metrics_table(table, D["var_level"])
                st.caption(
                    f"Your weights are rebalanced monthly back to target with {cost_bps:g} bps "
                    f"costs, on the same dates as the strategies ({mine.equity.index[0]:%d %b %Y} "
                    f"– {mine.equity.index[-1]:%d %b %Y}). The strategies choose among your "
                    f"holdings, at most {pct(max_weight)} each."
                )
                equity = pd.DataFrame({n: r.equity for n, r in results_mine.items()})
                plot(equity_chart(equity, reference=BENCHMARK_NAMES), width="stretch")
            with st.expander(
                "**Drawdown from peak** &nbsp; :gray[Falls from the running peak, against the "
                "strategies and the benchmark]"
            ):
                plot(
                    drawdown_chart(drawdown_series(rets), reference=BENCHMARK_NAMES),
                    width="stretch",
                )
            with st.expander(
                "**Share of money vs share of risk: table** &nbsp; :gray[Weight, share of risk "
                "and USD conversion for each holding]"
            ):
                st.table(
                    pd.DataFrame(
                        {
                            "Name": [holding_names.get(t, "") for t in w.index],
                            "Weight": w.map(pct),
                            "Share of risk": rc.map(pct),
                            "Converted to USD with": [
                                data.history.loc[t, "conversion"] for t in w.index
                            ],
                        },
                        index=w.index,
                    )
                )
            st.caption(
                "Monte Carlo for your portfolio: choose 'My portfolio' on the Monte Carlo tab."
            )

    with sec_bench:
        if not portfolio_ok:
            st.info("Add holdings with price history on the Holdings tab.")
        elif not enough_history:
            st.warning(SHORT_SAMPLE)
        elif bench is None:
            st.info("Choose a benchmark with data above.")
        else:
            mine = run_my_portfolio(my_key, loaded, start, end, day, cost_bps, lookback)
            port = mine.returns.iloc[1:]  # from the first full day held
            dates = port.index.intersection(bench.returns.index)
            for note in bench.notes:
                st.info(note)
            p, b = port.loc[dates], bench.returns.loc[dates]
            g_col, d_col = st.columns(2, gap="medium")
            with g_col.container(border=True, key="bench_growth_card"):
                st.markdown(
                    f'<div class="gpl-section">Your portfolio vs {bench_name}</div>',
                    unsafe_allow_html=True,
                    help=f"Backtest period {period_label(dates)}.",
                    width="content",
                )
                growth = pd.DataFrame(
                    {"Your portfolio": (1 + p).cumprod(), bench_name: (1 + b).cumprod()}
                )
                plot(equity_chart(growth, title="", reference=bench_name), key="bench_growth_chart")
            with d_col.container(border=True, key="bench_reldd_card"):
                st.markdown(
                    '<div class="gpl-section">Relative drawdown</div>',
                    unsafe_allow_html=True,
                    help="How far you fell behind the benchmark: your wealth divided by the "
                    "benchmark's, 0% at your best point relative to it, -10% when you have fallen "
                    "10% behind that point.",
                    width="content",
                )
                plot(
                    drawdown_chart(
                        pd.DataFrame(
                            {"Your portfolio relative to the benchmark": relative_drawdown(p, b)}
                        ),
                        title="",
                    ),
                    width="stretch",
                )
            with st.expander(
                f"**Your portfolio vs {bench_name}: table** &nbsp; :gray[Return, risk, beta, "
                "alpha, tracking error, information ratio and capture]",
                expanded=True,
            ):
                st.table(ws.benchmark_comparison(port, bench.returns, rf, periods, bench.is_cash))
                st.caption(
                    f"{dates[0]:%d %b %Y} – {dates[-1]:%d %b %Y}. Your portfolio is held at "
                    f"today's weights, rebalanced monthly with {cost_bps:g} bps costs; the "
                    "benchmark is an index, so it pays no costs. Alpha: annual return not "
                    "explained by the benchmark exposure (beta). Tracking error: how far your "
                    "returns stray from the benchmark's. Information ratio: extra return per unit "
                    "of tracking error. Up/down capture: your average return on the benchmark's "
                    "up (down) days as a share of its own; below 100% on down days means you fell "
                    "less."
                )

    with sec_opt:
        render_optimise()

    with sec_rebal:
        trade_kpis = st.container()
        trade_col, report_col = st.columns([1.75, 1], gap="medium")
        report_col = report_col.container(border=True, key="report_card")
        opt_now = st.session_state.get("ws_opt")
        opt_rec = opt_now["report"]["recommendation"] if opt_now else None
        has_target = bool(st.session_state.get("ctx") and st.session_state.ctx.proposed)
        if not portfolio_ok:
            trade_col.info("Add holdings with price history on the Holdings tab.")
        elif opt_rec is None and not has_target:
            with trade_col.container(border=True, key="no_trades_yet"):
                st.markdown(
                    '<div class="gpl-answer">No recommendation yet.</div>', unsafe_allow_html=True
                )
                st.markdown("Run Optimise to get target weights and trades.")
                st.button(
                    "Go to Optimise",
                    key="go_to_optimise",
                    type="primary",
                    on_click=layout.go_to,
                    args=(layout.STEPS[3],),
                )
        elif (
            opt_rec is not None
            and opt_rec["choice"] == KEEP
            and opt_rec["status"] != ("closest_fails")
        ):
            with trade_col.container(border=True, key="no_trades_keep"):
                st.markdown(
                    '<div class="gpl-answer">No trades needed: your current portfolio is kept.'
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.caption(main_reason(opt_rec["reasons"]))
        else:
            trade_col.__enter__()
            proposed = (st.session_state.get("ctx") and st.session_state.ctx.proposed) or None
            usd = st.session_state.get(ws.MODE) == ws.USD
            default_target = pd.Series(proposed["weights"]) if proposed else pd.Series(my_weights)
            need, value = cash_need_for_trades(), portfolio_value()
            if need and value:  # start from targets that keep the cash need after costs
                held = (
                    pd.Series({r["ticker"]: r["value"] for r in st.session_state[ws.ROWS]})
                    if usd
                    else pd.Series(my_weights) * value
                ).reindex(list(my_weights))
                latest_usd = pd.Series(
                    {t: getattr(get_quote(t), "price_usd", None) for t in default_target.index},
                    dtype=float,
                )
                default_target, _ = trades_covering_cash_need(
                    held,
                    default_target / default_target.sum(),
                    latest_usd,
                    cost_bps,
                    need["cash_like"],
                    need["need_usd"],
                    cost_by_suffix=COST_BY_SUFFIX,
                )
            rebalance = ws.render_rebalance(
                pd.Series(my_weights),
                default_target,
                f"{proposed['name']}" if proposed else "current weights",
                pd.Series(
                    {
                        r["ticker"]: r["value"]
                        for r in st.session_state[ws.ROWS]
                        if r["ticker"] in my_weights
                    }
                )
                if usd
                else None,
                portfolio_value(),
                lambda t: getattr(get_quote(t), "price_usd", None),
                cost_bps,
                cash_need_for_trades(),
                COST_BY_SUFFIX,
                kpis=trade_kpis,
            )
            trade_col.__exit__(None, None, None)
            if "ctx" in st.session_state:
                st.session_state.ctx.rebalance = rebalance

with tab_mc:
    st.markdown(
        '<div class="gpl-section">Monte Carlo: where could today\'s weights go?</div>',
        unsafe_allow_html=True,
    )
    if universe_issue:
        st.info(universe_issue)
    else:
        if not names:
            st.info("Pick at least one strategy in ⚙ Settings.")
        elif not enough_history:
            st.warning(SHORT_SAMPLE)
        else:
            results = run_backtests(
                tickers,
                loaded,
                start,
                end,
                day,
                tuple(names),
                max_weight,
                cost_bps,
                lookback,
                threshold if use_regime else None,
            )
            c1, c2, c3 = st.columns(3)
            mc_options = list(results) + (["My portfolio"] if portfolio_ok else [])
            mc_strategy = c1.selectbox("Use the latest weights of", mc_options, key="mc_strategy")
            years = c2.slider("Horizon (years)", 1, 30, 10)
            target = c3.number_input("Goal: end with at least × start", 0.5, 10.0, 1.5, 0.1)
            sim_box = st.expander(
                "**Simulation settings** &nbsp; :gray[Method, number of paths, seed and real terms]"
            )
            c4, c5, c6 = sim_box.columns(3)
            method = c4.radio(
                "Method",
                list(MC_METHODS),
                index=list(MC_METHODS).index(MC["method"]),
                format_func=MC_METHODS.get,
                key="mc_method",
                help=MC_METHOD_HELP,
            )
            n_paths = c5.select_slider("Paths", [1_000, 2_000, 5_000, 10_000], value=2_000)
            seed = int(c6.number_input("Seed", 0, 10_000, 42))
            real = sim_box.toggle(
                f"Show in real terms (after {pct(MC['inflation'])} inflation a year)",
                key="mc_real",
                help="Divides wealth by cumulative inflation, so 1.00× means the same purchasing "
                "power as today. The rate is fixed and set in config/universe.yaml.",
            )
            units = "real USD" if real else "nominal USD"
            if mc_strategy == "My portfolio":
                weights = pd.Series(my_weights)
            else:
                weights = results[mc_strategy].weights.iloc[-1]
                weights = weights[weights > 0]
            paths = simulate_paths(
                all_returns,
                weights,
                periods * years,
                n_paths,
                method=method,
                seed=seed,
                block_size=MC["block_size"],
            )
            if real:
                paths = to_real(paths, MC["inflation"], periods_per_year=periods)
            stats = terminal_stats(paths)
            st.markdown(f"**Over {years} years, in {units}**")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric(f"P(≥ {target:g}×)", pct(goal_probability(paths, target)))
            m2.metric("Median outcome", f"{stats['p50']:.2f}×")
            m3.metric("Bad case (5th pct)", f"{stats['p5']:.2f}×")
            m4.metric("P(lose money)", pct(1 - goal_probability(paths, 1.0)))
            m5.metric(
                "Drawdown (5th pct)",
                pct(np.percentile(path_max_drawdowns(paths), 5)),
                help="5th percentile of each path's worst fall from a previous high during the "
                "horizon: 1 in 20 simulations falls at least this far at some point. The other "
                "numbers only look at the end; this shows how rough the ride is.",
            )
            plot(
                fan_chart(
                    paths,
                    title=f"{mc_strategy}: simulated growth of $1 ({units})",
                    y_title="Wealth (real USD, today's money)" if real else "Wealth (nominal USD)",
                ),
                width="stretch",
            )
            st.caption(
                f"Outcomes in **{units}**"
                + (
                    f", assuming a constant {pct(MC['inflation'])} inflation a year. "
                    if real
                    else ". "
                )
                + "Simulations hold today's weights fixed, **rebalance daily and ignore "
                f"transaction costs**, unlike the backtest (monthly rebalancing, "
                f"{cost_bps:g} bps per unit of turnover). They assume the future resembles "
                f"{prices.index[0]:%b %Y} – {prices.index[-1]:%b %Y}."
            )

with tab_regime:
    st.markdown(
        '<div class="gpl-section">Will next month be stressed?</div>', unsafe_allow_html=True
    )
    regime = load_regime(end, day)
    p = regime["p_stressed"].dropna()
    folds, overall = regime["report"]["folds"], regime["report"]["overall"]
    model_scores, base_rate = overall.loc["model"], overall.loc["model", "stressed_share"]
    calm_accuracy = overall.loc["always_calm", "accuracy"]

    st.markdown(
        f"**Out-of-sample scores, {folds['test_start'].min():%b %Y} – "
        f"{folds['test_end'].max():%b %Y}** · base rate (actual stressed days) "
        f"{pct(base_rate)} · threshold {pct(threshold)}"
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        "ROC AUC",
        f"{model_scores['auc']:.2f}",
        help="How well the model ranks stressed days above calm ones, whatever the threshold. "
        "0.50 = coin flip, 1.00 = perfect.",
    )
    m2.metric(
        "Precision",
        pct(model_scores["precision"]),
        f"{(model_scores['precision'] - base_rate) * 100:+.2f} pp vs base",
        help="When the model says 'stressed', how often it was right, compared with the share "
        "of days that were actually stressed (what random guessing would score).",
    )
    m3.metric(
        "Recall", pct(model_scores["recall"]), help="Share of stressed days the model flagged."
    )
    m4.metric(
        "Accuracy",
        pct(model_scores["accuracy"]),
        f"{(model_scores['accuracy'] - calm_accuracy) * 100:+.2f} pp vs calm",
        help="Share of all days classified correctly, vs always saying 'calm'. Misleading on "
        "its own: always-calm is right on every calm day.",
    )
    m5.metric(
        "Latest P(stressed)",
        pct(p.iloc[-1]),
        help=f"Walk-forward probability on {p.index[-1]:%d %b %Y}, from a model trained only "
        "on labels known by then.",
    )

    monthly = p.resample("ME").last().dropna()
    realised = regime["labels"].reindex(p.index).resample("ME").last()
    plot(
        regime_chart(
            monthly, threshold, realised, title="Month-end probability of a stressed month ahead"
        ),
        width="stretch",
    )
    features = regime["features"]
    st.caption(
        f"The regime model uses SPY, AGG and VIX history from **{features.index[0]:%b %Y}** "
        f"(complete features from {features.dropna().index[0]:%b %Y}), much longer than the GCC "
        f"ETF sample ({prices.index[0]:%b %Y} onwards), so it learns from the 2008 crisis and "
        f"later episodes. First walk-forward prediction: {p.index[0]:%b %Y}. **'Stressed' is "
        f"relative:** SPY's volatility over the next {R['horizon']} trading days is above the "
        f"{R['quantile'] * 100:.0f}th percentile of its own past year, so the bar moves with "
        "recent conditions. Plotted at month ends (the dates the strategies rebalance); grey = "
        "months that turned out stressed. Unweighted logistic regression, so probabilities are "
        f"on the scale of the actual rate: {pct(model_scores['mean_p'])} predicted on average vs "
        f"{pct(base_rate)} actual."
    )

    folds_box = st.expander(
        "**By fold, with two baselines** &nbsp; :gray[Each test period's model scores against a "
        "volatility rule and always-calm]"
    )
    folds_box.markdown("**By fold, with two baselines**")
    fold_rows = folds.assign(
        when=folds["test_start"].dt.strftime("%b %Y")
        + " – "
        + folds["test_end"].dt.strftime("%b %Y")
    )
    all_rule = overall.loc["vol_rule"]
    fold_rows.loc[len(fold_rows)] = {
        "when": "All folds",
        "stressed_share": base_rate,
        "mean_p": model_scores["mean_p"],
        "auc": model_scores["auc"],
        "precision": model_scores["precision"],
        "recall": model_scores["recall"],
        "accuracy": model_scores["accuracy"],
        "rule_precision": all_rule["precision"],
        "rule_recall": all_rule["recall"],
        "rule_accuracy": all_rule["accuracy"],
        "baseline_accuracy": overall.loc["always_calm", "accuracy"],
    }
    fold_table = (
        fold_rows[
            [
                "when",
                "stressed_share",
                "mean_p",
                "auc",
                "precision",
                "recall",
                "accuracy",
                "rule_precision",
                "rule_recall",
                "rule_accuracy",
                "baseline_accuracy",
            ]
        ]
        .rename(
            columns={
                "when": "Test period",
                "stressed_share": "Stressed days (actual)",
                "mean_p": "Avg P(stressed)",
                "auc": "Model ROC AUC",
                "precision": "Model precision",
                "recall": "Model recall",
                "accuracy": "Model accuracy",
                "rule_precision": "Vol-rule precision",
                "rule_recall": "Vol-rule recall",
                "rule_accuracy": "Vol-rule accuracy",
                "baseline_accuracy": "Always-calm accuracy",
            }
        )
        .set_index("Test period")
    )
    folds_box.table(style_columns(fold_table, {"Model ROC AUC": "{:.2f}"}))
    folds_box.caption(
        "Vol rule = 'stressed if SPY's last-21-day volatility is already above its 80th "
        "percentile' (no model). Always-calm = never flag stress."
    )


def current_ports() -> dict[str, pd.Series]:
    """Today's weights and, after Optimise, the proposed portfolio."""
    ports = {"Current": pd.Series(my_weights)}
    proposed = st.session_state.get("ctx") and st.session_state.ctx.proposed
    if proposed:
        ports[proposed["name"]] = pd.Series(proposed["weights"], dtype=float)
    return ports


def latest_memo() -> tuple[str | None, list[str]]:
    """The AI's explanation of the Optimise recommendation, for the report (text and any
    flagged figures); None until "Explain this recommendation (AI)" has been pressed."""
    opt = st.session_state.get("ws_opt") or {}
    memo = opt.get("memo")
    if memo and memo.get("role") == "assistant":
        return memo["reply"].text, memo["reply"].unverified
    return None, []


def report_holdings(value: float | None) -> pd.DataFrame:
    """Holdings for the report: name, value in USD (when the portfolio value is known), weight."""
    w = pd.Series(my_weights)
    usd = st.session_state.get(ws.MODE) == ws.USD
    amounts = (
        pd.Series({r["ticker"]: r["value"] for r in st.session_state[ws.ROWS]}).reindex(w.index)
        if usd
        else (w * value if value else None)
    )
    table = {"Name": [getattr(get_quote(t), "name", None) or NAMES.get(t, "") for t in w.index]}
    if amounts is not None:
        table["Value (USD)"] = amounts.map("{:,.0f}".format)
    table["Weight"] = w.map(pct)
    return pd.DataFrame(table, index=w.index)


def report_recommendation() -> tuple[dict | None, dict | None]:
    """The Optimise run's recommendation and its trade list, formatted (None if not run).

    Always the optimiser's own result, never a what-if from the chat. Trades are proposed
    only for a recommendation that passes the IPS.
    """
    opt = st.session_state.get("ws_opt")
    if not opt:
        return None, None
    report, run = opt["report"], opt["run"]
    latest = optimise_weights(opt)
    rec = recommendation_section(report, latest)
    if full_move_chosen(opt) and rec["choice"]:
        full = report["full_restructure_to_target"][rec["choice"]]
        passes = full["passes_ips"] in ("yes", "no IPS")  # no IPS: no gate to pass
        rec["failed"] = [  # the rules the full target breaks, not today's limited move
            breach_text(c) for c in failures(run.full_checks.get(rec["choice"], []))
        ]
        cost = full.get("estimated_cost_usd")
        rec["reasons"].append(
            f"The adviser chose the full restructure to the {rec['choice']} target: "
            f"{full['turnover']} of the portfolio traded"
            + (f", estimated cost USD {cost}" if cost else "")
            + (
                "."
                if full["passes_ips"] == "no IPS"
                else "; it passes every IPS rule."
                if passes
                else f"; it does not pass the IPS ({full['passes_ips'][4:]})."
            )
        )
        if passes and rec["status"] == "closest_fails":
            rec["headline"] = (
                f"Within the {report['max_turnover']} turnover limit no move meets the IPS; "
                f"the adviser chose the full restructure to the {rec['choice']} portfolio, "
                "which passes every IPS rule."
            )
            rec["status"] = "recommended"
    elif (
        report.get("mode") == "improve"
        and rec["choice"] in report.get("full_restructure_to_target", {})
        and report["full_restructure_to_target"][rec["choice"]]["turnover"]
        != report["turnover_limited_move"][rec["choice"]]["turnover"]
    ):
        full = report["full_restructure_to_target"][rec["choice"]]
        lim = report["turnover_limited_move"][rec["choice"]]
        cost = full.get("estimated_cost_usd")
        rec["reasons"].append(
            f"This is the closest compliant move within the {report['max_turnover']} turnover "
            f"limit ({lim['turnover']} traded). A full restructure to the {rec['choice']} target "
            f"would trade {full['turnover']}"
            + (f" (estimated cost USD {cost})" if cost else "")
            + "; your adviser can choose it instead."
        )
    if rec["choice"] in run.equity and "Benchmark" in run.equity:
        daily = run.equity[[rec["choice"], "Benchmark"]].pct_change().dropna()
        rec["relative_drawdown"] = relative_drawdown(
            daily[rec["choice"]], daily["Benchmark"]
        ).rename(rec["choice"])
    value = portfolio_value()
    full_fails = full_move_chosen(opt) and report["full_restructure_to_target"][rec["choice"]][
        "passes_ips"
    ] not in ("yes", "no IPS")
    if rec["choice"] is None or rec["status"] == "closest_fails" or full_fails or not value:
        return rec, None  # trades only for a portfolio that passes the IPS
    usd = st.session_state.get(ws.MODE) == ws.USD
    current = pd.Series(my_weights)
    values = (
        pd.Series({r["ticker"]: r["value"] for r in st.session_state[ws.ROWS]}).reindex(
            current.index
        )
        if usd
        else current * value
    )
    target = latest.loc[rec["choice"]].fillna(0.0)
    target = target / target.sum()
    prices = pd.Series({t: getattr(get_quote(t), "price_usd", None) for t in target.index})
    need = cash_need_for_trades()
    if need:  # top up cash-like holdings so the need is covered after trading costs
        adjusted, trades = trades_covering_cash_need(
            values,
            target,
            prices.astype(float),
            cost_bps,
            need["cash_like"],
            need["need_usd"],
            cost_by_suffix=COST_BY_SUFFIX,
        )
        cash = cash_need_check(trades["target_usd"], need["cash_like"], need["need_usd"])
    else:
        adjusted, trades, cash = (
            target,
            trade_list(
                values, target, prices.astype(float), cost_bps, cost_by_suffix=COST_BY_SUFFIX
            ),
            None,
        )
    section = trade_section(trades, cost_bps, cash)
    raised = [t for t in target.index if adjusted[t] > target[t] + 5e-5]
    if raised:
        section["summary"] += (
            " To keep the cash you need after trading costs, "
            + ", ".join(f"{t} is set to {adjusted[t]:.2%} (not {target[t]:.2%})" for t in raised)
            + "."
        )
    return rec, section


def render_report_button() -> None:
    """Build the client report on request (charts take a moment), then offer the PDF."""
    rec_name = (report_recommendation()[0] or {}).get("choice")
    st.markdown('<div class="gpl-section">Client report</div>', unsafe_allow_html=True)
    st.caption(
        "PDF: IPS summary, analysis, "
        + (f"the {rec_name} recommendation" if rec_name else "the current portfolio")
        + ", suitability and trades, with an adviser sign-off."
    )
    a1 = a2 = st
    on_file = client_ui.current_profile()
    a1.text_input(
        "Adviser name",
        key="rep_adviser",
        placeholder=(on_file and on_file.adviser_name) or "",
        help="Blank: the adviser on the client's fact-find (Service & review).",
    )
    a2.text_input(
        "Licence / registration number",
        key="rep_licence",
        placeholder=(on_file and on_file.adviser_licence) or "",
    )
    ready = bool(st.session_state.get("client_report"))
    prep_box = st.container(key="prepare_box" if ready else "cta_prepare")
    if prep_box.button(
        "Prepare the client report again" if ready else "Prepare client report (PDF)",
        key="prepare_report",
        type="secondary" if ready else "primary",
        width="stretch",
    ):
        with st.spinner("Building the report…"):
            ctx = shared_context()
            charts = []
            for chart in ("growth_of_1", "drawdown_comparison", "stress_test"):
                try:
                    charts.append(chart_data(ctx, chart))
                except ValueError:
                    pass  # e.g. no benchmark: the report says what is missing
            memo, unverified = latest_memo()
            profile = client_ui.current_profile()
            rec, trades = report_recommendation()
            checked = {"Current": pd.Series(my_weights)}
            if rec and rec["choice"]:  # the optimiser's portfolio, never a chat what-if
                latest = optimise_weights(st.session_state.ws_opt)
                checked[rec["choice"]] = latest.loc[rec["choice"]].fillna(0.0)
            results = suitability_results(checked)
            record, edited = "", False
            if results:
                generated = generated_record(results)[1]
                record = profile.suitability_record or generated
                edited = bool(profile.suitability_record) and record != generated
            inputs = ReportInputs(
                report_date=date.today(),
                data_as_of=f"{data.price_as_of:%d %b %Y}",
                benchmark=bench_name,
                holdings=report_holdings(portfolio_value()),
                allocation=pd.Series(my_weights),
                client=profile,
                ips=client_ips(),
                kpis=portfolio_kpis(),
                charts=charts,
                suitability=results,
                suitability_record=record,
                record_edited=edited,
                memo=memo,
                memo_unverified=unverified,
                adviser=st.session_state.get("rep_adviser")
                or (profile and profile.adviser_name)
                or "",
                adviser_licence=st.session_state.get("rep_licence")
                or (profile and profile.adviser_licence)
                or "",
                domicile=domicile_section(),
                value=portfolio_value(),
                recommendation=rec,
                trades=trades,
            )
            st.session_state.client_report_flags = unverified
            ref = profile.reference if profile and profile.reference else "portfolio"
            st.session_state.client_report = (
                client_report(inputs),
                f"Client_report_{ref}_{date.today().isoformat()}.pdf",
            )
    if st.session_state.get("client_report"):
        pdf, name = st.session_state.client_report
        with st.container(key="cta_download_report"):
            st.download_button(
                "Download client report",
                pdf,
                file_name=name,
                mime="application/pdf",
                key="download_report",
                type="primary",
                icon=":material/download:",
                width="stretch",
            )
        st.caption(
            "Uses the Optimise recommendation (never a chat what-if) and the Optimise memo if "
            "one was written, else the latest analyst memo. The IPS PDF is on step 1. Prepare "
            "it again after changes."
        )
        flags = st.session_state.get("client_report_flags")
        if flags:
            st.warning(
                "Adviser check before sending: the memo has figures no tool returned: "
                + ", ".join(flags)
            )


with step_suitability:
    suit_counts = layout.page_header(
        "Suitability",
        "Every IPS rule, checked for today's portfolio and the recommendation.",
        help="Pass, fail or review for each rule of the client's IPS. A recommendation can only "
        "pass when every rule passes; review means the data can't decide and the adviser must "
        "check.",
    ).empty()
    if not portfolio_ok:
        st.info("Add holdings with price history on step 2 (Portfolio) to check suitability.")
    else:
        render_suitability(current_ports(), "suitability", record=True)

with report_col:
    if client_ui.current_profile() is None:
        st.info("Load or create a client on step 1 (Client) to issue the IPS and the report.")
    if portfolio_ok:
        render_report_button()


# ------------------------------------------------------------- workflow status and header
def step_statuses() -> list[layout.StepStatus]:
    """What each step has and still needs, from the app's own state (no new figures)."""
    profile = client_ui.current_profile()
    value = portfolio_value()
    if profile is None:
        client = layout.StepStatus(False, "no client loaded", ["load or create a client"])
    else:
        parts = completeness(profile)
        share = overall_completeness(parts)
        todo = [SECTION_BY_KEY[k].title for k, c in parts.items() if c.missing]
        client = layout.StepStatus(share == 1, f"fact-find {share:.0%} complete", todo)
    todo = []
    if not portfolio_ok:
        todo.append("holdings with price history")
    if value is None:
        todo.append("the portfolio value")
    portfolio = layout.StepStatus(
        not todo,
        f"{len(my_weights)} holdings" + (f" · {money(value)}" if value else ""),
        todo,
    )
    analyse = layout.StepStatus(
        portfolio_ok and not universe_issue,
        "analysis ready" if portfolio_ok and not universe_issue else "needs two or more holdings",
        [universe_issue] if universe_issue else [],
    )
    if profile is None and not ws.portfolio_started():  # first visit: nothing loaded yet
        start = ["load a client (▶ Try a demo client) or a portfolio"]
        idle = layout.StepStatus(False, "not started", start)
        return [client, idle, idle, idle, idle, idle]
    ran = bool(st.session_state.get("ws_opt"))
    optimise = layout.StepStatus(
        ran,
        "recommendation ready" if ran else "not run yet",
        [] if ran else ["choose a goal and press Optimise portfolio"],
    )
    suit_todo = (["a client"] if profile is None else []) + (
        ["holdings"] if not portfolio_ok else []
    )
    checked, fails = st.session_state.get("suit_fails", ("", []))
    if suit_todo:
        suitability = layout.StepStatus(False, "needs a client and holdings", suit_todo)
    elif fails:
        suitability = layout.StepStatus(
            False,
            f"{checked}: {len(fails)} rule{'s' if len(fails) != 1 else ''} fail",
            [f"{checked} fails " + ", ".join(fails).lower()],
        )
    else:
        suitability = layout.StepStatus(True, f"{checked}: passes every rule", [])
    prepared = bool(st.session_state.get("client_report"))
    report = layout.StepStatus(
        prepared,
        "report prepared" if prepared else "report not prepared",
        [] if prepared else ["prepare and download the client report"],
    )
    return [client, portfolio, analyse, optimise, suitability, report]


statuses = step_statuses()
for container, step, status in zip(
    (tab_client, step_portfolio, step_analyse, step_optimise, step_suitability, step_report),
    layout.STEPS,
    statuses,
    strict=True,
):
    with container:
        layout.step_footer(step, status)
st.session_state["step_statuses"] = statuses  # the next run draws the cards with these first
layout.update_stepper(step_slots, statuses, st.session_state.get(layout.STEP_KEY))
layout.render_header(chips_slot, header_info())
