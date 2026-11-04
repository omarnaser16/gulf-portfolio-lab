"""Gulf Portfolio Lab dashboard.

Presentation only: every number on screen comes from a ``gpl`` function. The app picks
inputs, calls the engine, and draws the results. Run with::

    streamlit run app/streamlit_app.py
"""

import json
import os
from datetime import date

import anthropic
import client_ui
import layout
import numpy as np
import pandas as pd
import streamlit as st
import workstation as ws
from inputs import asset_groups, combine_tickers
from layout import LAYOUT_CSS

from formatting import NUMBERS_CSS, pct, plot, show_metrics_table, style_columns, style_percent
from gpl import backtest as bt
from gpl import regime as rg
from gpl.agent import analyst as agent
from gpl.agent.charts import build_chart, chart_data
from gpl.agent.spec import (
    active_limits,
    describe_goal,
    propose_spec,
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
    build_ips,
    cash_like_holdings,
    ips_constraints,
    month_text,
    tighter,
)
from gpl.construct import BuildSettings, _groups, screen_universe
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
    allocation_chart,
    drawdown_chart,
    equity_chart,
    fan_chart,
    frontier_chart,
    lines_chart,
    paired_bars_chart,
    regime_chart,
    risk_contribution_chart,
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
from gpl.suitability import breach_text, check_suitability, rationale, suitability_table

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
DISCLAIMER = (
    "Educational project, **not investment advice**. "
    "Past performance does not predict future results."
)


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
st.markdown(NUMBERS_CSS + LAYOUT_CSS, unsafe_allow_html=True)
header = st.container(key="gpl_header")
head_title, head_chips, head_bench, head_demo, head_actions, head_settings = header.columns(
    [2.0, 4.2, 2.0, 1.7, 1.6, 0.7], vertical_alignment="center"
)
head_demo.button(
    "🎲 Fill demo client",
    key="fill_demo",
    on_click=client_ui.fill_demo,
    width="stretch",
    help="Fills every field with a random, realistic GCC client and a 5-8 holding portfolio "
    "of real tickers (no AI), then opens the analysis. Marked DEMO DATA everywhere; the seed "
    "in the header reproduces it (⚙ Settings › Demo seed).",
)
head_title.markdown("### Gulf Portfolio Lab")
stepper = st.container()  # drawn at the end of the run, when every status is known
head_actions.button(
    "💬 Ask the analyst",
    key="ask_analyst",
    on_click=layout.go_to,
    args=(layout.STEPS[-1], "AI analyst"),
    width="stretch",
    help="Open the AI analyst (Report & trades step) with this client and portfolio.",
)
with head_settings.popover("⚙", width="stretch", help="Settings"):
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
        "Fill demo client with this seed", key="fill_demo_seed", on_click=client_ui.fill_demo
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
st.markdown(
    f"**Data as of {data.price_as_of:%d %b %Y}** (latest price) · loaded {data.loaded_at} · "
    f"{'latest data' if end is None else 'fixed sample used in the README'}"
)
st.caption(
    f"Sample: **{data.prices.index[0]:%d %b %Y} – {data.prices.index[-1]:%d %b %Y}** "
    f"({len(returns):,} daily returns from {len(returns) + 1:,} price dates, about {periods} "
    "a year). Every result depends on this period."
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
with st.expander(
    "Data notes: start dates, currencies and trading calendars",
    expanded=bool(data.mixed_calendars or len(short)),
):
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
            "Sunday to a Monday week on 1 Jan 2022, but Yahoo has no Friday prices for them, so "
            "they still differ from US markets.) The analysis uses only the "
            f"{data.alignment['kept']:,} price dates ({data.alignment['kept'] - 1:,} returns) on "
            "which every selected market traded (about "
            f"{periods} a year), so some returns span more than one day; figures are annualised "
            f"with {periods} periods a year instead of 252."
        )
    for t, row in short.iterrows():
        st.warning(
            f"**{t}** has only {row['observations']:,} trading days of history (from "
            f"{row['first_date']:%d %b %Y}), less than the {lookback}-observation estimation "
            "window plus a month. Strategies that use it start late and its estimates are noisy."
        )
    setters = list(data.history.index[data.history["sets_start"]])
    st.caption(
        f"The common sample starts on {data.prices.index[0]:%d %b %Y}, when the most recent asset "
        f"({', '.join(setters)}) begins."
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
    except anthropic.AuthenticationError:
        return {"role": "error", "text": "The API key was rejected."}
    except anthropic.RateLimitError:
        return {"role": "error", "text": "Rate limited; try again in a minute."}
    except (anthropic.APIConnectionError, anthropic.APIStatusError) as e:
        return {"role": "error", "text": f"API error: {e}"}


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


def _interpret(holding_groups: dict[str, str]) -> None:
    """Callback: the AI turns the typed goals into a draft (goal, target, limits).

    Runs before the widgets are drawn, so it may also set the per-asset cap slider.
    """
    goals = st.session_state.get("ws_goals", "")
    st.session_state.pop("ws_interpret_error", None)
    if not goals.strip():
        st.session_state.ws_interpret_error = (
            "Describe your goals first, e.g. 'make it safer, at least 20% GCC'."
        )
        return
    try:
        proposal = propose_spec(agent.make_client(), A["model"], goals, holding_groups)
    except (ValueError, anthropic.APIError) as e:
        st.session_state.ws_interpret_error = str(e)
        return
    st.session_state.ws_draft = proposal.draft
    st.session_state.ws_proposal = proposal
    st.session_state.ws_draft_v = time_key()
    cap = proposal.draft.get("max_weight_pct")
    if cap is not None:
        st.session_state["max_weight"] = min(max(float(cap) / 100, 0.10), 1.0)
    turnover_cap = proposal.draft.get("max_turnover_pct")
    if turnover_cap is not None:
        st.session_state["opt_turnover"] = min(max(float(turnover_cap), 5.0), 200.0)
    te_cap = proposal.draft.get("tracking_error_max_pct")
    if te_cap is not None:
        st.session_state["build_te"] = min(max(float(te_cap), 0.5), 20.0)
    st.session_state.pop("ws_opt", None)


def _goal_changed(v: int) -> None:
    """Callback: a new goal starts from blank limits, so an old target (e.g. 8% target
    volatility) never carries over as a cap. The client's IPS limits stay (separate toggle)."""
    goal = st.session_state.get(f"ws_goal_{v}")
    st.session_state.ws_draft = {**BLANK_DRAFT, "goal": goal}
    st.session_state.ws_draft_v = time_key()
    st.session_state.pop("ws_proposal", None)


def _clear_limits() -> None:
    """Callback: clear every limit on the form, including the per-holding cap."""
    goal = st.session_state.get("ws_draft", {}).get("goal") or "max_risk_adjusted"
    st.session_state.ws_draft = {**BLANK_DRAFT, "goal": goal}
    st.session_state.ws_draft_v = time_key()
    st.session_state["max_weight"] = 1.0
    st.session_state.pop("ws_proposal", None)


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


def _diagnosis_section(ctx: AnalystContext) -> None:
    key = (settings_key, bench_name)
    if st.session_state.get("ws_diag_key") != key:
        st.session_state.ws_diag = diagnose_portfolio(ctx, my_weights)
        st.session_state.ws_diag_key = key
    diag = st.session_state.ws_diag
    ctx.diagnosis = diag
    st.markdown("**1. Diagnosis: where your risk comes from**")
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
    """Goal, target and limits (typed or interpreted by the AI); returns (goal, limits, target)."""
    st.markdown("**2. Your goal and limits**")
    with st.expander("Describe it in plain English (optional)", expanded=False):
        st.text_area(
            "Goals in plain English",
            key="ws_goals",
            placeholder="e.g. make it safer, at least 20% GCC, no single asset over 25%",
            help="The AI only fills in the goal and limits below for you to check; it never "
            "chooses weights or methods.",
        )
        st.button(
            "Interpret with AI",
            key="ws_interpret",
            type="primary",
            on_click=_interpret,
            args=(holding_groups,),
            disabled=not ai_ok,
            help=None if ai_ok else "Needs an API key (see Report & trades › AI analyst).",
        )
        if st.session_state.get("ws_interpret_error"):
            st.warning(st.session_state.ws_interpret_error)
        proposal = st.session_state.get("ws_proposal")
        if proposal:
            if proposal.summary:
                st.info(f"**The AI understood:** {proposal.summary}")
            for request in proposal.unmapped:
                st.warning(f"Not included (can't be expressed as a limit): “{request}”")
            if proposal.numbers_not_in_goals:
                st.warning(
                    "These numbers are not in your text, please check them: "
                    + ", ".join(proposal.numbers_not_in_goals)
                )

    draft = st.session_state.setdefault("ws_draft", initial_draft())
    v = st.session_state.setdefault("ws_draft_v", 0)

    def as_float(value):  # the model writes integers (25); number inputs need floats
        return None if value is None else float(value)

    goal = st.radio(
        "Goal",
        list(GOALS),
        format_func=GOALS.get,
        index=list(GOALS).index(draft.get("goal") or "max_risk_adjusted"),
        key=f"ws_goal_{v}",
        on_change=_goal_changed,
        args=(v,),
        help="Changing the goal clears the limits below (not the client's IPS limits).",
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
        f"USD {n['amount_usd']:,.0f} by {month_text(n['date'])}"
        for n in ips.limits["cash_needs_due"]
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
    st.markdown("**Fund domicile check (GCC clients)**")
    result = domicile_section()
    if result is None:
        st.info("Add holdings on step 2 (Portfolio) to check their domicile.")
        return
    st.info(result["label"])
    if portfolio_value() is None:
        c1, _ = st.columns([1, 2])
        ws.value_input(c1, f"pf_value_{where}", "Portfolio value (USD) for the estate-tax estimate")
    if result["rows"]:
        st.table(pd.DataFrame(result["rows"]).set_index("Ticker"))
    for line in result["summary"]:
        st.caption("• " + line)


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
    """The suitability record: generated from rules and checks, editable by the adviser."""
    profile = client_ui.current_profile()
    name, generated = generated_record(results)
    st.markdown(f"**Suitability record** ({name})")
    edited = bool(profile.suitability_record) and profile.suitability_record != generated
    key = (
        f"suit_record_{st.session_state.get(client_ui.VERSION, 0)}_"
        f"{st.session_state.get('suit_record_v', 0)}"
    )
    st.text_area(
        "Rationale (edit before it goes in the IPS and the client report)",
        profile.suitability_record or generated,
        height=220,
        key=key,
        on_change=_save_record,
        args=(key,),
    )
    c1, c2 = st.columns([1, 3])
    c1.button("Reset to the generated text", key="suit_record_reset", on_click=_reset_record)
    c2.caption(
        "Edited by the adviser."
        if edited
        else "Generated from the IPS rules and the checks "
        "above (no AI). Your edits are saved with the client profile."
    )


def render_suitability(portfolios: dict[str, pd.Series], key: str, record: bool = False) -> None:
    """Pass / fail / review per IPS rule for each portfolio, with the written reasons."""
    st.markdown("**Suitability against the client's IPS**")
    results = suitability_results(portfolios)
    if results is None:
        st.info("Load or create a client on step 1 (Client) to check suitability.")
        return
    ips = client_ips()
    last = list(results)[-1]
    st.session_state["suit_fails"] = (last, [c.rule for c in results[last] if c.status == "fail"])
    note = client_ui.closest_mix_note(ips)
    if note and record:
        fails = [breach_text(c) for c in results["Current"] if c.status == "fail"]
        st.warning(f"**{CLOSEST_MIX_LABEL}** {note}" + "".join(f"\n- {line}" for line in fails))
    table = suitability_table(results)
    for col in table.columns:
        statuses = {c.rule: c.status for c in results[col]}
        table[col] = [f"{SUITABILITY_ICON[statuses[r]]} {v}" for r, v in table[col].items()]
    st.table(table)
    for name, checks in results.items():
        with st.expander(f"Reasons: {name}"):
            for c in checks:
                st.markdown(f"{SUITABILITY_ICON[c.status]} **{c.rule}.** {c.reason}")
    full = period_label(all_returns.index)
    dates = headline_dates()
    st.caption(
        f"**Periods.** Policy risk checks (volatility, drawdown) use the full history, {full}: "
        "longer than the headline cards and including earlier crises. "
        + (
            f"The headline cards cover the backtest period, {period_label(dates)}; the reasons "
            "give both. "
            if dates is not None
            else ""
        )
        + f"Client {ips.client.reference or '(no reference)'}. Today's weights; ⚠️ Review means "
        "the data can't decide and the adviser must check. Decision support for a licensed "
        "adviser."
    )
    if record:
        render_suitability_record(results)


def client_ips():
    """The current client's IPS, or None if no client profile is loaded."""
    profile = client_ui.current_profile()
    return None if profile is None else build_ips(profile, CFG)


def initial_draft() -> dict:
    """Blank Optimise form; with a client return target, start on 'Reach a target return'."""
    draft = dict(BLANK_DRAFT)
    profile = client_ui.current_profile()
    if profile is not None and profile.return_target_pct is not None:
        draft.update(goal="target_return", min_return_pct=profile.return_target_pct)
    return draft


def _results_section(result: dict, ai_ok: bool) -> None:
    report, run = result["report"], result["run"]
    rec = report["recommendation"]
    st.markdown("**4. Recommendation**")
    reasons = "\n".join(f"- {r}" for r in rec["reasons"])
    gate = report.get("suitability_gate")
    if rec["status"] == "closest_fails":
        closest = "your current portfolio" if rec["choice"] == KEEP else rec["choice"]
        failed = gate["failed_rules"]["Current" if rec["choice"] == KEEP else rec["choice"]]
        st.warning(
            f"**No method meets the client's IPS. The closest is {closest}**, which still "
            "fails:\n\n"
            + "\n".join(f"- {line}" for line in failed)
            + f"\n\n{reasons}\n\nNot suitable as it stands: change the holdings (e.g. add a "
            "lower-risk or cash-like holding) or review the IPS with the client."
        )
    elif rec["choice"] == KEEP:
        st.info(f"**Keep your current portfolio.**\n\n{reasons}")
    elif rec["status"] == "next_best":
        st.info(
            f"**Next best option: {rec['choice']}.** No built portfolio beat the benchmark "
            f"out of sample after costs within the IPS, so this does not beat it.\n\n{reasons}"
        )
    else:
        label = "Recommended portfolio" if report.get("mode") == "build" else "Recommended method"
        st.success(f"**{label}: {rec['choice']}**\n\n{reasons}")
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
    st.caption(f"The rule for this goal: {rec['rule']}")

    rows = pd.DataFrame(report["walk_forward_after_costs"]).T
    columns = dict(CANDIDATE_TABLE_COLUMNS)
    if gate:
        columns["passes_ips"] = "Passes IPS?"
    rows = rows.reindex(columns=list(columns)).rename(columns=columns)
    rows = rows.astype(object).fillna("–")  # columns a row doesn't have (e.g. Benchmark)
    mark = "≈ " if rec["status"] == "closest_fails" else "★ "  # ≈: closest, still fails
    rows.index = [(mark if n == rec["choice"] else "") + n for n in rows.index]
    st.markdown(f"**Walk-forward results after costs ({report['backtest_period']})**")
    st.table(rows)
    build = report.get("mode") == "build"
    st.caption(
        (
            "Each candidate re-chooses its securities and weights every quarter from the data "
            "available then (risk from all history so far, returns from the last year), within "
            "the IPS and your limits, then holds them to the next quarter, after trading costs. "
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
    for name, why in report.get("infeasible_candidates", {}).items():
        st.warning(f"**{name}** can't meet your limits, so it wasn't run: " + " ".join(why))
    for name, why in {
        **report.get("identical_methods_shown_once", {}),
        **report.get("same_latest_weights_different_history", {}),
    }.items():
        st.caption(f"{name}: {why}.")
    chosen = report.get("securities_chosen")
    if chosen:
        st.markdown(f"**Securities chosen and why** ({rec['choice']})")
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
    for swap in report.get("swaps", []):
        st.markdown(f"🔁 **Swap ({swap['kind']})**: {swap['why']}")
    for idea in report.get("domicile_suggestions_information_not_tax_advice", []):
        st.caption(f"🏛 **Domicile (information, not tax advice):** {idea['why']}")
    if report.get("mode") == "improve":
        st.caption(
            f"Turnover limit {report['max_turnover']} of the portfolio: each month's move, and "
            "today's suggested change from your current weights, stay within it unless your "
            "limits need more."
        )
        for name, traded in report.get(
            "trades_more_than_the_turnover_limit_to_meet_the_limits", {}
        ).items():
            st.warning(
                f"**{name}** trades {traded} of the portfolio, above the "
                f"{report['max_turnover']} turnover limit, because meeting your limits needs it."
            )
    if len(run.latest_weights) > 1:
        st.markdown("**Latest weights of each method** (what it would hold now)")
        st.table(run.latest_weights.map(pct))
        if rec["choice"] != KEEP:
            plot(
                paired_bars_chart(
                    run.latest_weights.loc["Current"],
                    run.latest_weights.loc[rec["choice"]],
                    ("Current", rec["choice"]),
                    f"Current vs {rec['choice']}",
                ),
                width="stretch",
            )
    plot(equity_chart(run.equity, "Growth of $1, walk-forward after costs", reference="Benchmark"))
    if "Benchmark" in run.equity and rec["choice"] in run.equity:
        daily = run.equity[[rec["choice"], "Benchmark"]].pct_change().dropna()
        rel = relative_drawdown(daily[rec["choice"]], daily["Benchmark"]).rename(rec["choice"])
        plot(
            lines_chart(
                rel.to_frame(),
                f"{rec['choice']}: drawdown relative to the benchmark",
                "Relative drawdown",
            ),
            width="stretch",
        )
    st.info(report["caveats"])
    candidates = {"Current": run.latest_weights.loc["Current"]}
    if rec["choice"] != KEEP:
        candidates[rec["choice"]] = run.latest_weights.loc[rec["choice"]]
    render_suitability(candidates, "opt")

    if ai_ok and st.button("Explain the recommendation (AI memo)", key="ws_opt_memo"):
        memo_analyst = agent.Analyst(
            shared_context(),
            agent.make_client(),
            model=A["model"],
            max_tokens=A["max_tokens"],
            effort=A["effort"],
            max_tool_rounds=A["max_tool_rounds"],
        )
        result["memo"] = ask_analyst(
            memo_analyst,
            "Please explain the diagnosis of my portfolio and the recommendation from the "
            "Optimise section: why this method fits my goal, the trade-offs, and how much to "
            "trust the results.",
        )
    memo = result.get("memo")
    if memo:
        if memo["role"] == "error":
            st.error(memo["text"])
        else:
            with st.chat_message("assistant"):
                render_reply(memo["reply"])


def render_build(ctx, ai_ok: bool) -> None:
    """ "Build a portfolio for me": screen the universe, build candidates, recommend by rule."""
    st.markdown("**Build a portfolio for me**")
    st.caption(
        f"From a curated universe of {len(UNIVERSE_TAGS)} liquid securities (ETFs, GCC blue chips "
        "when the IPS asks for GCC exposure, bonds and sukuk, gold, cash), screened by the "
        "client's IPS. Candidates of 8-15 holdings aim for the highest information ratio "
        "against the benchmark, with volatility at most the benchmark's and a tracking-error "
        "limit; with a low-volatility, momentum or quality tilt; a diversified IPS portfolio; "
        "and a benchmark tracker. Judged walk-forward after costs."
    )
    if bench is None or bench.is_cash:
        st.warning("Choose a market benchmark (not cash) in the header: Build aims to beat it.")
        return
    c1, c2 = st.columns(2)
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
    c2.caption(
        f"At most {pct(max_weight)} in one holding (slider above); the client's IPS limits "
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
    )
    limits = OptimisationSpec(max_weight=max_weight)
    profile = client_ui.current_profile()
    client_key = None if profile is None else json.dumps(profile.to_dict(), sort_keys=True)
    run_key = ("build", settings_key, bench_name, te, max_weight, client_key, portfolio_value())
    if st.button("Build the candidate portfolios", key="ws_run_build", type="primary"):
        with st.spinner("Loading the universe, then choosing securities month by month…"):
            uni = universe_data(True)
            try:
                report, run = run_build(
                    ctx, uni, limits, my_weights, settings,
                    bench_choices.get(bench_name, {}).get("weights", {}),
                )  # fmt: skip
            except ToolError as e:
                st.error(str(e))
                return
        ctx.optimisation = report
        choice = run.recommendation.choice
        if choice in run.latest_weights.index:
            w = run.latest_weights.loc[choice]
            fails = run.recommendation.status == "closest_fails"
            ctx.proposed = {
                "name": f"{choice} ({'closest, fails the IPS' if fails else 'recommended'})",
                "weights": w[w > 0].to_dict(),
                "source": "Build (closest; fails the IPS)" if fails else "Build recommendation",
            }
        st.session_state.ws_opt = {
            "report": report, "run": run, "memo": None, "key": run_key, "universe": uni,
        }  # fmt: skip
    result = st.session_state.get("ws_opt")
    if result and result["key"] == run_key:
        with st.expander(
            f"Universe after the IPS screen: {result['report']['universe']['eligible']} eligible"
        ):
            out = result["report"]["universe"]["excluded"]
            if out:
                st.table(pd.DataFrame({"Left out because": out}))
        _results_section(result, ai_ok)
    elif result and result["key"][0] == "build":
        st.caption("The limits, client or benchmark changed: build the candidates again.")


def render_optimise() -> None:
    """Diagnose -> goal and limits -> candidate methods walk-forward -> recommendation -> memo."""
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
    if not portfolio_ok or len(my_weights) < 2:
        st.info("Add at least two holdings with price history on the Holdings tab.")
        return
    if not enough_history:
        st.warning(SHORT_SAMPLE)
        return
    ctx = shared_context()
    ctx.benchmark = bench
    holding_groups = {t: g for t, g in ws.groups_of_holdings().items() if t in my_weights}
    ai_ok = agent.has_credentials()
    mode = st.radio(
        "What should Optimise do?",
        [IMPROVE, BUILD],
        key="opt_mode",
        horizontal=True,
        help="Improve: start from your holdings, re-weight within a turnover limit, then try a "
        "few explained swaps. Build: choose 8-15 securities from a curated universe that aim to "
        "beat the benchmark on a risk-adjusted basis.",
    )
    if mode == BUILD:
        render_build(ctx, ai_ok)
        return
    _diagnosis_section(ctx)
    chosen = _goal_section(holding_groups, ai_ok)
    if chosen is None:
        return
    goal, limits, target = chosen
    if goal == "beat_benchmark_less_risk" and bench is None:
        st.warning("Choose a benchmark with data at the top of the Portfolio tab.")
        return
    st.markdown("**3. Candidate methods**")
    st.caption(
        "Max Sharpe, min variance, risk parity, max diversification and target volatility, each "
        "within your limits, judged on how they did out of sample after costs."
    )
    c1, c2, c3 = st.columns(3)
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
        "Also try swaps from the curated universe",
        value=True,
        key="opt_swaps",
        help="Sell a near-duplicate holding, or add a missing bond / sukuk diversifier: rules "
        "applied each month to the data available then.",
    )
    max_swaps = c3.number_input(
        "Max swaps", 1, 5, int(LAB.get("max_swaps", 3)), 1, key="opt_max_swaps"
    )
    profile = client_ui.current_profile()
    client_key = None if profile is None else json.dumps(profile.to_dict(), sort_keys=True)
    run_key = (
        settings_key, bench_name, goal, target, repr(limits), client_key, portfolio_value(),
        max_turnover, use_swaps, max_swaps,
    )  # fmt: skip
    if st.button("Run the candidate methods", key="ws_run_opt", type="primary"):
        with st.spinner("Re-optimising each method month by month and backtesting…"):
            uni = universe_data(True) if use_swaps else None
            report, run = run_candidates(
                ctx,
                goal,
                limits,
                my_weights,
                target,
                max_turnover=max_turnover / 100,
                universe=uni,
                max_swaps=int(max_swaps),
            )
        ctx.optimisation = report
        choice = run.recommendation.choice
        if choice != KEEP:  # charts on the Analyst page compare current with this
            w = run.latest_weights.loc[choice]
            fails = run.recommendation.status == "closest_fails"
            ctx.proposed = {
                "name": f"{choice} ({'closest, fails the IPS' if fails else 'recommended'})",
                "weights": w[w > 0].to_dict(),
                "source": "Optimise (closest; fails the IPS)"
                if fails
                else "Optimise recommendation",
            }
        st.session_state.ws_opt = {
            "report": report, "run": run, "memo": None, "key": run_key, "universe": uni,
        }  # fmt: skip
    result = st.session_state.get("ws_opt")
    if result and result["key"] == run_key:
        _results_section(result, ai_ok)
    elif result:
        st.caption("Your goal, limits or benchmark changed: run the candidate methods again.")


def time_key() -> int:
    """A fresh widget-key suffix so a new draft resets the spec form."""
    st.session_state["ws_spec_counter"] = st.session_state.get("ws_spec_counter", 0) + 1
    return st.session_state["ws_spec_counter"]


bench_choices = {
    **CFG["benchmark_choices"],
    **{
        f"{t} (an asset in your data)": {
            "weights": {t: 1.0},
            "represents": f"{NAMES.get(t, t)}, converted to USD.",
        }
        for t in data.prices.columns
    },
}
bench_name = head_bench.selectbox(
    "Benchmark",
    list(bench_choices),
    label_visibility="collapsed",
    key="ws_benchmark",
    help="What your portfolio is compared with across the dashboard. Total-return ETF "
    "proxies, so dividends count as they do in your portfolio.",
)
st.caption(f"Benchmark **{bench_name}**: {bench_choices[bench_name]['represents']}")
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
    st.subheader("Analyse the current portfolio")
    kpi_slot = st.container()  # filled once the KPI helpers are defined below
    sec_analyse, sec_bench, tab_assets, tab_mc, tab_regime = st.tabs(
        ["Performance & backtest", "Benchmark", "Assets", "Monte Carlo", "Regime"],
        key="analyse_tab",
        on_change="rerun",
    )
with step_optimise:
    sec_opt, tab_frontier, tab_backtest = st.tabs(
        ["Optimise", "Frontier", "Strategy backtests"], key="optimise_tab", on_change="rerun"
    )
with step_report:
    sec_rebal, sec_reports, tab_agent = st.tabs(
        layout.REPORT_TABS, key=layout.REPORT_KEY, on_change="rerun"
    )

with tab_client:
    client_ui.render_client_tab(
        CFG,
        list(my_weights),
        ws.groups_of_holdings(),
        get_attributes(tuple(my_weights)),
        agent.has_credentials(),
        agent.make_client,
        A["model"],
        cash_like_now() if portfolio_ok else [],
        holding_regions() if portfolio_ok else {},
        holding_products(tuple(my_weights)),
    )

with tab_assets:
    st.subheader("Risk and return by asset")
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
with tab_frontier:
    st.subheader("Efficient frontier (full sample, in hindsight)")
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
    if kpi is not None:
        st.caption(
            f"**Backtest period {kpi.attrs['period']}.** Return, volatility, Sharpe and max "
            "drawdown: your current weights rebalanced monthly after costs; deltas against "
            f"{bench_name}."
        )


with kpi_slot:
    render_portfolio_kpis()

with step_portfolio:
    st.subheader("Current portfolio")
    sec_hold, sec_screen = st.tabs(["Holdings", "Screen"], key="portfolio_tab", on_change="rerun")
    holding_names = {t: getattr(get_quote(t), "name", None) or NAMES.get(t, "") for t in my_weights}

    with sec_hold:
        ws.render_holdings(
            find_securities, get_quote, CFG["groups"], NAMES, HOLDING_PRESETS, price_data_ok
        )
        render_sample_limiters()
        rates = bt.cost_rates(list(my_weights), cost_bps, COST_BY_SUFFIX)
        local = rates[rates != cost_bps]
        if len(local):
            st.caption(
                "Trading costs used in backtests and trades: "
                + ", ".join(f"{t} {r:g} bps" for t, r in local.items())
                + f"; US listings {cost_bps:g} bps (⚙ Settings). Local-market rates are "
                "judgement estimates (config trading_costs)."
            )
        for problem in holding_problems:
            st.warning(problem)
        if holding_note:
            st.caption(holding_note)
        if missing_holdings:
            st.error(
                "No price history for " + ", ".join(missing_holdings) + ". Remove these "
                "holdings or check the symbols (see the warning at the top of the page)."
            )
        if my_weights:
            ws.render_allocation(pd.Series(my_weights), ws.groups_of_holdings())
        st.caption(
            "Prices are the latest quotes from Yahoo Finance (refreshed every 15 minutes), in "
            "each security's currency and in USD. The analysis uses daily history in USD."
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
            st.markdown("**Share of money vs share of risk**")
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
            plot(risk_contribution_chart(w, rc), width="stretch")
            st.caption(
                "Share of risk is each holding's contribution to the portfolio's volatility (full "
                "sample). A holding whose share of risk is much larger than its weight dominates "
                "how the portfolio moves."
            )
            mine = run_my_portfolio(my_key, loaded, start, end, day, cost_bps, lookback)
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
            st.markdown("**Backtest after costs, compared with the strategies and 60/40**")
            show_metrics_table(table, D["var_level"])
            st.caption(
                f"Your weights are rebalanced monthly back to target with {cost_bps:g} bps costs, "
                f"on the same dates as the strategies ({mine.equity.index[0]:%d %b %Y} – "
                f"{mine.equity.index[-1]:%d %b %Y}). The strategies choose among your holdings, "
                f"at most {pct(max_weight)} each."
            )
            equity = pd.DataFrame({n: r.equity for n, r in results_mine.items()})
            plot(equity_chart(equity, reference=BENCHMARK_NAMES), width="stretch")
            plot(drawdown_chart(drawdown_series(rets), reference=BENCHMARK_NAMES), width="stretch")
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
            st.markdown(
                f"**Your portfolio vs {bench_name}** · backtest period "
                f"{period_label(port.index.intersection(bench.returns.index))}"
            )
            st.table(ws.benchmark_comparison(port, bench.returns, rf, periods, bench.is_cash))
            dates = port.index.intersection(bench.returns.index)
            st.caption(
                f"{dates[0]:%d %b %Y} – {dates[-1]:%d %b %Y}. Your portfolio is held at today's "
                f"weights, rebalanced monthly with {cost_bps:g} bps costs (as on Analyse & "
                "backtest); the benchmark is an index, so it pays no costs. Alpha: annual return "
                "not explained by the benchmark exposure (beta). Tracking error: how far your "
                "returns stray from the benchmark's. Information ratio: extra return per unit of "
                "tracking error. Up/down capture: your average return on the benchmark's up "
                "(down) days as a share of its own; below 100% on down days means you fell less."
            )
            for note in bench.notes:
                st.info(note)
            p, b = port.loc[dates], bench.returns.loc[dates]
            growth = pd.DataFrame(
                {"Your portfolio": (1 + p).cumprod(), bench_name: (1 + b).cumprod()}
            )
            plot(equity_chart(growth, reference=bench_name))
            plot(
                drawdown_chart(
                    pd.DataFrame(
                        {"Your portfolio relative to the benchmark": relative_drawdown(p, b)}
                    ),
                    title="Relative drawdown: how far you fell behind the benchmark",
                ),
                width="stretch",
            )
            st.caption(
                "Relative drawdown tracks your wealth divided by the benchmark's: 0% at your best "
                "point relative to it, -10% when you have fallen 10% behind that point."
            )

    with sec_opt:
        render_optimise()

    with sec_rebal:
        if not portfolio_ok:
            st.info("Add holdings with price history on the Holdings tab.")
        else:
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
            )
            if "ctx" in st.session_state:
                st.session_state.ctx.rebalance = rebalance

with tab_mc:
    st.subheader("Monte Carlo: where could today's weights go?")
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
            c4, c5, c6 = st.columns(3)
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
            real = st.toggle(
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
    st.subheader("Will next month be stressed?")
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

    st.markdown("**By fold, with two baselines**")
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
    st.table(style_columns(fold_table, {"Model ROC AUC": "{:.2f}"}))
    st.caption(
        "Vol rule = 'stressed if SPY's last-21-day volatility is already above its 80th "
        "percentile' (no model). Always-calm = never flag stress."
    )

EXAMPLE_PROFILES = {
    "Dubai, moderate, 10 years": "I'm 35, based in Dubai, investing in USD for 10 years with "
    "moderate risk tolerance. I'd like to end with at least 1.5x my money.",
    "Riyadh, conservative, 5 years": "I'm 58 and live in Riyadh. I want to preserve capital over "
    "5 years with low risk; avoiding big drawdowns matters more than return.",
    "Doha, aggressive, 20 years": "I'm 27 in Doha, saving for 20 years, comfortable with large "
    "swings. I'd like to reach 3x my money.",
}


def _use_example() -> None:
    choice = st.session_state.get("profile_example")
    if choice in EXAMPLE_PROFILES:
        st.session_state.about_me = EXAMPLE_PROFILES[choice]


def render_portfolio_overview() -> None:
    """KPI cards, holdings (weight, share of risk, volatility) and the allocation chart."""
    render_portfolio_kpis()
    if not portfolio_ok:
        st.info("Add holdings with price history on step 2 (Portfolio).")
        return
    w = pd.Series(my_weights)
    rets = all_returns[list(w.index)]
    table = summary_table(rets, rf=rf, periods_per_year=periods)
    holdings = pd.DataFrame(
        {
            "Name": [holding_names.get(t, "") for t in w.index],
            "Weight": w.map(pct),
            "Share of risk": risk_contributions(w, rets.cov()).map(pct),
            "Volatility (ann.)": table["ann_vol"].map(pct),
        },
        index=w.index,
    )
    left, right = st.columns([3, 2])
    left.table(holdings)
    with right:
        plot(allocation_chart(w, "Allocation by holding"), key="analyst_allocation")


def current_ports() -> dict[str, pd.Series]:
    """Today's weights and, after Optimise, the proposed portfolio."""
    ports = {"Current": pd.Series(my_weights)}
    proposed = st.session_state.get("ctx") and st.session_state.ctx.proposed
    if proposed:
        ports[proposed["name"]] = pd.Series(proposed["weights"], dtype=float)
    return ports


def latest_memo() -> tuple[str | None, list[str]]:
    """The memo for the report: the Optimise memo if one was written (it explains the
    recommendation the report shows), else the latest analyst reply. Text and flagged figures.
    """
    opt = st.session_state.get("ws_opt") or {}
    memo = opt.get("memo")
    if memo and memo.get("role") == "assistant":
        return memo["reply"].text, memo["reply"].unverified
    for turn in reversed(st.session_state.get("chat", [])):
        if turn.get("role") == "assistant":
            return turn["reply"].text, turn["reply"].unverified
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
    rec = recommendation_section(report, run.latest_weights)
    if rec["choice"] in run.equity and "Benchmark" in run.equity:
        daily = run.equity[[rec["choice"], "Benchmark"]].pct_change().dropna()
        rec["relative_drawdown"] = relative_drawdown(
            daily[rec["choice"]], daily["Benchmark"]
        ).rename(rec["choice"])
    value = portfolio_value()
    if rec["choice"] is None or rec["status"] == "closest_fails" or not value:
        return rec, None
    usd = st.session_state.get(ws.MODE) == ws.USD
    current = pd.Series(my_weights)
    values = (
        pd.Series({r["ticker"]: r["value"] for r in st.session_state[ws.ROWS]}).reindex(
            current.index
        )
        if usd
        else current * value
    )
    target = run.latest_weights.loc[rec["choice"]].fillna(0.0)
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
    st.markdown("**Client report**")
    a1, a2 = st.columns(2)
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
    if st.button("Prepare client report (PDF)", key="prepare_report"):
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
                latest = st.session_state.ws_opt["run"].latest_weights
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
        st.download_button(
            "Download client report",
            pdf,
            file_name=name,
            mime="application/pdf",
            key="download_report",
            type="primary",
        )
        st.caption(
            "Uses the Optimise recommendation (never a chat what-if) and the Optimise memo if "
            "one was written, else the latest analyst memo. Prepare it again after changes."
        )
        flags = st.session_state.get("client_report_flags")
        if flags:
            st.warning(
                "Adviser check before sending: the memo has figures no tool returned: "
                + ", ".join(flags)
            )


with tab_agent:
    st.subheader("AI analyst")
    render_portfolio_overview()
    st.caption(
        f"The analyst ({A['model']}) calls this dashboard's own functions as tools on your "
        "holdings, the ⚙ Settings and the benchmark, and writes a memo with charts. It never "
        "calculates: every number must come from a tool and cite it (others are flagged), and "
        "every chart is drawn by code from the finance engine; the AI only chooses which charts "
        "to show and where. Not investment advice."
    )
    if universe_issue:
        st.info(universe_issue)
    elif not agent.has_credentials():
        st.info(
            "No API key found. Locally, put `ANTHROPIC_API_KEY=...` in `.env` at the project "
            "root. On Streamlit Community Cloud, add it under *App settings → Secrets*."
        )
    else:
        if st.session_state.get("analyst_key") != settings_key:
            st.session_state.analyst = agent.Analyst(
                shared_context(),
                agent.make_client(),
                model=A["model"],
                max_tokens=A["max_tokens"],
                effort=A["effort"],
                max_tool_rounds=A["max_tool_rounds"],
            )
            st.session_state.analyst_key = settings_key
            st.session_state.chat = []
        shared_context()  # keep the benchmark and groups current for the tools

        with st.expander("About me (optional): age, horizon, risk tolerance, goal"):
            st.selectbox(
                "Fill from an example",
                ["—", *EXAMPLE_PROFILES],
                key="profile_example",
                on_change=_use_example,
            )
            st.text_area("About me", key="about_me", height=90)
        about = st.session_state.get("about_me", "").strip()
        holdings_text = ", ".join(f"{t} {pct(v)}" for t, v in my_weights.items())
        b1, b2, b3, b4 = st.columns([2, 2, 2, 1])
        actions = {
            "review": b1.button(
                "Review my portfolio",
                type="primary",
                key="write_portfolio_memo",
                disabled=not portfolio_ok,
            ),
            "optimise": b2.button(
                "Explain the Optimise recommendation",
                key="explain_optimise",
                disabled=not shared_context().optimisation,
                help=None
                if shared_context().optimisation
                else "Run the candidate methods on step 4 (Optimise) first.",
            ),
            "stress": b3.button(
                "Stress test my portfolio", key="stress_memo", disabled=not portfolio_ok
            ),
        }
        if b4.button("New conversation", key="new_conversation"):
            st.session_state.analyst_key = None
            if (shared_context().proposed or {}).get("name") == "What-if":
                shared_context().proposed = None
            st.rerun()
        questions = {
            "review": f"Please review my portfolio ({holdings_text}) against the benchmark, "
            "including where its risk comes from and a stress-test section."
            + (f" About me: {about}" if about else ""),
            "optimise": "Please explain the diagnosis of my portfolio and the recommendation "
            "from the Optimise section: why this method fits my goal, the trade-offs, and how "
            "much to trust the results.",
            "stress": "Stress test my portfolio against the benchmark in the historical "
            "episodes, with their dates, and explain what drove the results.",
        }

        def ask(question: str) -> None:
            st.session_state.chat.append({"role": "user", "text": question})
            st.session_state.chat.append(ask_analyst(st.session_state.analyst, question))

        for name, pressed in actions.items():
            if pressed:
                ask(questions[name])
                st.rerun()

        for n, turn in enumerate(st.session_state.chat):
            if turn["role"] == "user":
                st.chat_message("user").markdown(turn["text"])
            elif turn["role"] == "error":
                st.error(turn["text"])
            else:
                with st.chat_message("assistant"):
                    render_reply(turn["reply"], key=f"chat_{n}")

        follow_up = st.chat_input(
            "Ask a follow-up, e.g. 'What if I cut SPY to 30%?' or 'How did I do in 2022?'",
            key="follow_up",
        )
        if follow_up:
            ask(follow_up)
            st.rerun()

with step_suitability:
    st.subheader("Suitability")
    if not portfolio_ok:
        st.info("Add holdings with price history on step 2 (Portfolio) to check suitability.")
    else:
        render_suitability(current_ports(), "suitability", record=True)
        st.divider()
        render_domicile("suitability")

with sec_reports:
    st.subheader("IPS and client report")
    profile = client_ui.current_profile()
    if profile is None:
        st.info("Load or create a client on step 1 (Client) to issue the IPS.")
    else:
        client_ui.render_ips_downloads(profile, client_ips(), "rep")
        st.divider()
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
        f"{len(my_weights)} holdings" + (f" · USD {value:,.0f}" if value else ""),
        todo,
    )
    analyse = layout.StepStatus(
        portfolio_ok and not universe_issue,
        "analysis ready" if portfolio_ok and not universe_issue else "needs two or more holdings",
        [universe_issue] if universe_issue else [],
    )
    ran = bool(st.session_state.get("ws_opt"))
    optimise = layout.StepStatus(
        ran,
        "recommendation ready" if ran else "not run yet",
        [] if ran else ["choose a goal and run the candidate methods"],
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
        [] if prepared else ["prepare the client report (Reports tab)"],
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
with stepper:
    layout.render_stepper(statuses, st.session_state.get(layout.STEP_KEY))
profile_now = client_ui.current_profile()
ips_now = client_ips()
layout.render_header(
    head_chips,
    layout.HeaderInfo(
        client=(profile_now.name or profile_now.reference or "unnamed") if profile_now else None,
        risk_score=ips_now.risk.score if ips_now else None,
        value=f"USD {portfolio_value():,.0f}" if portfolio_value() else "not set",
        ips_status="–" if profile_now is None else "Draft" if ips_missing(profile_now) else "Ready",
        demo=bool(profile_now and profile_now.demo),
        seed=st.session_state.get("demo_seed") if profile_now and profile_now.demo else None,
    ),
)

st.divider()
st.caption(DISCLAIMER)
