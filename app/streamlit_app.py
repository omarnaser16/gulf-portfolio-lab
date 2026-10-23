"""Gulf Portfolio Lab dashboard.

Presentation only: every number on screen comes from a ``gpl`` function. The app picks
inputs, calls the engine, and draws the results. Run with::

    streamlit run app/streamlit_app.py
"""

import json
import os
from datetime import date

import anthropic
import numpy as np
import pandas as pd
import streamlit as st
import workstation as ws
from inputs import asset_groups, combine_tickers

from formatting import pct, show_metrics_table, style_columns, style_percent
from gpl import backtest as bt
from gpl import regime as rg
from gpl.agent import analyst as agent
from gpl.agent.spec import describe_goal, describe_limits, propose_spec, spec_from_draft
from gpl.agent.tools import AnalystContext, diagnose_portfolio, run_candidates
from gpl.data import (
    AssetData,
    BenchmarkData,
    DataUnavailable,
    clean_prices,
    get_prices,
    get_risk_free,
    latest_quote,
    load_assets,
    load_benchmark,
    load_universe,
    search_securities,
    to_returns,
)
from gpl.metrics import drawdown_series, relative_drawdown, summary_table
from gpl.montecarlo import (
    goal_probability,
    path_max_drawdowns,
    simulate_paths,
    terminal_stats,
    to_real,
)
from gpl.optimise import (
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
    weights_chart,
)
from gpl.recommend import GOALS, KEEP

CFG = load_universe()

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
    run = dict(rebalance=D["rebalance"], lookback=lookback, cost_bps=cost_bps)
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
    )


# -------------------------------------------------------------------------- presentation


def first_eight(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape[1] > len(SERIES_COLORS):
        st.caption(f"Charts show the first {len(SERIES_COLORS)} series; the table has all of them.")
    return frame.iloc[:, : len(SERIES_COLORS)]


st.set_page_config(page_title="Gulf Portfolio Lab", page_icon="📈", layout="wide")
st.title("Gulf Portfolio Lab")
st.caption(
    "Risk, optimisation, backtests, simulation and regime forecasts for GCC and global ETFs."
)

with st.sidebar:
    st.header("Data")
    period = st.radio(
        "Period",
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
    st.header("Backtest settings")
    cost_bps = st.number_input("Costs (bps on turnover)", 0.0, 100.0, float(D["cost_bps"]), 5.0)
    lookback = st.select_slider(
        "Estimation window (observations)", [126, 252, 504], value=D["lookback"]
    )
    st.header("Strategies")
    names = st.multiselect(
        "Show", STRATEGY_NAMES + BENCHMARK_NAMES, default=STRATEGY_NAMES + BENCHMARK_NAMES
    )
    use_regime = st.toggle(
        "Add regime-aware versions",
        key="use_regime",
        value=False,
        help="Switch to min variance when the model forecasts a stressed month.",
    )
    threshold = st.slider(
        "De-risk when P(stressed) ≥", 0.2, 0.9, float(R["threshold"]), 0.05, disabled=not use_regime
    )
    st.divider()
    st.caption(DISCLAIMER)

end = None if period == LATEST else date.fromisoformat(D["end"])
# Set on Portfolio › Optimise; read here so every tab uses the current value.
max_weight = float(st.session_state.get("max_weight", D["max_weight"]))
day = date.today() if end is None else None
ws.init_state(CFG["groups"])
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
        "Portfolio › Holdings to compare strategies."
    )
elif max_weight * len(universe) < 1:
    universe_issue = (
        f"A {pct(max_weight)} cap can't hold {len(universe)} holdings fully invested. Raise "
        f"'{CAP_SETTING}' on Portfolio › Optimise, or add holdings."
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
    f"({len(returns):,} observations, {periods} a year). Every result depends on this period."
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
        sun_thu = [t for t in data.history.index if "Sun" in data.history.loc[t, "trading_days"]]
        st.info(
            f"**Different trading calendars.** {', '.join(sun_thu)} trade Sunday to Thursday, "
            "while US-listed assets trade Monday to Friday. The analysis uses only the "
            f"{data.alignment['kept']:,} dates on which every selected market traded (about "
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
    "date, or choose a shorter estimation window in the sidebar."
)

A = CFG["agent"]
settings_key = (tickers, loaded, start, end, day, max_weight, cost_bps, lookback, my_key)


def shared_context() -> AnalystContext:
    """One agent context per settings, shared by the Analyst tab and the optimiser."""
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
                "lookback": lookback,
                "periods_per_year": periods,
                "rebalance": D["rebalance"],
                "var_level": D["var_level"],
                "regime": R,
                "montecarlo": MC,
            },
            universe=universe,
            user_portfolio=my_weights if portfolio_ok else None,
            asset_notes=data.history,
        )
        ctx._cache["p_stressed"] = regime_data["p_stressed"]  # already computed
        st.session_state.ctx = ctx
        st.session_state.ctx_key = settings_key
        st.session_state.pop("ws_opt", None)  # an optimisation belongs to the old holdings
    return st.session_state.ctx


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


def render_reply(reply: agent.AnalystReply) -> None:
    """Memo text, unverified-number warning, cost line and audit trail."""
    st.markdown(reply.text)
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
}


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
    st.session_state.pop("ws_opt", None)


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
    st.plotly_chart(
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
            help=None if ai_ok else "Needs an API key (see the Analyst tab).",
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

    draft = st.session_state.setdefault("ws_draft", dict(BLANK_DRAFT))
    v = st.session_state.setdefault("ws_draft_v", 0)

    def as_float(value):  # the model writes integers (25); number inputs need floats
        return None if value is None else float(value)

    goal = st.radio(
        "Goal",
        list(GOALS),
        format_func=GOALS.get,
        index=list(GOALS).index(draft.get("goal") or "max_risk_adjusted"),
        key=f"ws_goal_{v}",
    )
    edited = {**draft, "goal": goal, "max_weight_pct": max_weight * 100}
    if goal == "target_return":
        edited["min_return_pct"] = st.number_input(
            "Target return (% a year)",
            -50.0,
            100.0,
            as_float(draft.get("min_return_pct")),
            0.5,
            key=f"ws_target_ret_{v}",
            placeholder="e.g. 8",
        )
    elif goal == "target_volatility":
        edited["max_volatility_pct"] = st.number_input(
            "Target volatility (% a year)",
            0.5,
            100.0,
            as_float(draft.get("max_volatility_pct")),
            0.5,
            key=f"ws_target_vol_{v}",
            placeholder="e.g. 10",
        )
    with st.expander("More limits (optional)"):
        n1, n2 = st.columns(2)
        if goal != "target_volatility":
            edited["max_volatility_pct"] = n1.number_input(
                "Volatility cap (% a year)",
                0.5,
                100.0,
                as_float(draft.get("max_volatility_pct")),
                0.5,
                key=f"ws_maxvol_{v}",
                placeholder="no cap",
            )
        if goal != "target_return":
            edited["min_return_pct"] = n2.number_input(
                "Return floor (% a year, historical)",
                -50.0,
                100.0,
                as_float(draft.get("min_return_pct")),
                0.5,
                key=f"ws_minret_{v}",
                placeholder="no floor",
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
    edited["group_limits"] = _limits_rows(group_table, "group")
    edited["asset_limits"] = _limits_rows(asset_table, "ticker")
    try:
        goal, limits, target = spec_from_draft(edited, holding_groups)
    except ValueError as e:
        st.warning(str(e))
        return None
    st.markdown(
        f"**Goal:** {describe_goal(goal, target)}  \n**Limits for every method:** "
        + "; ".join(describe_limits(limits))
        + f"  \n**Benchmark:** {bench_name}"
    )
    return goal, limits, target


def _results_section(result: dict, ai_ok: bool) -> None:
    report, run = result["report"], result["run"]
    rec = report["recommendation"]
    st.markdown("**4. Recommendation**")
    reasons = "\n".join(f"- {r}" for r in rec["reasons"])
    if rec["choice"] == KEEP:
        st.info(f"**Keep your current portfolio.**\n\n{reasons}")
    else:
        st.success(f"**Recommended method: {rec['choice']}**\n\n{reasons}")
    st.caption(f"The rule for this goal: {rec['rule']}")

    rows = pd.DataFrame(report["walk_forward_after_costs"]).T
    rows = rows.reindex(columns=list(CANDIDATE_TABLE_COLUMNS)).rename(
        columns=CANDIDATE_TABLE_COLUMNS
    )
    rows = rows.astype(object).fillna("–")  # columns a row doesn't have (e.g. Benchmark)
    rows.index = [("★ " if n == rec["choice"] else "") + n for n in rows.index]
    st.markdown(f"**Walk-forward results after costs ({report['backtest_period']})**")
    st.table(rows)
    st.caption(
        "Each method is re-optimised every month on the trailing window only (Ledoit-Wolf "
        "covariance), within your limits, then held to the next month, after trading costs. "
        "'Held up?': it kept at least half of its in-sample Sharpe ratio out of sample. "
        "'Months limits unmet': months when the limits couldn't be met with the data then "
        "available, so the previous weights were kept (no limit was broken). The "
        "target-volatility method aims at "
        f"{report['volatility_target_for_target_volatility_candidate']} a year."
    )
    for name, why in report["infeasible_candidates"].items():
        st.warning(f"**{name}** can't meet your limits, so it wasn't run: " + " ".join(why))
    if len(run.latest_weights) > 1:
        st.markdown("**Latest weights of each method** (what it would hold now)")
        st.table(run.latest_weights.map(pct))
        if rec["choice"] != KEEP:
            st.plotly_chart(
                paired_bars_chart(
                    run.latest_weights.loc["Current"],
                    run.latest_weights.loc[rec["choice"]],
                    ("Current", rec["choice"]),
                    f"Current vs {rec['choice']}",
                ),
                width="stretch",
            )
    st.plotly_chart(equity_chart(run.equity, "Growth of $1, walk-forward after costs"))
    st.info(report["caveats"])

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
        help="The most any one holding may weigh in an optimised portfolio. It also applies to "
        "the strategies on the Frontier, Backtest and Monte Carlo tabs.",
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
    run_key = (settings_key, bench_name, goal, target, repr(limits))
    if st.button("Run the candidate methods", key="ws_run_opt", type="primary"):
        with st.spinner("Re-optimising each method month by month and backtesting…"):
            report, run = run_candidates(ctx, goal, limits, my_weights, target)
        ctx.optimisation = report
        st.session_state.ws_opt = {"report": report, "run": run, "memo": None, "key": run_key}
    result = st.session_state.get("ws_opt")
    if result and result["key"] == run_key:
        _results_section(result, ai_ok)
    elif result:
        st.caption("Your goal, limits or benchmark changed: run the candidate methods again.")


def time_key() -> int:
    """A fresh widget-key suffix so a new draft resets the spec form."""
    st.session_state["ws_spec_counter"] = st.session_state.get("ws_spec_counter", 0) + 1
    return st.session_state["ws_spec_counter"]


tab_mine, tab_assets, tab_frontier, tab_backtest, tab_mc, tab_regime, tab_agent = st.tabs(
    ["Portfolio", "Assets", "Frontier", "Backtest", "Monte Carlo", "Regime", "Analyst"]
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
            f"('{CAP_SETTING}' on Portfolio › Optimise)."
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
        st.plotly_chart(
            frontier_chart(efficient_frontier(mu, cov, 40, max_weight), points), width="stretch"
        )
        st.dataframe(style_percent(pd.DataFrame(portfolios).T), width="stretch")
with tab_backtest:
    if universe_issue:
        st.info(universe_issue)
    else:
        if not names:
            st.info("Pick at least one strategy in the sidebar.")
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
                f"('{CAP_SETTING}' on Portfolio › Optimise). Benchmarks are "
                f"fixed by definition: {SIXTY_FORTY} holds "
                + " and ".join(f"{pct(w)} {t}" for t, w in CFG["benchmarks"]["sixty_forty"].items())
                + f"; SPY holds {pct(CFG['benchmarks']['spy']['SPY'])}."
            )
            equity = first_eight(pd.DataFrame({n: r.equity for n, r in results.items()}))
            st.plotly_chart(equity_chart(equity), width="stretch")
            st.plotly_chart(
                drawdown_chart(drawdown_series(strat_returns[equity.columns])), width="stretch"
            )
            chosen = st.selectbox("Weights over time for", list(results))
            st.plotly_chart(
                weights_chart(
                    results[chosen].weights,
                    groups=groups,
                    title=f"{chosen}: target weights by group",
                ),
                width="stretch",
            )
with tab_mine:
    st.subheader("My portfolio")
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
    bench_name = st.selectbox(
        "Benchmark",
        list(bench_choices),
        key="ws_benchmark",
        help="What your portfolio is compared with on the Screen, Benchmark and Optimise tabs. "
        "Total-return ETF proxies, so dividends count as they do in your portfolio.",
    )
    st.caption(f"**{bench_name}:** {bench_choices[bench_name]['represents']}")
    try:
        bench = get_benchmark(
            bench_name, bench_choices[bench_name], all_returns.index, loaded, start, end, day
        )
    except (ValueError, DataUnavailable) as e:
        bench = None
        st.warning(f"Couldn't load the benchmark: {e} Choose another one.")
    sec_hold, sec_screen, sec_analyse, sec_bench, sec_opt = st.tabs(
        ["Holdings", "Screen", "Analyse & backtest", "Benchmark", "Optimise"]
    )
    holding_names = {t: getattr(get_quote(t), "name", None) or NAMES.get(t, "") for t in my_weights}

    with sec_hold:
        ws.render_holdings(find_securities, get_quote, CFG["groups"], NAMES, HOLDING_PRESETS)
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
            st.plotly_chart(risk_contribution_chart(w, rc), width="stretch")
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
            st.plotly_chart(equity_chart(equity), width="stretch")
            st.plotly_chart(drawdown_chart(drawdown_series(rets)), width="stretch")
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
            st.markdown(f"**Your portfolio vs {bench_name}**")
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
            st.plotly_chart(equity_chart(growth), width="stretch")
            st.plotly_chart(
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

with tab_mc:
    st.subheader("Monte Carlo: where could today's weights go?")
    if universe_issue:
        st.info(universe_issue)
    else:
        if not names:
            st.info("Pick at least one strategy in the sidebar.")
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
            st.plotly_chart(
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
    st.plotly_chart(
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

with tab_agent:
    st.subheader("AI analyst")
    st.caption(
        f"Describe an investor profile. The analyst ({A['model']}) calls this dashboard's own "
        "functions as tools on your holdings and the sidebar settings, and writes a memo. It is "
        "not allowed to "
        "calculate: every number must be copied from a tool result and cite the tool, and any "
        "number that no tool returned is flagged below the answer. Not investment advice."
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

        examples = {
            "Dubai, moderate, 10 years": "I'm 35, based in Dubai, investing in USD for 10 years "
            "with moderate risk tolerance. I'd like to end with at least 1.5x my money.",
            "Riyadh, conservative, 5 years": "I'm 58 and live in Riyadh. I want to preserve "
            "capital over 5 years with low risk; avoiding big drawdowns matters more than return.",
            "Doha, aggressive, 20 years": "I'm 27 in Doha, saving for 20 years, comfortable with "
            "large swings. I'd like to reach 3x my money.",
        }
        if not st.session_state.chat:
            choice = st.selectbox("Start from an example", list(examples), key="profile_example")
            profile = st.text_area("Investor profile", examples[choice], height=100, key="profile")
            b1, b2 = st.columns(2)
            start_memo = b1.button("Write memo", type="primary", key="write_memo")
            portfolio_memo = b2.button(
                "Write memo about my portfolio",
                key="write_portfolio_memo",
                disabled=not portfolio_ok,
                help="Uses the holdings on the Portfolio tab."
                if portfolio_ok
                else "Enter valid holdings on the Portfolio tab first.",
            )
        else:
            profile, start_memo, portfolio_memo = None, False, False
            if st.button("New conversation", key="new_conversation"):
                st.session_state.analyst_key = None
                st.rerun()

        def ask(question: str) -> None:
            st.session_state.chat.append({"role": "user", "text": question})
            st.session_state.chat.append(ask_analyst(st.session_state.analyst, question))

        if start_memo and profile:
            ask(profile)
            st.rerun()  # redraw with the conversation instead of the profile form
        if portfolio_memo:
            holdings = ", ".join(f"{t} {pct(v)}" for t, v in my_weights.items())
            ask(
                f"Please review my portfolio ({holdings}), entered on the Portfolio tab. "
                f"About me: {profile}"
            )
            st.rerun()

        for turn in st.session_state.chat:
            if turn["role"] == "user":
                st.chat_message("user").markdown(turn["text"])
            elif turn["role"] == "error":
                st.error(turn["text"])
            else:
                with st.chat_message("assistant"):
                    render_reply(turn["reply"])

        if st.session_state.chat:
            follow_up = st.chat_input(
                "Ask a follow-up (e.g. 'What if I were more conservative?')", key="follow_up"
            )
            if follow_up:
                ask(follow_up)
                st.rerun()
