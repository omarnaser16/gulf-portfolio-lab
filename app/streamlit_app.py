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
from inputs import asset_groups, combine_tickers, parse_tickers

from formatting import pct, show_metrics_table, style_columns, style_percent
from gpl import backtest as bt
from gpl import regime as rg
from gpl.agent import analyst as agent
from gpl.agent.spec import describe_spec, propose_spec, spec_from_draft
from gpl.agent.tools import AnalystContext, run_optimisation
from gpl.data import (
    AssetData,
    DataUnavailable,
    clean_prices,
    get_prices,
    get_risk_free,
    latest_quote,
    load_assets,
    load_universe,
    search_securities,
    to_returns,
)
from gpl.metrics import drawdown_series, summary_table
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
    st.header("Assets")
    chosen_presets = st.multiselect(
        "Quick picks", list(PRESETS), default=["Default ETFs"], key="presets"
    )
    typed = st.text_input(
        "Add any Yahoo Finance tickers",
        key="custom_tickers",
        placeholder="e.g. AAPL, 7010.SR, VOD.L",
        help="Comma-separated Yahoo Finance symbols: 2222.SR (Saudi), EMAAR.AE (Dubai), "
        "QNBK.QA (Qatar), NBK.KW (Kuwait), VOD.L (London). Non-USD assets are converted to USD.",
    )
    requested = combine_tickers([PRESETS[p] for p in chosen_presets], parse_tickers(typed))
    st.caption(
        "Selected: "
        + (", ".join(f"{t} ({NAMES[t]})" if t in NAMES else t for t in requested) or "none")
    )
    st.header("Portfolio rules")
    max_weight = st.slider(
        "Max weight per asset", 0.10, 1.00, float(D["max_weight"]), 0.05, key="max_weight"
    )
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
day = date.today() if end is None else None
ws.init_state(CFG["groups"])
my_weights, holding_problems, holding_note = ws.current_weights()
# Benchmarks and the user's holdings are always loaded, on the same dates as the universe.
loaded = tuple(combine_tickers([requested, list(my_weights)], ["SPY", "AGG"]))
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

bad = {
    t: why
    for t, why in data.invalid.items()
    if t in requested or t in my_weights or t in ("SPY", "AGG")
}
if bad:
    st.warning(
        "Couldn't load "
        + "; ".join(f"**{t}** ({why.lower()})" for t, why in bad.items())
        + ". Check the symbol on finance.yahoo.com. Some exchanges, such as Abu Dhabi (ADX), "
        "are not covered by Yahoo Finance."
    )
universe = [t for t in requested if t in data.prices.columns]
if len(universe) < 2:
    st.info("Pick at least two assets with data (quick picks or typed tickers).")
    st.stop()
if max_weight * len(universe) < 1:
    st.error(
        f"A {pct(max_weight)} cap cannot hold {len(universe)} assets fully invested. Raise the cap."
    )
    st.stop()

tickers = tuple(universe)
periods = data.periods_per_year
prices = data.prices[universe]
returns = to_returns(prices)
all_returns = to_returns(data.prices)
groups = asset_groups(universe, CFG["groups"], GROUP_LABELS)
missing_holdings = [t for t in my_weights if t not in data.prices.columns]
portfolio_ok = bool(my_weights) and not missing_holdings
my_key = tuple(sorted(my_weights.items()))
st.markdown(
    f"**Data as of {data.price_as_of:%d %b %Y}** (latest price) · loaded {data.loaded_at} · "
    f"{'latest data' if end is None else 'fixed sample used in the README'}"
)
st.caption(
    f"Sample: **{prices.index[0]:%d %b %Y} – {prices.index[-1]:%d %b %Y}** "
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
        f"The common sample starts on {prices.index[0]:%d %b %Y}, when the most recent asset "
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


OBJECTIVE_NAMES = {
    "max_sharpe": "Maximise Sharpe ratio",
    "min_variance": "Minimise volatility",
    "max_return": "Maximise return",
    "target_volatility": "Highest return for a volatility cap",
    "target_return": "Lowest volatility for a return floor",
}
BLANK_DRAFT = {
    "objective": "max_sharpe",
    "max_weight_pct": None,
    "asset_limits": [],
    "group_limits": [],
    "max_volatility_pct": None,
    "min_return_pct": None,
    "unmapped_requests": [],
    "summary": "",
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


def render_optimise() -> None:
    """Goals in plain English -> spec the user confirms -> optimiser -> comparison and memo."""
    if not portfolio_ok or len(my_weights) < 2:
        st.info("Add at least two holdings with price history on the Holdings tab.")
        return
    if not enough_history:
        st.warning(SHORT_SAMPLE)
        return
    holding_groups = {t: g for t, g in ws.groups_of_holdings().items() if t in my_weights}
    st.markdown("**1. Describe your goals**")
    goals = st.text_area(
        "Goals in plain English",
        key="ws_goals",
        placeholder="e.g. max 12% vol, at least 20% GCC, no single asset over 25%, maximise Sharpe",
        help="The AI only turns this into limits you can check; it never chooses weights.",
    )
    c1, c2 = st.columns(2)
    ai_ok = agent.has_credentials()
    if c1.button(
        "Interpret with AI",
        key="ws_interpret",
        type="primary",
        disabled=not ai_ok,  # not tied to the text: typed text only registers on blur
        help=None if ai_ok else "Needs an API key (see the Analyst tab).",
    ):
        if not goals.strip():
            st.warning("Describe your goals first, e.g. 'max 12% vol, at least 20% GCC'.")
        else:
            try:
                with st.spinner("Turning your goals into a specification…"):
                    proposal = propose_spec(
                        agent.make_client(), A["model"], goals, holding_groups, effort="low"
                    )
                st.session_state.ws_spec = {
                    "draft": proposal.draft,
                    "proposal": proposal,
                    "v": time_key(),
                }
                st.session_state.pop("ws_opt", None)
            except (ValueError, anthropic.APIError) as e:
                st.error(str(e))
    if c2.button("Set the limits myself", key="ws_manual"):
        st.session_state.ws_spec = {"draft": dict(BLANK_DRAFT), "proposal": None, "v": time_key()}
        st.session_state.pop("ws_opt", None)

    spec_state = st.session_state.get("ws_spec")
    if not spec_state:
        return
    draft, proposal, v = spec_state["draft"], spec_state["proposal"], spec_state["v"]

    def as_float(value):  # the model writes integers (25); number inputs need floats
        return None if value is None else float(value)

    st.markdown("**2. Check and edit the specification**")
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
    objective = st.selectbox(
        "Objective",
        list(OBJECTIVE_NAMES),
        format_func=OBJECTIVE_NAMES.get,
        index=list(OBJECTIVE_NAMES).index(draft.get("objective") or "max_sharpe"),
        key=f"ws_obj_{v}",
    )
    n1, n2, n3 = st.columns(3)
    max_w = n1.number_input(
        "Max per asset (%)",
        0.0,
        100.0,
        as_float(draft.get("max_weight_pct")),
        1.0,
        key=f"ws_maxw_{v}",
        placeholder="no limit",
    )
    max_vol = n2.number_input(
        "Max volatility (% a year)",
        0.0,
        200.0,
        as_float(draft.get("max_volatility_pct")),
        0.5,
        key=f"ws_maxvol_{v}",
        placeholder="no limit",
    )
    min_ret = n3.number_input(
        "Min return (% a year)",
        -100.0,
        200.0,
        as_float(draft.get("min_return_pct")),
        0.5,
        key=f"ws_minret_{v}",
        placeholder="no limit",
    )
    g1, g2 = st.columns(2)
    g1.caption("Group limits (blank = none)")
    groups_present = sorted(set(holding_groups.values()))
    group_table = g1.data_editor(
        _limits_table(draft.get("group_limits", []), "group", groups_present),
        key=f"ws_groups_{v}",
        width="stretch",
    )
    g2.caption("Asset limits (blank = none)")
    asset_table = g2.data_editor(
        _limits_table(draft.get("asset_limits", []), "ticker", list(my_weights)),
        key=f"ws_assets_{v}",
        width="stretch",
    )
    edited = {
        **draft,
        "objective": objective,
        "max_weight_pct": max_w,
        "max_volatility_pct": max_vol,
        "min_return_pct": min_ret,
        "group_limits": _limits_rows(group_table, "group"),
        "asset_limits": _limits_rows(asset_table, "ticker"),
    }
    try:
        spec = spec_from_draft(edited, holding_groups)
    except ValueError as e:
        st.error(str(e))
        return
    st.markdown(
        "You are about to optimise for:\n" + "\n".join(f"- {line}" for line in describe_spec(spec))
    )
    if st.button("Run optimisation", key="ws_run_opt", type="primary"):
        ctx = shared_context()
        with st.spinner("Optimising and backtesting…"):
            report, weights = run_optimisation(ctx, spec, my_weights)
        ctx.optimisation = report
        st.session_state.ws_opt = {"report": report, "weights": weights, "memo": None}

    result = st.session_state.get("ws_opt")
    if not result:
        return
    report, weights = result["report"], result["weights"]
    st.markdown("**3. Results**")
    if not report["feasible"]:
        st.error(
            "**These limits can't all be met.**\n"
            + "\n".join(f"- {r}" for r in report["reasons_if_not_feasible"])
            + "\n\nRelax one of them and run again. No weights are shown."
        )
    else:
        now = pd.Series(my_weights)
        table = pd.DataFrame(report["weights"]).T.rename(
            columns={"current": "Current", "optimised": "Optimised", "change": "Change"}
        )
        w1, w2 = st.columns([2, 3])
        w1.table(table)
        w2.plotly_chart(
            paired_bars_chart(
                now, weights, ("Current", "Optimised"), "Current vs optimised weights"
            ),
            width="stretch",
        )
        if report["binding_constraints"]:
            st.caption("Binding limits: " + "; ".join(report["binding_constraints"]) + ".")
        hist = pd.DataFrame(report["in_sample_historical"]).T.rename(
            columns={
                "expected_return": "Return (ann.)",
                "volatility": "Volatility (ann.)",
                "sharpe": "Sharpe",
                "max_drawdown": "Max drawdown",
            }
        )
        st.markdown(f"**Historical, in sample ({report['estimation_sample']})**")
        st.table(hist)
    if "backtests_after_costs" in report:
        bt_table = pd.DataFrame(report["backtests_after_costs"]).T.rename(
            columns={
                "cagr": "CAGR",
                "annual_volatility": "Volatility (ann.)",
                "sharpe": "Sharpe",
                "max_drawdown": "Max drawdown",
                "turnover_per_year": "Turnover / yr",
            }
        )
        st.markdown(f"**Backtests after costs ({report['backtest_period']})**")
        st.table(bt_table)
    for note in report["notes"]:
        st.caption(note)
    st.info(report["caveats"])

    if ai_ok and st.button("Explain the changes (AI memo)", key="ws_opt_memo"):
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
            "Please explain the optimisation I just ran on the Portfolio tab: what changes, the "
            "trade-offs, and how much to trust the results.",
        )
    memo = result.get("memo")
    if memo:
        if memo["role"] == "error":
            st.error(memo["text"])
        else:
            with st.chat_message("assistant"):
                render_reply(memo["reply"])


def time_key() -> int:
    """A fresh widget-key suffix so a new draft resets the spec form."""
    st.session_state["ws_spec_counter"] = st.session_state.get("ws_spec_counter", 0) + 1
    return st.session_state["ws_spec_counter"]


tab_mine, tab_assets, tab_frontier, tab_backtest, tab_mc, tab_regime, tab_agent = st.tabs(
    ["Portfolio", "Assets", "Frontier", "Backtest", "Monte Carlo", "Regime", "Analyst"]
)

with tab_assets:
    st.subheader("Risk and return by asset")
    show_metrics_table(
        summary_table(returns, rf=rf, level=D["var_level"], periods_per_year=periods).sort_values(
            "sharpe", ascending=False
        ),
        D["var_level"],
    )
    st.caption(
        f"Full sample above. Annualised over {periods} observations a year. VaR and expected "
        "shortfall: one observation (usually 1 day), "
        f"{D['var_level']:.0%} confidence, shown as positive losses."
    )

with tab_frontier:
    st.subheader("Efficient frontier (full sample, in hindsight)")
    st.caption(
        "Uses the whole sample's means and covariances, so nobody could have held these at "
        "the time. The backtest tab is the honest test."
    )
    st.caption(
        f"Every portfolio here is long-only with at most **{pct(max_weight)}** per asset (sidebar)."
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
        table = summary_table(strat_returns, rf=rf, level=D["var_level"], periods_per_year=periods)
        table["annual_turnover"] = pd.Series(
            {n: r.turnover.iloc[1:].mean() * 12 for n, r in results.items()}
        )
        first_day = next(iter(results.values())).equity.index[0]
        st.subheader("Strategies after costs")
        st.caption(
            f"Monthly rebalancing on a trailing {lookback}-observation window; new weights apply "
            f"from the next observation; {cost_bps:g} bps per unit of turnover."
        )
        show_metrics_table(table.sort_values("sharpe", ascending=False), D["var_level"])
        st.caption(
            f"**Backtest period: {first_day:%d %b %Y} – {prices.index[-1]:%d %b %Y}.** Each "
            f"strategy needs {lookback} observations of history before its first rebalance, so the "
            f"backtest starts {lookback} observations after the data does. Figures (even for SPY) "
            "therefore differ from the Assets tab, which uses the full sample."
        )
        st.caption(
            f"Strategies hold at most **{pct(max_weight)}** per asset (sidebar). Benchmarks are "
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
                results[chosen].weights, groups=groups, title=f"{chosen}: target weights by group"
            ),
            width="stretch",
        )

with tab_mine:
    st.subheader("My portfolio")
    sec_hold, sec_screen, sec_analyse, sec_opt = st.tabs(
        ["Holdings", "Screen", "Analyse & backtest", "Optimise"]
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
                ws.screen_table(all_returns, pd.Series(my_weights), rf, D["var_level"], periods),
                holding_names,
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
                tuple(STRATEGY_NAMES + [SIXTY_FORTY]),
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
                f"{mine.equity.index[-1]:%d %b %Y}). The strategies choose from the assets in the "
                "sidebar; your holdings can include others."
            )
            equity = pd.DataFrame({n: r.equity for n, r in results_mine.items()})
            st.plotly_chart(equity_chart(equity), width="stretch")
            st.plotly_chart(drawdown_chart(drawdown_series(rets)), width="stretch")
            st.caption(
                "Monte Carlo for your portfolio: choose 'My portfolio' on the Monte Carlo tab."
            )

    with sec_opt:
        render_optimise()

with tab_mc:
    st.subheader("Monte Carlo: where could today's weights go?")
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
            + (f", assuming a constant {pct(MC['inflation'])} inflation a year. " if real else ". ")
            + "Simulations hold today's weights fixed, **rebalance daily and ignore transaction "
            f"costs**, unlike the backtest (monthly rebalancing, {cost_bps:g} bps per unit of "
            f"turnover). They assume the future resembles {prices.index[0]:%b %Y} – "
            f"{prices.index[-1]:%b %Y}."
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
        "functions as tools, using the sidebar settings, and writes a memo. It is not allowed to "
        "calculate: every number must be copied from a tool result and cite the tool, and any "
        "number that no tool returned is flagged below the answer. Not investment advice."
    )
    if not agent.has_credentials():
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
