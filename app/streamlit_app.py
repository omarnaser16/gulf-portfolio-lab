"""Gulf Portfolio Lab dashboard.

Presentation only: every number on screen comes from a ``gpl`` function. The app picks
inputs, calls the engine, and draws the results. Run with::

    streamlit run app/streamlit_app.py
"""

from datetime import date

import pandas as pd
import streamlit as st

from gpl import backtest as bt
from gpl import regime as rg
from gpl.data import clean_prices, get_prices, get_risk_free, load_universe, to_returns
from gpl.metrics import drawdown_series, summary_table
from gpl.montecarlo import goal_probability, simulate_paths, terminal_stats
from gpl.optimise import (
    efficient_frontier,
    max_sharpe,
    min_variance,
    portfolio_performance,
    risk_parity,
)
from gpl.plots import (
    SERIES_COLORS,
    drawdown_chart,
    equity_chart,
    fan_chart,
    frontier_chart,
    regime_chart,
    weights_chart,
)

CFG = load_universe()
D, R = CFG["defaults"], CFG["regime"]
STRATEGY_NAMES = ["Equal weight", "Min variance", "Max Sharpe", "Risk parity"]
BENCHMARK_NAMES = ["60/40", "SPY"]
PERCENT_COLS = [
    "ann_return",
    "cagr",
    "ann_vol",
    "max_drawdown",
    "hist_var",
    "param_var",
    "expected_shortfall",
]
DISCLAIMER = (
    "Educational project, **not investment advice**. "
    "Past performance does not predict future results."
)


# ----------------------------------------------------------------------------- data layer
# Cached wrappers around gpl functions. Arguments are plain values so Streamlit can hash them.


@st.cache_data(show_spinner="Loading prices…")
def load_prices(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    return clean_prices(get_prices(list(tickers), start, end), max_ffill=D["max_ffill"])


@st.cache_data(show_spinner="Loading the risk-free rate…")
def load_rf(start: date, end: date) -> pd.Series:
    return get_risk_free(start, end)


@st.cache_data(show_spinner="Fitting the regime model month by month…")
def load_regime(end: date) -> dict:
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
    report = rg.evaluate_model(features, labels, model=R["model"], horizon=R["horizon"])
    return {"features": features, "labels": labels, "p_stressed": p_stressed, "report": report}


def _strategy(name: str, rf: pd.Series, max_weight: float) -> bt.Strategy:
    match name:
        case "Equal weight":
            return bt.equal_weight
        case "Min variance":
            return bt.min_variance_strategy(max_weight=max_weight)
        case "Max Sharpe":
            return bt.max_sharpe_strategy(rf=rf, max_weight=max_weight)
        case "Risk parity":
            return bt.risk_parity_strategy(max_weight=max_weight)
        case "60/40":
            return bt.fixed_weights(CFG["benchmarks"]["sixty_forty"])
        case "SPY":
            return bt.fixed_weights(CFG["benchmarks"]["spy"])
    raise ValueError(name)


@st.cache_data(show_spinner="Running backtests…")
def run_backtests(
    tickers: tuple[str, ...],
    start: date,
    end: date,
    names: tuple[str, ...],
    max_weight: float,
    cost_bps: float,
    lookback: int,
    regime_threshold: float | None,
) -> dict[str, bt.BacktestResult]:
    prices = load_prices(tickers, start, end)
    rf = load_rf(start, end)
    strategies = {name: _strategy(name, rf, max_weight) for name in names}
    if regime_threshold is not None:
        p = load_regime(end)["p_stressed"]
        defensive = bt.min_variance_strategy(max_weight=max_weight)
        for name in [n for n in names if n in STRATEGY_NAMES and n != "Min variance"]:
            strategies[f"{name} + regime"] = bt.regime_switch_strategy(
                strategies[name], defensive, p, regime_threshold
            )
    return {
        name: bt.run_backtest(
            prices, s, rebalance=D["rebalance"], lookback=lookback, cost_bps=cost_bps
        )
        for name, s in strategies.items()
    }


# -------------------------------------------------------------------------- presentation


def show_table(table: pd.DataFrame) -> None:
    config = {c: st.column_config.NumberColumn(format="percent") for c in PERCENT_COLS}
    config |= {
        "sharpe": st.column_config.NumberColumn(format="%.2f"),
        "sortino": st.column_config.NumberColumn(format="%.2f"),
        "annual_turnover": st.column_config.NumberColumn("turnover / yr", format="%.2f×"),
        "start": st.column_config.DateColumn(),
        "end": st.column_config.DateColumn(),
    }
    st.dataframe(table, column_config=config, width="stretch")


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
    st.header("Universe")
    assets = st.multiselect("Assets", CFG["assets"], default=CFG["assets"])
    start = st.date_input("Start", date.fromisoformat(D["start"]), min_value=date(2005, 1, 1))
    end = st.date_input("End", date.fromisoformat(D["end"]), min_value=date(2006, 1, 1))
    st.header("Portfolio rules")
    max_weight = st.slider(
        "Max weight per asset", 0.10, 1.00, float(D["max_weight"]), 0.05, key="max_weight"
    )
    cost_bps = st.number_input("Costs (bps on turnover)", 0.0, 100.0, float(D["cost_bps"]), 5.0)
    lookback = st.select_slider("Estimation window (days)", [126, 252, 504], value=D["lookback"])
    st.header("Strategies")
    benchmarks_ok = {"SPY", "AGG"} <= set(assets)
    names = st.multiselect(
        "Show",
        STRATEGY_NAMES + (BENCHMARK_NAMES if benchmarks_ok else []),
        default=STRATEGY_NAMES + (BENCHMARK_NAMES if benchmarks_ok else []),
    )
    if not benchmarks_ok:
        st.caption("Benchmarks need SPY and AGG in the universe.")
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

if len(assets) < 2:
    st.info("Pick at least two assets.")
    st.stop()
if max_weight * len(assets) < 1:
    st.error(
        f"A {max_weight:.0%} cap cannot hold {len(assets)} assets fully invested. Raise the cap."
    )
    st.stop()
if start >= end:
    st.error("Start must be before end.")
    st.stop()

tickers = tuple(assets)
prices = load_prices(tickers, start, end)
returns = to_returns(prices)
rf = load_rf(start, end)
groups = {g: [t for t in ts if t in assets] for g, ts in CFG["groups"].items()}
groups = {g: ts for g, ts in groups.items() if ts}
st.caption(
    f"Sample: **{prices.index[0]:%d %b %Y} – {prices.index[-1]:%d %b %Y}** "
    f"({len(returns):,} trading days). Every result depends on this period."
)

tab_assets, tab_frontier, tab_backtest, tab_mc, tab_regime, tab_agent = st.tabs(
    ["Assets", "Frontier", "Backtest", "Monte Carlo", "Regime", "Analyst"]
)

with tab_assets:
    st.subheader("Risk and return by asset")
    show_table(
        summary_table(returns, rf=rf, level=D["var_level"]).sort_values("sharpe", ascending=False)
    )
    st.caption(
        "Annualised over 252 days. VaR and Expected Shortfall: 1-day, 95%, "
        "shown as positive losses."
    )

with tab_frontier:
    st.subheader("Efficient frontier (full sample, in hindsight)")
    st.caption(
        "Uses the whole sample's means and covariances, so nobody could have held these at "
        "the time. The backtest tab is the honest test."
    )
    st.caption(
        f"Every portfolio here is long-only with at most **{max_weight:.0%}** per asset (sidebar)."
    )
    mu, cov = returns.mean() * 252, returns.cov() * 252
    rf_annual = float(
        (1 + rf.loc[returns.index[0] :]).prod() ** (252 / len(rf.loc[returns.index[0] :])) - 1
    )
    portfolios = {
        "Min variance": min_variance(cov, max_weight),
        "Max Sharpe": max_sharpe(mu, cov, rf_annual, max_weight),
        "Risk parity": risk_parity(cov, max_weight),
        "Equal weight": pd.Series(1 / len(assets), index=assets),
    }
    points = {}
    for name, w in portfolios.items():
        ret, vol = portfolio_performance(w, mu, cov)
        points[name] = (vol, ret)
    st.plotly_chart(
        frontier_chart(efficient_frontier(mu, cov, 40, max_weight), points), width="stretch"
    )
    st.dataframe(
        pd.DataFrame(portfolios).T,
        column_config={t: st.column_config.NumberColumn(format="percent") for t in assets},
        width="stretch",
    )

with tab_backtest:
    if not names:
        st.info("Pick at least one strategy in the sidebar.")
    else:
        results = run_backtests(
            tickers,
            start,
            end,
            tuple(names),
            max_weight,
            cost_bps,
            lookback,
            threshold if use_regime else None,
        )
        strat_returns = pd.DataFrame({n: r.returns for n, r in results.items()})
        table = summary_table(strat_returns, rf=rf, level=D["var_level"])
        table["annual_turnover"] = pd.Series(
            {n: r.turnover.iloc[1:].mean() * 12 for n, r in results.items()}
        )
        st.subheader("Strategies after costs")
        st.caption(
            f"Monthly rebalancing on a trailing {lookback}-day window; new weights apply the next "
            f"day; {cost_bps:g} bps per unit of turnover."
        )
        show_table(table.sort_values("sharpe", ascending=False))
        st.caption(
            f"Strategies hold at most **{max_weight:.0%}** per asset (sidebar). Benchmarks are "
            "fixed by definition (60/40 holds 60% SPY; SPY holds 100%)."
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

with tab_mc:
    st.subheader("Monte Carlo: where could today's weights go?")
    if not names:
        st.info("Pick at least one strategy in the sidebar.")
    else:
        results = run_backtests(
            tickers,
            start,
            end,
            tuple(names),
            max_weight,
            cost_bps,
            lookback,
            threshold if use_regime else None,
        )
        c1, c2, c3 = st.columns(3)
        mc_strategy = c1.selectbox("Use the latest weights of", list(results), key="mc_strategy")
        years = c2.slider("Horizon (years)", 1, 30, 10)
        target = c3.number_input("Goal: end with at least × start", 0.5, 10.0, 1.5, 0.1)
        c4, c5, c6 = st.columns(3)
        method = c4.radio(
            "Method",
            ["bootstrap", "gbm"],
            horizontal=True,
            help="Bootstrap reshuffles real days; GBM assumes normal log returns.",
        )
        n_paths = c5.select_slider("Paths", [1_000, 2_000, 5_000, 10_000], value=2_000)
        seed = int(c6.number_input("Seed", 0, 10_000, 42))
        weights = results[mc_strategy].weights.iloc[-1]
        weights = weights[weights > 0]
        paths = simulate_paths(returns, weights, 252 * years, n_paths, method=method, seed=seed)
        stats = terminal_stats(paths)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"P(≥ {target:g}×)", f"{goal_probability(paths, target):.0%}")
        m2.metric("Median outcome", f"{stats['p50']:.2f}×")
        m3.metric("Bad case (5th pct)", f"{stats['p5']:.2f}×")
        m4.metric("P(lose money)", f"{1 - goal_probability(paths, 1.0):.0%}")
        st.plotly_chart(
            fan_chart(paths, title=f"{mc_strategy}: simulated growth of $1"), width="stretch"
        )
        st.caption(
            "Weights held fixed and rebalanced daily. Simulations assume the future resembles "
            "this sample."
        )

with tab_regime:
    st.subheader("Will next month be stressed?")
    regime = load_regime(end)
    p = regime["p_stressed"].dropna()
    folds = regime["report"]["folds"]
    conf = regime["report"]["confusion"].to_numpy()
    tn, fp, fn, tp = conf.ravel()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"P(stressed) on {p.index[-1]:%d %b %Y}", f"{p.iloc[-1]:.0%}")
    m2.metric("Recall (stressed caught)", f"{tp / (tp + fn):.0%}")
    m3.metric("Precision (calls right)", f"{tp / (tp + fp):.0%}")
    m4.metric(
        "Accuracy vs always-calm",
        f"{(tp + tn) / conf.sum():.0%}",
        f"{(tp + tn) / conf.sum() - (tn + fp) / conf.sum():+.0%}",
        delta_color="normal",
    )
    st.plotly_chart(regime_chart(p, threshold, regime["labels"]), width="stretch")
    st.caption(
        f"Stressed = SPY's volatility over the next {R['horizon']} trading days is above the "
        f"{R['quantile']:.0%} percentile of the past year. "
        f"{R['model'].replace('_', ' ').title()} model, "
        "refit each month on labels already known, scored on 5 chronological folds with a "
        f"{R['horizon']}-day gap. Grey = months that turned out stressed."
    )
    st.dataframe(
        folds[
            [
                "test_start",
                "test_end",
                "stressed_share",
                "precision",
                "recall",
                "accuracy",
                "baseline_accuracy",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

with tab_agent:
    st.subheader("AI analyst")
    st.info(
        "Coming in Week 3: describe your profile and an agent calls these same functions "
        "as tools and writes a memo in which every number is traceable to a tool call."
    )
