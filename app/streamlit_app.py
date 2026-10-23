"""Gulf Portfolio Lab dashboard.

Presentation only: every number on screen comes from a ``gpl`` function. The app picks
inputs, calls the engine, and draws the results. Run with::

    streamlit run app/streamlit_app.py
"""

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from formatting import pct, show_metrics_table, style_columns, style_percent
from gpl import backtest as bt
from gpl import regime as rg
from gpl.data import clean_prices, get_prices, get_risk_free, load_universe, to_returns
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
        case "60/40 (SPY/AGG)":
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
        f"A {pct(max_weight)} cap cannot hold {len(assets)} assets fully invested. Raise the cap."
    )
    st.stop()
if start >= end:
    st.error("Start must be before end.")
    st.stop()

tickers = tuple(assets)
prices = load_prices(tickers, start, end)
returns = to_returns(prices)
rf = load_rf(start, end)
groups = {
    GROUP_LABELS.get(g, g.replace("_", " ").capitalize()): [t for t in ts if t in assets]
    for g, ts in CFG["groups"].items()
}
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
    show_metrics_table(
        summary_table(returns, rf=rf, level=D["var_level"]).sort_values("sharpe", ascending=False),
        D["var_level"],
    )
    st.caption(
        f"Full sample above. Annualised over 252 days. VaR and expected shortfall: 1-day, "
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
    st.dataframe(style_percent(pd.DataFrame(portfolios).T), width="stretch")

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
        first_day = next(iter(results.values())).equity.index[0]
        st.subheader("Strategies after costs")
        st.caption(
            f"Monthly rebalancing on a trailing {lookback}-day window; new weights apply the next "
            f"day; {cost_bps:g} bps per unit of turnover."
        )
        show_metrics_table(table.sort_values("sharpe", ascending=False), D["var_level"])
        st.caption(
            f"**Backtest period: {first_day:%d %b %Y} – {prices.index[-1]:%d %b %Y}.** Each "
            f"strategy needs {lookback} trading days of history before its first rebalance, so the "
            f"backtest starts {lookback} days after the data does. Figures (even for SPY) "
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
        weights = results[mc_strategy].weights.iloc[-1]
        weights = weights[weights > 0]
        paths = simulate_paths(
            returns,
            weights,
            252 * years,
            n_paths,
            method=method,
            seed=seed,
            block_size=MC["block_size"],
        )
        if real:
            paths = to_real(paths, MC["inflation"])
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
    regime = load_regime(end)
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
    st.info(
        "Coming in Week 3: describe your profile and an agent calls these same functions "
        "as tools and writes a memo in which every number is traceable to a tool call."
    )
