"""Charts the analyst can ask for, built only from data that code computes.

The model chooses a chart **type** from ``CHART_TYPES``, its order and the memo section it
belongs to (the ``show_chart`` tool has no numeric fields). ``chart_data`` computes everything
by calling the finance engine on the analyst's context; ``build_chart`` turns that data, and
nothing else, into a figure or table; ``chart_summary`` gives the model formatted figures it
may quote. The current portfolio is the user's holdings; the proposed one is the latest
what-if or Optimise recommendation, if any; the benchmark is the one chosen in the app.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from gpl import backtest as bt
from gpl.metrics import drawdown_series, kpi_table, rolling_volatility
from gpl.montecarlo import simulate_paths
from gpl.optimise import efficient_frontier, portfolio_performance, risk_contributions
from gpl.plots import (
    correlation_heatmap,
    drawdown_chart,
    equity_chart,
    exposure_chart,
    fan_compare_chart,
    frontier_chart,
    lines_chart,
    paired_bars_chart,
    risk_contribution_chart,
    stress_chart,
)
from gpl.scenarios import classify_holdings, episode_performance, exposures

CHART_TYPES: dict[str, str] = {
    "weights_current_vs_proposed": "Current vs proposed weights by holding.",
    "growth_of_1": "Growth of $1 after costs: current, proposed (hindsight) and benchmark.",
    "drawdown_comparison": "Drawdown from peak: current, proposed and benchmark.",
    "metrics_table": "CAGR, volatility, Sharpe and max drawdown with deltas vs the benchmark.",
    "money_vs_risk": "Each holding's share of the money vs its share of the risk.",
    "correlation_heatmap": "Correlation of daily returns between the holdings.",
    "rolling_volatility_1y": "Rolling one-year volatility: current, proposed and benchmark.",
    "efficient_frontier": "Efficient frontier (hindsight) with current and proposed marked.",
    "monte_carlo_fan": "Simulated wealth over 10 years: median and 5-95% band, current vs "
    "proposed.",
    "exposure": "Weight by region, asset type and currency (current and proposed).",
    "stress_test": "Return in historical stress episodes: current, proposed and benchmark.",
}
CURRENT = "Current"
MC_YEARS, MC_PATHS, MC_SEED = 10, 2_000, 7


@dataclass
class ChartData:
    """Everything needed to draw one chart: computed by code, never written by the model."""

    chart: str
    title: str
    frames: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    reference: str | None = None  # the benchmark series, drawn grey and dashed


def _portfolios(ctx) -> dict[str, pd.Series]:
    """Current and (if any) proposed weights, as Series without zero weights."""
    out = {CURRENT: pd.Series(ctx.user_portfolio, dtype=float)}
    if ctx.proposed:
        out[ctx.proposed["name"]] = pd.Series(ctx.proposed["weights"], dtype=float)
    return {n: w[w > 0] for n, w in out.items()}


def _backtest_returns(ctx, w: pd.Series) -> pd.Series:
    """Fixed weights rebalanced monthly after costs, from the first full day held."""
    run = dict(
        rebalance=ctx.settings["rebalance"],
        lookback=ctx.settings["lookback"],
        cost_bps=ctx.settings["cost_bps"],
    )
    return bt.run_backtest(
        ctx.prices[list(w.index)], bt.fixed_weights(w.to_dict()), **run
    ).returns.iloc[1:]


def _held_returns(ctx, w: pd.Series) -> pd.Series:
    """Daily returns at fixed weights over the full common sample (no costs)."""
    return ctx.all_returns[list(w.index)] @ w


def _with_benchmark(ctx, frame: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if ctx.benchmark is None:
        return frame, None
    name = f"Benchmark: {ctx.benchmark.name}"
    frame[name] = ctx.benchmark.returns.reindex(frame.index)
    return frame.dropna(), name


def chart_data(ctx, chart: str) -> ChartData:
    """Compute the data for ``chart`` from the engine. Raises ValueError for an unknown type."""
    if chart not in CHART_TYPES:
        raise ValueError(f"unknown chart {chart!r}; choose from {list(CHART_TYPES)}")
    if not ctx.user_portfolio:
        raise ValueError("no portfolio entered on the Portfolio tab")
    ports = _portfolios(ctx)
    notes = [] if ctx.proposed else ["No proposed portfolio yet (no what-if or Optimise run)."]
    current = ports[CURRENT]

    if chart == "weights_current_vs_proposed":
        names = list(ports)
        second = ports[names[1]] if len(names) > 1 else current
        return ChartData(
            chart,
            "Current vs proposed weights" if len(names) > 1 else "Current weights",
            {"first": current, "second": second, "names": (CURRENT, names[-1])},
            notes,
        )
    if chart in ("growth_of_1", "drawdown_comparison", "metrics_table"):
        rets = pd.DataFrame({n: _backtest_returns(ctx, w) for n, w in ports.items()}).dropna()
        rets, ref = _with_benchmark(ctx, rets)
        notes.append(
            "Rebalanced monthly after costs. The proposed portfolio uses today's proposed weights "
            "over the past, so it is hindsight."
            if ctx.proposed
            else "Rebalanced monthly after costs."
        )
        if chart == "growth_of_1":
            return ChartData(
                chart, "Growth of $1 after costs", {"values": (1 + rets).cumprod()}, notes, ref
            )
        if chart == "drawdown_comparison":
            return ChartData(
                chart, "Drawdown from peak", {"values": drawdown_series(rets)}, notes, ref
            )
        if ref is None:
            raise ValueError("choose a benchmark to compare the metrics with")
        tables = {n: kpi_table(rets[n], rets[ref], ctx.rf, ctx.periods) for n in ports}
        return ChartData(
            chart, f"Key figures vs {ctx.benchmark.name}", {"kpis": tables}, notes, ref
        )
    if chart == "money_vs_risk":
        cov = ctx.all_returns[list(current.index)].cov()
        return ChartData(
            chart,
            "Share of money vs share of risk (current)",
            {"weights": current, "risk": risk_contributions(current, cov)},
            [],
        )
    if chart == "correlation_heatmap":
        return ChartData(
            chart,
            "Correlation of daily returns",
            {"corr": ctx.all_returns[list(current.index)].corr()},
            [],
        )
    if chart == "rolling_volatility_1y":
        held = pd.DataFrame({n: _held_returns(ctx, w) for n, w in ports.items()})
        held, ref = _with_benchmark(ctx, held)
        return ChartData(
            chart,
            "Rolling one-year volatility",
            {"values": rolling_volatility(held, ctx.periods, ctx.periods)},
            notes + ["Today's weights held over the past, rebalanced daily, no costs."],
            ref,
        )
    if chart == "efficient_frontier":
        tickers = sorted(set().union(*[set(w.index) for w in ports.values()]))
        rets = ctx.all_returns[tickers]
        mu, cov = rets.mean() * ctx.periods, rets.cov() * ctx.periods
        cap = ctx.settings["max_weight"] if ctx.settings["max_weight"] * len(tickers) >= 1 else 1.0
        points = {
            n: portfolio_performance(w.reindex(tickers).fillna(0), mu, cov)[::-1]
            for n, w in ports.items()
        }
        return ChartData(
            chart,
            "Efficient frontier (full sample, hindsight)",
            {"frontier": efficient_frontier(mu, cov, 40, cap), "points": points},
            notes + [f"Long-only, at most {cap:.2%} per asset; full-sample estimates."],
        )
    if chart == "monte_carlo_fan":
        mc = ctx.settings["montecarlo"]
        bands = {}
        for n, w in ports.items():
            paths = simulate_paths(
                ctx.all_returns[list(w.index)],
                w,
                ctx.periods * MC_YEARS,
                MC_PATHS,
                method=mc["method"],
                seed=MC_SEED,
                block_size=mc["block_size"],
            )
            step = max(ctx.periods // 12, 1)
            sampled = paths[::step]
            bands[n] = pd.DataFrame(
                {
                    q: np.percentile(sampled, p, axis=1)
                    for q, p in (("p5", 5), ("p50", 50), ("p95", 95))
                },
                index=np.arange(len(sampled)) * step / ctx.periods,
            )
        return ChartData(
            chart,
            f"Simulated growth of $1 over {MC_YEARS} years",
            {"bands": bands},
            notes
            + [
                f"{MC_PATHS:,} paths, {mc['block_size']}-day block bootstrap "
                "of the sample, daily rebalancing, no costs, nominal USD."
            ],
        )
    if chart == "exposure":
        tickers = sorted(set().union(*[set(w.index) for w in ports.values()]))
        currencies = ctx.asset_notes["currency"].to_dict() if ctx.asset_notes is not None else {}
        meta = classify_holdings(
            tickers, ctx.settings.get("exposure_meta", {}), ctx.holding_groups or {}, currencies
        )
        return ChartData(
            chart,
            "Exposure by region, asset type and currency",
            {"exposures": {n: exposures(w, meta) for n, w in ports.items()}},
            notes
            + [
                "Currency is the trading currency (everything is analysed in "
                "USD); region by known fund exposure or listing."
            ],
        )
    # stress_test
    held = pd.DataFrame({n: _held_returns(ctx, w) for n, w in ports.items()})
    held, ref = _with_benchmark(ctx, held)
    episodes = ctx.settings.get("stress_episodes", [])
    return ChartData(
        chart,
        "Historical stress tests",
        {"table": episode_performance(held, episodes), "episodes": episodes},
        notes + ["Today's weights held through each episode, rebalanced daily, no costs."],
        ref,
    )


def build_chart(data: ChartData) -> go.Figure | pd.DataFrame:
    """Figure (or formatted table) from ``ChartData`` alone."""
    f, c = data.frames, data.chart
    if c == "weights_current_vs_proposed":
        return paired_bars_chart(f["first"], f["second"], f["names"], data.title)
    if c == "growth_of_1":
        return equity_chart(f["values"], data.title, reference=data.reference)
    if c == "drawdown_comparison":
        return drawdown_chart(f["values"], data.title, reference=data.reference)
    if c == "metrics_table":
        return metrics_display(f["kpis"])
    if c == "money_vs_risk":
        return risk_contribution_chart(f["weights"], f["risk"], data.title)
    if c == "correlation_heatmap":
        return correlation_heatmap(f["corr"], data.title)
    if c == "rolling_volatility_1y":
        return lines_chart(f["values"], data.title, "Volatility (annual)", reference=data.reference)
    if c == "efficient_frontier":
        return frontier_chart(f["frontier"], f["points"], data.title)
    if c == "monte_carlo_fan":
        return fan_compare_chart(f["bands"], data.title, "Wealth (nominal USD)")
    if c == "exposure":
        return exposure_chart(f["exposures"], data.title)
    if c == "stress_test":
        return stress_chart(f["table"], data.reference, data.title)
    raise ValueError(f"unknown chart {c!r}")


LABELS = {
    "cagr": "Return (CAGR)",
    "volatility": "Volatility",
    "sharpe": "Sharpe",
    "max_drawdown": "Max drawdown",
}


def _fmt(metric: str, x: float, signed: bool = False) -> str:
    if pd.isna(x):
        return "–"
    if metric == "sharpe":
        return f"{x:+.2f}" if signed else f"{x:.2f}"
    return f"{x * 100:+.2f} pp" if signed else f"{x:.2%}"


def metrics_display(kpis: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Formatted metrics table: each portfolio, the benchmark, and each portfolio's delta."""
    first = next(iter(kpis.values()))
    out = {}
    for name, k in kpis.items():
        out[name] = [_fmt(m, k.loc[m, "portfolio"]) for m in LABELS]
    out["Benchmark"] = [_fmt(m, first.loc[m, "benchmark"]) for m in LABELS]
    for name, k in kpis.items():
        out[f"{name} vs benchmark"] = [_fmt(m, k.loc[m, "delta"], signed=True) for m in LABELS]
    return pd.DataFrame(out, index=list(LABELS.values()))


def chart_summary(data: ChartData) -> dict:
    """Formatted key figures the model may quote (it never sees or writes the raw series)."""
    f, c = data.frames, data.chart

    def pct(x: float) -> str:
        return "n/a" if pd.isna(x) else f"{x:.2%}"

    if c == "weights_current_vs_proposed":
        return {
            "current": {t: pct(v) for t, v in f["first"].items()},
            f["names"][1]: {t: pct(v) for t, v in f["second"].items()},
        }
    if c == "growth_of_1":
        end = f["values"].iloc[-1]
        return {
            "period": f"{f['values'].index[0].date()} to {f['values'].index[-1].date()}",
            "ending_value_of_1": {n: f"{v:.2f}" for n, v in end.items()},
        }
    if c == "drawdown_comparison":
        return {"max_drawdown": {n: pct(v) for n, v in f["values"].min().items()}}
    if c == "metrics_table":
        return metrics_display(f["kpis"]).to_dict()
    if c == "money_vs_risk":
        return {
            t: {"weight": pct(f["weights"][t]), "share_of_risk": pct(f["risk"][t])}
            for t in f["weights"].index
        }
    if c == "correlation_heatmap":
        corr = f["corr"]
        pairs = [
            (a, b, corr.loc[a, b]) for i, a in enumerate(corr.index) for b in corr.columns[i + 1 :]
        ]
        top = sorted(pairs, key=lambda p: -p[2])[:3]
        return {"most_correlated_pairs": {f"{a}-{b}": f"{v:.2f}" for a, b, v in top}}
    if c == "rolling_volatility_1y":
        v = f["values"]
        return {
            "latest": {n: pct(x) for n, x in v.iloc[-1].items()},
            "highest": {n: pct(x) for n, x in v.max().items()},
        }
    if c == "efficient_frontier":
        return {
            n: {"volatility": pct(vol), "expected_return": pct(ret)}
            for n, (vol, ret) in f["points"].items()
        }
    if c == "monte_carlo_fan":
        return {
            n: {
                "median_after_10_years": f"{b['p50'].iloc[-1]:.2f}",
                "5th_percentile": f"{b['p5'].iloc[-1]:.2f}",
                "95th_percentile": f"{b['p95'].iloc[-1]:.2f}",
            }
            for n, b in f["bands"].items()
        }
    if c == "exposure":
        return {
            n: {k: {i: pct(v) for i, v in s.items()} for k, s in e.items()}
            for n, e in f["exposures"].items()
        }
    t = f["table"]
    return {
        ep["name"]: {
            "dates": f"{ep['start']} to {ep['end']}",
            "what_happened": ep.get("description", ""),
            **{
                r.series: (
                    "not covered by the data"
                    if not r.covered
                    else {"return": pct(r.total_return), "max_drawdown": pct(r.max_drawdown)}
                )
                for r in t[t.episode == ep["name"]].itertuples()
            },
        }
        for ep in f["episodes"]
    }
