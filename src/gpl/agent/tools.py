"""Agent tools: JSON schemas plus executors that wrap the gpl finance functions.

The tools never reimplement the maths: each executor calls ``gpl.metrics``, ``gpl.optimise``,
``gpl.backtest``, ``gpl.montecarlo`` or ``gpl.regime`` and returns the numbers **already
formatted** (``"15.27%"``, ``"0.49"``, ``"1.98×"``). The model is told to copy them verbatim,
which keeps every figure in a memo traceable to a tool call.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from gpl import backtest as bt
from gpl import regime as rg
from gpl.data import clean_prices, get_prices, get_risk_free, to_returns
from gpl.metrics import PERIODS_PER_YEAR, summary_table
from gpl.montecarlo import (
    goal_probability,
    path_max_drawdowns,
    simulate_paths,
    terminal_stats,
    to_real,
)
from gpl.optimise import (
    max_sharpe,
    min_variance,
    portfolio_performance,
    risk_contributions,
    risk_parity,
)

STRATEGIES = ["Equal weight", "Min variance", "Max Sharpe", "Risk parity"]
BENCHMARKS = ["60/40 (SPY/AGG)", "SPY"]


class ToolError(Exception):
    """A tool could not run with the given input; the message is returned to the model."""


def pct(x: float) -> str:
    return f"{x:.2%}"


def num(x: float) -> str:
    return f"{x:.2f}"


def times(x: float) -> str:
    return f"{x:.2f}×"


@dataclass
class AnalystContext:
    """Everything the tools need, loaded once per dashboard session or memo."""

    prices: pd.DataFrame
    rf: pd.Series
    groups: Mapping[str, Sequence[str]]
    benchmarks: Mapping[str, Mapping[str, float]]
    regime_features: pd.DataFrame
    regime_labels: pd.Series
    settings: Mapping[str, Any]
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def assets(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def returns(self) -> pd.DataFrame:
        if "returns" not in self._cache:
            self._cache["returns"] = to_returns(self.prices)
        return self._cache["returns"]

    def strategy(self, name: str) -> bt.Strategy:
        mw, rf = self.settings["max_weight"], self.rf
        builders: dict[str, Callable[[], bt.Strategy]] = {
            "Equal weight": lambda: bt.equal_weight,
            "Min variance": lambda: bt.min_variance_strategy(max_weight=mw),
            "Max Sharpe": lambda: bt.max_sharpe_strategy(rf=rf, max_weight=mw),
            "Risk parity": lambda: bt.risk_parity_strategy(max_weight=mw),
            "60/40 (SPY/AGG)": lambda: bt.fixed_weights(self.benchmarks["sixty_forty"]),
            "SPY": lambda: bt.fixed_weights(self.benchmarks["spy"]),
        }
        if name not in builders:
            raise ToolError(f"unknown strategy {name!r}; choose from {list(builders)}")
        return builders[name]()

    def p_stressed(self) -> pd.Series:
        if "p_stressed" not in self._cache:
            r = self.settings["regime"]
            self._cache["p_stressed"] = rg.walk_forward_probabilities(
                self.regime_features,
                self.regime_labels,
                model=r["model"],
                refit=r["refit"],
                min_train=r["min_train"],
                horizon=r["horizon"],
            )
        return self._cache["p_stressed"]

    def backtest(self, name: str, regime_aware: bool = False) -> bt.BacktestResult:
        key = ("backtest", name, regime_aware)
        if key not in self._cache:
            strategy = self.strategy(name)
            if regime_aware:
                defensive = bt.min_variance_strategy(max_weight=self.settings["max_weight"])
                strategy = bt.regime_switch_strategy(
                    strategy, defensive, self.p_stressed(), self.settings["regime"]["threshold"]
                )
            self._cache[key] = bt.run_backtest(
                self.prices,
                strategy,
                rebalance=self.settings["rebalance"],
                lookback=self.settings["lookback"],
                cost_bps=self.settings["cost_bps"],
            )
        return self._cache[key]


def build_context(
    cfg: Mapping[str, Any],
    tickers: Sequence[str] | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
    max_weight: float | None = None,
    cost_bps: float | None = None,
    lookback: int | None = None,
) -> AnalystContext:
    """Load prices, risk-free rate and regime data via ``gpl.data`` (all I/O happens there)."""
    d, r = cfg["defaults"], cfg["regime"]
    start, end = start or d["start"], end or d["end"]
    prices = clean_prices(get_prices(list(tickers or cfg["assets"]), start, end), d["max_ffill"])
    regime_px = clean_prices(get_prices([r["equity"], r["bond"]], r["start"], end))
    vix = get_prices(["^VIX"], r["start"], end)["^VIX"]
    return AnalystContext(
        prices=prices,
        rf=get_risk_free(start, end),
        groups=cfg["groups"],
        benchmarks=cfg["benchmarks"],
        regime_features=rg.make_features(regime_px, vix, equity=r["equity"], bond=r["bond"]),
        regime_labels=rg.make_labels(
            regime_px[r["equity"]].pct_change().iloc[1:], r["horizon"], r["quantile"], r["window"]
        ),
        settings={
            "max_weight": d["max_weight"] if max_weight is None else max_weight,
            "cost_bps": d["cost_bps"] if cost_bps is None else cost_bps,
            "lookback": d["lookback"] if lookback is None else lookback,
            "rebalance": d["rebalance"],
            "var_level": d["var_level"],
            "regime": r,
            "montecarlo": cfg["montecarlo"],
        },
    )


# ------------------------------------------------------------------------------ executors


def _group_of(ctx: AnalystContext, ticker: str) -> str:
    return next((g for g, ts in ctx.groups.items() if ticker in ts), "other")


def _metrics_rows(table: pd.DataFrame) -> dict[str, dict[str, str]]:
    return {
        name: {
            "annual_return": pct(row["ann_return"]),
            "cagr": pct(row["cagr"]),
            "annual_volatility": pct(row["ann_vol"]),
            "sharpe": num(row["sharpe"]),
            "sortino": num(row["sortino"]),
            "max_drawdown": pct(row["max_drawdown"]),
            "var_95_1day": pct(row["hist_var"]),
            "expected_shortfall_95_1day": pct(row["expected_shortfall"]),
        }
        for name, row in table.iterrows()
    }


def _weights(w: pd.Series) -> dict[str, str]:
    w = w[w > 0.0005].sort_values(ascending=False)
    return {t: pct(v) for t, v in w.items()}


def get_universe(ctx: AnalystContext) -> dict:
    s = ctx.settings
    return {
        "assets": {t: _group_of(ctx, t) for t in ctx.assets},
        "sample_start": str(ctx.prices.index[0].date()),
        "sample_end": str(ctx.prices.index[-1].date()),
        "trading_days": len(ctx.returns),
        "currency": "USD (all assets are US-listed ETFs priced in USD)",
        "max_weight_per_asset": pct(s["max_weight"]),
        "transaction_cost_bps": f"{s['cost_bps']:g}",
        "estimation_window_days": s["lookback"],
        "benchmarks": {"60/40 (SPY/AGG)": "60.00% SPY, 40.00% AGG", "SPY": "100.00% SPY"},
    }


def get_asset_metrics(ctx: AnalystContext, tickers: list[str]) -> dict:
    chosen = tickers or ctx.assets
    unknown = sorted(set(chosen) - set(ctx.assets))
    if unknown:
        raise ToolError(f"unknown tickers {unknown}; available: {ctx.assets}")
    table = summary_table(ctx.returns[chosen], rf=ctx.rf, level=ctx.settings["var_level"])
    return {
        "period": f"{ctx.returns.index[0].date()} to {ctx.returns.index[-1].date()}",
        "note": "Full-sample, annualised over 252 days; VaR/ES are 1-day 95% losses.",
        "metrics": _metrics_rows(table),
    }


def optimise_portfolio(ctx: AnalystContext, method: str) -> dict:
    mu, cov = ctx.returns.mean() * PERIODS_PER_YEAR, ctx.returns.cov() * PERIODS_PER_YEAR
    mw = ctx.settings["max_weight"]
    rf_daily = ctx.rf.loc[ctx.returns.index[0] :]
    rf_annual = float((1 + rf_daily).prod() ** (PERIODS_PER_YEAR / len(rf_daily)) - 1)
    match method:
        case "min_variance":
            w = min_variance(cov, mw)
        case "max_sharpe":
            w = max_sharpe(mu, cov, rf_annual, mw)
        case "risk_parity":
            w = risk_parity(cov, mw)
        case "equal_weight":
            w = pd.Series(1 / len(ctx.assets), index=ctx.assets)
        case _:
            raise ToolError(f"unknown method {method!r}")
    ret, vol = portfolio_performance(w, mu, cov)
    rc = risk_contributions(w, cov)
    return {
        "method": method,
        "weights": _weights(w),
        "risk_contributions": {t: pct(rc[t]) for t in _weights(w)},
        "expected_annual_return_in_sample": pct(ret),
        "expected_annual_volatility_in_sample": pct(vol),
        "max_weight_per_asset": pct(mw),
        "caveat": "Estimated on the full sample (hindsight). Use backtest_strategies for an "
        "out-of-sample view.",
    }


def backtest_strategies(ctx: AnalystContext, strategies: list[str], regime_aware: bool) -> dict:
    names = strategies or STRATEGIES + BENCHMARKS
    results = {}
    for name in names:
        results[name] = ctx.backtest(name)
        if regime_aware and name in STRATEGIES and name != "Min variance":
            results[f"{name} + regime"] = ctx.backtest(name, regime_aware=True)
    rets = pd.DataFrame({n: r.returns for n, r in results.items()})
    rows = _metrics_rows(summary_table(rets, rf=ctx.rf, level=ctx.settings["var_level"]))
    for n, r in results.items():
        rows[n]["turnover_per_year"] = times(r.turnover.iloc[1:].mean() * 12)
        rows[n]["latest_weights"] = _weights(r.weights.iloc[-1])
    first = next(iter(results.values())).equity.index
    return {
        "backtest_period": f"{first[0].date()} to {first[-1].date()}",
        "method": f"Monthly rebalancing on a trailing {ctx.settings['lookback']}-day window; "
        f"weights apply from the next day; {ctx.settings['cost_bps']:g} bps per unit traded. "
        "'+ regime' versions switch to min variance when P(stressed) >= threshold.",
        "results": rows,
    }


def simulate_future(
    ctx: AnalystContext, strategy: str, horizon_years: int, target_multiple: float, real_terms: bool
) -> dict:
    if not 1 <= horizon_years <= 30:
        raise ToolError("horizon_years must be between 1 and 30")
    if target_multiple <= 0:
        raise ToolError("target_multiple must be positive")
    mc = ctx.settings["montecarlo"]
    w = ctx.backtest(strategy).weights.iloc[-1]
    w = w[w > 0]
    n_paths, seed = 5_000, 42
    paths = simulate_paths(
        ctx.returns,
        w,
        PERIODS_PER_YEAR * horizon_years,
        n_paths,
        method=mc["method"],
        seed=seed,
        block_size=mc["block_size"],
    )
    if real_terms:
        paths = to_real(paths, mc["inflation"])
    stats = terminal_stats(paths)
    return {
        "strategy": strategy,
        "weights_used": _weights(w),
        "horizon_years": horizon_years,
        "units": f"real USD after {pct(mc['inflation'])} inflation a year"
        if real_terms
        else "nominal USD",
        f"probability_of_reaching_{times(target_multiple)}": pct(
            goal_probability(paths, target_multiple)
        ),
        "probability_of_losing_money": pct(1 - goal_probability(paths, 1.0)),
        "median_outcome": times(stats["p50"]),
        "bad_case_5th_percentile_outcome": times(stats["p5"]),
        "good_case_95th_percentile_outcome": times(stats["p95"]),
        "bad_case_drawdown_along_the_way_5th_percentile": pct(
            np.percentile(path_max_drawdowns(paths), 5)
        ),
        "method": f"{mc['block_size']}-day block bootstrap of the sample's daily returns, "
        f"{n_paths} paths, seed {seed}; weights rebalanced daily, no costs.",
    }


def get_regime_view(ctx: AnalystContext) -> dict:
    r = ctx.settings["regime"]
    p = ctx.p_stressed().dropna()
    rule = rg.naive_stress_signal(ctx.regime_features, r["quantile"], r["window"]).dropna()
    report = rg.evaluate_model(
        ctx.regime_features,
        ctx.regime_labels,
        model=r["model"],
        horizon=r["horizon"],
        threshold=r["threshold"],
        rule=rg.naive_stress_signal(ctx.regime_features, r["quantile"], r["window"]),
    )
    m, v = report["overall"].loc["model"], report["overall"].loc["vol_rule"]
    latest = float(p.iloc[-1])
    return {
        "as_of": str(p.index[-1].date()),
        "probability_next_month_stressed": pct(latest),
        "threshold": pct(r["threshold"]),
        "model_signal": "stressed" if latest >= r["threshold"] else "calm",
        "vol_rule_signal": "stressed" if rule.iloc[-1] >= 0.5 else "calm",
        "definition": f"Stressed = SPY volatility over the next {r['horizon']} trading days above "
        f"the {r['quantile'] * 100:.0f}th percentile of its past year (relative, not absolute).",
        "out_of_sample_track_record": {
            "roc_auc": num(m["auc"]),
            "precision": pct(m["precision"]),
            "recall": pct(m["recall"]),
            "base_rate": pct(m["stressed_share"]),
            "vol_rule_precision": pct(v["precision"]),
            "vol_rule_recall": pct(v["recall"]),
        },
        "caveats": "Modest skill (a simple volatility rule scores as well at this threshold); "
        "month-to-month forecasts can whipsaw across the threshold, as in 2022.",
    }


# --------------------------------------------------------------------------------- schemas


def tool_definitions(ctx: AnalystContext) -> list[dict]:
    """Anthropic tool definitions. Enums are built from the context's universe."""

    def tool(name: str, description: str, properties: dict) -> dict:
        return {
            "name": name,
            "description": description,
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        }

    all_strategies = STRATEGIES + [b for b in BENCHMARKS if {"SPY", "AGG"} <= set(ctx.assets)]
    return [
        tool(
            "get_universe",
            "List the investable assets (US-listed ETFs in USD) with their asset groups, the "
            "data sample period and the portfolio rules (max weight, costs). Call this first.",
            {},
        ),
        tool(
            "get_asset_metrics",
            "Full-sample risk and return for individual assets: annual return, CAGR, volatility, "
            "Sharpe, Sortino, max drawdown, 1-day 95% VaR and expected shortfall. Pass an empty "
            "list for all assets.",
            {"tickers": {"type": "array", "items": {"type": "string", "enum": ctx.assets}}},
        ),
        tool(
            "optimise_portfolio",
            "Weights for one construction method on the full sample (in hindsight), with "
            "in-sample expected return, volatility and each asset's share of risk.",
            {
                "method": {
                    "type": "string",
                    "enum": ["min_variance", "max_sharpe", "risk_parity", "equal_weight"],
                }
            },
        ),
        tool(
            "backtest_strategies",
            "Out-of-sample monthly-rebalanced backtest after costs: CAGR, volatility, Sharpe, "
            "Sortino, max drawdown, VaR/ES, turnover and latest weights. Pass an empty list for "
            "all strategies and benchmarks. regime_aware adds versions that switch to min "
            "variance when the regime model forecasts stress.",
            {
                "strategies": {
                    "type": "array",
                    "items": {"type": "string", "enum": all_strategies},
                },
                "regime_aware": {"type": "boolean"},
            },
        ),
        tool(
            "simulate_future",
            "Monte Carlo of a strategy's latest weights: probability of reaching a target "
            "multiple of starting wealth, probability of losing money, median and bad/good-case "
            "outcomes, and the bad-case drawdown along the way.",
            {
                "strategy": {"type": "string", "enum": all_strategies},
                "horizon_years": {"type": "integer"},
                "target_multiple": {"type": "number"},
                "real_terms": {"type": "boolean"},
            },
        ),
        tool(
            "get_regime_view",
            "Latest probability that next month is a stressed market, the model's out-of-sample "
            "track record vs a simple volatility rule, and caveats.",
            {},
        ),
    ]


EXECUTORS: dict[str, Callable[..., dict]] = {
    "get_universe": get_universe,
    "get_asset_metrics": get_asset_metrics,
    "optimise_portfolio": optimise_portfolio,
    "backtest_strategies": backtest_strategies,
    "simulate_future": simulate_future,
    "get_regime_view": get_regime_view,
}


def execute_tool(ctx: AnalystContext, name: str, arguments: Mapping[str, Any]) -> dict:
    """Run one tool. Raises ``ToolError`` for bad input or an unknown tool."""
    if name not in EXECUTORS:
        raise ToolError(f"unknown tool {name!r}")
    try:
        return EXECUTORS[name](ctx, **arguments)
    except TypeError as e:  # wrong or missing arguments
        raise ToolError(f"bad arguments for {name}: {e}") from e
    except ValueError as e:  # e.g. infeasible weight cap, too little history
        raise ToolError(f"{name} could not run: {e}") from e
