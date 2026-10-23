"""Periodically rebalanced backtester with transaction costs.

Pure functions: no I/O, no printing, no plotting.

No look-ahead bias: weights set at rebalance date *t* use returns up to and including *t*
only, and apply from the next trading day. Between rebalances, weights drift with prices.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gpl.data import to_returns
from gpl.metrics import PERIODS_PER_YEAR
from gpl.optimise import OptimisationSpec, max_sharpe, min_variance, optimise, risk_parity

type Strategy = Callable[[pd.DataFrame], pd.Series]
"""Maps a trailing window of daily returns (rows = dates, columns = tickers) to target
weights indexed by ticker."""


@dataclass(frozen=True)
class BacktestResult:
    """Output of ``run_backtest``.

    Attributes
    ----------
    equity : pd.Series
        Portfolio wealth, from 1.0 invested at the close of the first rebalance date.
        The first value is ``1 - initial cost``: buying from cash is turnover of 1.
    returns : pd.Series
        Daily portfolio returns, net of costs.
    weights : pd.DataFrame
        Target weights at each rebalance date (rows = rebalance dates, columns = tickers).
    turnover : pd.Series
        ``sum(|w_target - w_drifted|)`` at each rebalance date.
    costs : pd.Series
        Cost charged at each rebalance, as a fraction of wealth: ``turnover * cost_bps / 1e4``.
    """

    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    rebalance: str = "ME",
    lookback: int = 252,
    cost_bps: float = 10.0,
) -> BacktestResult:
    """Simulate a strategy rebalanced on a calendar schedule.

    Parameters
    ----------
    prices : pd.DataFrame
        Cleaned prices, one column per ticker.
    strategy : Strategy
        Called at each rebalance date with the trailing ``lookback`` days of returns.
    rebalance : str, default "ME"
        Pandas offset alias; rebalance on the last trading day of each period.
    lookback : int, default 252
        Trading days of history passed to ``strategy``. Rebalance dates with fewer days
        of history are skipped, so the backtest starts after the first full window.
    cost_bps : float, default 10.0
        Transaction cost in basis points on turnover.

    Returns
    -------
    BacktestResult
    """
    returns = to_returns(prices)
    if returns.isna().any().any():
        raise ValueError("prices contain NaN gaps; run clean_prices and handle long gaps first")
    tickers = returns.columns
    r = returns.to_numpy()

    rebalance_dates = _rebalance_dates(returns.index, rebalance)
    is_rebalance = returns.index.isin(rebalance_dates)
    # A rebalance needs a full window ending on that day (returns are known at the close).
    has_history = np.arange(1, len(returns) + 1) >= lookback
    can_rebalance = is_rebalance & has_history
    if not can_rebalance.any():
        raise ValueError(f"no rebalance date has {lookback} days of history")
    start = int(np.argmax(can_rebalance))

    w = np.zeros(len(tickers))  # start in cash
    net_returns, weight_rows, turnover, costs = [], {}, {}, {}
    for i in range(start, len(returns)):
        date = returns.index[i]
        if i == start:
            day_return = 0.0  # we buy at today's close, so today's move isn't ours
        else:
            day_return = float(w @ r[i])
            w = w * (1 + r[i]) / (1 + day_return)  # weights drift with prices

        cost = 0.0
        if can_rebalance[i]:
            window = returns.iloc[i - lookback + 1 : i + 1]  # up to and including t
            target = _validate_weights(strategy(window), tickers)
            traded = float(np.abs(target - w).sum())
            cost = traded * cost_bps / 1e4
            w = target  # earns returns from the next day (i + 1) onwards
            weight_rows[date] = target
            turnover[date] = traded
            costs[date] = cost
        net_returns.append((1 + day_return) * (1 - cost) - 1)

    index = returns.index[start:]
    port = pd.Series(net_returns, index=index, name="returns")
    return BacktestResult(
        equity=(1 + port).cumprod().rename("equity"),
        returns=port,
        weights=pd.DataFrame.from_dict(weight_rows, orient="index", columns=tickers),
        turnover=pd.Series(turnover, name="turnover"),
        costs=pd.Series(costs, name="costs"),
    )


def _rebalance_dates(index: pd.DatetimeIndex, rebalance: str) -> pd.DatetimeIndex:
    """Last trading day of each period, excluding the final day of data (nothing follows)."""
    last_days = index.to_series().resample(rebalance).last().dropna()
    return pd.DatetimeIndex(last_days[last_days < index[-1]])


def _validate_weights(weights: pd.Series, tickers: pd.Index) -> np.ndarray:
    """Align strategy output to the price columns and check it is long-only and sums to 1."""
    unknown = weights.index.difference(tickers)
    if len(unknown):
        raise ValueError(f"strategy returned weights for unknown tickers: {list(unknown)}")
    w = weights.reindex(tickers, fill_value=0.0).to_numpy(dtype=float)
    if np.isnan(w).any() or (w < -1e-9).any() or not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(
            f"strategy weights must be long-only and sum to 1, got {weights.to_dict()}"
        )
    return w


def _annual_rf_at(rf: float | pd.Series, date: pd.Timestamp) -> float:
    """Annual risk-free rate known on ``date``; a daily Series is never read past it."""
    if not isinstance(rf, pd.Series):
        return rf
    known = rf.loc[:date].dropna()
    if known.empty:
        raise ValueError(f"no risk-free rate available on or before {date.date()}")
    return float((1 + known.iloc[-1]) ** PERIODS_PER_YEAR - 1)


def equal_weight(window: pd.DataFrame) -> pd.Series:
    """Strategy: ``1 / n`` in every asset.

    Parameters
    ----------
    window : pd.DataFrame
        Trailing returns (only the columns are used).

    Returns
    -------
    pd.Series
        Equal weights indexed by ticker.
    """
    n = len(window.columns)
    return pd.Series(1 / n, index=window.columns)


def fixed_weights(weights: Mapping[str, float]) -> Strategy:
    """Build a strategy that always targets the same weights (e.g. a 60/40 benchmark).

    Parameters
    ----------
    weights : Mapping[str, float]
        Target weights, e.g. ``{"SPY": 0.6, "AGG": 0.4}``. Must sum to 1.

    Returns
    -------
    Strategy
        A function returning these weights for any window.
    """
    target = pd.Series(dict(weights), dtype=float)
    if not np.isclose(target.sum(), 1.0):
        raise ValueError(f"weights must sum to 1, got {target.sum():.6f}")

    def strategy(window: pd.DataFrame) -> pd.Series:
        missing = target.index.difference(window.columns)
        if len(missing):
            raise ValueError(f"tickers not in prices: {list(missing)}")
        return target.reindex(window.columns, fill_value=0.0)

    return strategy


def min_variance_strategy(max_weight: float = 0.40) -> Strategy:
    """Build a strategy that rebalances to ``optimise.min_variance`` on the window."""

    def strategy(window: pd.DataFrame) -> pd.Series:
        return min_variance(window.cov() * PERIODS_PER_YEAR, max_weight)

    return strategy


def max_sharpe_strategy(
    rf: float | pd.Series = 0.0,
    max_weight: float = 0.40,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> Strategy:
    """Build a strategy that rebalances to ``optimise.max_sharpe`` on the window.

    ``rf`` is an annual float, or a daily rate Series (from ``data.get_risk_free``) of
    which only the latest value on or before the window's last date is used.
    ``periods_per_year`` annualises the window's mean and covariance (lower when assets on
    different calendars are aligned on common dates), so they are comparable with ``rf``.
    """

    def strategy(window: pd.DataFrame) -> pd.Series:
        mu = window.mean() * periods_per_year
        cov = window.cov() * periods_per_year
        return max_sharpe(mu, cov, _annual_rf_at(rf, window.index[-1]), max_weight)

    return strategy


def risk_parity_strategy(max_weight: float = 0.40) -> Strategy:
    """Build a strategy that rebalances to ``optimise.risk_parity`` on the window."""

    def strategy(window: pd.DataFrame) -> pd.Series:
        return risk_parity(window.cov() * PERIODS_PER_YEAR, max_weight)

    return strategy


def regime_switch_strategy(
    calm: Strategy,
    stressed: Strategy,
    p_stressed: pd.Series,
    threshold: float = 0.5,
) -> Strategy:
    """Build a strategy that switches to a defensive one when stress is forecast.

    Parameters
    ----------
    calm : Strategy
        Used when the latest P(stressed) is below ``threshold``.
    stressed : Strategy
        Used when the latest P(stressed) is at or above ``threshold`` (e.g. min variance).
    p_stressed : pd.Series
        Daily out-of-sample probabilities, e.g. from ``regime.walk_forward_probabilities``.
        Only the latest non-NaN value **on or before** the window's last date is read.
    threshold : float, default 0.5
        Probability at which to switch.

    Returns
    -------
    Strategy
        Falls back to ``calm`` until the first probability is available.
    """

    def strategy(window: pd.DataFrame) -> pd.Series:
        known = p_stressed.loc[: window.index[-1]].dropna()
        if not known.empty and known.iloc[-1] >= threshold:
            return stressed(window)
        return calm(window)

    return strategy


def spec_strategy(
    spec: OptimisationSpec,
    rf: float | pd.Series = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> Strategy:
    """Re-optimise ``spec`` each rebalance on the trailing window only (walk-forward).

    If the constraints can't be met on a window, the previous weights are kept (that month
    is recorded in ``strategy.infeasible_dates``); if they can't be met on the first window,
    ``ValueError`` explains why.
    """
    state: dict = {"last": None}

    def strategy(window: pd.DataFrame) -> pd.Series:
        mu = window.mean() * periods_per_year
        cov = window.cov() * periods_per_year
        result = optimise(mu, cov, spec, _annual_rf_at(rf, window.index[-1]))
        if result.feasible:
            state["last"] = result.weights
        elif state["last"] is None:
            raise ValueError(
                "constraints can't be met on the first window: " + " ".join(result.reasons)
            )
        else:
            strategy.infeasible_dates.append(window.index[-1])
        return state["last"]

    strategy.infeasible_dates = []
    return strategy
