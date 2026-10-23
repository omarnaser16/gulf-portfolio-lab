"""Periodically rebalanced backtester with transaction costs.

Pure functions: no I/O, no printing, no plotting.

No look-ahead bias: weights set at rebalance date *t* use returns up to and including *t*
only, and apply from the next trading day. Between rebalances, weights drift with prices.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd

type Strategy = Callable[[pd.DataFrame], pd.Series]
"""Maps a trailing window of daily returns (rows = dates, columns = tickers) to target
weights indexed by ticker."""


@dataclass(frozen=True)
class BacktestResult:
    """Output of ``run_backtest``.

    Attributes
    ----------
    equity : pd.Series
        Portfolio wealth, starting at 1.0.
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def min_variance_strategy(max_weight: float = 0.40) -> Strategy:
    """Build a strategy that rebalances to ``optimise.min_variance`` on the window."""
    raise NotImplementedError


def max_sharpe_strategy(rf: float = 0.0, max_weight: float = 0.40) -> Strategy:
    """Build a strategy that rebalances to ``optimise.max_sharpe`` on the window."""
    raise NotImplementedError


def risk_parity_strategy() -> Strategy:
    """Build a strategy that rebalances to ``optimise.risk_parity`` on the window."""
    raise NotImplementedError
