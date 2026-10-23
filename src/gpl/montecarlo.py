"""Monte Carlo simulation of portfolio wealth paths.

Pure functions: no I/O, no printing, no plotting. Every stochastic function takes a
``seed`` so results are reproducible.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd


def simulate_paths(
    returns: pd.DataFrame,
    weights: pd.Series,
    horizon_days: int,
    n_paths: int,
    method: Literal["block_bootstrap", "bootstrap", "gbm"] = "block_bootstrap",
    *,
    seed: int,
    block_size: int = 21,
) -> np.ndarray:
    """Simulate future wealth paths for a fixed-weight portfolio.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical simple daily returns, one column per ticker.
    weights : pd.Series
        Portfolio weights indexed by ticker (rebalanced daily to these weights, no costs).
    horizon_days : int
        Number of trading days to simulate.
    n_paths : int
        Number of paths.
    method : {"block_bootstrap", "bootstrap", "gbm"}, default "block_bootstrap"
        ``"block_bootstrap"`` stitches together runs of ``block_size`` consecutive
        historical days (circular: a run may wrap from the last day to the first), so
        calm and turbulent stretches stay together. ``"bootstrap"`` resamples single
        days independently, which keeps fat tails but shuffles away volatility
        clustering. Both draw whole rows, so cross-asset correlation is kept.
        ``"gbm"`` uses geometric Brownian motion fitted to the portfolio's **log** returns.
    seed : int
        Seed for ``np.random.default_rng``.
    block_size : int, default 21
        Block length in trading days for ``"block_bootstrap"`` (21 = about a month).
        Capped at the length of the history.

    Returns
    -------
    np.ndarray
        Shape ``(horizon_days + 1, n_paths)``. Row 0 is 1.0 (starting wealth). Nominal,
        in the currency of ``returns`` (USD here); see ``to_real``.
    """
    if method not in ("block_bootstrap", "bootstrap", "gbm"):
        raise ValueError(f"method must be 'block_bootstrap', 'bootstrap' or 'gbm', got {method!r}")
    if block_size < 1:
        raise ValueError(f"block_size must be at least 1, got {block_size}")
    if returns.isna().any().any():
        raise ValueError("returns contain NaN; clean the data first")

    # Daily-rebalanced portfolio return for each historical day (whole rows keep correlation).
    port = (returns[weights.index] @ weights).to_numpy()
    n_hist = len(port)
    rng = np.random.default_rng(seed)

    if method == "block_bootstrap":
        block = min(block_size, n_hist)
        n_blocks = -(-horizon_days // block)  # ceiling division
        starts = rng.integers(0, n_hist, size=(n_blocks, 1, n_paths))
        offsets = np.arange(block)[None, :, None]
        days = ((starts + offsets) % n_hist).reshape(n_blocks * block, n_paths)[:horizon_days]
        log_growth = np.log1p(port[days])
    elif method == "bootstrap":
        days = rng.integers(0, n_hist, size=(horizon_days, n_paths))
        log_growth = np.log1p(port[days])
    else:
        # GBM: daily log returns are normal. Fitting mu to *log* returns already includes the
        # volatility drag, so no extra -sigma^2/2 term is needed.
        log_port = np.log1p(port)
        mu, sigma = log_port.mean(), log_port.std(ddof=1)
        log_growth = mu + sigma * rng.standard_normal(size=(horizon_days, n_paths))

    wealth = np.exp(np.cumsum(log_growth, axis=0))
    return np.vstack([np.ones(n_paths), wealth])


def to_real(paths: np.ndarray, inflation: float, periods_per_year: int = 252) -> np.ndarray:
    """Convert nominal wealth paths to real terms (today's purchasing power).

    Parameters
    ----------
    paths : np.ndarray
        Nominal paths from ``simulate_paths``; row *t* is wealth after *t* trading days.
    inflation : float
        Constant annual inflation rate, e.g. ``0.025``.
    periods_per_year : int, default 252
        Trading days per year.

    Returns
    -------
    np.ndarray
        ``paths[t] / (1 + inflation) ** (t / periods_per_year)``, same shape.
    """
    years = np.arange(paths.shape[0]) / periods_per_year
    return paths / ((1 + inflation) ** years)[:, None]


def terminal_stats(
    paths: np.ndarray,
    percentiles: Sequence[float] = (5, 25, 50, 75, 95),
) -> pd.Series:
    """Percentiles and mean of terminal wealth.

    Parameters
    ----------
    paths : np.ndarray
        Output of ``simulate_paths``.
    percentiles : Sequence[float], default (5, 25, 50, 75, 95)
        Percentiles to report.

    Returns
    -------
    pd.Series
        Index ``p5``, ``p25``, ... and ``mean``.
    """
    terminal = paths[-1]
    values = np.percentile(terminal, percentiles)
    stats = pd.Series(values, index=[f"p{p:g}" for p in percentiles])
    stats["mean"] = terminal.mean()
    return stats


def goal_probability(paths: np.ndarray, target: float) -> float:
    """Probability that terminal wealth reaches a target multiple.

    Parameters
    ----------
    paths : np.ndarray
        Output of ``simulate_paths``.
    target : float
        Target terminal wealth as a multiple of starting wealth, e.g. ``1.5``.

    Returns
    -------
    float
        Share of paths with terminal wealth ``>= target``.
    """
    return float((paths[-1] >= target).mean())
