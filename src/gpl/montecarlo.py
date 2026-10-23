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
    method: Literal["bootstrap", "gbm"] = "bootstrap",
    *,
    seed: int,
) -> np.ndarray:
    """Simulate future wealth paths for a fixed-weight portfolio.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical simple daily returns, one column per ticker.
    weights : pd.Series
        Portfolio weights indexed by ticker (rebalanced daily to these weights).
    horizon_days : int
        Number of trading days to simulate.
    n_paths : int
        Number of paths.
    method : {"bootstrap", "gbm"}, default "bootstrap"
        ``"bootstrap"`` resamples historical days (whole rows, so cross-asset correlation
        is kept). ``"gbm"`` uses geometric Brownian motion fitted to the portfolio's
        **log** returns.
    seed : int
        Seed for ``np.random.default_rng``.

    Returns
    -------
    np.ndarray
        Shape ``(horizon_days + 1, n_paths)``. Row 0 is 1.0 (starting wealth).
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
