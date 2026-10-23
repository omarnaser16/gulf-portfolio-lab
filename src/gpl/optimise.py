"""Portfolio construction: mean-variance and risk parity.

Pure functions: no I/O, no printing, no plotting.

Conventions
-----------
``mu`` is a ``pd.Series`` of **annualised** expected returns and ``cov`` an annualised
covariance ``pd.DataFrame``, both indexed by ticker. Weights are long-only, sum to 1 and
are returned as a ``pd.Series`` indexed by ticker. Default max weight is 0.40 per asset.
"""

import numpy as np
import pandas as pd

type Weights = pd.Series | np.ndarray

DEFAULT_MAX_WEIGHT = 0.40


def portfolio_performance(
    weights: Weights,
    mu: pd.Series,
    cov: pd.DataFrame,
) -> tuple[float, float]:
    """Expected return and volatility of a portfolio.

    Parameters
    ----------
    weights : pd.Series or np.ndarray
        Portfolio weights, same order as ``mu`` and ``cov``.
    mu : pd.Series
        Annualised expected returns.
    cov : pd.DataFrame
        Annualised covariance matrix.

    Returns
    -------
    tuple of float
        ``(w @ mu, sqrt(w @ cov @ w))``.
    """
    raise NotImplementedError


def min_variance(cov: pd.DataFrame, max_weight: float = DEFAULT_MAX_WEIGHT) -> pd.Series:
    """Long-only minimum-variance portfolio (SLSQP).

    Parameters
    ----------
    cov : pd.DataFrame
        Annualised covariance matrix.
    max_weight : float, default 0.40
        Upper bound on each weight.

    Returns
    -------
    pd.Series
        Optimal weights indexed by ticker.

    Raises
    ------
    ValueError
        If ``max_weight * n_assets < 1`` (no feasible portfolio), or the optimiser fails.
    """
    raise NotImplementedError


def max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> pd.Series:
    """Long-only maximum-Sharpe (tangency) portfolio (SLSQP).

    Parameters
    ----------
    mu : pd.Series
        Annualised expected returns.
    cov : pd.DataFrame
        Annualised covariance matrix.
    rf : float, default 0.0
        Annual risk-free rate.
    max_weight : float, default 0.40
        Upper bound on each weight.

    Returns
    -------
    pd.Series
        Optimal weights indexed by ticker.

    Raises
    ------
    ValueError
        If no feasible portfolio exists, or the optimiser fails.
    """
    raise NotImplementedError


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 50,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> pd.DataFrame:
    """Minimum-variance portfolios for a grid of target returns.

    Parameters
    ----------
    mu : pd.Series
        Annualised expected returns.
    cov : pd.DataFrame
        Annualised covariance matrix.
    n_points : int, default 50
        Number of target returns, from the min-variance return to the highest
        achievable return under the weight bounds.
    max_weight : float, default 0.40
        Upper bound on each weight.

    Returns
    -------
    pd.DataFrame
        One row per point. Columns ``ret``, ``vol``, then one weight column per ticker.
    """
    raise NotImplementedError


def risk_contributions(weights: Weights, cov: pd.DataFrame) -> pd.Series:
    """Each asset's share of portfolio variance.

    Parameters
    ----------
    weights : pd.Series or np.ndarray
        Portfolio weights.
    cov : pd.DataFrame
        Covariance matrix.

    Returns
    -------
    pd.Series
        ``w_i * (cov @ w)_i / (w @ cov @ w)``, indexed by ticker; sums to 1.
    """
    raise NotImplementedError


def risk_parity(cov: pd.DataFrame) -> pd.Series:
    """Long-only equal-risk-contribution portfolio.

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix.

    Returns
    -------
    pd.Series
        Weights indexed by ticker such that every asset's risk contribution is
        ``1 / n_assets``.
    """
    raise NotImplementedError
