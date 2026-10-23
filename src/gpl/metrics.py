"""Risk and performance metrics.

Pure functions: no I/O, no printing, no plotting.

Conventions
-----------
Inputs are simple daily returns, either a ``pd.Series`` (one asset or strategy) or a
``pd.DataFrame`` (one column each). A Series in gives a float out; a DataFrame in gives a
``pd.Series`` indexed by column. Annualisation uses 252 trading days. VaR and Expected
Shortfall are 1-day figures reported as positive losses. Max drawdown is negative.
"""

import pandas as pd

type Returns = pd.Series | pd.DataFrame
type Scalar = float | pd.Series

PERIODS_PER_YEAR = 252


def annualised_return(returns: Returns, periods_per_year: int = PERIODS_PER_YEAR) -> Scalar:
    """Arithmetic mean daily return scaled to a year.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Simple periodic returns.
    periods_per_year : int, default 252
        Periods in one year.

    Returns
    -------
    float or pd.Series
        ``mean(r) * periods_per_year``.
    """
    raise NotImplementedError


def cagr(returns: Returns, periods_per_year: int = PERIODS_PER_YEAR) -> Scalar:
    """Compound annual growth rate (geometric average annual return).

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Simple periodic returns.
    periods_per_year : int, default 252
        Periods in one year.

    Returns
    -------
    float or pd.Series
        ``prod(1 + r) ** (periods_per_year / n) - 1`` where ``n`` is the number of returns.
    """
    raise NotImplementedError


def annualised_volatility(returns: Returns, periods_per_year: int = PERIODS_PER_YEAR) -> Scalar:
    """Sample standard deviation of returns scaled to a year.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Simple periodic returns.
    periods_per_year : int, default 252
        Periods in one year.

    Returns
    -------
    float or pd.Series
        ``std(r, ddof=1) * sqrt(periods_per_year)``.
    """
    raise NotImplementedError


def sharpe_ratio(
    returns: Returns,
    rf: float | pd.Series = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> Scalar:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Simple daily returns.
    rf : float or pd.Series, default 0.0
        A float is an **annual** risk-free rate, converted to daily with
        ``(1 + rf) ** (1/252) - 1``. A Series is a **daily** rate aligned on dates
        (e.g. from ``data.get_risk_free``).
    periods_per_year : int, default 252
        Periods in one year.

    Returns
    -------
    float or pd.Series
        Annualised mean excess return divided by annualised volatility of excess returns.
        NaN where volatility is zero.
    """
    raise NotImplementedError


def sortino_ratio(
    returns: Returns,
    rf: float | pd.Series = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> Scalar:
    """Annualised Sortino ratio: like Sharpe, but only downside deviation in the denominator.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Simple daily returns.
    rf : float or pd.Series, default 0.0
        As in ``sharpe_ratio``. Also used as the target return for downside deviation.
    periods_per_year : int, default 252
        Periods in one year.

    Returns
    -------
    float or pd.Series
        Annualised mean excess return divided by annualised downside deviation,
        ``sqrt(mean(min(r - rf, 0) ** 2)) * sqrt(periods_per_year)``.
        NaN where downside deviation is zero.
    """
    raise NotImplementedError


def drawdown_series(returns: Returns) -> Returns:
    """Drawdown from the running peak of cumulative wealth, at every date.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Simple daily returns.

    Returns
    -------
    pd.Series or pd.DataFrame
        Same shape as ``returns``. ``wealth / running_max(wealth) - 1``, so values are
        ``<= 0``, and 0 at a new high. Wealth starts at 1 before the first return.
    """
    raise NotImplementedError


def max_drawdown(returns: Returns) -> Scalar:
    """Worst peak-to-trough fall in cumulative wealth.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Simple daily returns.

    Returns
    -------
    float or pd.Series
        Minimum of ``drawdown_series``, a negative number (e.g. ``-0.34``) or 0.
    """
    raise NotImplementedError


def historical_var(returns: Returns, level: float = 0.95) -> Scalar:
    """1-day historical Value at Risk.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Simple daily returns.
    level : float, default 0.95
        Confidence level.

    Returns
    -------
    float or pd.Series
        Minus the ``1 - level`` empirical quantile of returns, a **positive loss**.
    """
    raise NotImplementedError


def parametric_var(returns: Returns, level: float = 0.95) -> Scalar:
    """1-day Gaussian (variance-covariance) Value at Risk.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Simple daily returns.
    level : float, default 0.95
        Confidence level.

    Returns
    -------
    float or pd.Series
        ``-(mean + z * std)`` with ``z = norm.ppf(1 - level)``, a **positive loss**.
    """
    raise NotImplementedError


def expected_shortfall(returns: Returns, level: float = 0.95) -> Scalar:
    """1-day historical Expected Shortfall (CVaR).

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Simple daily returns.
    level : float, default 0.95
        Confidence level.

    Returns
    -------
    float or pd.Series
        Minus the mean of returns at or below the ``1 - level`` quantile, a
        **positive loss**. Always ``>= historical_var`` at the same level.
    """
    raise NotImplementedError


def summary_table(
    returns: Returns,
    rf: float | pd.Series = 0.0,
    level: float = 0.95,
) -> pd.DataFrame:
    """All headline metrics in one table.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Simple daily returns, one column per asset or strategy.
    rf : float or pd.Series, default 0.0
        As in ``sharpe_ratio``.
    level : float, default 0.95
        Confidence level for VaR and Expected Shortfall.

    Returns
    -------
    pd.DataFrame
        One row per asset or strategy. Columns: ``ann_return``, ``cagr``, ``ann_vol``,
        ``sharpe``, ``sortino``, ``max_drawdown``, ``hist_var``, ``param_var``,
        ``expected_shortfall``, ``start``, ``end``, ``n_obs``.
    """
    raise NotImplementedError
