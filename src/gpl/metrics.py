"""Risk and performance metrics.

Pure functions: no I/O, no printing, no plotting.

Conventions
-----------
Inputs are simple daily returns, either a ``pd.Series`` (one asset or strategy) or a
``pd.DataFrame`` (one column each). A Series in gives a float out; a DataFrame in gives a
``pd.Series`` indexed by column. Annualisation uses 252 trading days. VaR and Expected
Shortfall are 1-day figures reported as positive losses. Max drawdown is negative.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

type Returns = pd.Series | pd.DataFrame
type Scalar = float | pd.Series

PERIODS_PER_YEAR = 252
_ZERO_TOL = 1e-12  # risk below this is treated as zero (constant returns)


def _excess_returns(returns: Returns, rf: float | pd.Series, periods_per_year: int) -> Returns:
    """Subtract the risk-free return per observation.

    A float ``rf`` is annual. A Series is a daily rate (per 252-day year); when observations are
    sparser than daily (assets on different calendars aligned on common dates), each one is
    charged ``252 / periods_per_year`` days of interest so the annual total stays right.
    """
    if isinstance(rf, pd.Series):
        daily_rf = rf.reindex(returns.index).ffill()
        if daily_rf.isna().any():
            raise ValueError("rf series does not cover the start of the returns period")
        if periods_per_year != PERIODS_PER_YEAR:
            daily_rf = (1 + daily_rf) ** (PERIODS_PER_YEAR / periods_per_year) - 1
        if isinstance(returns, pd.DataFrame):
            return returns.sub(daily_rf, axis=0)
        return returns - daily_rf
    return returns - ((1 + rf) ** (1 / periods_per_year) - 1)


def _check_level(level: float) -> None:
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1, got {level}")


def _safe_divide(numerator: Scalar, denominator: Scalar) -> Scalar:
    """Ratio that is NaN, not inf, where the denominator (a risk measure) is ~zero."""
    if isinstance(denominator, pd.Series):
        return numerator / denominator.where(denominator > _ZERO_TOL)
    return float(numerator / denominator) if denominator > _ZERO_TOL else float("nan")


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
    return returns.mean() * periods_per_year


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
    n = returns.count()  # non-NaN observations
    growth = (1 + returns).prod()  # NaNs are skipped
    if isinstance(n, pd.Series):
        # Empty column -> NaN, not a fake 0% (numpy says 1 ** nan == 1).
        return (growth ** (periods_per_year / n) - 1).where(n > 0)
    if n == 0:
        return float("nan")
    return growth ** (periods_per_year / n) - 1


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
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


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
    excess = _excess_returns(returns, rf, periods_per_year)
    return _safe_divide(
        annualised_return(excess, periods_per_year),
        annualised_volatility(excess, periods_per_year),
    )


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
    excess = _excess_returns(returns, rf, periods_per_year)
    downside = np.sqrt((excess.clip(upper=0) ** 2).mean()) * np.sqrt(periods_per_year)
    return _safe_divide(annualised_return(excess, periods_per_year), downside)


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
    wealth = (1 + returns).cumprod()
    # Include the starting wealth of 1, so a loss on day one counts as a drawdown.
    peak = wealth.cummax().clip(lower=1.0)
    return wealth / peak - 1


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
    return drawdown_series(returns).min()


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
    _check_level(level)
    return -returns.quantile(1 - level)


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
    _check_level(level)
    z = norm.ppf(1 - level)  # e.g. -1.645 at 95%
    return -(returns.mean() + z * returns.std(ddof=1))


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
    _check_level(level)
    cutoff = returns.quantile(1 - level)
    return -returns.where(returns <= cutoff).mean()


def summary_table(
    returns: Returns,
    rf: float | pd.Series = 0.0,
    level: float = 0.95,
    periods_per_year: int = PERIODS_PER_YEAR,
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
    periods_per_year : int, default 252
        Return observations per year, used to annualise. Lower when assets on different
        trading calendars are aligned on common dates (see ``data.load_assets``).

    Returns
    -------
    pd.DataFrame
        One row per asset or strategy. Columns: ``ann_return``, ``cagr``, ``ann_vol``,
        ``sharpe``, ``sortino``, ``max_drawdown``, ``hist_var``, ``param_var``,
        ``expected_shortfall``, ``start``, ``end``, ``n_obs``.
    """
    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
    table = pd.DataFrame(
        {
            "ann_return": annualised_return(frame, periods_per_year),
            "cagr": cagr(frame, periods_per_year),
            "ann_vol": annualised_volatility(frame, periods_per_year),
            "sharpe": sharpe_ratio(frame, rf, periods_per_year),
            "sortino": sortino_ratio(frame, rf, periods_per_year),
            "max_drawdown": max_drawdown(frame),
            "hist_var": historical_var(frame, level),
            "param_var": parametric_var(frame, level),
            "expected_shortfall": expected_shortfall(frame, level),
            "start": frame.apply(lambda col: col.first_valid_index()),
            "end": frame.apply(lambda col: col.last_valid_index()),
            "n_obs": frame.count(),
        }
    )
    return table
