"""ML regime classifier: predict next month's calm vs stressed regime (Week 2).

Pure functions: no I/O, no printing, no plotting.

Every feature at date *t* uses data up to *t* only; the label looks forward. Evaluate
with ``TimeSeriesSplit`` only, never shuffled splits.
"""

from typing import Any, Literal

import pandas as pd


def make_features(prices: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """Build features known at each date *t*.

    Parameters
    ----------
    prices : pd.DataFrame
        Cleaned prices including ``SPY`` and ``AGG``.
    vix : pd.Series
        ``^VIX`` level.

    Returns
    -------
    pd.DataFrame
        One row per date. Columns include trailing volatility, drawdown, VIX level and
        change, momentum and rolling bond-equity correlation.
    """
    raise NotImplementedError


def make_labels(
    spy_returns: pd.Series,
    horizon: int = 21,
    quantile: float = 0.80,
    window: int = 252,
) -> pd.Series:
    """Label each date as stressed (1) or calm (0) using **forward** realised volatility.

    Parameters
    ----------
    spy_returns : pd.Series
        Simple daily returns of ``SPY``.
    horizon : int, default 21
        Forward window for realised volatility.
    quantile : float, default 0.80
        Threshold quantile of the rolling distribution of realised volatility.
    window : int, default 252
        Rolling window for the threshold (uses past data only).

    Returns
    -------
    pd.Series
        0/1 labels; NaN for the last ``horizon`` dates, where the future is unknown.
    """
    raise NotImplementedError


def fit_model(
    features: pd.DataFrame,
    labels: pd.Series,
    model: Literal["logistic", "random_forest"] = "logistic",
    seed: int = 0,
) -> Any:
    """Fit a classifier for the stressed regime.

    Parameters
    ----------
    features : pd.DataFrame
        Output of ``make_features``.
    labels : pd.Series
        Output of ``make_labels``, aligned on dates.
    model : {"logistic", "random_forest"}, default "logistic"
        Model type.
    seed : int, default 0
        Random seed.

    Returns
    -------
    Any
        A fitted scikit-learn estimator (scikit-learn to be added as a dependency in Week 2).
    """
    raise NotImplementedError


def predict_regime(model: Any, features: pd.DataFrame) -> pd.Series:
    """Probability of the stressed regime at each date.

    Parameters
    ----------
    model : Any
        Output of ``fit_model``.
    features : pd.DataFrame
        Output of ``make_features``.

    Returns
    -------
    pd.Series
        P(stressed next month), indexed like ``features``.
    """
    raise NotImplementedError
