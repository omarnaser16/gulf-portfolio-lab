"""ML regime classifier: predict next month's calm vs stressed regime.

Pure functions: no I/O, no printing, no plotting.

Timing rules (the whole point of this module)
---------------------------------------------
- **Features** at date *t* use data up to the close of *t* only.
- **Labels** at date *t* look *forward*: they describe the next ``horizon`` trading days.
  So the label for day *p* is only known at the close of day *p + horizon*.
- **Evaluation** uses ``TimeSeriesSplit`` with a ``gap`` of ``horizon`` rows, so no training
  label overlaps the test period. Never shuffled splits.
- **Walk-forward** models refit at date *t* only train on labels known by *t*
  (dates up to *t - horizon*).
"""

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gpl.metrics import PERIODS_PER_YEAR

logger = logging.getLogger(__name__)

type ModelKind = Literal["logistic", "random_forest"]

FEATURE_COLUMNS = [
    "vol_21",
    "vol_63",
    "drawdown_252",
    "mom_63",
    "mom_252",
    "vix",
    "vix_chg_21",
    "stock_bond_corr_63",
]


def make_features(
    prices: pd.DataFrame,
    vix: pd.Series,
    equity: str = "SPY",
    bond: str = "AGG",
) -> pd.DataFrame:
    """Build features known at the close of each date *t*.

    Parameters
    ----------
    prices : pd.DataFrame
        Cleaned prices including the ``equity`` and ``bond`` columns.
    vix : pd.Series
        ``^VIX`` level. Aligned to ``prices`` dates; a missing day takes the previous value.
    equity, bond : str, default "SPY", "AGG"
        Columns used for the equity and bond features.

    Returns
    -------
    pd.DataFrame
        Indexed like ``prices``, columns ``FEATURE_COLUMNS``:

        - ``vol_21``, ``vol_63``: trailing annualised volatility of equity returns.
        - ``drawdown_252``: equity price vs its 1-year high (<= 0).
        - ``mom_63``, ``mom_252``: equity return over 3 and 12 months.
        - ``vix``: VIX level; ``vix_chg_21``: log change in VIX over 21 days.
        - ``stock_bond_corr_63``: 3-month correlation of equity and bond daily returns.

        Early rows are NaN until each window has enough history.
    """
    px = prices[equity]
    r = px.pct_change()
    bond_r = prices[bond].pct_change()
    v = vix.reindex(prices.index).ffill()
    features = pd.DataFrame(
        {
            "vol_21": r.rolling(21).std() * np.sqrt(PERIODS_PER_YEAR),
            "vol_63": r.rolling(63).std() * np.sqrt(PERIODS_PER_YEAR),
            "drawdown_252": px / px.rolling(252).max() - 1,
            "mom_63": px / px.shift(63) - 1,
            "mom_252": px / px.shift(252) - 1,
            "vix": v,
            "vix_chg_21": np.log(v / v.shift(21)),
            "stock_bond_corr_63": r.rolling(63).corr(bond_r),
        },
        index=prices.index,
    )
    return features[FEATURE_COLUMNS]


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
        Forward window for realised volatility (about one month).
    quantile : float, default 0.80
        Threshold quantile of the rolling distribution of realised volatility.
    window : int, default 252
        Rolling window for the threshold (uses past data only).

    Returns
    -------
    pd.Series
        1.0 if volatility over days *t+1 ... t+horizon* is above the ``quantile`` of the
        past ``window`` days' trailing ``horizon``-day volatility, else 0.0. NaN for the
        last ``horizon`` dates (future unknown) and before the threshold has a full window.
    """
    trailing_vol = spy_returns.rolling(horizon).std()
    forward_vol = trailing_vol.shift(-horizon)  # std of returns t+1 .. t+horizon
    threshold = trailing_vol.rolling(window).quantile(quantile)  # known at t
    labels = (forward_vol > threshold).astype(float)
    labels[forward_vol.isna() | threshold.isna()] = np.nan
    return labels.rename("stressed")


def _clean_xy(
    features: pd.DataFrame, labels: pd.Series, log: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Rows where every feature and the label are known; logs how many were dropped."""
    data = features.join(labels.rename("_y"), how="inner")
    clean = data.dropna()
    dropped = len(data) - len(clean)
    if dropped and log:
        logger.warning("Dropped %d of %d rows with missing features or labels", dropped, len(data))
    return clean.drop(columns="_y"), clean["_y"]


def _new_model(model: ModelKind, seed: int) -> Any:
    # No class weighting: weighting the rare "stressed" class pushed the average predicted
    # probability to ~47% against a ~21% actual rate, so the output was a score rather than
    # a probability. Unweighted, the logistic intercept matches the training base rate.
    if model == "logistic":
        # Scaling matters for regularised logistic regression.
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if model == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=20,  # shallow-ish leaves: daily rows are highly autocorrelated
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"model must be 'logistic' or 'random_forest', got {model!r}")


def fit_model(
    features: pd.DataFrame,
    labels: pd.Series,
    model: ModelKind = "logistic",
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
        Logistic regression (with standardised features) or a random forest. Neither uses
        class weights, so ``predict_proba`` is a probability on the scale of the actual
        stressed rate (class weighting inflated it; see ``_new_model``).
    seed : int, default 0
        Random seed (used by the random forest).

    Returns
    -------
    Any
        A fitted scikit-learn estimator with ``predict_proba``.

    Raises
    ------
    ValueError
        If ``model`` is unknown or the labels contain only one class.
    """
    estimator = _new_model(model, seed)
    x, y = _clean_xy(features, labels)
    if y.nunique() < 2:
        raise ValueError("labels contain only one class; cannot fit a classifier")
    return estimator.fit(x.to_numpy(), y.to_numpy().astype(int))


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
        P(stressed next month), indexed like ``features``. NaN where any feature is missing.
    """
    complete = features.notna().all(axis=1)
    p = pd.Series(np.nan, index=features.index, name="p_stressed")
    if complete.any():
        p[complete] = model.predict_proba(features[complete].to_numpy())[:, 1]
    return p


def _scores(actual: np.ndarray, p: np.ndarray | None, flagged: np.ndarray) -> dict[str, float]:
    """Classification scores for one set of predictions (``p`` None for a 0/1 rule)."""
    both = len(np.unique(actual)) == 2
    score = p if p is not None else flagged
    return {
        "auc": roc_auc_score(actual, score) if both else np.nan,
        "precision": precision_score(actual, flagged, zero_division=0),
        "recall": recall_score(actual, flagged, zero_division=0),
        "accuracy": accuracy_score(actual, flagged),
        "flagged_share": flagged.mean(),
        "mean_p": p.mean() if p is not None else np.nan,
        "brier": brier_score_loss(actual, p) if p is not None else np.nan,
    }


def evaluate_model(
    features: pd.DataFrame,
    labels: pd.Series,
    model: ModelKind = "logistic",
    n_splits: int = 5,
    horizon: int = 21,
    threshold: float = 0.5,
    seed: int = 0,
    rule: pd.Series | None = None,
) -> dict[str, Any]:
    """Out-of-sample evaluation with expanding-window ``TimeSeriesSplit``.

    Parameters
    ----------
    features, labels
        As in ``fit_model``.
    model : {"logistic", "random_forest"}, default "logistic"
        Model type.
    n_splits : int, default 5
        Number of chronological train/test folds.
    horizon : int, default 21
        Gap (in rows) between each training set and its test fold. Labels look forward
        ``horizon`` days, so without the gap the last training labels would use returns
        from inside the test period.
    threshold : float, default 0.5
        Probability above which a day is classified as stressed.
    seed : int, default 0
        Random seed.
    rule : pd.Series, optional
        A 0/1 baseline signal (e.g. ``naive_stress_signal``) scored on the same test rows.

    Returns
    -------
    dict
        ``"folds"``: DataFrame, one row per fold: ``auc`` (ROC AUC: how well the model
        ranks stressed days above calm ones; 0.5 = coin flip), ``precision`` and
        ``recall`` on the stressed class, ``accuracy``, ``mean_p`` (average predicted
        probability) vs ``stressed_share`` (actual rate), the ``baseline_accuracy`` of
        always predicting calm and, if ``rule`` is given, ``rule_precision``,
        ``rule_recall`` and ``rule_accuracy``.
        ``"overall"``: DataFrame with rows ``model``, ``vol_rule`` (if given) and
        ``always_calm``, scored on all test rows together, plus ``stressed_share``.
        ``"confusion"``: summed 2x2 confusion matrix (rows actual, columns predicted).
        ``"probabilities"``: out-of-sample P(stressed) for every test row.
    """
    x, y = _clean_xy(features, labels)
    if rule is not None:
        rule = rule.reindex(x.index)
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    rows, confusion = [], np.zeros((2, 2), dtype=int)
    probabilities = pd.Series(np.nan, index=x.index, name="p_stressed")
    tested = np.zeros(len(x), dtype=bool)
    for train, test in splitter.split(x):
        fitted = _new_model(model, seed).fit(x.iloc[train].to_numpy(), y.iloc[train].astype(int))
        p = fitted.predict_proba(x.iloc[test].to_numpy())[:, 1]
        predicted = (p >= threshold).astype(int)
        actual = y.iloc[test].to_numpy().astype(int)
        probabilities.iloc[test] = p
        tested[test] = True
        confusion += confusion_matrix(actual, predicted, labels=[0, 1])
        row = {
            "test_start": x.index[test[0]],
            "test_end": x.index[test[-1]],
            "train_end_pos": int(train[-1]),
            "test_start_pos": int(test[0]),
            "n_test": len(test),
            "stressed_share": actual.mean(),
            **_scores(actual, p, predicted),
            "baseline_accuracy": 1 - actual.mean(),  # always predict calm
        }
        if rule is not None:
            r = rule.iloc[test]
            if r.isna().any():
                raise ValueError("rule has no value for some test rows")
            rule_scores = _scores(actual, None, r.to_numpy().astype(int))
            row |= {f"rule_{k}": rule_scores[k] for k in ("precision", "recall", "accuracy")}
        rows.append(row)

    actual_all = y[tested].to_numpy().astype(int)
    p_all = probabilities[tested].to_numpy()
    overall = {"model": _scores(actual_all, p_all, (p_all >= threshold).astype(int))}
    if rule is not None:
        overall["vol_rule"] = _scores(actual_all, None, rule[tested].to_numpy().astype(int))
    overall["always_calm"] = _scores(actual_all, None, np.zeros_like(actual_all))
    overall_table = pd.DataFrame(overall).T
    overall_table["stressed_share"] = actual_all.mean()

    index = pd.Index(["actual calm", "actual stressed"])
    return {
        "folds": pd.DataFrame(rows),
        "overall": overall_table,
        "confusion": pd.DataFrame(
            confusion, index=index, columns=["predicted calm", "predicted stressed"]
        ),
        "probabilities": probabilities.dropna(),
    }


def walk_forward_probabilities(
    features: pd.DataFrame,
    labels: pd.Series,
    model: ModelKind = "logistic",
    refit: str = "ME",
    min_train: int = 504,
    horizon: int = 21,
    seed: int = 0,
) -> pd.Series:
    """Out-of-sample P(stressed) for every date, as a live investor would have had it.

    At the last trading day *r* of each ``refit`` period, a fresh model is trained on rows
    up to *r - horizon* (the labels known by then) and used for dates *r* up to the next
    refit.

    Parameters
    ----------
    features, labels
        As in ``fit_model``; ``labels`` is aligned to ``features`` by date.
    model : {"logistic", "random_forest"}, default "logistic"
        Model type.
    refit : str, default "ME"
        Pandas offset alias for the refit schedule (month end).
    min_train : int, default 504
        Minimum number of complete training rows (about two years) before predicting.
    horizon : int, default 21
        Label horizon, used to hold back labels not yet known.
    seed : int, default 0
        Random seed.

    Returns
    -------
    pd.Series
        Indexed like ``features``. NaN before the first model, and where features are missing.
    """
    labels = labels.reindex(features.index)
    index = features.index
    refit_dates = index.to_series().resample(refit).last().dropna()
    refit_pos = [index.get_loc(d) for d in refit_dates]
    p = pd.Series(np.nan, index=index, name="p_stressed")
    for k, pos in enumerate(refit_pos):
        train_end = pos - horizon  # label at train_end is fully known at the close of pos
        if train_end < 0:
            continue
        x, y = _clean_xy(features.iloc[: train_end + 1], labels.iloc[: train_end + 1], log=False)
        if len(y) < min_train:
            continue
        stop = refit_pos[k + 1] if k + 1 < len(refit_pos) else len(index)
        block = features.iloc[pos:stop]
        if y.nunique() < 2:  # e.g. a long calm stretch: the only honest estimate is the base rate
            p.iloc[pos:stop] = np.where(block.notna().all(axis=1), y.iloc[0], np.nan)
            continue
        fitted = _new_model(model, seed).fit(x.to_numpy(), y.to_numpy().astype(int))
        p.iloc[pos:stop] = predict_regime(fitted, block).to_numpy()
    return p


def naive_stress_signal(
    features: pd.DataFrame, quantile: float = 0.80, window: int = 252
) -> pd.Series:
    """Rule-based baseline: "stressed" when trailing 21-day volatility is already high.

    Parameters
    ----------
    features : pd.DataFrame
        Output of ``make_features`` (uses ``vol_21``).
    quantile : float, default 0.80
        Same threshold quantile as the labels.
    window : int, default 252
        Rolling window for the threshold.

    Returns
    -------
    pd.Series
        1.0 when ``vol_21`` at *t* is above its rolling ``quantile`` up to *t*, else 0.0;
        NaN until the threshold is available. A "volatility persists" rule with no model,
        so the ML forecast has to beat it to be worth its complexity.
    """
    vol = features["vol_21"]
    threshold = vol.rolling(window).quantile(quantile)
    signal = (vol > threshold).astype(float)
    signal[threshold.isna()] = np.nan
    return signal.rename("p_stressed")
