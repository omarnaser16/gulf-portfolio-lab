"""Tests for gpl.regime. Most checks are look-ahead properties: change the future, the past
must not move."""

import logging

import numpy as np
import pandas as pd
import pytest

from gpl.regime import (
    FEATURE_COLUMNS,
    evaluate_model,
    fit_model,
    make_features,
    make_labels,
    naive_stress_signal,
    predict_regime,
    walk_forward_probabilities,
)

# ---------------------------------------------------------------------------- fixtures


@pytest.fixture
def block_returns() -> pd.Series:
    """Deterministic: +/-0.5% alternating for 400 days, +/-3% for 60 days, then +/-0.5%."""
    calm, wild = 0.005, 0.03
    signs = np.where(np.arange(560) % 2 == 0, 1.0, -1.0)
    size = np.r_[np.full(400, calm), np.full(60, wild), np.full(100, calm)]
    return pd.Series(signs * size, index=pd.bdate_range("2010-01-01", periods=560), name="SPY")


@pytest.fixture
def market() -> tuple[pd.DataFrame, pd.Series]:
    """1,000 days of SPY/AGG prices with two stress blocks, and a VIX that tracks SPY vol."""
    rng = np.random.default_rng(7)
    n = 1000
    vol = np.full(n, 0.006)
    vol[400:460] = 0.025
    vol[750:800] = 0.02
    spy = rng.normal(0.0003, vol)
    agg = rng.normal(0.0001, 0.003, n) - 0.1 * spy
    idx = pd.bdate_range("2012-01-02", periods=n)
    prices = pd.DataFrame(
        {"SPY": 100 * np.cumprod(1 + spy), "AGG": 100 * np.cumprod(1 + agg)}, index=idx
    )
    realised = pd.Series(spy, index=idx).rolling(10, min_periods=1).std() * np.sqrt(252) * 100
    vix = (realised + 12 + rng.normal(0, 1, n)).rename("^VIX")
    return prices, vix


@pytest.fixture
def separable() -> tuple[pd.DataFrame, pd.Series]:
    """1,500 rows where stress is (mostly) high x1: an easy problem a model must learn."""
    rng = np.random.default_rng(3)
    n = 1500
    x = pd.DataFrame(
        {"x1": rng.normal(size=n), "x2": rng.normal(size=n)},
        index=pd.bdate_range("2010-01-01", periods=n),
    )
    y = ((x["x1"] + 0.3 * rng.normal(size=n)) > 0.85).astype(float)  # ~20% stressed
    return x, y


# ------------------------------------------------------------------------------ labels


def test_labels_unknown_at_the_end(block_returns: pd.Series) -> None:
    labels = make_labels(block_returns)
    assert labels.iloc[-21:].isna().all()
    assert labels.iloc[:252].isna().all()  # threshold needs a year of history
    assert set(labels.dropna().unique()) <= {0.0, 1.0}


def test_labels_flag_stress_before_it_happens(block_returns: pd.Series) -> None:
    labels = make_labels(block_returns)
    # day 399 is the last calm day: its next 21 days are all wild
    assert labels.iloc[399] == 1.0
    # day 385: 7 of its next 21 days are wild -> forward vol above the calm threshold
    assert labels.iloc[385] == 1.0


def test_labels_calm_after_stress(block_returns: pd.Series) -> None:
    # day 480: next 21 days are calm, and the past year's threshold now includes the wild days
    assert make_labels(block_returns).iloc[480] == 0.0


def test_labels_ignore_returns_beyond_the_horizon(market) -> None:
    prices, _ = market
    r = prices["SPY"].pct_change().iloc[1:]
    base = make_labels(r)
    t = 600
    shocked = r.copy()
    shocked.iloc[t + 22 :] *= 5  # only returns after t+21 change
    pd.testing.assert_series_equal(make_labels(shocked).iloc[: t + 1], base.iloc[: t + 1])


# ---------------------------------------------------------------------------- features


def test_feature_columns(market) -> None:
    prices, vix = market
    feats = make_features(prices, vix)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert feats.index.equals(prices.index)


def test_features_use_only_the_past(market) -> None:
    prices, vix = market
    full = make_features(prices, vix)
    for t in [300, 420, 900]:
        cut = prices.index[t]
        truncated = make_features(prices.loc[:cut], vix.loc[:cut])
        pd.testing.assert_series_equal(truncated.loc[cut], full.loc[cut])


def test_vol_21_hand_value(market) -> None:
    prices, vix = market
    feats = make_features(prices, vix)
    r = prices["SPY"].pct_change()
    expected = r.iloc[480 - 20 : 481].std() * np.sqrt(252)
    assert feats["vol_21"].iloc[480] == pytest.approx(expected)


def test_missing_vix_day_is_carried_forward(market) -> None:
    prices, vix = market
    feats = make_features(prices, vix.drop(vix.index[500]))
    assert feats["vix"].iloc[500] == vix.iloc[499]


# -------------------------------------------------------------------- fit and predict


@pytest.mark.parametrize("kind", ["logistic", "random_forest"])
def test_model_learns_an_easy_signal(kind, separable) -> None:
    x, y = separable
    model = fit_model(x.iloc[:1000], y.iloc[:1000], model=kind, seed=0)
    p = predict_regime(model, x.iloc[1000:])
    y_test = y.iloc[1000:]
    assert p.between(0, 1).all()
    assert p[y_test == 1].mean() > p[y_test == 0].mean() + 0.3


def test_random_forest_is_reproducible(separable) -> None:
    x, y = separable
    a = predict_regime(fit_model(x, y, "random_forest", seed=1), x)
    b = predict_regime(fit_model(x, y, "random_forest", seed=1), x)
    pd.testing.assert_series_equal(a, b)


def test_fit_logs_dropped_rows(separable, caplog: pytest.LogCaptureFixture) -> None:
    x, y = separable
    y = y.copy()
    y.iloc[:10] = np.nan
    with caplog.at_level(logging.WARNING, logger="gpl.regime"):
        fit_model(x, y)
    assert any("10" in rec.getMessage() for rec in caplog.records)


def test_predict_keeps_index_and_nans(separable) -> None:
    x, y = separable
    model = fit_model(x, y)
    x_gap = x.copy()
    x_gap.iloc[5, 0] = np.nan
    p = predict_regime(model, x_gap)
    assert p.index.equals(x.index)
    assert np.isnan(p.iloc[5])
    assert p.drop(x.index[5]).notna().all()


def test_bad_inputs_raise(separable) -> None:
    x, y = separable
    with pytest.raises(ValueError):
        fit_model(x, y, model="xgboost")
    with pytest.raises(ValueError, match="one class"):
        fit_model(x, y * 0)


# ---------------------------------------------------------------------------- evaluate


def test_evaluate_model_reports_folds_and_baseline(separable) -> None:
    x, y = separable
    report = evaluate_model(x, y, model="logistic", n_splits=4, horizon=21)
    folds = report["folds"]
    assert len(folds) == 4
    assert {"precision", "recall", "accuracy", "baseline_accuracy", "stressed_share"} <= set(folds)
    np.testing.assert_allclose(folds["baseline_accuracy"], 1 - folds["stressed_share"])
    assert folds["recall"].mean() > 0.5  # learns the easy signal
    assert report["confusion"].to_numpy().sum() == folds["n_test"].sum()
    assert report["probabilities"].notna().sum() == folds["n_test"].sum()


def test_evaluate_model_leaves_a_gap(separable) -> None:
    x, y = separable
    folds = evaluate_model(x, y, n_splits=3, horizon=21)["folds"]
    # each test fold starts more than 21 rows after its training data ends
    assert (folds["test_start_pos"] - folds["train_end_pos"] > 21).all()


# ------------------------------------------------------------------------ walk-forward


def test_walk_forward_only_uses_labels_known_at_refit(separable) -> None:
    x, y = separable
    base = walk_forward_probabilities(x, y, min_train=300, horizon=21)
    t = 1100
    changed = y.copy()
    changed.iloc[t - 21 + 1 :] = 1 - changed.iloc[t - 21 + 1 :]  # rewrite labels not yet known at t
    other = walk_forward_probabilities(x, changed, min_train=300, horizon=21)
    pd.testing.assert_series_equal(other.iloc[: t + 1], base.iloc[: t + 1])
    assert not other.iloc[t + 1 :].equals(base.iloc[t + 1 :])  # later refits do see them


def test_walk_forward_starts_after_min_train(separable) -> None:
    x, y = separable
    p = walk_forward_probabilities(x, y, min_train=500, horizon=21)
    first = p.first_valid_index()
    # a refit at position pos trains on rows 0 .. pos - 21, i.e. pos - 20 rows
    assert x.index.get_loc(first) - 20 >= 500
    assert p.loc[first:].notna().all()
    assert p.dropna().between(0, 1).all()


# -------------------------------------------------------------------------- naive signal


def test_naive_signal_fires_once_volatility_is_high(block_returns: pd.Series) -> None:
    feats = pd.DataFrame({"vol_21": block_returns.rolling(21).std()})
    signal = naive_stress_signal(feats)
    assert signal.iloc[:271].isna().all()
    assert signal.iloc[399] == 0.0  # stress hasn't started: a backward-looking rule can't see it
    assert signal.iloc[410] == 1.0  # ten wild days in: now it fires


def test_naive_signal_uses_only_the_past(market) -> None:
    prices, vix = market
    feats = make_features(prices, vix)
    base = naive_stress_signal(feats)
    t = 700
    truncated = naive_stress_signal(feats.iloc[: t + 1])
    pd.testing.assert_series_equal(truncated, base.iloc[: t + 1])
