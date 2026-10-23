"""Tests for gpl.optimise.

With **uncorrelated** assets the optimal weights have closed forms, so expected values
can be checked by hand:

- min variance: w_i proportional to 1 / variance_i
- max Sharpe:   w_i proportional to (mu_i - rf) / variance_i
"""

import numpy as np
import pandas as pd
import pytest

from gpl.optimise import (
    efficient_frontier,
    max_sharpe,
    min_variance,
    portfolio_performance,
    risk_contributions,
    risk_parity,
)

TICKERS = ["A", "B", "C"]


@pytest.fixture
def mu3() -> pd.Series:
    return pd.Series([0.10, 0.06, 0.04], index=TICKERS)


@pytest.fixture
def cov3() -> pd.DataFrame:
    """Uncorrelated: vols 20%, 10%, 10% -> variances 0.04, 0.01, 0.01."""
    return pd.DataFrame(np.diag([0.04, 0.01, 0.01]), index=TICKERS, columns=TICKERS)


@pytest.fixture
def cov_corr() -> pd.DataFrame:
    """Correlated 3-asset covariance (vols 20%, 15%, 5%; correlations 0.6, 0.2, -0.1)."""
    vols = np.array([0.20, 0.15, 0.05])
    corr = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, -0.1], [0.2, -0.1, 1.0]])
    return pd.DataFrame(np.outer(vols, vols) * corr, index=TICKERS, columns=TICKERS)


# ------------------------------------------------------------------ portfolio_performance


def test_portfolio_performance_hand_values(mu3, cov3) -> None:
    w = np.array([0.2, 0.4, 0.4])
    ret, vol = portfolio_performance(w, mu3, cov3)
    assert ret == pytest.approx(0.2 * 0.10 + 0.4 * 0.06 + 0.4 * 0.04)  # 0.06
    # variance = 0.2^2 * 0.04 + 0.4^2 * 0.01 + 0.4^2 * 0.01 = 0.0048
    assert vol == pytest.approx(np.sqrt(0.0048))


def test_portfolio_performance_accepts_series(mu3, cov3) -> None:
    w = pd.Series([0.4, 0.4, 0.2], index=["B", "C", "A"])  # different order: align by name
    ret, _ = portfolio_performance(w, mu3, cov3)
    assert ret == pytest.approx(0.06)


# ---------------------------------------------------------------------------- min_variance


def test_min_variance_unconstrained_is_inverse_variance(cov3) -> None:
    # 1/var = 25, 100, 100 -> weights 1/9, 4/9, 4/9
    w = min_variance(cov3, max_weight=1.0)
    np.testing.assert_allclose(w.to_numpy(), [1 / 9, 4 / 9, 4 / 9], atol=1e-5)
    assert list(w.index) == TICKERS


def test_min_variance_respects_cap(cov3) -> None:
    # B and C would like 44% each; capped at 40%, so A takes the remaining 20%
    w = min_variance(cov3, max_weight=0.40)
    np.testing.assert_allclose(w.to_numpy(), [0.2, 0.4, 0.4], atol=1e-5)


def test_min_variance_long_only_and_sums_to_one(cov_corr) -> None:
    w = min_variance(cov_corr, max_weight=0.6)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()
    assert (w <= 0.6 + 1e-9).all()


def test_min_variance_beats_equal_weight(cov_corr) -> None:
    mu = pd.Series(0.0, index=TICKERS)
    w = min_variance(cov_corr, max_weight=1.0)
    _, vol_opt = portfolio_performance(w, mu, cov_corr)
    _, vol_eq = portfolio_performance(np.full(3, 1 / 3), mu, cov_corr)
    assert vol_opt < vol_eq


def test_min_variance_infeasible_cap_raises(cov3) -> None:
    with pytest.raises(ValueError):
        min_variance(cov3, max_weight=0.30)  # 3 x 30% < 100%


# ------------------------------------------------------------------------------ max_sharpe


def test_max_sharpe_unconstrained_closed_form() -> None:
    # (mu - rf) / var = 0.10/0.04, 0.05/0.01 = 2.5, 5 -> weights 1/3, 2/3
    mu = pd.Series([0.10, 0.05], index=["A", "B"])
    cov = pd.DataFrame(np.diag([0.04, 0.01]), index=["A", "B"], columns=["A", "B"])
    w = max_sharpe(mu, cov, rf=0.0, max_weight=1.0)
    np.testing.assert_allclose(w.to_numpy(), [1 / 3, 2 / 3], atol=1e-4)


def test_max_sharpe_uses_rf() -> None:
    # rf = 2%: (0.08/0.04, 0.03/0.01) = 2, 3 -> weights 0.4, 0.6
    mu = pd.Series([0.10, 0.05], index=["A", "B"])
    cov = pd.DataFrame(np.diag([0.04, 0.01]), index=["A", "B"], columns=["A", "B"])
    w = max_sharpe(mu, cov, rf=0.02, max_weight=1.0)
    np.testing.assert_allclose(w.to_numpy(), [0.4, 0.6], atol=1e-4)


def test_max_sharpe_beats_min_variance_on_sharpe(mu3, cov_corr) -> None:
    def sharpe(w):
        ret, vol = portfolio_performance(w, mu3, cov_corr)
        return ret / vol

    w_ms = max_sharpe(mu3, cov_corr, max_weight=0.6)
    w_mv = min_variance(cov_corr, max_weight=0.6)
    assert sharpe(w_ms) >= sharpe(w_mv) - 1e-9
    assert (w_ms <= 0.6 + 1e-9).all()


# ---------------------------------------------------------------------- efficient_frontier


def test_efficient_frontier_shape_and_bounds(mu3, cov_corr) -> None:
    front = efficient_frontier(mu3, cov_corr, n_points=10, max_weight=0.6)
    assert len(front) == 10
    assert list(front.columns) == ["ret", "vol", *TICKERS]
    np.testing.assert_allclose(front[TICKERS].sum(axis=1), 1.0, atol=1e-6)
    assert (front[TICKERS] >= -1e-9).all().all()
    assert (front[TICKERS] <= 0.6 + 1e-9).all().all()


def test_efficient_frontier_starts_at_min_variance(mu3, cov_corr) -> None:
    front = efficient_frontier(mu3, cov_corr, n_points=10, max_weight=0.6)
    _, vol_mv = portfolio_performance(min_variance(cov_corr, 0.6), mu3, cov_corr)
    assert front["vol"].iloc[0] == pytest.approx(vol_mv, rel=1e-4)


def test_efficient_frontier_more_return_costs_more_risk(mu3, cov_corr) -> None:
    front = efficient_frontier(mu3, cov_corr, n_points=10, max_weight=0.6)
    assert front["ret"].is_monotonic_increasing
    assert (front["vol"].diff().dropna() >= -1e-8).all()
    # top point: best achievable return under the cap = 0.6 * 10% + 0.4 * 6%
    assert front["ret"].iloc[-1] == pytest.approx(0.6 * 0.10 + 0.4 * 0.06, abs=1e-6)


# ---------------------------------------------------------------------- risk_contributions


def test_risk_contributions_hand_values(cov3) -> None:
    # w = (0.2, 0.4, 0.4); variance pieces w_i^2 * var_i = 0.0016, 0.0016, 0.0016 (equal!)
    rc = risk_contributions(np.array([0.2, 0.4, 0.4]), cov3)
    np.testing.assert_allclose(rc.to_numpy(), [1 / 3, 1 / 3, 1 / 3])
    assert list(rc.index) == TICKERS


def test_risk_contributions_equal_weight_is_not_equal_risk(cov3) -> None:
    # Equal weights: pieces are 0.04, 0.01, 0.01 (/9) -> A carries 4/6 of the risk
    rc = risk_contributions(np.full(3, 1 / 3), cov3)
    np.testing.assert_allclose(rc.to_numpy(), [4 / 6, 1 / 6, 1 / 6])


def test_risk_contributions_sum_to_one(cov_corr) -> None:
    rc = risk_contributions(pd.Series([0.5, 0.3, 0.2], index=TICKERS), cov_corr)
    assert rc.sum() == pytest.approx(1.0)


# ----------------------------------------------------------------------------- risk_parity


def test_risk_parity_uncorrelated_is_inverse_vol(cov3) -> None:
    # 1/vol = 5, 10, 10 -> weights 0.2, 0.4, 0.4
    w = risk_parity(cov3)
    np.testing.assert_allclose(w.to_numpy(), [0.2, 0.4, 0.4], atol=1e-6)


def test_risk_parity_equalises_risk_when_correlated(cov_corr) -> None:
    w = risk_parity(cov_corr, max_weight=1.0)
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()
    np.testing.assert_allclose(risk_contributions(w, cov_corr).to_numpy(), 1 / 3, atol=1e-6)


def test_risk_parity_gives_low_vol_asset_more_weight(cov_corr) -> None:
    w = risk_parity(cov_corr, max_weight=1.0)
    assert w["C"] > w["B"] > w["A"]  # vols 5% < 15% < 20%


def test_risk_parity_rejects_zero_variance() -> None:
    cov = pd.DataFrame(np.diag([0.04, 0.0]), index=["A", "B"], columns=["A", "B"])
    with pytest.raises(ValueError, match="variance"):
        risk_parity(cov, max_weight=1.0)


@pytest.fixture
def cov_one_safe() -> pd.DataFrame:
    """Uncorrelated: A has 5% vol, B, C, D have 20%. Uncapped risk parity is
    1/vol = 20, 5, 5, 5 -> A = 20/35 = 57%, above the 40% cap."""
    tickers = ["A", "B", "C", "D"]
    return pd.DataFrame(np.diag([0.0025, 0.04, 0.04, 0.04]), index=tickers, columns=tickers)


def test_risk_parity_uncapped_breaches_40pct(cov_one_safe) -> None:
    w = risk_parity(cov_one_safe, max_weight=1.0)
    assert w["A"] == pytest.approx(20 / 35)


def test_risk_parity_cap_hand_values(cov_one_safe) -> None:
    # A is held at 40%; B, C, D are identical so they split the remaining 60% equally
    w = risk_parity(cov_one_safe, max_weight=0.40)
    np.testing.assert_allclose(w.to_numpy(), [0.4, 0.2, 0.2, 0.2], atol=1e-9)
    rc = risk_contributions(w, cov_one_safe)
    assert rc["A"] < rc["B"]  # the capped asset carries less risk than the others


def test_risk_parity_cap_with_correlation(cov_corr) -> None:
    uncapped = risk_parity(cov_corr, max_weight=1.0)
    assert uncapped.max() > 0.6  # C (5% vol) wants most of the portfolio
    w = risk_parity(cov_corr, max_weight=0.6)
    assert w.sum() == pytest.approx(1.0)
    assert w.max() == pytest.approx(0.6)
    rc = risk_contributions(w, cov_corr)
    assert rc["A"] == pytest.approx(rc["B"], rel=1e-8)  # uncapped assets still equal
    assert rc["C"] < rc["A"]


def test_risk_parity_cap_not_binding_is_unchanged(cov_corr) -> None:
    pd.testing.assert_series_equal(risk_parity(cov_corr, 1.0), risk_parity(cov_corr, 0.99))


def test_risk_parity_infeasible_cap_raises(cov3) -> None:
    with pytest.raises(ValueError):
        risk_parity(cov3, max_weight=0.30)
