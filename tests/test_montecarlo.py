"""Tests for gpl.montecarlo."""

import numpy as np
import pandas as pd
import pytest

from gpl.montecarlo import goal_probability, simulate_paths, terminal_stats


@pytest.fixture
def hist_returns() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data = rng.normal([0.0004, 0.0002], [0.012, 0.004], size=(500, 2))
    return pd.DataFrame(data, columns=["EQ", "BD"], index=pd.bdate_range("2022-01-03", periods=500))


@pytest.fixture
def weights() -> pd.Series:
    return pd.Series({"EQ": 0.6, "BD": 0.4})


@pytest.mark.parametrize("method", ["bootstrap", "gbm"])
def test_paths_shape_and_start(method, hist_returns, weights) -> None:
    paths = simulate_paths(
        hist_returns, weights, horizon_days=20, n_paths=50, method=method, seed=0
    )
    assert paths.shape == (21, 50)
    np.testing.assert_array_equal(paths[0], 1.0)
    assert (paths > 0).all()


@pytest.mark.parametrize("method", ["bootstrap", "gbm"])
def test_seed_makes_it_reproducible(method, hist_returns, weights) -> None:
    a = simulate_paths(hist_returns, weights, 30, 100, method=method, seed=42)
    b = simulate_paths(hist_returns, weights, 30, 100, method=method, seed=42)
    c = simulate_paths(hist_returns, weights, 30, 100, method=method, seed=43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("method", ["bootstrap", "gbm"])
def test_constant_returns_are_deterministic(method) -> None:
    # +0.1% every day, no risk: every path is 1.001 ** t whatever the method
    r = pd.DataFrame({"X": [0.001] * 50}, index=pd.bdate_range("2024-01-01", periods=50))
    paths = simulate_paths(r, pd.Series({"X": 1.0}), 10, 5, method=method, seed=0)
    expected = 1.001 ** np.arange(11)
    np.testing.assert_allclose(paths, np.tile(expected[:, None], (1, 5)))


def test_bootstrap_only_reuses_history() -> None:
    r = pd.DataFrame({"X": [0.01, -0.02, 0.03]}, index=pd.bdate_range("2024-01-01", periods=3))
    paths = simulate_paths(r, pd.Series({"X": 1.0}), 5, 200, seed=0)
    daily = paths[1:] / paths[:-1] - 1
    assert set(np.round(daily.ravel(), 10)) <= {0.01, -0.02, 0.03}


def test_bootstrap_draws_whole_days_keeping_correlation() -> None:
    # Two assets that always move in opposite directions: a 50/50 portfolio never moves
    # if days are drawn as whole rows.
    r = pd.DataFrame(
        {"A": [0.02, -0.01, 0.03], "B": [-0.02, 0.01, -0.03]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    paths = simulate_paths(r, pd.Series({"A": 0.5, "B": 0.5}), 10, 20, seed=0)
    np.testing.assert_allclose(paths, 1.0)


def test_gbm_matches_log_return_drift(hist_returns, weights) -> None:
    port = np.log1p(hist_returns @ weights)
    paths = simulate_paths(hist_returns, weights, 252, 20_000, method="gbm", seed=0)
    mean_log_growth = np.log(paths[-1]).mean()
    assert mean_log_growth == pytest.approx(
        252 * port.mean(), abs=4 * np.sqrt(252) * port.std() / np.sqrt(20_000)
    )


def test_unknown_method_and_nans_raise(hist_returns, weights) -> None:
    with pytest.raises(ValueError):
        simulate_paths(hist_returns, weights, 5, 5, method="magic", seed=0)
    bad = hist_returns.copy()
    bad.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        simulate_paths(bad, weights, 5, 5, seed=0)


# ------------------------------------------------------------------------- terminal stats


@pytest.fixture
def hand_paths() -> np.ndarray:
    """101 paths whose terminal wealth is 0.00, 0.01, ..., 1.00 (start row is 1)."""
    return np.vstack([np.ones(101), np.linspace(0, 1, 101)])


def test_terminal_stats_hand_values(hand_paths) -> None:
    stats = terminal_stats(hand_paths)
    assert list(stats.index) == ["p5", "p25", "p50", "p75", "p95", "mean"]
    assert stats["p5"] == pytest.approx(0.05)
    assert stats["p50"] == pytest.approx(0.50)
    assert stats["mean"] == pytest.approx(0.50)


def test_goal_probability_hand_value() -> None:
    paths = np.array([[1.0, 1.0, 1.0, 1.0], [0.5, 1.0, 1.5, 2.0]])
    assert goal_probability(paths, target=1.5) == pytest.approx(0.5)  # 1.5 and 2.0 hit it
    assert goal_probability(paths, target=3.0) == 0.0
