"""Tests for gpl.montecarlo."""

import numpy as np
import pandas as pd
import pytest

from gpl.metrics import max_drawdown
from gpl.montecarlo import (
    goal_probability,
    path_max_drawdowns,
    simulate_paths,
    terminal_stats,
    to_real,
)


@pytest.fixture
def hist_returns() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data = rng.normal([0.0004, 0.0002], [0.012, 0.004], size=(500, 2))
    return pd.DataFrame(data, columns=["EQ", "BD"], index=pd.bdate_range("2022-01-03", periods=500))


@pytest.fixture
def weights() -> pd.Series:
    return pd.Series({"EQ": 0.6, "BD": 0.4})


@pytest.mark.parametrize("method", ["block_bootstrap", "bootstrap", "gbm"])
def test_paths_shape_and_start(method, hist_returns, weights) -> None:
    paths = simulate_paths(
        hist_returns, weights, horizon_days=20, n_paths=50, method=method, seed=0
    )
    assert paths.shape == (21, 50)
    np.testing.assert_array_equal(paths[0], 1.0)
    assert (paths > 0).all()


@pytest.mark.parametrize("method", ["block_bootstrap", "bootstrap", "gbm"])
def test_seed_makes_it_reproducible(method, hist_returns, weights) -> None:
    a = simulate_paths(hist_returns, weights, 30, 100, method=method, seed=42)
    b = simulate_paths(hist_returns, weights, 30, 100, method=method, seed=42)
    c = simulate_paths(hist_returns, weights, 30, 100, method=method, seed=43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("method", ["block_bootstrap", "bootstrap", "gbm"])
def test_constant_returns_are_deterministic(method) -> None:
    # +0.1% every day, no risk: every path is 1.001 ** t whatever the method
    r = pd.DataFrame({"X": [0.001] * 50}, index=pd.bdate_range("2024-01-01", periods=50))
    paths = simulate_paths(r, pd.Series({"X": 1.0}), 10, 5, method=method, seed=0)
    expected = 1.001 ** np.arange(11)
    np.testing.assert_allclose(paths, np.tile(expected[:, None], (1, 5)))


def test_bootstrap_only_reuses_history() -> None:
    r = pd.DataFrame({"X": [0.01, -0.02, 0.03]}, index=pd.bdate_range("2024-01-01", periods=3))
    paths = simulate_paths(r, pd.Series({"X": 1.0}), 5, 200, method="bootstrap", seed=0)
    daily = paths[1:] / paths[:-1] - 1
    assert set(np.round(daily.ravel(), 10)) <= {0.01, -0.02, 0.03}


def test_bootstrap_draws_whole_days_keeping_correlation() -> None:
    # Two assets that always move in opposite directions: a 50/50 portfolio never moves
    # if days are drawn as whole rows.
    r = pd.DataFrame(
        {"A": [0.02, -0.01, 0.03], "B": [-0.02, 0.01, -0.03]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    paths = simulate_paths(r, pd.Series({"A": 0.5, "B": 0.5}), 10, 20, method="bootstrap", seed=0)
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


# ------------------------------------------------------------------------ block bootstrap


def _drawn_days(paths: np.ndarray) -> np.ndarray:
    """Recover which historical day each simulated day used (history day i returned i * 1e-5)."""
    daily = paths[1:] / paths[:-1] - 1
    return np.rint(daily / 1e-5).astype(int)


@pytest.fixture
def numbered_days() -> pd.DataFrame:
    """100 days whose return encodes the day number: day i returns i * 0.00001."""
    return pd.DataFrame(
        {"X": np.arange(100) * 1e-5}, index=pd.bdate_range("2024-01-01", periods=100)
    )


def test_block_bootstrap_is_the_default(numbered_days) -> None:
    default = simulate_paths(numbered_days, pd.Series({"X": 1.0}), 42, 30, seed=3)
    explicit = simulate_paths(
        numbered_days, pd.Series({"X": 1.0}), 42, 30, method="block_bootstrap", seed=3
    )
    np.testing.assert_array_equal(default, explicit)


def test_block_bootstrap_draws_consecutive_days_in_blocks(numbered_days) -> None:
    paths = simulate_paths(
        numbered_days,
        pd.Series({"X": 1.0}),
        63,
        40,
        method="block_bootstrap",
        block_size=21,
        seed=0,
    )
    days = _drawn_days(paths)  # shape (63, 40)
    for t in range(62):
        same_block = (t + 1) % 21 != 0
        follows = days[t + 1] == (days[t] + 1) % 100  # next day, wrapping round the end
        if same_block:
            assert follows.all(), f"day {t} -> {t + 1} broke a block"
    # block starts are random: across blocks, consecutive days are rare
    assert not (days[21] == (days[20] + 1) % 100).all()


def test_block_size_one_is_plain_bootstrap_shape(numbered_days) -> None:
    days = _drawn_days(
        simulate_paths(numbered_days, pd.Series({"X": 1.0}), 50, 200, block_size=1, seed=0)
    )
    follows = days[1:] == (days[:-1] + 1) % 100
    assert follows.mean() < 0.05  # independent days: ~1% follow on by chance


def test_block_longer_than_history_is_capped(numbered_days) -> None:
    short = numbered_days.iloc[:10]
    paths = simulate_paths(short, pd.Series({"X": 1.0}), 30, 5, block_size=21, seed=0)
    assert paths.shape == (31, 5)
    assert set(_drawn_days(paths).ravel()) <= set(range(10))


def test_bad_block_size_raises(numbered_days) -> None:
    with pytest.raises(ValueError):
        simulate_paths(numbered_days, pd.Series({"X": 1.0}), 10, 5, block_size=0, seed=0)


def test_block_bootstrap_keeps_volatility_clustering() -> None:
    """History: calm and wild months alternate. Squared returns are positively
    autocorrelated; blocks keep that, shuffling single days destroys it."""
    rng = np.random.default_rng(5)
    vol = np.repeat(np.tile([0.005, 0.03], 20), 21)  # 40 alternating 21-day regimes
    r = pd.DataFrame(
        {"X": rng.normal(0, vol)}, index=pd.bdate_range("2010-01-01", periods=len(vol))
    )

    def lag1_autocorr_of_squares(method: str) -> float:
        paths = simulate_paths(r, pd.Series({"X": 1.0}), 252, 400, method=method, seed=1)
        sq = (paths[1:] / paths[:-1] - 1) ** 2
        a, b = sq[:-1].ravel(), sq[1:].ravel()
        return float(np.corrcoef(a, b)[0, 1])

    sq = r["X"].to_numpy() ** 2
    history = float(np.corrcoef(sq[:-1], sq[1:])[0, 1])  # ~0.2: volatility clusters
    assert history > 0.1
    assert lag1_autocorr_of_squares("block_bootstrap") == pytest.approx(history, abs=0.05)
    assert abs(lag1_autocorr_of_squares("bootstrap")) < 0.03  # shuffling destroys it


# ------------------------------------------------------------------------------ real terms


def test_to_real_hand_values() -> None:
    # nominal wealth 1.025 after exactly one year at 2.5% inflation is 1.0 in real terms
    nominal = np.ones((253, 2))
    nominal[252] = 1.025
    real = to_real(nominal, inflation=0.025)
    np.testing.assert_allclose(real[0], 1.0)
    np.testing.assert_allclose(real[252], 1.0)
    assert real[126, 0] == pytest.approx(1 / 1.025**0.5)  # half a year of inflation


def test_to_real_zero_inflation_is_identity(hist_returns, weights) -> None:
    paths = simulate_paths(hist_returns, weights, 30, 10, seed=0)
    np.testing.assert_array_equal(to_real(paths, 0.0), paths)


# ------------------------------------------------------------------ drawdown along the way


def test_path_max_drawdowns_hand_values() -> None:
    # columns are paths: 1 -> 1.2 -> 0.9 -> 1.3   (worst: 0.9 / 1.2 - 1 = -25%)
    #                    1 -> 0.8 -> 1.0 -> 1.1   (worst: 0.8 / 1 - 1 = -20%; start is a peak)
    #                    1 -> 1.1 -> 1.2 -> 1.3   (never falls: 0)
    paths = np.array([[1.0, 1.0, 1.0], [1.2, 0.8, 1.1], [0.9, 1.0, 1.2], [1.3, 1.1, 1.3]])
    np.testing.assert_allclose(path_max_drawdowns(paths), [-0.25, -0.20, 0.0])


def test_path_drawdowns_match_the_metrics_module(hist_returns, weights) -> None:
    paths = simulate_paths(hist_returns, weights, 60, 3, seed=0)
    daily = pd.DataFrame(paths[1:] / paths[:-1] - 1)
    np.testing.assert_allclose(path_max_drawdowns(paths), max_drawdown(daily).to_numpy())


def test_blocks_give_deeper_drawdowns_when_crashes_cluster() -> None:
    # A history where all the bad days come in one 21-day run
    r = np.full(500, 0.001)
    r[200:221] = -0.02
    hist = pd.DataFrame({"X": r}, index=pd.bdate_range("2015-01-01", periods=500))
    w = pd.Series({"X": 1.0})
    block = path_max_drawdowns(simulate_paths(hist, w, 252, 2000, seed=0))
    daily = path_max_drawdowns(simulate_paths(hist, w, 252, 2000, method="bootstrap", seed=0))
    assert np.percentile(block, 5) < np.percentile(daily, 5) - 0.05
