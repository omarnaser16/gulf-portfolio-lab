"""Tests for gpl.backtest. Synthetic prices over Jan-Mar 2024 so every number is checkable.

Month-end rebalance dates in the data: 31 Jan, 29 Feb (29 Mar is the last day of data,
so nothing can be applied after it and it is skipped).
"""

import numpy as np
import pandas as pd
import pytest

from gpl.backtest import (
    BacktestResult,
    equal_weight,
    fixed_weights,
    max_sharpe_strategy,
    min_variance_strategy,
    regime_switch_strategy,
    risk_parity_strategy,
    run_backtest,
    spec_strategy,
)
from gpl.optimise import OptimisationSpec

DATES = pd.bdate_range("2024-01-01", "2024-03-29")  # 65 business days


def flat_prices(**jumps: tuple[str, float]) -> pd.DataFrame:
    """Two assets at 100. ``jumps={"A": ("2024-02-01", 0.10)}`` moves A +10% from that day on."""
    prices = pd.DataFrame(100.0, index=DATES, columns=["A", "B"])
    for ticker, (day, size) in jumps.items():
        prices.loc[day:, ticker] *= 1 + size
    return prices


@pytest.fixture
def random_returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = rng.normal(0.0004, 0.01, size=(300, 3)) * np.array([1.0, 2.0, 0.5])
    return pd.DataFrame(
        data, columns=["A", "B", "C"], index=pd.bdate_range("2023-01-02", periods=300)
    )


# ------------------------------------------------------------------------------ mechanics


def test_returns_backtest_result() -> None:
    res = run_backtest(flat_prices(), equal_weight, lookback=5, cost_bps=0)
    assert isinstance(res, BacktestResult)
    assert list(res.weights.index) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    assert res.equity.index[0] == pd.Timestamp("2024-01-31")
    assert res.equity.index[-1] == pd.Timestamp("2024-03-29")


def test_single_asset_tracks_its_price() -> None:
    prices = flat_prices()
    prices["A"] = 100 * 1.001 ** np.arange(len(DATES))  # +0.1% a day
    res = run_backtest(prices, fixed_weights({"A": 1.0}), lookback=5, cost_bps=0)
    expected = prices["A"].loc["2024-01-31":] / prices["A"].loc["2024-01-31"]
    np.testing.assert_allclose(res.equity.to_numpy(), expected.to_numpy())


def test_initial_purchase_costs_turnover_of_one() -> None:
    res = run_backtest(flat_prices(), equal_weight, lookback=5, cost_bps=10)
    assert res.turnover.iloc[0] == pytest.approx(1.0)  # from cash into the portfolio
    assert res.costs.iloc[0] == pytest.approx(0.001)
    assert res.equity.iloc[0] == pytest.approx(0.999)


def test_weights_drift_and_turnover_hand_value() -> None:
    # A +10% on 1 Feb. 50/50 drifts to 0.55/0.45 of 1.05 = 0.5238/0.4762 by 29 Feb.
    res = run_backtest(flat_prices(A=("2024-02-01", 0.10)), equal_weight, lookback=5, cost_bps=0)
    assert res.returns.loc["2024-02-01"] == pytest.approx(0.05)
    drifted_a = 0.5 * 1.10 / 1.05
    assert res.turnover.loc["2024-02-29"] == pytest.approx(2 * (drifted_a - 0.5))  # 0.047619


def test_costs_reduce_wealth() -> None:
    prices = flat_prices(A=("2024-02-01", 0.10))
    free = run_backtest(prices, equal_weight, lookback=5, cost_bps=0)
    costly = run_backtest(prices, equal_weight, lookback=5, cost_bps=10)
    expected_drag = (1 - 0.001) * (1 - 0.001 * free.turnover.loc["2024-02-29"])
    assert costly.equity.iloc[-1] == pytest.approx(free.equity.iloc[-1] * expected_drag)


def test_short_history_rebalances_are_skipped() -> None:
    # 22 returns exist by 31 Jan: not enough for lookback=30, so start on 29 Feb
    res = run_backtest(flat_prices(), equal_weight, lookback=30, cost_bps=0)
    assert res.weights.index[0] == pd.Timestamp("2024-02-29")


def test_nan_prices_raise() -> None:
    prices = flat_prices()
    prices.iloc[10, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_backtest(prices, equal_weight, lookback=5)


# ----------------------------------------------------------------------- no look-ahead bias


def test_strategy_only_sees_the_past() -> None:
    seen: list[pd.DataFrame] = []

    def spy_on_window(window: pd.DataFrame) -> pd.Series:
        seen.append(window)
        return equal_weight(window)

    res = run_backtest(flat_prices(), spy_on_window, lookback=5, cost_bps=0)
    assert [w.index[-1] for w in seen] == list(res.weights.index)  # window ends at t
    assert all(len(w) == 5 for w in seen)


def test_new_weights_apply_from_next_day() -> None:
    # A jumps +10% ON 29 Feb (a rebalance date). A "cheating" strategy that sees the jump
    # and buys A must not earn it: the trade happens at the close, after the jump.
    def chase_winner(window: pd.DataFrame) -> pd.Series:
        last = window.iloc[-1]
        return pd.Series({"A": 1.0, "B": 0.0}) if last["A"] > 0 else pd.Series({"A": 0.0, "B": 1.0})

    res = run_backtest(flat_prices(A=("2024-02-29", 0.10)), chase_winner, lookback=5, cost_bps=0)
    assert res.weights.loc["2024-02-29", "A"] == 1.0  # it did switch...
    assert res.returns.loc["2024-02-29"] == 0.0  # ...but held B on the jump day
    assert res.equity.iloc[-1] == pytest.approx(1.0)


# ------------------------------------------------------------------------------ strategies


def test_equal_weight(random_returns: pd.DataFrame) -> None:
    w = equal_weight(random_returns)
    np.testing.assert_allclose(w.to_numpy(), 1 / 3)


def test_fixed_weights_validates() -> None:
    with pytest.raises(ValueError):
        fixed_weights({"A": 0.6, "B": 0.3})  # sums to 0.9
    strat = fixed_weights({"A": 1.0})
    with pytest.raises(ValueError, match="Z"):
        fixed_weights({"Z": 1.0})(pd.DataFrame(columns=["A", "B"]))
    w = strat(pd.DataFrame(columns=["A", "B"], dtype=float))
    assert w.to_dict() == {"A": 1.0, "B": 0.0}


@pytest.mark.parametrize("make", [min_variance_strategy, max_sharpe_strategy, risk_parity_strategy])
def test_optimised_strategies_give_valid_weights(make, random_returns: pd.DataFrame) -> None:
    w = make()(random_returns)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()
    assert list(w.index) == ["A", "B", "C"]
    assert (w <= 0.40 + 1e-9).all()  # every strategy respects the default 40% cap


def test_min_variance_strategy_favours_low_vol(random_returns: pd.DataFrame) -> None:
    w = min_variance_strategy(max_weight=1.0)(random_returns)
    assert w["C"] > w["A"] > w["B"]  # vol multipliers 0.5 < 1 < 2


def test_max_sharpe_strategy_ignores_future_rf(random_returns: pd.DataFrame) -> None:
    # A daily rf series that is 2% a year up to the window end and absurd afterwards
    end = random_returns.index[-1]
    idx = pd.bdate_range(random_returns.index[0], end + pd.Timedelta(days=60))
    daily = pd.Series((1.02) ** (1 / 252) - 1, index=idx)
    daily.loc[daily.index > end] = 0.01  # ~1100% a year: must never be used
    w_series = max_sharpe_strategy(rf=daily)(random_returns)
    w_float = max_sharpe_strategy(rf=0.02)(random_returns)
    np.testing.assert_allclose(w_series.to_numpy(), w_float.to_numpy(), atol=1e-6)


# -------------------------------------------------------------------------- regime switch

CALM = fixed_weights({"A": 1.0})
STRESSED = fixed_weights({"B": 1.0})


def _p(values: dict[str, float]) -> pd.Series:
    return pd.Series(values, dtype=float).rename(index=pd.Timestamp)


def test_regime_switch_picks_by_threshold() -> None:
    p = _p({"2024-01-31": 0.2, "2024-02-29": 0.7})
    res = run_backtest(flat_prices(), regime_switch_strategy(CALM, STRESSED, p), lookback=5)
    assert res.weights.loc["2024-01-31", "A"] == 1.0  # 0.2 < 0.5: calm
    assert res.weights.loc["2024-02-29", "B"] == 1.0  # 0.7 >= 0.5: stressed


def test_regime_switch_uses_latest_known_probability() -> None:
    # Daily probabilities: high only on 1 Feb. At 29 Feb the latest value is 0.1 -> calm.
    p = pd.Series(0.1, index=DATES)
    p.loc["2024-02-01"] = 0.9
    w = regime_switch_strategy(CALM, STRESSED, p)(
        pd.DataFrame(columns=["A", "B"], index=DATES[:42])
    )
    assert w["A"] == 1.0  # window ends 29 Feb (index 41)


def test_regime_switch_never_reads_future_probabilities() -> None:
    # p jumps to 1.0 the day AFTER the window ends: must still be calm
    p = pd.Series(0.0, index=DATES)
    p.loc["2024-03-01":] = 1.0
    window = pd.DataFrame(0.0, columns=["A", "B"], index=DATES[DATES <= "2024-02-29"])
    assert regime_switch_strategy(CALM, STRESSED, p)(window)["A"] == 1.0


def test_regime_switch_without_signal_stays_calm() -> None:
    p = pd.Series([np.nan, 0.9], index=pd.to_datetime(["2024-01-02", "2024-12-31"]))
    window = pd.DataFrame(0.0, columns=["A", "B"], index=DATES[:20])
    assert regime_switch_strategy(CALM, STRESSED, p)(window)["A"] == 1.0


def test_max_sharpe_strategy_periods_per_year_scale(random_returns: pd.DataFrame) -> None:
    # With rf = 0, rescaling mu and cov by the same factor leaves the tangency weights unchanged;
    # with rf > 0 the annualisation matters, so the parameter must reach the optimiser.
    a = max_sharpe_strategy(rf=0.0, periods_per_year=252)(random_returns)
    b = max_sharpe_strategy(rf=0.0, periods_per_year=183)(random_returns)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-5)
    c = max_sharpe_strategy(rf=0.03, max_weight=1.0, periods_per_year=252)(random_returns)
    d = max_sharpe_strategy(rf=0.03, max_weight=1.0, periods_per_year=120)(random_returns)
    assert not np.allclose(c.to_numpy(), d.to_numpy(), atol=1e-3)


# ------------------------------------------------------------------ spec walk-forward strategy


def test_spec_strategy_respects_limits_each_month(random_returns: pd.DataFrame) -> None:
    spec = OptimisationSpec(
        "max_sharpe", max_weight=0.6, groups={"A": "GCC"}, group_bounds={"GCC": (0.3, None)}
    )
    prices = 100 * (1 + random_returns).cumprod()
    res = run_backtest(prices, spec_strategy(spec), lookback=60, cost_bps=0)
    assert (res.weights["A"] >= 0.3 - 1e-6).all()
    assert (res.weights.max(axis=1) <= 0.6 + 1e-6).all()


def test_spec_strategy_infeasible_first_window_explains(random_returns: pd.DataFrame) -> None:
    impossible = OptimisationSpec("min_variance", max_volatility=0.001)
    with pytest.raises(ValueError, match="lowest volatility"):
        spec_strategy(impossible)(random_returns)


def test_spec_strategy_keeps_last_weights_when_a_window_fails(random_returns: pd.DataFrame) -> None:
    strategy = spec_strategy(OptimisationSpec("min_variance", max_volatility=0.12))
    first = strategy(random_returns.iloc[:120])  # calm enough to meet the cap
    stormy = random_returns.iloc[120:240] * 6  # far too volatile for a 12% cap
    assert strategy(stormy).equals(first)
    assert strategy.infeasible_dates == [stormy.index[-1]]
