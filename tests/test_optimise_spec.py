"""Tests for the constrained optimiser: objectives, asset and group limits, targets, and the
plain-language reasons when a request can't be met.

Hand-checkable two-asset problem (uncorrelated):
    A: return 10%, volatility 20% (variance 0.04)      B: return 5%, volatility 10% (0.01)
"""

import math

import numpy as np
import pandas as pd
import pytest

from gpl.optimise import (
    OptimisationSpec,
    max_sharpe,
    min_variance,
    optimise,
)

MU = pd.Series({"A": 0.10, "B": 0.05})
COV = pd.DataFrame(np.diag([0.04, 0.01]), index=["A", "B"], columns=["A", "B"])
GCC = {"A": "GCC", "B": "US"}


def weights(result) -> tuple[float, float]:
    assert result.feasible, result.reasons
    return tuple(result.weights[["A", "B"]].round(6))


# ------------------------------------------------------------------------- objectives


def test_matches_existing_functions_without_extra_limits() -> None:
    mv = optimise(MU, COV, OptimisationSpec("min_variance"))
    np.testing.assert_allclose(mv.weights, min_variance(COV, 1.0), atol=1e-5)  # 0.2 / 0.8
    ms = optimise(MU, COV, OptimisationSpec("max_sharpe"))
    np.testing.assert_allclose(ms.weights, max_sharpe(MU, COV, 0.0, 1.0), atol=1e-4)  # 1/3, 2/3
    assert ms.sharpe == pytest.approx(ms.expected_return / ms.volatility)


def test_group_minimum_binds() -> None:
    # Unconstrained min variance holds 20% in A; "at least 60% GCC" forces A to 60%
    spec = OptimisationSpec("min_variance", groups=GCC, group_bounds={"GCC": (0.6, None)})
    result = optimise(MU, COV, spec)
    assert weights(result) == pytest.approx((0.6, 0.4), abs=1e-5)
    assert "group 'GCC' at its 60.00% minimum" in result.binding


def test_asset_bounds() -> None:
    result = optimise(MU, COV, OptimisationSpec("min_variance", asset_bounds={"B": (None, 0.5)}))
    assert weights(result) == pytest.approx((0.5, 0.5), abs=1e-5)
    assert "B at its 50.00% maximum" in result.binding


def test_target_return_hand_value() -> None:
    # lowest volatility with return >= 8%: 0.10a + 0.05(1 - a) = 0.08  ->  a = 0.6
    result = optimise(MU, COV, OptimisationSpec("target_return", min_return=0.08))
    assert weights(result) == pytest.approx((0.6, 0.4), abs=1e-5)
    assert result.expected_return == pytest.approx(0.08)
    assert "return at the 8.00% floor" in result.binding


def test_target_volatility_hand_value() -> None:
    # highest return with vol <= 12%: 0.04a^2 + 0.01(1-a)^2 = 0.0144
    a = (0.02 + math.sqrt(0.00128)) / 0.1  # 0.5578
    result = optimise(MU, COV, OptimisationSpec("target_volatility", max_volatility=0.12))
    assert weights(result) == pytest.approx((a, 1 - a), abs=1e-4)
    assert result.volatility == pytest.approx(0.12, abs=1e-6)


def test_max_return_with_cap_is_greedy() -> None:
    result = optimise(MU, COV, OptimisationSpec("max_return", max_weight=0.7))
    assert weights(result) == pytest.approx((0.7, 0.3), abs=1e-6)


def test_max_sharpe_with_volatility_cap() -> None:
    free = optimise(MU, COV, OptimisationSpec("max_sharpe"))
    capped = optimise(MU, COV, OptimisationSpec("max_sharpe", max_volatility=0.09))
    assert capped.volatility <= 0.09 + 1e-6 < free.volatility


# ---------------------------------------------------------------------- infeasible requests


@pytest.mark.parametrize(
    ("spec", "fragment"),
    [
        (
            OptimisationSpec(
                groups={"A": "G1", "B": "G2"}, group_bounds={"G1": (0.6, None), "G2": (0.6, None)}
            ),
            "Group minimums add up to 120.00%",
        ),
        (
            OptimisationSpec(asset_bounds={"A": (0.5, 0.3)}),
            "A: minimum 50.00% is above its maximum 30.00%",
        ),
        (OptimisationSpec(max_weight=0.3), "Asset caps add up to only 60.00%"),
        (OptimisationSpec(groups=GCC, group_bounds={"Gold": (0.1, None)}), "no holding is in it"),
        (
            OptimisationSpec(
                groups=GCC, group_bounds={"GCC": (0.5, None)}, asset_bounds={"A": (None, 0.3)}
            ),
            "Group 'GCC' needs at least 50.00%, but its assets' caps allow only 30.00%",
        ),
        (
            OptimisationSpec("min_variance", max_volatility=0.05),
            "lowest volatility these limits allow is 8.94% a year, above the 5.00% cap",
        ),
        (
            OptimisationSpec("target_return", min_return=0.20),
            "allow is 10.00% a year (historical), below the 20.00% floor",
        ),
        (OptimisationSpec(asset_bounds={"TSLA": (None, 0.1)}), "not in the portfolio: TSLA"),
    ],
)
def test_infeasible_requests_explain_why(spec, fragment) -> None:
    result = optimise(MU, COV, spec)
    assert not result.feasible and result.weights is None
    assert any(fragment in r for r in result.reasons), result.reasons


def test_return_floor_and_volatility_cap_together() -> None:
    # 9% return needs a = 0.8: vol = sqrt(0.64*0.04 + 0.04*0.01) = 16.12%, above a 12% cap
    result = optimise(
        MU, COV, OptimisationSpec("min_variance", min_return=0.09, max_volatility=0.12)
    )
    assert not result.feasible
    assert "together with the return floor is 16.12%" in result.reasons[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"objective": "target_volatility"},
        {"objective": "target_return"},
        {"objective": "magic"},
        {"max_weight": 0},
    ],
)
def test_spec_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        OptimisationSpec(**kwargs)


# --------------------------------------------------------------- property: never breaks a rule


def _random_problem(n: int, rng) -> tuple[pd.Series, pd.DataFrame]:
    tickers = [f"T{i}" for i in range(n)]
    rets = rng.normal(0.0004, 0.01, size=(400, n)) * rng.uniform(0.4, 2.0, n) + rng.normal(
        0, 0.005, (400, 1)
    )
    frame = pd.DataFrame(rets, columns=tickers)
    return frame.mean() * 252, frame.cov() * 252


@pytest.mark.parametrize("seed", range(25))
def test_results_always_respect_every_constraint(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 8))
    mu, cov = _random_problem(n, rng)
    groups = {t: ("GCC" if i % 2 == 0 else "Other") for i, t in enumerate(mu.index)}
    spec = OptimisationSpec(
        objective=str(rng.choice(["max_sharpe", "min_variance", "max_return"])),
        max_weight=float(rng.uniform(1 / n + 0.05, 1.0)),
        groups=groups,
        group_bounds={"GCC": (float(rng.uniform(0, 0.5)), None)},
        asset_bounds={mu.index[0]: (None, float(rng.uniform(0.05, 0.5)))},
        max_volatility=float(rng.uniform(0.08, 0.3)) if rng.random() < 0.5 else None,
    )
    result = optimise(mu, cov, spec, rf=0.02)
    if not result.feasible:
        assert result.reasons and result.weights is None
        return
    w = result.weights
    assert w.sum() == pytest.approx(1.0) and (w >= -1e-9).all()
    assert (w <= spec.max_weight + 1e-6).all()
    assert w[mu.index[0]] <= spec.asset_bounds[mu.index[0]][1] + 1e-6
    gcc = sum(w[t] for t, g in groups.items() if g == "GCC")
    assert gcc >= spec.group_bounds["GCC"][0] - 1e-6
    if spec.max_volatility is not None:
        assert result.volatility <= spec.max_volatility + 1e-6
