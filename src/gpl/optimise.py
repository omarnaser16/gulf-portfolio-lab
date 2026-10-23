"""Portfolio construction: mean-variance and risk parity.

Pure functions: no I/O, no printing, no plotting.

Conventions
-----------
``mu`` is a ``pd.Series`` of **annualised** expected returns and ``cov`` an annualised
covariance ``pd.DataFrame``, both indexed by ticker. Weights are long-only, sum to 1 and
are returned as a ``pd.Series`` indexed by ticker. Default max weight is 0.40 per asset.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

type Weights = pd.Series | np.ndarray

DEFAULT_MAX_WEIGHT = 0.40


def _as_array(weights: Weights, index: pd.Index) -> np.ndarray:
    """Weights as a numpy vector in ``index`` order (a Series is aligned by ticker)."""
    if isinstance(weights, pd.Series):
        aligned = weights.reindex(index)
        if aligned.isna().any():
            raise ValueError(f"weights missing for: {list(aligned[aligned.isna()].index)}")
        return aligned.to_numpy(dtype=float)
    return np.asarray(weights, dtype=float)


def _check_feasible(n_assets: int, max_weight: float) -> None:
    if max_weight * n_assets < 1 - 1e-12:
        raise ValueError(
            f"max_weight={max_weight} is infeasible for {n_assets} assets: "
            f"weights capped at {max_weight} cannot sum to 1"
        )


def _max_return_weights(mu: np.ndarray, max_weight: float) -> np.ndarray:
    """Highest-return long-only portfolio under the cap: fill the best assets first."""
    w = np.zeros(len(mu))
    remaining = 1.0
    for i in np.argsort(-mu):
        w[i] = min(max_weight, remaining)
        remaining -= w[i]
    return w


def _solve(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    index: pd.Index,
    max_weight: float,
    extra_constraints: list[dict] | None = None,
    x0: np.ndarray | None = None,
) -> pd.Series:
    """Minimise ``objective`` over long-only, fully invested, capped weights (SLSQP)."""
    n = len(index)
    _check_feasible(n, max_weight)
    budget = {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}
    result = minimize(
        objective,
        x0=np.full(n, 1 / n) if x0 is None else x0,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[budget, *(extra_constraints or [])],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise ValueError(f"optimiser failed: {result.message}")
    # Remove tiny numerical noise (e.g. -1e-17) and renormalise.
    w = np.clip(result.x, 0.0, max_weight)
    return pd.Series(w / w.sum(), index=index)


def portfolio_performance(
    weights: Weights,
    mu: pd.Series,
    cov: pd.DataFrame,
) -> tuple[float, float]:
    """Expected return and volatility of a portfolio.

    Parameters
    ----------
    weights : pd.Series or np.ndarray
        Portfolio weights, same order as ``mu`` and ``cov``.
    mu : pd.Series
        Annualised expected returns.
    cov : pd.DataFrame
        Annualised covariance matrix.

    Returns
    -------
    tuple of float
        ``(w @ mu, sqrt(w @ cov @ w))``.
    """
    w = _as_array(weights, cov.index)
    ret = float(w @ mu.reindex(cov.index).to_numpy())
    vol = float(np.sqrt(w @ cov.to_numpy() @ w))
    return ret, vol


def min_variance(cov: pd.DataFrame, max_weight: float = DEFAULT_MAX_WEIGHT) -> pd.Series:
    """Long-only minimum-variance portfolio (SLSQP).

    Parameters
    ----------
    cov : pd.DataFrame
        Annualised covariance matrix.
    max_weight : float, default 0.40
        Upper bound on each weight.

    Returns
    -------
    pd.Series
        Optimal weights indexed by ticker.

    Raises
    ------
    ValueError
        If ``max_weight * n_assets < 1`` (no feasible portfolio), or the optimiser fails.
    """
    sigma = cov.to_numpy()

    def variance(w: np.ndarray) -> float:
        return w @ sigma @ w

    def variance_grad(w: np.ndarray) -> np.ndarray:
        return 2 * sigma @ w

    return _solve(variance, variance_grad, cov.index, max_weight)


def max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> pd.Series:
    """Long-only maximum-Sharpe (tangency) portfolio (SLSQP).

    Parameters
    ----------
    mu : pd.Series
        Annualised expected returns.
    cov : pd.DataFrame
        Annualised covariance matrix.
    rf : float, default 0.0
        Annual risk-free rate.
    max_weight : float, default 0.40
        Upper bound on each weight.

    Returns
    -------
    pd.Series
        Optimal weights indexed by ticker.

    Raises
    ------
    ValueError
        If no feasible portfolio exists, or the optimiser fails.
    """
    sigma = cov.to_numpy()
    m = mu.reindex(cov.index).to_numpy()

    def neg_sharpe(w: np.ndarray) -> float:
        return -(w @ m - rf) / np.sqrt(w @ sigma @ w)

    def neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
        excess, var = w @ m - rf, w @ sigma @ w
        vol = np.sqrt(var)
        return -(m / vol - excess * (sigma @ w) / (var * vol))

    return _solve(neg_sharpe, neg_sharpe_grad, cov.index, max_weight)


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 50,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> pd.DataFrame:
    """Minimum-variance portfolios for a grid of target returns.

    Parameters
    ----------
    mu : pd.Series
        Annualised expected returns.
    cov : pd.DataFrame
        Annualised covariance matrix.
    n_points : int, default 50
        Number of target returns, from the min-variance return to the highest
        achievable return under the weight bounds.
    max_weight : float, default 0.40
        Upper bound on each weight.

    Returns
    -------
    pd.DataFrame
        One row per point. Columns ``ret``, ``vol``, then one weight column per ticker.
    """
    sigma = cov.to_numpy()
    m = mu.reindex(cov.index).to_numpy()

    w_start = min_variance(cov, max_weight)
    ret_min = float(w_start.to_numpy() @ m)
    ret_max = float(_max_return_weights(m, max_weight) @ m)

    rows = []
    x0 = w_start.to_numpy()
    for target in np.linspace(ret_min, ret_max, n_points):
        hits_target = {"type": "eq", "fun": lambda w, t=target: w @ m - t, "jac": lambda w: m}
        w = _solve(
            lambda w: w @ sigma @ w,
            lambda w: 2 * sigma @ w,
            cov.index,
            max_weight,
            extra_constraints=[hits_target],
            x0=x0,
        )
        x0 = w.to_numpy()  # warm start the next, slightly riskier point
        ret, vol = portfolio_performance(w, mu, cov)
        rows.append({"ret": ret, "vol": vol, **w.to_dict()})
    return pd.DataFrame(rows, columns=["ret", "vol", *cov.index])


def risk_contributions(weights: Weights, cov: pd.DataFrame) -> pd.Series:
    """Each asset's share of portfolio variance.

    Parameters
    ----------
    weights : pd.Series or np.ndarray
        Portfolio weights.
    cov : pd.DataFrame
        Covariance matrix.

    Returns
    -------
    pd.Series
        ``w_i * (cov @ w)_i / (w @ cov @ w)``, indexed by ticker; sums to 1.
    """
    w = _as_array(weights, cov.index)
    sigma = cov.to_numpy()
    marginal = sigma @ w  # how much each asset adds to risk per unit of weight
    return pd.Series(w * marginal / (w @ marginal), index=cov.index)


def _equal_risk_step(
    sigma: np.ndarray, w: np.ndarray, free: np.ndarray, budget: float, tol: float, max_iter: int
) -> np.ndarray:
    """Coordinate descent for ``w_i * (S w)_i = budget`` on the ``free`` assets.

    Holding the other weights fixed, each condition is a quadratic in ``w_i``,
    ``var_i * w_i^2 + c_i * w_i - budget = 0`` with ``c_i = sum_{j != i} S_ij w_j``,
    whose positive root has a closed form. Sweep the free assets until nothing moves.
    """
    variances = np.diag(sigma)
    w = w.copy()
    for _ in range(max_iter):
        w_previous = w.copy()
        for i in np.flatnonzero(free):
            c = sigma[i] @ w - variances[i] * w[i]
            w[i] = (-c + np.sqrt(c * c + 4 * variances[i] * budget)) / (2 * variances[i])
        if np.max(np.abs(w - w_previous)) <= tol * np.max(np.abs(w)):
            return w
    raise ValueError(f"risk parity did not converge in {max_iter} iterations")


def _capped_equal_risk(
    sigma: np.ndarray, capped: np.ndarray, max_weight: float, tol: float, max_iter: int
) -> np.ndarray:
    """Capped assets fixed at ``max_weight``; the rest get equal risk contributions.

    The free weights solve ``w_i * (S w)_i = lam``. Their total rises with ``lam``, so a
    bisection on ``lam`` finds the value at which all weights sum to 1.
    """
    free = ~capped
    target = 1 - max_weight * capped.sum()
    w = np.where(capped, max_weight, target / free.sum())

    def free_total(lam: float) -> tuple[float, np.ndarray]:
        solved = _equal_risk_step(sigma, w, free, lam, tol, max_iter)
        return solved[free].sum(), solved

    lo, hi = 0.0, float(np.max(np.diag(sigma)))
    while free_total(hi)[0] < target:
        hi *= 2
    if free_total(lo)[0] > target:
        raise ValueError("no capped risk parity solution: hedging assets exceed the budget")
    for _ in range(200):
        mid = (lo + hi) / 2
        total, w = free_total(mid)
        if abs(total - target) <= 1e-13:
            break
        lo, hi = (mid, hi) if total < target else (lo, mid)
    return w / w.sum()


def risk_parity(
    cov: pd.DataFrame,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> pd.Series:
    """Long-only equal-risk-contribution portfolio, with a cap on each weight.

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix.
    max_weight : float, default 0.40
        Upper bound on each weight.
    tol : float, default 1e-12
        Convergence tolerance on the relative change in weights.
    max_iter : int, default 10000
        Maximum number of full passes over the assets.

    Returns
    -------
    pd.Series
        Weights indexed by ticker. If no weight exceeds ``max_weight``, every asset's
        risk contribution is ``1 / n_assets``. Otherwise assets that would exceed the
        cap are held at it (their risk contribution is then *below* the others') and the
        remaining assets share equal risk contributions.

    Raises
    ------
    ValueError
        If an asset has zero variance, the cap is infeasible, or the solver fails.
    """
    sigma = cov.to_numpy(dtype=float)
    n = len(cov)
    variances = np.diag(sigma)
    if (variances <= 0).any():
        raise ValueError("risk parity needs every asset to have positive variance")
    _check_feasible(n, max_weight)

    # Uncapped: solve y_i * (S y)_i = 1/n for all i, then rescale (scaling keeps RCs equal).
    y0 = 1 / np.sqrt(variances)  # inverse-vol start: exact when assets are uncorrelated
    y = _equal_risk_step(sigma, y0, np.ones(n, dtype=bool), 1 / n, tol, max_iter)
    w = y / y.sum()

    # Cap: hold offenders at max_weight and re-solve the rest; repeat while anyone breaches.
    capped = np.zeros(n, dtype=bool)
    while (w > max_weight + 1e-10).any():
        capped |= w > max_weight + 1e-10
        w = _capped_equal_risk(sigma, capped, max_weight, tol, max_iter)
    return pd.Series(w, index=cov.index)
