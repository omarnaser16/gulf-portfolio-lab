"""Portfolio construction: mean-variance and risk parity.

Pure functions: no I/O, no printing, no plotting.

Conventions
-----------
``mu`` is a ``pd.Series`` of **annualised** expected returns and ``cov`` an annualised
covariance ``pd.DataFrame``, both indexed by ticker. Weights are long-only, sum to 1 and
are returned as a ``pd.Series`` indexed by ticker. Default max weight is 0.40 per asset.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

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
    return _checked_weights(result.x, index, max_weight)


def _checked_weights(
    x: np.ndarray, index: pd.Index, max_weight: float, tol: float = 1e-6
) -> pd.Series:
    """Remove numerical noise, then refuse (rather than repair) any real constraint breach.

    Clipping and renormalising a solution that doesn't sum to 1 would silently scale weights
    *above* the cap, so a genuine violation raises instead.
    """
    if abs(x.sum() - 1) > tol or x.min() < -tol or x.max() > max_weight + tol:
        raise ValueError(
            f"optimiser returned infeasible weights (sum={x.sum():.6f}, "
            f"min={x.min():.6f}, max={x.max():.6f}, cap={max_weight})"
        )
    w = np.clip(x, 0.0, max_weight)  # e.g. -1e-17 -> 0
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
    """Long-only maximum-Sharpe (tangency) portfolio.

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

    Notes
    -----
    Maximising a ratio directly is a non-convex problem that SLSQP can fail on (e.g. when
    the cap forces weight into assets with negative excess returns). When some feasible
    portfolio beats ``rf``, the problem is solved in its equivalent convex form instead
    (Cornuejols & Tutuncu): with ``y = k * w`` for a scale ``k > 0``,

        minimise  y' S y   subject to   (mu - rf)' y = 1,  sum(y) = k,  0 <= y <= cap * k,

    and ``w = y / k``. If no portfolio beats ``rf`` (or the convex solve fails), the ratio
    is maximised directly from several starting points.
    """
    _check_feasible(len(cov), max_weight)
    sigma = cov.to_numpy()
    m = mu.reindex(cov.index).to_numpy()
    excess = m - rf
    best_excess = float(_max_return_weights(m, max_weight) @ excess)
    if best_excess > 1e-12:
        try:
            # Dividing by best_excess only rescales y (w = y / k is unchanged) but keeps the
            # problem well-scaled when the best excess return is tiny.
            return _max_sharpe_convex(sigma, excess / best_excess, cov.index, max_weight)
        except ValueError:
            pass  # numerically awkward case: fall through to the multi-start direct method
    return _max_sharpe_direct(sigma, excess, cov.index, max_weight)


def _max_sharpe_convex(
    sigma: np.ndarray, excess: np.ndarray, index: pd.Index, max_weight: float
) -> pd.Series:
    """Tangency portfolio via the convex quadratic reformulation. Variables z = [y, k]."""
    n = len(index)
    w0 = _max_return_weights(excess, max_weight)  # feasible, positive excess return
    y0 = w0 / (w0 @ excess)
    z0 = np.append(y0, y0.sum())
    constraints = [
        {
            "type": "eq",
            "fun": lambda z: z[:n] @ excess - 1,
            "jac": lambda z: np.append(excess, 0.0),
        },
        {
            "type": "eq",
            "fun": lambda z: z[:n].sum() - z[n],
            "jac": lambda z: np.append(np.ones(n), -1.0),
        },
        {  # y_i <= cap * k  for every asset
            "type": "ineq",
            "fun": lambda z: max_weight * z[n] - z[:n],
            "jac": lambda z: np.hstack([-np.eye(n), np.full((n, 1), max_weight)]),
        },
    ]
    result = minimize(
        lambda z: z[:n] @ sigma @ z[:n],
        x0=z0,
        jac=lambda z: np.append(2 * sigma @ z[:n], 0.0),
        method="SLSQP",
        bounds=[(0.0, None)] * (n + 1),
        constraints=constraints,
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    if not result.success or result.x[n] <= 0:
        raise ValueError(f"max Sharpe optimiser failed: {result.message}")
    return _checked_weights(result.x[:n] / result.x[n], index, max_weight)


def _max_sharpe_direct(
    sigma: np.ndarray, excess: np.ndarray, index: pd.Index, max_weight: float
) -> pd.Series:
    """Maximise the (non-positive) Sharpe ratio directly, keeping the best of several starts."""
    n = len(index)

    def neg_sharpe(w: np.ndarray) -> float:
        return -(w @ excess) / np.sqrt(w @ sigma @ w)

    def neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
        ex, var = w @ excess, w @ sigma @ w
        vol = np.sqrt(var)
        return -(excess / vol - ex * (sigma @ w) / (var * vol))

    starts = [
        np.full(n, 1 / n),
        _max_return_weights(excess, max_weight),
        min_variance(pd.DataFrame(sigma, index=index, columns=index), max_weight).to_numpy(),
    ]
    best, errors = None, []
    for x0 in starts:
        try:
            w = _solve(neg_sharpe, neg_sharpe_grad, index, max_weight, x0=x0)
        except ValueError as e:
            errors.append(str(e))
            continue
        if best is None or neg_sharpe(w.to_numpy()) < neg_sharpe(best.to_numpy()):
            best = w
    if best is None:
        raise ValueError(f"max Sharpe optimiser failed from every start: {errors}")
    return best


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
    return _checked_weights(w, cov.index, max_weight)


# ------------------------------------------------------------- constrained optimisation

Objective = Literal[
    "max_sharpe", "min_variance", "max_return", "target_volatility", "target_return"
]
OBJECTIVES: tuple[str, ...] = (
    "max_sharpe",
    "min_variance",
    "max_return",
    "target_volatility",
    "target_return",
)
_TOL = 1e-6


@dataclass(frozen=True)
class OptimisationSpec:
    """What to optimise and under which constraints (all weights and rates as fractions).

    Attributes
    ----------
    objective : str
        ``max_sharpe``; ``min_variance``; ``max_return``; ``target_volatility`` (highest return
        with volatility at most ``max_volatility``); ``target_return`` (lowest volatility with
        return at least ``min_return``).
    max_weight : float
        Cap on every asset (1.0 = no cap).
    asset_bounds : mapping
        Ticker -> (minimum, maximum); either may be ``None``.
    groups : mapping
        Ticker -> group name, e.g. ``{"2222.SR": "GCC"}``.
    group_bounds : mapping
        Group -> (minimum, maximum) total weight; either may be ``None``.
    max_volatility, min_return : float, optional
        Annual volatility cap and return floor, applied with any objective.
    """

    objective: Objective = "max_sharpe"
    max_weight: float = 1.0
    asset_bounds: Mapping[str, tuple[float | None, float | None]] = field(default_factory=dict)
    groups: Mapping[str, str] = field(default_factory=dict)
    group_bounds: Mapping[str, tuple[float | None, float | None]] = field(default_factory=dict)
    max_volatility: float | None = None
    min_return: float | None = None

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")
        if self.objective == "target_volatility" and self.max_volatility is None:
            raise ValueError("target_volatility needs max_volatility")
        if self.objective == "target_return" and self.min_return is None:
            raise ValueError("target_return needs min_return")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")


@dataclass
class OptimisationResult:
    """Outcome of ``optimise``: weights and statistics, or the reasons it is infeasible."""

    feasible: bool
    weights: pd.Series | None
    reasons: list[str]
    expected_return: float | None = None
    volatility: float | None = None
    sharpe: float | None = None
    binding: list[str] = field(default_factory=list)


def _linear_constraints(
    spec: OptimisationSpec, index: pd.Index
) -> tuple[
    np.ndarray, np.ndarray, list[tuple[str, np.ndarray, float | None, float | None]], list[str]
]:
    """Per-asset lower/upper bounds, group rows (name, indicator, min, max) and early reasons."""
    n = len(index)
    lo, hi = np.zeros(n), np.full(n, spec.max_weight)
    reasons = []
    unknown = sorted(set(spec.asset_bounds) - set(index))
    if unknown:
        reasons.append(f"Limits were given for assets not in the portfolio: {', '.join(unknown)}.")
    for i, t in enumerate(index):
        a_lo, a_hi = spec.asset_bounds.get(t, (None, None))
        lo[i] = max(0.0, a_lo or 0.0)
        hi[i] = min(spec.max_weight, 1.0 if a_hi is None else a_hi)
        if lo[i] > hi[i] + _TOL:
            reasons.append(f"{t}: minimum {lo[i]:.2%} is above its maximum {hi[i]:.2%}.")
    rows = []
    for g, (g_lo, g_hi) in spec.group_bounds.items():
        member = np.array([spec.groups.get(t) == g for t in index], dtype=float)
        if not member.any():
            if (g_lo or 0) > 0:
                reasons.append(
                    f"Group '{g}' must hold at least {g_lo:.2%}, but no holding is in it."
                )
            continue
        rows.append((g, member, g_lo, g_hi))
        if g_lo is not None and g_lo > member @ hi + _TOL:
            reasons.append(
                f"Group '{g}' needs at least {g_lo:.2%}, but its assets' caps allow only "
                f"{member @ hi:.2%}."
            )
        if g_hi is not None and g_hi < member @ lo - _TOL:
            reasons.append(
                f"Group '{g}' is capped at {g_hi:.2%}, but its assets' minimums add up to "
                f"{member @ lo:.2%}."
            )
    if lo.sum() > 1 + _TOL:
        reasons.append(f"Asset minimums add up to {lo.sum():.2%}, more than 100%.")
    if hi.sum() < 1 - _TOL:
        reasons.append(
            f"Asset caps add up to only {hi.sum():.2%}, so the portfolio can't be fully invested."
        )
    group_mins = sum((g_lo or 0) for _, _, g_lo, _ in rows)
    if group_mins > 1 + _TOL:
        reasons.append(f"Group minimums add up to {group_mins:.2%}, more than 100%.")
    return lo, hi, rows, reasons


def _lp(c: np.ndarray, lo, hi, rows, extra_ub=None) -> np.ndarray | None:
    """Solve min c·w over the linear constraints (sum = 1, bounds, groups); None if infeasible."""
    n = len(c)
    a_ub, b_ub = [], []
    for _, member, g_lo, g_hi in rows:
        if g_hi is not None:
            a_ub.append(member)
            b_ub.append(g_hi)
        if g_lo is not None:
            a_ub.append(-member)
            b_ub.append(-g_lo)
    if extra_ub is not None:
        a_ub.append(extra_ub[0])
        b_ub.append(extra_ub[1])
    res = linprog(
        c,
        A_ub=np.array(a_ub) if a_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.ones((1, n)),
        b_eq=[1.0],
        bounds=list(zip(lo, hi, strict=True)),
        method="highs",
    )
    return res.x if res.success else None


def _slsqp(objective, gradient, x0, lo, hi, rows, extra=()) -> np.ndarray | None:
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(len(w))}]
    for _, member, g_lo, g_hi in rows:
        if g_lo is not None:
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda w, m=member, b=g_lo: m @ w - b,
                    "jac": lambda w, m=member: m,
                }
            )
        if g_hi is not None:
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda w, m=member, b=g_hi: b - m @ w,
                    "jac": lambda w, m=member: -m,
                }
            )
    cons += list(extra)
    res = minimize(
        objective,
        x0=x0,
        jac=gradient,
        method="SLSQP",
        bounds=list(zip(lo, hi, strict=True)),
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    return res.x if res.success else None


def _satisfies(w, lo, hi, rows, sigma, m, spec, tol=1e-5) -> bool:
    if w is None or abs(w.sum() - 1) > tol or (w < lo - tol).any() or (w > hi + tol).any():
        return False
    for _, member, g_lo, g_hi in rows:
        if (g_lo is not None and member @ w < g_lo - tol) or (
            g_hi is not None and member @ w > g_hi + tol
        ):
            return False
    if spec.max_volatility is not None and np.sqrt(w @ sigma @ w) > spec.max_volatility + tol:
        return False
    return not (spec.min_return is not None and w @ m < spec.min_return - tol)


def optimise(
    mu: pd.Series, cov: pd.DataFrame, spec: OptimisationSpec, rf: float = 0.0
) -> OptimisationResult:
    """Optimise weights for ``spec``, or explain in plain language why it can't be done.

    Parameters
    ----------
    mu : pd.Series
        Annualised expected returns.
    cov : pd.DataFrame
        Annualised covariance matrix.
    spec : OptimisationSpec
        Objective and constraints.
    rf : float, default 0.0
        Annual risk-free rate (for the Sharpe objective and the reported Sharpe).

    Returns
    -------
    OptimisationResult
        ``feasible=True`` with weights, expected return, volatility, Sharpe and the binding
        constraints; or ``feasible=False`` with ``reasons``. Weights that break a constraint
        are never returned or rescaled.

    Notes
    -----
    Feasibility is checked exactly before solving: linear limits with a linear programme,
    the volatility cap against the lowest volatility the limits allow, and the return floor
    against the highest return they allow.
    """
    index = cov.index
    sigma, m = cov.to_numpy(), mu.reindex(index).to_numpy()
    lo, hi, rows, reasons = _linear_constraints(spec, index)
    if reasons:
        return OptimisationResult(False, None, reasons)
    start = _lp(np.zeros(len(index)), lo, hi, rows)
    if start is None:
        return OptimisationResult(
            False, None, ["The asset and group limits can't all be met at the same time."]
        )

    def variance(w):
        return w @ sigma @ w

    def variance_grad(w):
        return 2 * sigma @ w

    floor = (
        []
        if spec.min_return is None
        else [{"type": "ineq", "fun": lambda w: w @ m - spec.min_return, "jac": lambda w: m}]
    )
    vol_cap = (
        []
        if spec.max_volatility is None
        else [
            {
                "type": "ineq",
                "fun": lambda w: spec.max_volatility**2 - w @ sigma @ w,
                "jac": lambda w: -2 * sigma @ w,
            }
        ]
    )
    best_return = _lp(-m, lo, hi, rows)
    if spec.min_return is not None and best_return @ m < spec.min_return - _TOL:
        return OptimisationResult(
            False,
            None,
            [
                f"The highest return these limits allow is {best_return @ m:.2%} a year "
                f"(historical), below the {spec.min_return:.2%} floor."
            ],
        )
    if spec.max_volatility is not None:
        lowest = _slsqp(variance, variance_grad, start, lo, hi, rows, floor)
        if lowest is None:
            return OptimisationResult(False, None, ["The optimiser could not solve this problem."])
        if np.sqrt(variance(lowest)) > spec.max_volatility + _TOL:
            needs = " together with the return floor" if spec.min_return is not None else ""
            return OptimisationResult(
                False,
                None,
                [
                    f"The lowest volatility these limits allow{needs} is "
                    f"{np.sqrt(variance(lowest)):.2%} a year, above the "
                    f"{spec.max_volatility:.2%} cap."
                ],
            )

    match spec.objective:
        case "min_variance" | "target_return":
            w = _slsqp(variance, variance_grad, start, lo, hi, rows, floor + vol_cap)
        case "max_return" | "target_volatility":
            if spec.max_volatility is None:
                w = best_return
            else:
                w = _slsqp(lambda w: -(w @ m), lambda w: -m, start, lo, hi, rows, floor + vol_cap)
        case "max_sharpe":
            w = _constrained_max_sharpe(sigma, m, rf, start, lo, hi, rows, floor + vol_cap)
    if not _satisfies(w, lo, hi, rows, sigma, m, spec):
        return OptimisationResult(
            False, None, ["The optimiser could not find weights meeting every limit."]
        )

    w = np.clip(w, 0.0, None)
    weights = pd.Series(w / w.sum(), index=index)
    ret, vol = float(weights @ m), float(np.sqrt(weights @ sigma @ weights))
    return OptimisationResult(
        True,
        weights,
        [],
        ret,
        vol,
        (ret - rf) / vol if vol > 0 else None,
        _binding(weights.to_numpy(), lo, hi, rows, vol, ret, spec, index),
    )


def _constrained_max_sharpe(sigma, m, rf, start, lo, hi, rows, extra) -> np.ndarray | None:
    """Max Sharpe under general constraints: best of several SLSQP starts (incl. the LP point)."""
    excess = m - rf

    def neg_sharpe(w):
        return -(w @ excess) / np.sqrt(w @ sigma @ w)

    def neg_sharpe_grad(w):
        ex, var = w @ excess, w @ sigma @ w
        vol = np.sqrt(var)
        return -(excess / vol - ex * (sigma @ w) / (var * vol))

    starts = [start, _lp(-m, lo, hi, rows)]
    mv = _slsqp(lambda w: w @ sigma @ w, lambda w: 2 * sigma @ w, start, lo, hi, rows, extra)
    if mv is not None:
        starts.append(mv)
    best = None
    for x0 in [s for s in starts if s is not None]:
        w = _slsqp(neg_sharpe, neg_sharpe_grad, x0, lo, hi, rows, extra)
        if w is not None and (best is None or neg_sharpe(w) < neg_sharpe(best)):
            best = w
    return best


def _binding(w, lo, hi, rows, vol, ret, spec, index, tol=1e-4) -> list[str]:
    """Human-readable list of the constraints the solution sits on."""
    out = []
    for i, t in enumerate(index):
        if w[i] >= hi[i] - tol and hi[i] < 1 - tol:
            out.append(f"{t} at its {hi[i]:.2%} maximum")
        elif w[i] <= lo[i] + tol and lo[i] > tol:
            out.append(f"{t} at its {lo[i]:.2%} minimum")
    for g, member, g_lo, g_hi in rows:
        total = member @ w
        if g_lo is not None and abs(total - g_lo) < tol:
            out.append(f"group '{g}' at its {g_lo:.2%} minimum")
        if g_hi is not None and abs(total - g_hi) < tol:
            out.append(f"group '{g}' at its {g_hi:.2%} maximum")
    if spec.max_volatility is not None and abs(vol - spec.max_volatility) < tol:
        out.append(f"volatility at the {spec.max_volatility:.2%} cap")
    if spec.min_return is not None and abs(ret - spec.min_return) < tol:
        out.append(f"return at the {spec.min_return:.2%} floor")
    return out
