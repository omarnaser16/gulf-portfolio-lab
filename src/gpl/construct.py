"""Portfolio construction from a curated universe, and improvements to a current portfolio.

Pure functions: no I/O, no printing, no plotting. Shared by both Optimise modes:
"Build a portfolio for me" (screen the universe by the client's IPS, choose 8-15 holdings that
aim to beat the benchmark on a risk-adjusted basis) and "Improve my current portfolio"
(re-weight within a turnover limit, then a few explained swaps). Every estimate uses only the
data available at the date it is made; the recommendation itself is made by a tested rule in
``gpl.recommend``.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from gpl.backtest import _annual_rf_at
from gpl.client import IPS, excluded_holdings, ips_constraints, tighter
from gpl.optimise import OptimisationSpec, active_moments, ledoit_wolf_cov, optimise


def turnover(before: pd.Series, after: pd.Series) -> float:
    """Sum of absolute weight changes (one-way buys plus sells): 0.30 = 30% of the portfolio
    traded in total, e.g. selling 15% of one holding and buying 15% of another."""
    idx = before.index.union(after.index)
    return float(
        (after.reindex(idx, fill_value=0.0) - before.reindex(idx, fill_value=0.0)).abs().sum()
    )


def limit_turnover(
    current: pd.Series, target: pd.Series, max_turnover: float
) -> tuple[pd.Series, float]:
    """Move from ``current`` towards ``target`` only as far as ``max_turnover`` allows.

    Returns the weights and the share of the move taken (1.0 = all the way). The result is
    ``current + s (target - current)`` with the largest s <= 1 whose turnover is within the
    limit; turnover grows in proportion to s. Every IPS limit is linear or convex (volatility,
    tracking error), so this point meets a limit whenever both ends do. E.g. from 100% A to
    100% B (turnover 2.00) with a 0.50 limit: s = 0.25, so 75% A and 25% B.
    """
    idx = current.index.union(target.index)
    w0 = current.reindex(idx, fill_value=0.0)
    w1 = target.reindex(idx, fill_value=0.0)
    full = turnover(w0, w1)
    share = 1.0 if full <= max_turnover or full == 0 else max_turnover / full
    w = w0 + share * (w1 - w0)
    return w[w.abs() > 1e-12], share


def limit_turnover_clean(
    current: pd.Series, target: pd.Series, max_turnover: float, dust: float = 0.005
) -> pd.Series:
    """``limit_turnover`` without a tail of tiny leftovers.

    Moving part of the way each quarter leaves a sliver of every holding the target no longer
    wants. Here a holding the target doesn't hold that would fall below ``dust`` is sold
    outright and the money goes to the holdings the target wants more of (in proportion to the
    shortfall). That extra trading counts towards the limit: the share of the move is the
    largest whose total turnover stays within ``max_turnover`` (found by bisection; turnover
    only rises with the share). E.g. from 99.4% A and 0.6% D towards 100% B with a 0.40 limit:
    D (0.48% after the move) is sold, and the result is 80% A and 20% B (turnover exactly 0.40).
    """
    idx = current.index.union(target.index)
    w0 = current.reindex(idx, fill_value=0.0)
    w1 = target.reindex(idx, fill_value=0.0)
    full = turnover(w0, w1)
    if full <= max_turnover or full == 0:
        return w1[w1 > 1e-12]

    def move(share: float) -> pd.Series:
        w = w0 + share * (w1 - w0)
        drop = (w1 <= 1e-12) & (w > 1e-12) & (w < dust)
        freed = float(w[drop].sum())
        w[drop] = 0.0
        need = (w1 - w).clip(lower=0.0)
        if freed > 0 and need.sum() > 0:
            w = w + need * min(1.0, freed / float(need.sum()))
            w = w + w1 * max(0.0, freed - float(need.sum()))  # only if more was freed than needed
        return w

    lo, hi = 0.0, max_turnover / full
    if turnover(w0, move(hi)) <= max_turnover + 1e-12:
        lo = hi
    else:
        for _ in range(50):
            mid = (lo + hi) / 2
            if turnover(w0, move(mid)) <= max_turnover + 1e-12:
                lo = mid
            else:
                hi = mid
    w = move(lo)
    return w[w > 1e-12]


# ------------------------------------------------------------------------ universe screen

GCC_REGIONS = {"Saudi Arabia", "UAE", "Qatar", "Kuwait", "Bahrain", "Oman"}
DEFENSIVE = ("Bonds", "Sukuk", "Cash", "Gold")


@dataclass
class Screen:
    """The universe after the client's IPS: who may be picked, and why the others may not."""

    eligible: list[str]
    excluded: dict[str, str]  # ticker -> reason


def gcc_stocks_wanted(ips: IPS | None) -> bool:
    """GCC single stocks join the universe only when the IPS asks for GCC exposure: a GCC
    minimum, a minimum in a GCC country, or Sharia-only (Aramco and Al Rajhi are screened)."""
    if ips is None:
        return False
    lim = ips.limits
    return bool(
        lim["home_bias_gcc"]
        or lim["sharia_only"]
        or any(r in GCC_REGIONS for r in lim["region_minimums"])
    )


def screen_universe(
    universe: Mapping[str, Mapping],
    ips: IPS | None,
    today: date,
    min_history_years: float = 7.0,
    attributes: Mapping[str, Mapping[str, str]] | None = None,
) -> Screen:
    """Apply the client's IPS and data rules to the curated universe.

    A security is left out (with the reason) if it is: a GCC single stock the IPS doesn't ask
    for; restricted by the client; a non-US fund for a US person (PFIC tax rules); verified
    with less than ``min_history_years`` of prices; or excluded by the IPS (excluded sector, or
    not Sharia-compliant for a Sharia-only client). With Sharia-only, a security whose Sharia
    status is unknown is not picked automatically.
    """
    excluded: dict[str, str] = {}
    client = ips.client if ips is not None else None
    restricted = {s.upper() for s in (client.restricted_securities if client else [])}
    gcc = gcc_stocks_wanted(ips)
    cutoff = today - timedelta(days=round(min_history_years * 365.25))
    left = []
    for t, r in universe.items():
        if r.get("gcc_stock") and not gcc:
            excluded[t] = "GCC single stock: included only when the IPS asks for GCC exposure"
        elif t.upper() in restricted:
            excluded[t] = "restricted by the client"
        elif (
            client
            and client.us_person
            and r["domicile"] not in ("United States",)
            and not (r.get("gcc_stock"))
        ):
            excluded[t] = "non-US fund: US persons face PFIC tax rules on it"
        elif not r.get("history_from"):
            excluded[t] = "price history not verified"
        elif date.fromisoformat(r["history_from"]) > cutoff:
            excluded[t] = (
                f"history only from {r['history_from']}: under {min_history_years:g} years"
            )
        else:
            left.append(t)
    if ips is not None:
        attrs = {
            t: {**(attributes or {}).get(t, {}), "sharia": universe[t]["sharia"]} for t in left
        }
        out, review = excluded_holdings(ips, left, attrs)
        for t, why in out.items():
            excluded[t] = f"excluded by the IPS: {why}"
        for t in review:
            excluded[t] = "Sharia status unknown: not picked automatically"
    return Screen([t for t in left if t not in excluded], excluded)


# ------------------------------------------------------------------------ factor scores


def momentum_score(window: pd.DataFrame, length: int = 252, skip: int = 21) -> pd.Series:
    """12-1 month momentum: the return from 12 months ago to 1 month ago (the last month is
    skipped, as it tends to reverse). Uses only the window passed in."""
    past = window.iloc[-length:-skip] if len(window) > skip else window.iloc[:0]
    return (1 + past).prod() - 1


def low_vol_score(window: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """Minus the annualised volatility over the window: higher = calmer."""
    return -window.std() * np.sqrt(periods_per_year)


def risk_adjusted_score(window: pd.DataFrame) -> pd.Series:
    """Mean ÷ standard deviation of returns over the window (a Sharpe ratio without the
    risk-free rate; only used to rank securities within a sleeve)."""
    sd = window.std().replace(0, np.nan)
    return (window.mean() / sd).fillna(0.0)


# ------------------------------------------------------------------------ build one portfolio

DUPLICATE_CORRELATION = 0.95  # two funds this correlated track nearly the same thing

SLEEVE_SIZES = {"Equity": 10, "Bonds": 4, "Sukuk": 2, "Cash": 1, "Gold": 1, "Real estate": 1}
TILTS = {
    "none": "no tilt",
    "low_vol": "low volatility (trailing volatility, from prices)",
    "momentum": "momentum (12-1 month return, from prices)",
    "quality": "quality (through quality-factor ETFs)",
}


@dataclass(frozen=True)
class BuildSettings:
    """Limits for building a portfolio (fractions). Defaults: config ``optimise_lab``."""

    tracking_error_max: float = 0.04
    n_min: int = 8
    n_max: int = 15
    min_weight: float = 0.02
    tilt: str = "none"
    tilt_min: float = 0.25  # at least this much in the tilt's top-scored names
    vol_vs_benchmark: bool = True  # volatility at most the benchmark's
    max_turnover: float | None = 0.25  # per rebalance after the first build; None = no limit
    periods_per_year: int = 252


@dataclass
class Built:
    """One portfolio built on one window: weights and why each security is there."""

    feasible: bool
    weights: pd.Series | None
    reasons: dict[str, str]  # ticker -> why it was chosen
    notes: list[str]  # why it couldn't be built, or what was relaxed


def _groups(tickers, universe) -> dict[str, str]:
    """Holding groups for the IPS limits: 'GCC' for GCC-country holdings, else asset class."""
    return {
        t: "GCC" if universe[t]["region"] in GCC_REGIONS else universe[t]["asset_class"]
        for t in tickers
    }


def _drop_near_duplicates(
    picked: dict[str, str],
    window: pd.DataFrame,
    keep_first: tuple[str, ...] = (),
    universe: Mapping[str, Mapping] | None = None,
) -> dict[str, str]:
    """Drop a fund whose weekly returns correlate ``DUPLICATE_CORRELATION`` or more with one
    ranked higher (``picked`` is in rank order; the benchmark's own funds rank first). Weekly,
    because daily returns of the same index listed in London and New York close at different
    times and understate how alike they are. A UCITS fund and the US fund it mirrors
    (``ucits_of``) always count as duplicates."""
    twin = {}
    for t, r in (universe or {}).items():
        if r.get("ucits_of"):
            twin[t], twin[r["ucits_of"]] = r["ucits_of"], t
    order = [t for t in keep_first if t in picked] + [t for t in picked if t not in keep_first]
    weekly = (1 + window[order]).resample("W").prod() - 1
    corr = weekly.corr()
    kept: list[str] = []
    for t in order:
        if twin.get(t) not in kept and all(corr.loc[t, k] < DUPLICATE_CORRELATION for k in kept):
            kept.append(t)
    return {t: picked[t] for t in picked if t in kept}


def shortlist(
    window: pd.DataFrame,
    eligible: list[str],
    universe: Mapping[str, Mapping],
    tilt: str = "none",
    required_regions: tuple[str, ...] = (),
    periods_per_year: int = 252,
    core: tuple[str, ...] = (),
) -> tuple[list[str], dict[str, str], list[str]]:
    """Up to ~25 candidates: the best of each sleeve, plus what the IPS minimums need.

    Each sleeve (asset class) keeps its top names (``SLEEVE_SIZES``) by risk-adjusted return
    over the window; with a tilt, equities are ranked by the tilt's score instead (quality:
    quality-factor ETFs first). Each region the IPS needs (a region minimum or GCC) adds its
    two best names. ``core`` (the benchmark's own funds, or their UCITS equivalents) is always
    included, so the tracking-error limit can be met. Returns (tickers, reason per ticker, the
    tilt's names).
    """
    cols = [t for t in eligible if t in window.columns]
    base = risk_adjusted_score(window[cols])
    tilt_score = {
        "low_vol": low_vol_score(window[cols], periods_per_year),
        "momentum": momentum_score(window[cols]),
    }.get(tilt)
    days = len(window)
    picked: dict[str, str] = {}
    tilt_names: list[str] = []
    for sleeve, size in SLEEVE_SIZES.items():
        members = [t for t in cols if universe[t]["asset_class"] == sleeve]
        if not members:
            continue
        if sleeve == "Equity" and tilt == "quality":
            score = base[members] + 1e3 * pd.Series(
                {t: universe[t].get("factor") == "quality" for t in members}, dtype=float
            )
            criterion = "quality-factor ETFs first, then risk-adjusted return"
        elif sleeve == "Equity" and tilt_score is not None:
            score, criterion = tilt_score[members], TILTS[tilt].split(" (")[0]
        else:
            score, criterion = base[members], "risk-adjusted return"
        ranked = score.sort_values(ascending=False).index.tolist()
        for k, t in enumerate(ranked[:size], 1):
            picked[t] = (
                f"{sleeve}, {universe[t]['region']}: ranked {k} of {len(ranked)} in its sleeve "
                f"by {criterion} over the last {days} trading days"
            )
        if sleeve == "Equity" and tilt != "none":
            tilt_names = (
                ranked[:5]
                if tilt != "quality"
                else [t for t in ranked if universe[t].get("factor") == "quality"]
            )
    picked = _drop_near_duplicates(picked, window, core, universe)
    for t in core:
        if t in cols:
            picked.setdefault(
                t, f"{universe[t]['name']}: in the benchmark (keeps tracking error low)"
            )
    for region in required_regions:
        members = [
            t
            for t in cols
            if (universe[t]["region"] in GCC_REGIONS if region == "GCC"
                else universe[t]["region"] == region)
        ]  # fmt: skip
        for t in base[members].sort_values(ascending=False).index[:2]:
            picked.setdefault(t, f"{universe[t]['region']}: needed for the IPS {region} minimum")
    return list(picked), picked, tilt_names


def build_once(
    window: pd.DataFrame,
    bench: pd.Series,
    eligible: list[str],
    universe: Mapping[str, Mapping],
    ips: IPS | None,
    base: OptimisationSpec,
    settings: BuildSettings,
    rf: float = 0.0,
    objective: str = "max_information_ratio",
    cash_like: tuple[str, ...] = (),
    core: tuple[str, ...] = (),
    history: pd.DataFrame | None = None,
) -> Built:
    """Build one portfolio on one window: shortlist, optimise, then prune to 8-15 holdings.

    The objective (default: highest information ratio against ``bench``) runs under the IPS
    limits (``client.ips_constraints``), the ``base`` limits from the Optimise form, the
    tracking-error cap and volatility at most the benchmark's, with the tilt's names held at
    ``tilt_min`` or more. Holdings under ``min_weight`` are dropped and the rest re-optimised;
    if fewer than ``n_min`` remain, the largest ``n_min`` are kept at ``min_weight`` or more.

    Risk (covariance, volatility cap, tracking error) is estimated on ``history``: all returns
    up to the window's last date (default: the window), because the IPS judges volatility and
    drawdown over the full history; expected returns come from the window. If the result's
    largest historical fall exceeds the IPS drawdown limit, the volatility cap is tightened by
    10% and the portfolio rebuilt (up to six times).
    """
    s = settings
    needed = []
    if ips is not None:
        if ips.limits["home_bias_gcc"]:
            needed.append("GCC")
        needed += list(ips.limits["region_minimums"])
    tickers, reasons, tilt_names = shortlist(
        window, eligible, universe, s.tilt, tuple(needed), s.periods_per_year, core
    )
    if len(tickers) < s.n_min:
        return Built(False, None, {}, [f"Only {len(tickers)} eligible securities."])
    data = window[tickers].join(bench.rename("__b__"), how="inner").dropna()
    win, b = data[tickers], data["__b__"]
    hist = window if history is None else history.loc[: window.index[-1]]
    hist_data = hist[tickers].join(bench.rename("__b__"), how="inner").dropna()
    hist_r, hist_b = hist_data[tickers], hist_data["__b__"]
    bench_vol = float(hist_b.std() * np.sqrt(s.periods_per_year))
    dd_limit = ips.limits["max_drawdown"] if ips is not None else None
    squeeze = {"factor": 1.0}

    def solve(names: list[str], floors: Mapping[str, float] | None = None):
        groups = _groups(names, universe)
        cash = tuple(t for t in names if t in cash_like or universe[t]["asset_class"] == "Cash")
        regions = {t: universe[t]["region"] for t in names}
        if ips is not None:
            spec, _ = ips_constraints(ips, names, groups, {}, list(cash), regions)
        else:
            spec = OptimisationSpec(groups=groups)
        spec = tighter(spec, replace(base, asset_bounds={}, sets={}, groups={}, group_bounds={}))
        vs_bench = s.vol_vs_benchmark and objective == "max_information_ratio"
        caps = [v for v in (spec.max_volatility, bench_vol if vs_bench else None) if v]
        caps = [c * squeeze["factor"] for c in caps]
        sets = dict(spec.sets)
        held = [t for t in tilt_names if t in names]
        if held and objective == "max_information_ratio":
            sets["Tilt"] = (frozenset(held), min(s.tilt_min, spec.max_weight * len(held)), None)
        bounds = {**spec.asset_bounds, **{t: (lo, None) for t, lo in (floors or {}).items()}}
        spec = replace(
            spec,
            objective=objective,
            max_volatility=min(caps) if caps else None,
            tracking_error_max=s.tracking_error_max if objective == "max_information_ratio"
            else None,
            sets=sets,
            asset_bounds=bounds,
        )  # fmt: skip
        mu = win[names].mean() * s.periods_per_year
        cov, _ = ledoit_wolf_cov(hist_r[names], s.periods_per_year)
        active = active_moments(hist_r[names], hist_b, s.periods_per_year)
        active.mu = (win[names].sub(b, axis=0)).mean() * s.periods_per_year
        return optimise(mu, cov, spec, rf, active)

    for _ in range(7):
        built = _prune_and_solve(solve, tickers, s)
        if not built.feasible or dd_limit is None:
            break
        port = hist_r[list(built.weights.index)] @ built.weights
        wealth = (1 + port).cumprod()
        fall = float((wealth / wealth.cummax() - 1).min())
        if fall >= -dd_limit:
            break
        squeeze["factor"] *= 0.9  # too deep a fall in the history so far: take less risk
    if not built.feasible:
        return built
    return Built(True, built.weights, {t: reasons[t] for t in built.weights.index}, built.notes)


def _prune_and_solve(solve, tickers: list[str], s: BuildSettings) -> Built:
    """Optimise the shortlist, keep 8-15 holdings at ``min_weight`` or more, re-optimise."""
    first = solve(tickers)
    if not first.feasible:
        return Built(False, None, {}, first.reasons)
    w = first.weights.sort_values(ascending=False)
    keep = w[w >= s.min_weight].index[: s.n_max].tolist()
    if len(keep) < s.n_min:
        keep = w.index[: s.n_min].tolist()
    # every holding kept is held at min_weight or more, so the count stays n_min..n_max
    second = solve(keep, dict.fromkeys(keep, s.min_weight))
    if not second.feasible:  # pruning broke a limit: keep the first solution
        final, notes = first.weights[first.weights > 1e-6], ["Kept unpruned: " + second.reasons[0]]
    else:
        final, notes = second.weights[second.weights > 1e-6], []
    final = final / final.sum()
    return Built(True, final.sort_values(ascending=False), {}, notes)


# ------------------------------------------------------------------------ walk-forward strategies


def build_strategy(
    eligible: list[str],
    universe: Mapping[str, Mapping],
    bench: pd.Series,
    ips: IPS | None,
    base: OptimisationSpec,
    settings: BuildSettings,
    rf: float | pd.Series = 0.0,
    objective: str = "max_information_ratio",
    cash_like: tuple[str, ...] = (),
    fallback: Mapping[str, float] | None = None,
    core: tuple[str, ...] = (),
    history: pd.DataFrame | None = None,
):
    """Walk-forward Build: at each rebalance, screen-select-optimise on the trailing window
    only (``build_once``), so the securities as well as the weights use no future data.

    If a window can't be built, the previous weights are kept (``strategy.infeasible_dates``,
    with the optimiser's reason in ``strategy.last_reason``); before the first feasible
    window, ``fallback`` (the client's current portfolio), or equal weights without one.
    ``strategy.latest`` is the last ``Built`` (weights and reasons).
    """
    state: dict = {"last": None if fallback is None else pd.Series(fallback, dtype=float)}

    def strategy(window: pd.DataFrame) -> pd.Series:
        known = bench.loc[: window.index[-1]]
        built = build_once(
            window, known, eligible, universe, ips, base, settings,
            _annual_rf_at(rf, window.index[-1]), objective, cash_like, core, history,
        )  # fmt: skip
        if built.feasible:
            state["last"] = built.weights
            strategy.latest = built
        else:
            strategy.infeasible_dates.append(window.index[-1])
            strategy.last_reason = " ".join(built.notes)
            if state["last"] is None:
                cols = [t for t in eligible if t in window.columns]
                state["last"] = pd.Series(1 / len(cols), index=cols)
        return state["last"]

    strategy.infeasible_dates = []
    strategy.latest = None
    strategy.last_reason = ""
    return strategy


def turnover_limited(
    strategy,
    start: Mapping[str, float] | None,
    max_turnover: float,
    spec: OptimisationSpec | None = None,
    dust: float | None = None,
):
    """Wrap a walk-forward strategy so each rebalance moves at most ``max_turnover`` from the
    previous target, starting from the client's current weights. With ``spec``, each move is
    the closest compliant move within the limit (``compliant_move``): if the current weights
    break a weight or group limit, the portfolio moves towards meeting it month by month.

    ``start=None`` (Build): the first target is taken in full (the initial purchase), and only
    later rebalances are limited. Without ``spec`` the move is along the straight line between
    two targets, so any linear or convex limit both meet (weights, groups, volatility, tracking
    error) is met on the way too. With ``dust`` (Build), leftovers the target doesn't want are
    sold once under that weight, within the same limit (``limit_turnover_clean``)."""
    state = {"last": None if start is None else pd.Series(start, dtype=float)}

    def limited(window: pd.DataFrame) -> pd.Series:
        target = strategy(window)
        if state["last"] is None:
            w = target
        elif spec is None and dust is not None:
            w = limit_turnover_clean(state["last"], target, max_turnover, dust)
        elif spec is None:
            w, _ = limit_turnover(state["last"], target, max_turnover)
        else:
            w = compliant_move(state["last"], target, max_turnover, spec).weights
        w = w / w.sum()
        state["last"] = w
        return w

    limited.infeasible_dates = getattr(strategy, "infeasible_dates", [])
    return limited


def benchmark_core(
    weights: Mapping[str, float], universe: Mapping[str, Mapping], eligible: list[str]
) -> tuple[str, ...]:
    """The benchmark's funds that may be held, or their UCITS equivalents where the US fund
    is not eligible (e.g. for a client preferring Irish funds)."""
    ucits = {r["ucits_of"]: t for t, r in universe.items() if r.get("ucits_of")}
    out = []
    for t in weights:
        if t in eligible:
            out.append(t)
        elif ucits.get(t) in eligible:
            out.append(ucits[t])
    return tuple(out)


# ------------------------------------------------------------------------ improve: swaps


@dataclass(frozen=True)
class Swap:
    """One suggested change to the current holdings, with the rule that produced it."""

    kind: str  # "duplicate" | "diversifier" | "domicile"
    remove: str | None
    add: str | None
    reason: str


def _weekly(window: pd.DataFrame) -> pd.DataFrame:
    return (1 + window).resample("W").prod() - 1


def swap_ideas(
    window: pd.DataFrame,
    current: pd.Series,
    eligible: list[str],
    universe: Mapping[str, Mapping],
    classes: Mapping[str, str],
    max_swaps: int = 3,
    defensive_wanted: tuple[str, ...] = ("Bonds", "Sukuk"),
) -> list[Swap]:
    """Rule-based swap ideas from the trailing window only, at most ``max_swaps``.

    1. Duplicates: two holdings whose weekly returns correlate ``DUPLICATE_CORRELATION`` or
       more (or a UCITS fund and the US fund it mirrors): sell the one with the lower
       risk-adjusted return; its weight goes to the other.
    2. Missing diversifier: no bond, sukuk or cash holding (``classes``: ticker -> asset
       class), so add the eligible fund in ``defensive_wanted`` least correlated with the
       current portfolio.
    UCITS domicile changes are not here: they are information, not performance
    (``domicile_suggestions``).
    """
    held = [t for t in current.index if t in window.columns and current[t] > 0]
    ideas: list[Swap] = []
    score = risk_adjusted_score(window[held]) if held else pd.Series(dtype=float)
    weekly = _weekly(window[held]) if held else pd.DataFrame()
    twin = {t: r["ucits_of"] for t, r in universe.items() if r.get("ucits_of")}
    twin |= {v: k for k, v in twin.items()}
    gone: set[str] = set()
    for i, a in enumerate(held):
        for b in held[i + 1 :]:
            if a in gone or b in gone:
                continue
            c = float(weekly[a].corr(weekly[b]))
            if c >= DUPLICATE_CORRELATION or twin.get(a) == b:
                drop, keep = (a, b) if score[a] < score[b] else (b, a)
                gone.add(drop)
                ideas.append(
                    Swap(
                        "duplicate",
                        drop,
                        None,
                        f"{drop} and {keep} move almost together (weekly correlation {c:.2f}) "
                        f"and {drop} had the lower risk-adjusted return, so the swap rule sells "
                        f"{drop} and keeps {keep}.",
                    )
                )
    has_defensive = any(classes.get(t) in DEFENSIVE[:3] for t in held)
    if not has_defensive:
        port = (window[held] * (current[held] / current[held].sum())).sum(axis=1)
        options = [
            t
            for t in eligible
            if t in window.columns and universe[t]["asset_class"] in defensive_wanted
            and t not in held
        ]  # fmt: skip
        if options:
            corr = window[options].corrwith(port)
            pick = corr.idxmin()
            ideas.append(
                Swap(
                    "diversifier",
                    None,
                    pick,
                    f"No bond, sukuk or cash holding, so the swap rule adds {pick} "
                    f"({universe[pick]['name']}), the eligible "
                    f"{universe[pick]['asset_class'].lower()} fund least correlated with the "
                    f"portfolio ({corr[pick]:.2f}).",
                )
            )
    return ideas[:max_swaps]


def domicile_suggestions(
    current: pd.Series, universe: Mapping[str, Mapping], us_person: bool
) -> list[Swap]:
    """US-domiciled funds with an Irish (UCITS) equivalent, for a non-US person.

    Information, not tax advice: Irish funds pay 15% US withholding on US dividends at fund
    level (30% on a US fund for a non-US person) and are not US-situs for US estate tax.
    These are outside the 0.05 Sharpe rule: they change the wrapper, not the exposure.
    """
    if us_person:
        return []
    twins = {r["ucits_of"]: t for t, r in universe.items() if r.get("ucits_of")}
    out = []
    for t in current.index:
        if current[t] <= 0 or t not in twins:
            continue
        alt = twins[t]
        income = universe[alt]["asset_class"] in ("Equity", "Real estate")
        benefit = (
            "lower US withholding on its dividends (15% at fund level vs 30%) and no US "
            "estate-tax exposure"
            if income
            else "no US estate-tax exposure (it pays no dividends to withhold on: "
            f"{universe[alt]['asset_class'].lower()})"
            if universe[alt]["asset_class"] == "Gold"
            else "no US estate-tax exposure"
        )
        out.append(
            Swap(
                "domicile",
                t,
                alt,
                f"{t} is US-domiciled; {alt} ({universe[alt]['name']}) is the Irish "
                f"equivalent: {benefit}. Information, not tax advice: check costs, tracking "
                "and share class.",
            )
        )
    return out


def swap_outcome(
    swap: Swap,
    choice: str,
    is_swap_candidate: bool,
    now: Mapping[str, float],
    recommended: Mapping[str, float] | None,
    max_turnover: str | None = None,
) -> str:
    """What today's recommended move does with a swap idea, in one sentence.

    The swap rule's reason says what the rule found; the recommended portfolio may still differ:
    another method may be recommended, or the turnover limit may allow only part of the move.
    ``now`` / ``recommended``: today's and the recommended weights (``recommended`` None when
    the advice is to keep the current portfolio); ``max_turnover`` as shown, e.g. "30.00%".
    E.g. "Within the 30.00% turnover limit, today's move only trims IEF from 7.31% to 4.97%;
    a full restructure would sell it all."
    """
    if recommended is None:
        return "The advice is to keep the current portfolio, so this swap is not made today."
    ticker = swap.remove or swap.add
    before, after = float(now.get(ticker, 0.0)), float(recommended.get(ticker, 0.0))
    limit = f"Within the {max_turnover} turnover limit, " if max_turnover else ""
    if swap.remove:
        if after <= 1e-9:
            return (
                "The recommended move sells it."
                if is_swap_candidate
                else f"The recommended {choice} portfolio also sells it."
            )
        if is_swap_candidate:
            return (
                f"{limit}today's move only trims {ticker} from {before:.2%} to {after:.2%}; "
                "a full restructure would sell it all."
            )
        return (
            f"The recommended {choice} portfolio doesn't make this swap: it holds {ticker} at "
            f"{after:.2%} (from {before:.2%})."
        )
    if after > 1e-9:
        who = "move" if is_swap_candidate else f"{choice} portfolio"
        return f"The recommended {who} buys {after:.2%}."
    return f"The recommended {choice} portfolio doesn't add it."


def apply_swaps(current: pd.Series, swaps: list[Swap]) -> list[str]:
    """The holdings after the swaps: removed ones out, added ones in (order kept)."""
    removed = {s.remove for s in swaps if s.remove}
    added = [s.add for s in swaps if s.add]
    return [t for t in current.index if current[t] > 0 and t not in removed] + [
        t for t in added if t not in current.index or current[t] == 0
    ]


def improve_strategy(
    current: pd.Series,
    eligible: list[str],
    universe: Mapping[str, Mapping],
    classes: Mapping[str, str],
    spec_for,
    rf: float | pd.Series = 0.0,
    max_swaps: int = 3,
    max_turnover: float | None = None,
    periods_per_year: int = 252,
    defensive_wanted: tuple[str, ...] = ("Bonds", "Sukuk"),
):
    """Walk-forward "re-weight + swaps": at each rebalance, the swap rules run on the trailing
    window only, the resulting holdings are re-optimised with ``spec_for(holdings)`` (the
    goal's objective under the IPS and form limits), and the move from the previous target is
    capped at ``max_turnover``. ``strategy.latest_swaps`` holds the last window's swaps."""
    state: dict = {"last": current / current.sum()}

    def strategy(window: pd.DataFrame) -> pd.Series:
        swaps = swap_ideas(window, current, eligible, universe, classes, max_swaps,
                           defensive_wanted)  # fmt: skip
        holdings = apply_swaps(current, swaps)
        win = window[holdings]
        mu = win.mean() * periods_per_year
        cov, _ = ledoit_wolf_cov(win, periods_per_year)
        result = optimise(mu, cov, spec_for(holdings), _annual_rf_at(rf, window.index[-1]))
        strategy.latest_swaps = swaps  # the ideas, even in a month the limits can't be met
        if result.feasible:
            target = result.weights
        else:
            strategy.infeasible_dates.append(window.index[-1])
            target = state["last"]
        if max_turnover is not None:
            target, _ = limit_turnover(state["last"], target, max_turnover)
            target = target / target.sum()
        state["last"] = target
        return target

    strategy.infeasible_dates = []
    strategy.latest_swaps = []
    return strategy


def _linear_rows(
    spec: OptimisationSpec, tickers: list[str]
) -> list[tuple[pd.Series, float | None, float | None]]:
    """The spec's linear limits as (coefficients, minimum, maximum) rows over ``tickers``."""
    rows = []
    for t in tickers:
        lo, hi = spec.asset_bounds.get(t, (None, None))
        cap = spec.max_weight if hi is None else min(hi, spec.max_weight)
        rows.append((pd.Series({t: 1.0}), lo, cap))
    for group, (lo, hi) in spec.group_bounds.items():
        members = [t for t in tickers if spec.groups.get(t) == group]
        rows.append((pd.Series(1.0, index=members), lo, hi))
    for members, lo, hi in spec.sets.values():
        rows.append((pd.Series(1.0, index=[t for t in tickers if t in members]), lo, hi))
    return rows


@dataclass(frozen=True)
class Move:
    """Today's change from the current weights under a turnover limit (``compliant_move``)."""

    weights: pd.Series
    turnover: float  # of this move
    full_turnover: float  # all the way to the target ("full restructure")
    least_turnover: float  # the least that meets every weight and group limit (NaN: none can)
    meets_limits: bool  # every weight, group and set limit met by ``weights``


def _meets_linear(w: pd.Series, spec: OptimisationSpec, tol: float = 1e-9) -> bool:
    for coef, lo, hi in _linear_rows(spec, list(w.index)):
        g = float(coef.reindex(w.index, fill_value=0.0) @ w)
        if (hi is not None and g > hi + tol) or (lo is not None and g < lo - tol):
            return False
    return True


def least_turnover_point(
    current: pd.Series, spec: OptimisationSpec, toward: pd.Series | None = None
) -> pd.Series | None:
    """The portfolio closest to ``current`` (least turnover) that meets every weight, group
    and set limit, by linear programming; None if no long-only portfolio meets them.

    Minimise sum(d) with d >= w - w0 and d >= w0 - w, so sum(d) is the turnover; the limits
    are linear, and the weights are long-only and add up to 1. Many portfolios can share the
    least turnover (e.g. which holding receives what is sold); with ``toward``, a tiny second
    term picks the one closest to it, so the move heads for the method's target.
    """
    from scipy.optimize import linprog

    idx = list(current.index)
    n = len(idx)
    w0 = current.to_numpy(dtype=float)
    w1 = w0 if toward is None else toward.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
    eye, zero = np.eye(n), np.zeros((n, n))
    # variables: w, d (|w - w0|), e (|w - w1|)
    a_ub = [
        np.hstack([eye, -eye, zero]),  # w - d <= w0
        np.hstack([-eye, -eye, zero]),  # -w - d <= -w0
        np.hstack([eye, zero, -eye]),  # w - e <= w1
        np.hstack([-eye, zero, -eye]),  # -w - e <= -w1
    ]
    b_ub = [w0, -w0, w1, -w1]
    for coef, lo, hi in _linear_rows(spec, idx):
        c = coef.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
        row = np.hstack([c, np.zeros(2 * n)])[None, :]
        if hi is not None:
            a_ub.append(row)
            b_ub.append(np.array([hi]))
        if lo is not None:
            a_ub.append(-row)
            b_ub.append(np.array([-lo]))
    tie = 0.0 if toward is None else 1e-4  # far too small to buy extra turnover
    res = linprog(
        np.hstack([np.zeros(n), np.ones(n), np.full(n, tie)]),
        A_ub=np.vstack(a_ub),
        b_ub=np.hstack(b_ub),
        A_eq=np.hstack([np.ones(n), np.zeros(2 * n)])[None, :],
        b_eq=[1.0],
        bounds=[(0, None)] * (3 * n),
        method="highs",
    )
    if not res.success:
        return None
    return pd.Series(np.clip(res.x[:n], 0.0, None), index=idx)


def compliant_move(
    current: pd.Series, target: pd.Series, max_turnover: float, spec: OptimisationSpec
) -> Move:
    """The closest compliant move within ``max_turnover``: the turnover limit is kept.

    1. If the whole move to ``target`` is within the limit, take it.
    2. If ``current`` already meets the weight and group limits, move part-way towards
       ``target`` (``limit_turnover``): every point on that line meets them too.
    3. Otherwise find the least-turnover portfolio that meets them (``least_turnover_point``).
       If it is within the limit, go there and spend what is left of the limit moving
       towards ``target`` (both ends meet the limits, so the whole segment does). If not, go
       as far towards it as the limit allows: closer to the limits, but still outside them.

    E.g. 50% in A with a 40% cap: meeting the cap needs a turnover of 0.20 (sell 10% of A,
    buy 10% elsewhere); a 0.10 limit gets half way, to 45% in A.
    """
    idx = current.index.union(target.index)
    w0 = current.reindex(idx, fill_value=0.0).astype(float)
    w1 = target.reindex(idx, fill_value=0.0).astype(float)
    full = turnover(w0, w1)

    def done(w: pd.Series, least: float) -> Move:
        w = w.clip(lower=0.0)
        w = w / w.sum()
        return Move(w[w > 1e-12], turnover(w0, w), full, least, _meets_linear(w, spec))

    if full <= max_turnover + 1e-12:
        return done(w1, 0.0 if _meets_linear(w0, spec) else float("nan"))
    if _meets_linear(w0, spec):
        w, _ = limit_turnover(w0, w1, max_turnover)
        return done(w.reindex(idx, fill_value=0.0), 0.0)
    p = least_turnover_point(w0, spec, toward=w1)
    if p is None:  # no portfolio meets the limits: keep the turnover limit along the line
        w, _ = limit_turnover(w0, w1, max_turnover)
        return done(w.reindex(idx, fill_value=0.0), float("nan"))
    least = turnover(w0, p)
    if least > max_turnover:
        return done(w0 + (max_turnover / least) * (p - w0), least)
    if not _meets_linear(w1, spec):
        return done(p, least)
    lo, hi = 0.0, 1.0  # turnover from w0 grows (convexly) along p -> w1: bisect for the limit
    for _ in range(60):
        mid = (lo + hi) / 2
        if turnover(w0, p + mid * (w1 - p)) <= max_turnover:
            lo = mid
        else:
            hi = mid
    return done(p + lo * (w1 - p), least)


# ------------------------------------------------------------------------ policy benchmark

POLICY_BLOCKS = {  # sleeve -> funds (a sleeve over the per-holding cap is split across them)
    "conventional": {
        "Equity": ("ACWI",),  # one global equity fund
        "GCC": ("KSA", "UAE", "QAT"),
        "Bonds": ("AGG", "BNDX"),
        "Cash": ("BIL", "SHV"),
    },
    "sharia": {
        "Equity": ("SPUS", "HLAL"),
        "GCC": ("2222.SR", "1120.SR"),
        "Bonds": ("SPSK",),
        "Cash": ("SPSK",),
    },
}
POLICY_NAME = "Policy benchmark (fits the IPS)"
# Broad index funds the per-holding cap doesn't apply to in the benchmark: a benchmark is not a
# holding, and ACWI spreads its weight over about 2,300 companies in 47 markets.
POLICY_CAP_EXEMPT = {"ACWI"}


@dataclass
class PolicyBenchmark:
    weights: dict[str, float]  # ticker -> weight
    sleeves: dict[str, float]  # sleeve -> weight
    passes: bool
    description: str
    reasons: list[str]  # why it passes, or what it can't meet


def _split(sleeve_weight: float, funds: tuple[str, ...], cap: float) -> dict[str, float] | None:
    """Spread a sleeve over as few of its funds as the cap allows (None if it can't fit)."""
    if sleeve_weight <= 1e-12:
        return {}
    for n in range(1, len(funds) + 1):
        if sleeve_weight / n <= cap + 1e-12:
            return {f: sleeve_weight / n for f in funds[:n]}
    return None


def policy_benchmark(
    ips: IPS, returns: pd.DataFrame, periods_per_year: int = 252, step: float = 0.05
) -> PolicyBenchmark:
    """A simple benchmark mix that itself passes the client's IPS limits.

    Sleeves: global equity, GCC equity (country ETFs; Aramco and Al Rajhi for Sharia), bonds
    (sukuk for Sharia) and cash. GCC is the IPS's GCC minimum, cash its cash need (as a share
    of the portfolio); equity is the largest share, in ``step`` steps, whose mix keeps
    volatility and the largest fall over ``returns`` (all history) within the IPS limits;
    bonds take the rest. A sleeve above the per-holding cap is split across its funds, except
    a broad global index fund (``POLICY_CAP_EXEMPT``, ACWI), which is one fund at any weight.
    Returns the mix and whether it passes (if no mix does, the most defensive one and why).
    """
    lim = ips.limits
    blocks = POLICY_BLOCKS["sharia" if lim["sharia_only"] else "conventional"]
    cap = lim["max_weight"] or 1.0
    gcc = lim["home_bias_gcc"] or 0.0
    cash = lim["liquidity_reserve"] or 0.0
    vol_cap, dd_cap = lim["max_volatility"], lim["max_drawdown"]
    best, last_fail = None, None
    grid = np.arange(0.0, 1.0 - gcc - cash + 1e-9, step)
    for equity in grid[::-1]:  # the most equity first: closest to a growth benchmark
        bonds = max(0.0, 1.0 - gcc - cash - equity)
        sleeves = {"Equity": equity, "GCC": gcc, "Bonds": bonds, "Cash": cash}
        weights: dict[str, float] = {}
        ok = True
        for sleeve, w in sleeves.items():
            exempt = set(blocks[sleeve]) <= POLICY_CAP_EXEMPT
            part = _split(w, blocks[sleeve], 1.0 if exempt else cap)
            if part is None:
                ok, last_fail = False, f"{sleeve} at {w:.2%} can't stay within the {cap:.2%} cap"
                break
            for t, x in part.items():
                weights[t] = weights.get(t, 0.0) + x
        if not ok or any(x > cap + 1e-9 for t, x in weights.items() if t not in POLICY_CAP_EXEMPT):
            last_fail = last_fail or f"a fund would exceed the {cap:.2%} cap"
            continue
        if not set(weights) <= set(returns.columns):
            missing = sorted(set(weights) - set(returns.columns))
            return PolicyBenchmark({}, sleeves, False, "", [f"No prices for {', '.join(missing)}."])
        w = pd.Series(weights)
        port = returns[list(w.index)].dropna() @ w
        vol = float(port.std() * np.sqrt(periods_per_year))
        wealth = (1 + port).cumprod()
        fall = float((wealth / wealth.cummax() - 1).min())
        if (vol_cap is None or vol <= vol_cap) and (dd_cap is None or fall >= -dd_cap):
            best = (weights, sleeves, vol, fall)
            break
        last_fail = f"volatility {vol:.2%}, largest fall {fall:.2%}"
    if best is None:
        return PolicyBenchmark({}, {}, False, "", [
            "No mix of the policy funds meets every limit"
            + (f" (the most defensive tried: {last_fail})." if last_fail else ".")
        ])  # fmt: skip
    weights, sleeves, vol, fall = best
    parts = [f"{v:.0%} {k.lower()}" for k, v in sleeves.items() if v > 1e-9]
    funds = ", ".join(f"{t} {x:.2%}" for t, x in weights.items())
    return PolicyBenchmark(
        weights,
        sleeves,
        True,
        f"{', '.join(parts)} ({funds}); rebalanced daily",
        [
            f"Volatility {vol:.2%} and largest fall {fall:.2%} over the full history, within the "
            f"IPS limits; GCC at least {gcc:.2%}, cash need {cash:.2%}, no fund over {cap:.2%}"
            + (
                " except ACWI, one broad global index fund (about 2,300 companies), to which "
                "the per-holding cap doesn't apply."
                if any(weights.get(t, 0.0) > cap + 1e-9 for t in POLICY_CAP_EXEMPT)
                else "."
            )
        ],
    )
