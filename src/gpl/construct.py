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
) -> Built:
    """Build one portfolio on one window: shortlist, optimise, then prune to 8-15 holdings.

    The objective (default: highest information ratio against ``bench``) runs under the IPS
    limits (``client.ips_constraints``), the ``base`` limits from the Optimise form, the
    tracking-error cap and volatility at most the benchmark's, with the tilt's names held at
    ``tilt_min`` or more. Holdings under ``min_weight`` are dropped and the rest re-optimised;
    if fewer than ``n_min`` remain, the largest ``n_min`` are kept at ``min_weight`` or more.
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
    bench_vol = float(b.std() * np.sqrt(s.periods_per_year))

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
        cov, _ = ledoit_wolf_cov(win[names], s.periods_per_year)
        active = active_moments(win[names], b, s.periods_per_year)
        return optimise(mu, cov, spec, rf, active)

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
    return Built(
        True, final.sort_values(ascending=False), {t: reasons[t] for t in final.index}, notes
    )


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
):
    """Walk-forward Build: at each rebalance, screen-select-optimise on the trailing window
    only (``build_once``), so the securities as well as the weights use no future data.

    If a window can't be built, the previous weights are kept (``strategy.infeasible_dates``);
    before the first feasible window, ``fallback`` (or equal weights in the eligible set).
    ``strategy.latest`` is the last ``Built`` (weights and reasons).
    """
    state: dict = {"last": None if fallback is None else pd.Series(fallback, dtype=float)}

    def strategy(window: pd.DataFrame) -> pd.Series:
        known = bench.loc[: window.index[-1]]
        built = build_once(
            window, known, eligible, universe, ips, base, settings,
            _annual_rf_at(rf, window.index[-1]), objective, cash_like, core,
        )  # fmt: skip
        if built.feasible:
            state["last"] = built.weights
            strategy.latest = built
        else:
            strategy.infeasible_dates.append(window.index[-1])
            if state["last"] is None:
                cols = [t for t in eligible if t in window.columns]
                state["last"] = pd.Series(1 / len(cols), index=cols)
        return state["last"]

    strategy.infeasible_dates = []
    strategy.latest = None
    return strategy


def turnover_limited(strategy, start: Mapping[str, float], max_turnover: float):
    """Wrap a walk-forward strategy so each rebalance moves at most ``max_turnover`` from the
    previous target, starting from the client's current weights (``limit_turnover``)."""
    state = {"last": pd.Series(start, dtype=float)}

    def limited(window: pd.DataFrame) -> pd.Series:
        target = strategy(window)
        w, _ = limit_turnover(state["last"], target, max_turnover)
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
                        f"{drop} and {keep} move almost together (weekly correlation {c:.2f}): "
                        f"sell {drop}, which had the lower risk-adjusted return, and keep {keep}.",
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
                    f"No bond, sukuk or cash holding: add {pick} ({universe[pick]['name']}), the "
                    f"eligible {universe[pick]['asset_class'].lower()} fund least correlated "
                    f"with the portfolio ({corr[pick]:.2f}).",
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
    return [
        Swap(
            "domicile",
            t,
            twins[t],
            f"{t} is US-domiciled; {twins[t]} ({universe[twins[t]]['name']}) is the Irish "
            "equivalent: lower US dividend withholding (15% vs 30%) and no US estate-tax "
            "exposure. Information, not tax advice: check costs, tracking and share class.",
        )
        for t in current.index
        if current[t] > 0 and t in twins
    ]


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
        if result.feasible:
            target = result.weights
            strategy.latest_swaps = swaps
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
