"""Goals, candidate methods and the rule that picks one, the way an analyst would.

Pure functions: no I/O, no printing, no plotting.

Several standard methods are run within the user's limits and each is evaluated **walk-forward,
after costs** (re-optimised every month on past data only). ``recommend`` then applies a fixed,
transparent rule for the user's goal to those out-of-sample results. It prefers methods whose
out-of-sample Sharpe ratio stayed close to their in-sample one ("held up"), and it recommends
keeping the current portfolio when no method improves on it. Results are historical and
risk-adjusted; they are not a forecast or a promise of higher returns.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import pandas as pd

from gpl.optimise import OptimisationSpec

GOALS: dict[str, str] = {
    "max_risk_adjusted": "Maximise risk-adjusted return (Sharpe)",
    "reduce_risk": "Reduce risk",
    "target_return": "Reach a target return with the least risk",
    "target_volatility": "Highest return at a target volatility",
    "beat_benchmark_less_risk": "Beat the benchmark with less risk",
}
CANDIDATES: dict[str, str] = {  # display name -> optimiser objective
    "Max Sharpe": "max_sharpe",
    "Min variance": "min_variance",
    "Risk parity": "risk_parity",
    "Max diversification": "max_diversification",
    "Target volatility": "target_volatility",
}
CURRENT, BENCHMARK, KEEP = "Current", "Benchmark", "Keep your current portfolio"
SHARPE_GOALS = ("max_risk_adjusted", "beat_benchmark_less_risk")  # decided on Sharpe ratios
_EPS = 1e-9  # so a gain of exactly the minimum counts despite floating-point rounding


def volatility_target(
    goal: str,
    target: float | None,
    current_vol: float,
    benchmark_vol: float | None,
    cap: float | None = None,
) -> float:
    """Volatility for the target-volatility candidate.

    The user's target for the target-volatility goal; the benchmark's volatility for "beat
    the benchmark with less risk"; otherwise the current portfolio's (best return for the
    risk you already take). Never above a volatility cap the user set as a limit.
    """
    if goal == "target_volatility" and target is not None:
        level = target
    elif goal == "beat_benchmark_less_risk" and benchmark_vol is not None and benchmark_vol > 0:
        level = benchmark_vol
    else:
        level = current_vol
    return level if cap is None else min(level, cap)


def candidate_specs(limits: OptimisationSpec, vol_target: float) -> dict[str, OptimisationSpec]:
    """One spec per candidate method, each carrying every one of the user's limits."""
    specs = {}
    for name, objective in CANDIDATES.items():
        cap = vol_target if objective == "target_volatility" else limits.max_volatility
        specs[name] = replace(limits, objective=objective, max_volatility=cap)
    return specs


@dataclass
class Recommendation:
    """The chosen method (or ``KEEP``), the rule applied, and why, in plain English."""

    choice: str
    rule: str
    reasons: list[str]
    ranking: list[str] = field(default_factory=list)  # qualifying candidates, best first
    held_up: dict[str, bool] = field(default_factory=dict)
    # "recommended", "keep", or "closest_fails" (no portfolio passes the IPS; ``choice`` is the
    # closest one, or KEEP when the current portfolio is closest)
    status: str = "recommended"
    current_passes_ips: bool | None = None  # None: no IPS to check against


def _closest(pool: pd.DataFrame) -> str:
    """Fewest failed IPS rules, then the smallest summed relative breach."""
    return str(pool.sort_values(["ips_fails", "ips_breach"], kind="stable").index[0])


RULES = {
    "max_risk_adjusted": (
        "Highest walk-forward Sharpe ratio after costs, and at least {gain} above your current "
        "portfolio's."
    ),
    "reduce_risk": (
        "Lowest walk-forward volatility, below your current portfolio's; when two are within "
        "{tol} of each other, the higher Sharpe ratio wins."
    ),
    "target_return": (
        "Lowest walk-forward volatility among methods whose walk-forward CAGR reached your "
        "target; if none did, the highest CAGR."
    ),
    "target_volatility": (
        "Highest walk-forward CAGR among methods whose walk-forward volatility stayed within "
        "{tol} of your target; if none did, the lowest volatility."
    ),
    "beat_benchmark_less_risk": (
        "Among methods with lower walk-forward volatility than the benchmark that also beat its "
        "CAGR, the highest Sharpe ratio; if none beat its CAGR, the highest Sharpe ratio among "
        "those with lower volatility. If your portfolio already beats the benchmark with less "
        "risk, a change must raise the Sharpe ratio by at least {gain}."
    ),
}


def held_up(in_sample_sharpe: float, walk_forward_sharpe: float, ratio: float = 0.5) -> bool:
    """Did the method keep at least ``ratio`` of its in-sample Sharpe out of sample?

    If the in-sample Sharpe was zero or negative there was nothing to lose, so it held up
    unless it got worse.
    """
    if pd.isna(in_sample_sharpe) or pd.isna(walk_forward_sharpe):
        return False
    if in_sample_sharpe <= 0:
        return walk_forward_sharpe >= in_sample_sharpe
    return walk_forward_sharpe >= ratio * in_sample_sharpe


def _ranked(goal, cands, current, benchmark, target, tol) -> tuple[list[str], list[str]]:
    """Qualifying candidates, best first, and notes (e.g. 'none reached the target')."""
    notes: list[str] = []
    by_sharpe = cands.sort_values("sharpe", ascending=False)
    match goal:
        case "max_risk_adjusted":
            return list(by_sharpe.index), notes
        case "reduce_risk":
            lower = cands[cands["volatility"] < current["volatility"]]
            if lower.empty:
                notes.append("No method had lower volatility than your current portfolio.")
                return [], notes
            # bucket volatility by the tolerance, then prefer the higher Sharpe within a bucket
            lowest = lower["volatility"].min()
            near = lower[lower["volatility"] <= lowest + tol].sort_values("sharpe", ascending=False)
            rest = lower.drop(near.index).sort_values("volatility")
            return list(near.index) + list(rest.index), notes
        case "target_return":
            reached = cands[cands["cagr"] >= target]
            if reached.empty:
                notes.append(f"No method reached the {target:.2%} target return.")
                return list(cands.sort_values("cagr", ascending=False).index), notes
            return list(reached.sort_values("volatility").index), notes
        case "target_volatility":
            within = cands[cands["volatility"] <= target + tol]
            if within.empty:
                notes.append(f"No method stayed within {tol:.2%} of the {target:.2%} target.")
                return list(cands.sort_values("volatility").index), notes
            return list(within.sort_values("cagr", ascending=False).index), notes
        case "beat_benchmark_less_risk":
            safer = cands[cands["volatility"] < benchmark["volatility"]]
            if safer.empty:
                notes.append("No method had lower volatility than the benchmark.")
                return [], notes
            beat = safer[safer["cagr"] > benchmark["cagr"]]
            if beat.empty:
                notes.append(
                    "No method with less risk than the benchmark also beat its return; the "
                    "best risk-adjusted one is shown."
                )
                return list(safer.sort_values("sharpe", ascending=False).index), notes
            return list(beat.sort_values("sharpe", ascending=False).index), notes
    raise ValueError(f"unknown goal {goal!r}; choose one of {list(GOALS)}")


def _improves(goal, pick, current, benchmark, target, tol, min_gain) -> bool:
    """Does the pick do better than the current portfolio on the goal's own terms?

    Where the comparison is on the Sharpe ratio, the pick must beat the current portfolio by at
    least ``min_gain``: a smaller gain is within estimation noise and not worth the trading.
    """
    match goal:
        case "max_risk_adjusted":
            return pick["sharpe"] >= current["sharpe"] + min_gain - _EPS
        case "reduce_risk":
            return pick["volatility"] < current["volatility"]
        case "target_return":
            if pick["cagr"] >= target:
                return current["cagr"] < target or pick["volatility"] < current["volatility"]
            return pick["cagr"] > current["cagr"]
        case "target_volatility":
            if pick["volatility"] <= target + tol:
                return current["volatility"] > target + tol or pick["cagr"] > current["cagr"]
            return pick["volatility"] < current["volatility"]
        case "beat_benchmark_less_risk":
            current_ok = (
                current["volatility"] < benchmark["volatility"]
                and current["cagr"] > benchmark["cagr"]
            )
            pick_ok = (
                pick["volatility"] < benchmark["volatility"] and pick["cagr"] > benchmark["cagr"]
            )
            if pick_ok and not current_ok:
                return True
            better = pick["sharpe"] >= current["sharpe"] + min_gain - _EPS
            return better and (pick_ok or not current_ok)
    return False


def recommend(
    goal: str,
    table: pd.DataFrame,
    target: float | None = None,
    tolerance: float = 0.005,
    held_up_ratio: float = 0.5,
    min_sharpe_gain: float = 0.05,
    extra: tuple[str, ...] = (),
) -> Recommendation:
    """Apply the goal's rule to walk-forward results.

    ``extra``: further candidate rows judged by the same rule (e.g. "Re-weight + swaps").

    Parameters
    ----------
    goal : str
        A key of ``GOALS``.
    table : pd.DataFrame
        One row per candidate (names from ``CANDIDATES``) plus ``"Current"`` and, for the
        benchmark goal, ``"Benchmark"``. Columns: ``cagr``, ``volatility``, ``sharpe``
        (walk-forward, after costs), ``in_sample_sharpe`` and ``feasible`` (candidates only).
    target : float, optional
        Target return (``target_return``) or volatility (``target_volatility``), annual.
    tolerance : float, default 0.005
        Volatility tolerance (0.50 percentage points) for "about the same" and "on target".
    min_sharpe_gain : float, default 0.05
        Where the rule compares Sharpe ratios (maximise risk-adjusted return; the Sharpe test in
        "beat the benchmark with less risk"), the minimum gain over the current portfolio for a
        change to be recommended. Smaller gains are within estimation noise.
    held_up_ratio : float, default 0.5
        Share of its in-sample Sharpe a method must keep out of sample to count as holding up.

    Suitability gate
    ----------------
    If ``table`` has the columns ``passes_ips`` (bool), ``ips_fails`` (number of failed IPS
    rules) and ``ips_breach`` (summed relative breach), only candidates that pass every IPS
    rule can be recommended:

    - if none passes, the closest portfolio (candidates and current) is returned with status
      ``closest_fails``; it must not be presented as suitable;
    - if the current portfolio fails and a candidate passes, the best passing candidate under
      the goal's ranking is recommended even without an improvement on the goal, because the
      current portfolio is outside the client's limits;
    - the current portfolio is kept only when it passes.

    Returns
    -------
    Recommendation
    """
    if goal in ("target_return", "target_volatility") and target is None:
        raise ValueError(f"{goal} needs a target")
    tol_text = f"{tolerance * 100:.2f} percentage points"
    rule = RULES[goal].format(tol=tol_text, gain=f"{min_sharpe_gain:.2f}")
    current = table.loc[CURRENT]
    benchmark = table.loc[BENCHMARK] if BENCHMARK in table.index else None
    if goal == "beat_benchmark_less_risk" and benchmark is None:
        raise ValueError("beat_benchmark_less_risk needs a Benchmark row")
    cands = table[table.index.isin([*CANDIDATES, *extra])]
    cands = cands[cands["feasible"].astype(bool)]
    survived = {
        n: held_up(r["in_sample_sharpe"], r["sharpe"], held_up_ratio) for n, r in cands.iterrows()
    }
    gate = "passes_ips" in table.columns
    current_ok = bool(current["passes_ips"]) if gate else None

    def keep(reasons: list[str], ranking: list[str]) -> Recommendation:
        if gate:
            reasons = [*reasons, "Your current portfolio passes every IPS rule."]
        return Recommendation(KEEP, rule, reasons, ranking, survived, "keep", current_ok)

    if gate:
        passing = cands[cands["passes_ips"].astype(bool)]
        if passing.empty:
            pool = pd.concat([cands, table.loc[[CURRENT]]])
            closest = _closest(pool)
            reasons = [
                "No method meets every rule of the client's IPS"
                + ("" if not cands.empty else " (no method could meet the limits at all)")
                + "."
            ]
            if current_ok:
                return keep(reasons, [])
            reasons.append("Your current portfolio fails the IPS too.")
            reasons.append(
                "Your current portfolio is the closest to passing."
                if closest == CURRENT
                else f"{closest} is the closest to passing (fewest rules failed, smallest "
                "breaches). It is shown for discussion, not as a suitable portfolio."
            )
            choice = KEEP if closest == CURRENT else closest
            return Recommendation(choice, rule, reasons, [], survived, "closest_fails", current_ok)
        if not current_ok:
            return _move_off_failing(
                goal, passing, current, benchmark, target, tolerance, rule, survived, held_up_ratio
            )
        cands = passing
    if cands.empty:
        return keep(["No method could meet your limits."], [])

    ranking, reasons = _ranked(goal, cands, current, benchmark, target, tolerance)
    if not ranking:
        return keep(reasons, ranking)
    pick = next((n for n in ranking if survived[n]), None)
    if pick is None:
        reasons.append(
            "None of the qualifying methods held up out of sample (each kept less than "
            f"{held_up_ratio:.0%} of its in-sample Sharpe ratio), so none is recommended."
        )
        return keep(reasons, ranking)
    if pick != ranking[0]:
        reasons.append(
            f"{ranking[0]} ranked first but did not hold up out of sample (it kept less than "
            f"{held_up_ratio:.0%} of its in-sample Sharpe ratio), so the next method that did "
            f"is recommended."
        )
    if not _improves(goal, cands.loc[pick], current, benchmark, target, tolerance, min_sharpe_gain):
        gain = cands.loc[pick, "sharpe"] - current["sharpe"]
        if goal in SHARPE_GOALS and 0 < gain < min_sharpe_gain - _EPS:
            reasons.append(
                f"{pick} had a higher Sharpe ratio than your current portfolio, but by less than "
                f"the {min_sharpe_gain:.2f} minimum, which is within estimation noise and not "
                "worth the extra trading."
            )
            return keep(reasons, ranking)
        reasons.append(
            f"{pick} was the best method for this goal but did not improve on your current "
            "portfolio over the same period."
        )
        return keep(reasons, ranking)
    reasons.insert(0, f"{pick} ranks first under the rule: {rule[0].lower()}{rule[1:]}")
    if gate:
        reasons.append(f"{pick} passes every IPS rule.")
    return Recommendation(pick, rule, reasons, ranking, survived, "recommended", current_ok)


def _move_off_failing(
    goal, passing, current, benchmark, target, tol, rule, survived, held_up_ratio
) -> Recommendation:
    """The current portfolio fails the IPS: recommend the best candidate that passes.

    The goal's ranking still orders the candidates (falling back to Sharpe when none qualifies
    for the goal), and one that held up out of sample is preferred, but no improvement on the
    goal is required: staying outside the client's limits is not an option.
    """
    ranking, notes = _ranked(goal, passing, current, benchmark, target, tol)
    rest = [n for n in passing.sort_values("sharpe", ascending=False).index if n not in ranking]
    order = ranking + rest
    pick = next((n for n in order if survived[n]), order[0])
    reasons = [
        "Your current portfolio fails the client's IPS, so a change is needed even if it does "
        "not improve on the goal.",
        f"{pick} is the best of the methods that pass every IPS rule "
        + (
            "under the goal's rule."
            if pick in ranking
            else "by Sharpe ratio (none of the passing methods qualified under the goal's rule)."
        ),
        *notes,
    ]
    if not survived[pick]:
        reasons.append(
            f"Caution: {pick} kept less than {held_up_ratio:.0%} of its in-sample Sharpe ratio "
            "out of sample; no passing method held up."
        )
    return Recommendation(pick, rule, reasons, order, survived, "recommended", False)


# ------------------------------------------------------------------------ Build mode

TRACKER = "Benchmark tracker"
DIVERSIFIED = "Diversified (IPS)"
BUILD_RULE = (
    "Every candidate (each build, the benchmark tracker and the diversified portfolio) is tested "
    "the same way: it must pass the IPS, hold up out of sample, keep volatility at most the "
    "benchmark's (within {tol}) and tracking error within {te}. Among builds that pass every "
    "test and beat the benchmark out of sample (excess return above zero and an information "
    "ratio of at least {min_ir}), the highest information ratio is recommended. If none did, the "
    "next best is the candidate with the highest walk-forward Sharpe ratio among those that pass "
    "every test (if none pass them all: among those that pass the IPS, the one failing the "
    "fewest tests, then the highest Sharpe ratio, with the tests it failed named). A portfolio "
    "that another IPS-passing candidate matches or beats on Sharpe ratio, volatility, drawdown, "
    "tracking error and turnover is never chosen. If the current portfolio passes the IPS and "
    "its walk-forward Sharpe ratio is at least the chosen option's, keep it."
)

# measure -> True if higher is better (the dominance check)
DOMINANCE = {
    "sharpe": True,
    "volatility": False,
    "max_drawdown": True,  # negative numbers: closer to zero is better
    "tracking_error": False,
    "turnover_per_year": False,
}
TEST_LABELS = {
    "ips": "breaks the IPS",
    "held_up": "didn't hold up out of sample",
    "volatility": "more volatile than the benchmark",
    "tracking_error": "too much tracking error",
}


def build_test_failures(
    table: pd.DataFrame,
    tracking_error_max: float,
    tolerance: float = 0.005,
    held_up_ratio: float = 0.5,
) -> dict[str, list[str]]:
    """Which of the four Build tests each candidate fails (keys of ``TEST_LABELS``).

    The tests apply to every candidate alike: IPS, held up out of sample, volatility at most
    the benchmark's (within ``tolerance``) and tracking error within the limit (same tolerance).
    """
    bench_vol = float(table.loc[BENCHMARK, "volatility"])
    out: dict[str, list[str]] = {}
    for name in table.index.drop([BENCHMARK, CURRENT], errors="ignore"):
        r = table.loc[name]
        failed = []
        if "passes_ips" in table and not (pd.notna(r["passes_ips"]) and bool(r["passes_ips"])):
            failed.append("ips")
        if not held_up(r["in_sample_sharpe"], r["sharpe"], held_up_ratio):
            failed.append("held_up")
        if not (pd.notna(r["volatility"]) and r["volatility"] <= bench_vol + tolerance):
            failed.append("volatility")
        te = r["tracking_error"]
        if not (pd.notna(te) and te <= tracking_error_max + tolerance):
            failed.append("tracking_error")
        out[str(name)] = failed
    return out


def dominated_by(table: pd.DataFrame, pick: str, pool: list[str]) -> str | None:
    """The best candidate in ``pool`` that is at least as good as ``pick`` on every measure in
    ``DOMINANCE`` (those both have) and strictly better on one; None if there is none."""
    measures = [m for m in DOMINANCE if m in table]
    better = []
    for other in pool:
        if other == pick:
            continue
        pairs = [
            (table.loc[other, m], table.loc[pick, m], DOMINANCE[m])
            for m in measures
            if pd.notna(table.loc[other, m]) and pd.notna(table.loc[pick, m])
        ]
        if not pairs:
            continue
        at_least = all((o >= p) if up else (o <= p) for o, p, up in pairs)
        strictly = any((o > p) if up else (o < p) for o, p, up in pairs)
        if at_least and strictly:
            better.append(other)
    if not better:
        return None
    return str(table.loc[better, "sharpe"].astype(float).idxmax())


def recommend_build(
    table: pd.DataFrame,
    tracking_error_max: float,
    tolerance: float = 0.005,
    held_up_ratio: float = 0.5,
    min_information_ratio: float = 0.10,
) -> Recommendation:
    """Choose among built portfolios by a fixed rule (``BUILD_RULE``); never promises returns.

    ``table``: one row per candidate plus ``Benchmark``, with ``volatility``, ``sharpe``,
    ``in_sample_sharpe``, ``excess_return``, ``information_ratio``, ``tracking_error``,
    optionally ``max_drawdown``, ``turnover_per_year`` and ``feasible``, and (with an IPS)
    ``passes_ips``, ``ips_fails``, ``ips_breach``. Statuses: "recommended" (beats the benchmark
    under the rule), "next_best" (does not beat it reliably: the best of the rest), "keep" or
    "closest_fails" (nothing passes the IPS: the closest, with its breaches).
    """
    rule = BUILD_RULE.format(
        tol=f"{tolerance * 100:.2f} pp",
        te=f"{tracking_error_max:.2%}",
        min_ir=f"{min_information_ratio:.2f}",
    )
    table = table.copy()
    for col in ("sharpe", "volatility", "tracking_error", "information_ratio", "excess_return"):
        table[col] = pd.to_numeric(table[col], errors="coerce")
    cands = table.drop(index=[BENCHMARK, CURRENT], errors="ignore")
    if "feasible" in cands:
        cands = cands[cands["feasible"].fillna(True).astype(bool)]
    fails = {
        n: f
        for n, f in build_test_failures(table, tracking_error_max, tolerance, held_up_ratio).items()
        if n in cands.index
    }
    survived = {n: "held_up" not in f for n, f in fails.items()}
    passing_ips = [n for n in cands.index if "ips" not in fails[n]]
    all_tests = [n for n in cands.index if not fails[n]]
    builds_ok = [
        n
        for n in all_tests
        if n not in (TRACKER, DIVERSIFIED)
        and table.loc[n, "excess_return"] > 0
        and table.loc[n, "information_ratio"] >= min_information_ratio
    ]

    def sharpe_list(names: list[str], skip: str) -> str:
        rest = [n for n in names if n != skip]
        rest = sorted(rest, key=lambda n: -float(table.loc[n, "sharpe"]))
        return ", ".join(f"{n} {table.loc[n, 'sharpe']:.2f}" for n in rest)

    if builds_ok:
        ranking = table.loc[builds_ok].sort_values("information_ratio", ascending=False)
        ranking = ranking.index.tolist()
        pick, status = ranking[0], "recommended"
        r = table.loc[pick]
        reasons = [
            f"{pick} ranks first under the rule: walk-forward information ratio "
            f"{r['information_ratio']:.2f}, {r['excess_return']:.2%} a year above the benchmark "
            f"after costs, tracking error {r['tracking_error']:.2%}, volatility "
            f"{r['volatility']:.2%} against the benchmark's "
            f"{float(table.loc[BENCHMARK, 'volatility']):.2%}.",
            "Historical and risk-adjusted: not a promise that it will beat the benchmark.",
        ]
    else:
        ranking = []
        why = (
            "No built portfolio beat the benchmark out of sample after costs within every test "
            f"(an information ratio of at least {min_information_ratio:.2f} is needed: below "
            "that, the difference is within noise)."
        )
        if not passing_ips:
            if "ips_fails" not in cands or cands.empty:
                return Recommendation(KEEP, rule, [why, "No candidate could be built."], [],
                                      survived, "closest_fails")  # fmt: skip
            closest = _closest(cands)
            return Recommendation(
                closest,
                rule,
                [why, f"No portfolio passes every IPS rule; the closest is {closest}, shown "
                      "with its breaches. Not suitable as it stands."],
                [],
                survived,
                "closest_fails",
            )  # fmt: skip
        pool = all_tests or passing_ips
        order = table.loc[pool].assign(failed=[len(fails[n]) for n in pool])
        order = order.sort_values(
            ["failed", "sharpe", "information_ratio"], ascending=[True, False, False]
        )
        pick, status = str(order.index[0]), "next_best"
        if all_tests:
            how = (
                "it has the highest walk-forward Sharpe ratio after costs among the portfolios "
                "that pass the IPS, held up out of sample and stay within the volatility and "
                "tracking-error limits"
            )
        else:
            failed = "; ".join(TEST_LABELS[k] for k in fails[pick])
            how = (
                "no portfolio passed every test, so among those that pass the IPS it fails the "
                f"fewest tests ({failed}) and then has the highest walk-forward Sharpe ratio"
            )
        others = sharpe_list(pool, pick)
        reasons = [
            why,
            f"{pick} is the next best option: {how}, {table.loc[pick, 'sharpe']:.2f}"
            + (f" (others: {others})" if others else "")
            + ". It does not beat the benchmark reliably; it is the rule's fallback, not a "
            "forecast.",
        ]
    better = dominated_by(table, pick, passing_ips)
    while better is not None:
        reasons.append(
            f"{better} was chosen instead of {pick}: it passes the IPS and is at least as good "
            "on Sharpe ratio, volatility, drawdown, tracking error and turnover, and better on "
            "at least one."
        )
        pick = better
        status = "recommended" if pick in builds_ok else "next_best"
        better = dominated_by(table, pick, passing_ips)
    current_ok = (
        CURRENT in table.index
        and bool(table.loc[CURRENT].get("passes_ips", True))
        and pd.notna(table.loc[CURRENT, "sharpe"])
    )
    if (
        status != "recommended"
        and current_ok
        and float(table.loc[CURRENT, "sharpe"]) >= float(table.loc[pick, "sharpe"])
    ):
        return Recommendation(
            KEEP,
            rule,
            [
                *reasons[:1],
                f"The next best option, {pick}, had a walk-forward Sharpe ratio of "
                f"{table.loc[pick, 'sharpe']:.2f}; your current portfolio passes the IPS with "
                f"{float(table.loc[CURRENT, 'sharpe']):.2f}, so keep it.",
            ],
            [],
            survived,
            "keep",
            True,
        )
    return Recommendation(pick, rule, reasons, ranking, survived, status)


# ------------------------------------------------------------------------ plain-English status

STATUS_LABELS = {
    "ips": "✗ Breaks IPS",
    "held_up": "✗ Didn't hold up out of sample",
    "tracking_error": "✗ Too much tracking error",
    "volatility": "✗ More volatile than the benchmark",
}


def candidate_statuses(
    table: pd.DataFrame,
    rec: Recommendation,
    mode: str,
    ips_reasons: Mapping[str, list[str]] | None = None,
    tracking_error_max: float | None = None,
    tolerance: float = 0.005,
    held_up_ratio: float = 0.5,
    min_information_ratio: float = 0.10,
) -> dict[str, tuple[str, str]]:
    """A short status and a one-line reason for every row of a results table, for readers who
    aren't analysts: e.g. ("✗ Too much tracking error", "Tracking error 4.87%, over the 4.00%
    limit."). Derived from the same tests the rule applies (``mode``: "build" or "improve").
    """
    ips_reasons = ips_reasons or {}
    out: dict[str, tuple[str, str]] = {}
    closest = rec.status == "closest_fails"
    chosen_label = (
        "≈ Closest (still fails the IPS)"
        if closest
        else {"recommended": "✓ Recommended", "next_best": "✓ Next best"}.get(
            rec.status, "✓ Recommended"
        )
    )
    build_fails = (
        build_test_failures(table, tracking_error_max, tolerance, held_up_ratio)
        if mode == "build" and tracking_error_max is not None
        else {}
    )

    def f2(x) -> str:
        return "–" if pd.isna(x) else f"{float(x):.2f}"

    def why_failed(name: str, keys: list[str]) -> str:
        r = table.loc[name]
        parts = []
        for k in keys:
            if k == "ips":
                parts.append("Fails the IPS: " + "; ".join(ips_reasons.get(name, ["a rule"])))
            elif k == "held_up":
                parts.append(
                    f"Sharpe {f2(r['sharpe'])} in the test against {f2(r['in_sample_sharpe'])} "
                    f"when fitted: it kept less than {held_up_ratio:.0%} of it"
                )
            elif k == "tracking_error":
                parts.append(
                    f"Tracking error {float(r['tracking_error']):.2%}, over the "
                    f"{tracking_error_max:.2%} limit"
                )
            elif k == "volatility":
                bench = float(table.loc[BENCHMARK, "volatility"])
                parts.append(
                    f"Volatility {float(r['volatility']):.2%}, above the benchmark's {bench:.2%}"
                )
        return "; ".join(parts) + "."

    for name in table.index:
        name = str(name)
        r = table.loc[name]
        if name == BENCHMARK:
            out[name] = ("— Reference", "The benchmark the portfolios are compared with.")
            continue
        if name == CURRENT:
            if rec.choice == KEEP:
                rest = [
                    x for x in rec.reasons if x != "Your current portfolio passes every IPS rule."
                ]
                out[name] = (
                    chosen_label if closest else "✓ Keep (recommended)",
                    (rest or rec.reasons or ["Your current portfolio."])[-1],
                )
            else:
                out[name] = ("— Today's portfolio", "What the client holds now, for comparison.")
            continue
        if name == rec.choice:
            out[name] = (chosen_label, "Chosen by the rule (see the summary above).")
            continue
        if "feasible" in table and not bool(pd.isna(r["feasible"]) or r["feasible"]):
            out[name] = ("✗ Couldn't meet the limits", "No portfolio met every limit.")
            continue
        if mode == "build":
            keys = build_fails.get(name, [])
            if keys:
                out[name] = (STATUS_LABELS[keys[0]], why_failed(name, keys))
            elif name not in (TRACKER, DIVERSIFIED) and not (
                r["excess_return"] > 0 and r["information_ratio"] >= min_information_ratio
            ):
                out[name] = (
                    "✗ Didn't beat the benchmark enough",
                    f"Information ratio {f2(r['information_ratio'])} (at least "
                    f"{min_information_ratio:.2f} is needed), "
                    f"{float(r['excess_return']):.2%} a year against the benchmark.",
                )
            else:
                out[name] = ("Ranked lower", "Passed every test; another option ranked higher.")
            continue
        # improve
        if "passes_ips" in table and not bool(r["passes_ips"]):
            out[name] = (STATUS_LABELS["ips"], why_failed(name, ["ips"]))
        elif not rec.held_up.get(name, True):
            out[name] = (STATUS_LABELS["held_up"], why_failed(name, ["held_up"]))
        elif name in rec.ranking:
            if rec.choice == KEEP:
                out[name] = (
                    "✗ Not enough better than today",
                    "Qualified under the rule but didn't improve enough on the current portfolio.",
                )
            else:
                place = rec.ranking.index(name) + 1
                out[name] = (
                    "Ranked lower",
                    f"Qualified under the rule; ranked {place} of {len(rec.ranking)}.",
                )
        else:
            out[name] = ("✗ Didn't meet the rule for this goal", "See the rule above.")
    return out
