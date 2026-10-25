"""Suitability checker: does a portfolio respect the client's IPS? Pass, fail or review.

Pure functions: no I/O, no printing, no plotting.

Each rule measures one thing with tested engine functions and writes its reason from the
numbers, so the record says exactly why a portfolio passed or failed (regulators such as the
DFSA have criticised generic, tick-box suitability notes). "Review" means the data can't
decide (e.g. unknown Sharia status, no portfolio value); it never counts as a pass. The
checks support a licensed adviser's judgement; they don't replace it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from gpl.client import IPS, PEGGED, excluded_holdings
from gpl.data import SUBUNITS
from gpl.metrics import annualised_volatility, max_drawdown
from gpl.scenarios import episode_performance

STATUS = {"pass": "Pass", "fail": "Fail", "review": "Review"}


@dataclass
class Check:
    rule: str
    status: str  # pass | fail | review
    measured: str
    limit: str
    reason: str
    breach: float | None = None  # how far past the limit (limit's units: fraction or USD)
    scale: float | None = None  # the limit's size, to compare breaches across rules


def _pct(x: float | None) -> str:
    return "not set" if x is None else f"{x:.2%}"


def check_suitability(
    weights: pd.Series,
    returns: pd.DataFrame,
    ips: IPS,
    attributes: Mapping[str, Mapping[str, str]],
    currencies: Mapping[str, str],
    groups: Mapping[str, str],
    periods_per_year: int,
    episodes: Sequence[Mapping[str, str]],
    value: float | None = None,
    liquid_max_volatility: float = 0.06,
) -> list[Check]:
    """Check ``weights`` against every IPS rule.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights (zero weights are ignored).
    returns : pd.DataFrame
        Daily returns of the holdings (USD, common dates); the portfolio is held at
        ``weights``, rebalanced daily, over the full sample.
    value : float, optional
        Portfolio value in USD (for the liquidity rule).
    liquid_max_volatility : float, default 0.06
        Holdings in the Bonds group, or with annual volatility at most this, count as the
        low-volatility reserve for near-term cash needs.
    """
    w = weights[weights > 0]
    port = returns[list(w.index)] @ w
    lim = ips.limits
    band = (
        f"band {ips.risk.score}: capacity {ips.risk.capacity}, willingness {ips.risk.willingness}"
        if ips.risk.score
        else "no risk band"
    )
    checks: list[Check] = []

    def source(key: str) -> str:
        s = lim["source"][key]
        return band if s.startswith("band") else s

    # Volatility
    vol = float(annualised_volatility(port, periods_per_year))
    if lim["max_volatility"] is None:
        checks.append(
            Check(
                "Volatility",
                "review",
                _pct(vol),
                "not set",
                "No volatility limit: complete the risk questionnaire or set one.",
            )
        )
    else:
        ok = vol <= lim["max_volatility"]
        checks.append(
            Check(
                "Volatility",
                "pass" if ok else "fail",
                _pct(vol),
                _pct(lim["max_volatility"]),
                f"Historical volatility {vol:.2%} is {'within' if ok else 'above'} the "
                f"{lim['max_volatility']:.2%} IPS limit ({source('max_volatility')}).",
                None if ok else vol - lim["max_volatility"],
                lim["max_volatility"],
            )
        )

    # Max drawdown and worst stress episode
    mdd = float(max_drawdown(port))
    limit_dd = lim["max_drawdown"]
    if limit_dd is None:
        checks.append(
            Check(
                "Max drawdown",
                "review",
                _pct(mdd),
                "not set",
                "No drawdown limit: complete the risk questionnaire or set one.",
            )
        )
    else:
        ok = mdd >= -limit_dd
        checks.append(
            Check(
                "Max drawdown",
                "pass" if ok else "fail",
                _pct(mdd),
                f"-{limit_dd:.2%}",
                f"The worst historical fall was {mdd:.2%} "
                f"({returns.index[0]:%b %Y} - {returns.index[-1]:%b %Y}), "
                f"{'within' if ok else 'beyond'} the {limit_dd:.2%} the client can accept "
                f"({source('max_drawdown')}).",
                None if ok else -mdd - limit_dd,
                limit_dd,
            )
        )
    stress = episode_performance(port.to_frame("Portfolio"), episodes)
    covered = stress[stress["covered"]] if "covered" in stress else stress  # none configured
    if covered.empty:
        checks.append(
            Check(
                "Worst stress episode",
                "review",
                "–",
                _pct(limit_dd),
                "No stress episode falls within the holdings' common history.",
            )
        )
    else:
        worst = covered.loc[covered["total_return"].idxmin()]
        ok = limit_dd is None or worst["total_return"] >= -limit_dd
        checks.append(
            Check(
                "Worst stress episode",
                "pass" if ok else "fail",
                _pct(worst["total_return"]),
                "–" if limit_dd is None else f"-{limit_dd:.2%}",
                f"In the {worst['episode']} ({worst['start']:%d %b %Y} - {worst['end']:%d %b %Y}) "
                f"today's weights would have returned {worst['total_return']:.2%}"
                + (
                    ""
                    if limit_dd is None
                    else f", {'within' if ok else 'beyond'} the {limit_dd:.2%} drawdown limit"
                )
                + ".",
                None if ok else -worst["total_return"] - limit_dd,
                limit_dd,
            )
        )

    # Concentration
    top, top_w = w.idxmax(), float(w.max())
    cap = lim["max_weight"]
    if cap is None:
        checks.append(
            Check("Concentration", "review", _pct(top_w), "not set", "No single-holding limit set.")
        )
    else:
        ok = top_w <= cap + 1e-9
        checks.append(
            Check(
                "Concentration",
                "pass" if ok else "fail",
                _pct(top_w),
                _pct(cap),
                f"The largest holding, {top}, is {top_w:.2%} of the portfolio, "
                f"{'within' if ok else 'above'} the {cap:.2%} limit ({source('max_weight')}).",
                None if ok else top_w - cap,
                cap,
            )
        )

    # Home-market bias
    gcc = float(sum(v for t, v in w.items() if groups.get(t) == "GCC"))
    home = lim["home_bias_gcc"]
    if not home:
        checks.append(
            Check(
                "Home-market bias",
                "pass",
                _pct(gcc),
                "none",
                "The client set no minimum in GCC holdings.",
            )
        )
    else:
        ok = gcc >= home - 1e-9
        checks.append(
            Check(
                "Home-market bias",
                "pass" if ok else "fail",
                _pct(gcc),
                f"at least {home:.2%}",
                f"GCC holdings are {gcc:.2%} of the portfolio; the client asked for at least "
                f"{home:.2%}.",
                None if ok else home - gcc,
                home,
            )
        )

    # Liquidity
    vols = returns[list(w.index)].std() * (periods_per_year**0.5)
    liquid = [t for t in w.index if groups.get(t) == "Bonds" or vols[t] <= liquid_max_volatility]
    liquid_share = float(w[liquid].sum()) if liquid else 0.0
    need = lim["liquidity_reserve_usd"]
    if need <= 0:
        checks.append(
            Check(
                "Liquidity",
                "pass",
                _pct(liquid_share),
                "none",
                "The client has no near-term cash need recorded.",
            )
        )
    elif not value:
        checks.append(
            Check(
                "Liquidity",
                "review",
                _pct(liquid_share),
                f"USD {need:,.0f}",
                f"Enter the portfolio value to check the cash reserve of USD {need:,.0f}.",
            )
        )
    else:
        held = liquid_share * value
        ok = held >= need - 0.5
        names = ", ".join(liquid) if liquid else "none"
        checks.append(
            Check(
                "Liquidity",
                "pass" if ok else "fail",
                f"USD {held:,.0f}",
                f"USD {need:,.0f}",
                f"Low-volatility holdings ({names}) are worth USD {held:,.0f}; the client needs "
                f"USD {need:,.0f} within {ips.client.liquidity_months} months (including any "
                "reserve).",
                None if ok else need - held,
                need,
            )
        )

    # Exclusions
    excluded, review = excluded_holdings(ips, list(w.index), dict(attributes))
    if not (lim["sharia_only"] or lim["excluded_sectors"]):
        checks.append(Check("Exclusions", "pass", "–", "none", "The client set no exclusions."))
    elif excluded:
        checks.append(
            Check(
                "Exclusions",
                "fail",
                ", ".join(excluded),
                "none held",
                "Held but excluded by the IPS: "
                + "; ".join(f"{t} ({w[t]:.2%}): {why}" for t, why in excluded.items())
                + ".",
                float(w[list(excluded)].sum()),
                1.0,
            )
        )
    elif review:
        checks.append(
            Check(
                "Exclusions",
                "review",
                ", ".join(review),
                "none held",
                "Needs checking: " + "; ".join(f"{t}: {why}" for t, why in review.items()) + ".",
            )
        )
    else:
        checks.append(
            Check("Exclusions", "pass", "–", "none held", "No holding is excluded by the IPS.")
        )

    # Currency
    def base(code: str) -> str:
        return SUBUNITS.get(code, (code, 1))[0]

    non_pegged = {
        t: base(currencies.get(t, "USD"))
        for t in w.index
        if base(currencies.get(t, "USD")) not in PEGGED
    }
    share = float(w[list(non_pegged)].sum()) if non_pegged else 0.0
    cap_fx = lim["max_non_pegged_currency"]
    ok = share <= cap_fx + 1e-9
    detail = ", ".join(f"{t} in {c}" for t, c in non_pegged.items()) if non_pegged else "none"
    checks.append(
        Check(
            "Currency",
            "pass" if ok else "fail",
            _pct(share),
            _pct(cap_fx),
            f"Holdings in currencies not pegged to the US dollar ({detail}) are {share:.2%} of the "
            f"portfolio; the IPS allows {cap_fx:.2%}. Pegged: USD, SAR, AED, QAR, BHD, OMR.",
            None if ok else share - cap_fx,
            cap_fx or 1.0,
        )
    )
    return checks


def failures(checks: Sequence[Check]) -> list[Check]:
    """The rules a portfolio fails ("review" is not a fail, but never a pass either)."""
    return [c for c in checks if c.status == "fail"]


def closeness(checks: Sequence[Check]) -> tuple[int, float]:
    """How far a portfolio is from passing: (number of failed rules, summed relative breach).

    Each breach is divided by its limit, so 2 pp over a 12% volatility limit (0.17) and USD
    100,000 short of a USD 600,000 cash need (0.17) weigh about the same. Smaller is closer.
    """
    failed = failures(checks)
    return len(failed), float(sum((c.breach or 0) / (c.scale or 1) for c in failed))


def breach_text(check: Check) -> str:
    """'Volatility: 14.20%, 2.20 pp above the 12.00% limit' for a failed check."""
    if check.breach is None:
        return f"{check.rule}: {check.measured} (limit {check.limit})"
    if check.rule == "Liquidity":
        return (
            f"{check.rule}: {check.measured} in cash-like holdings, USD {check.breach:,.0f} "
            f"short of the {check.limit} needed"
        )
    if check.rule == "Exclusions":
        return f"{check.rule}: {check.measured} held ({check.breach:.2%} of the portfolio)"
    if check.rule in MINIMUM_RULES:
        minimum = check.limit.removeprefix("at least ")
        return (
            f"{check.rule}: {check.measured}, {check.breach * 100:.2f} pp below the {minimum} "
            "minimum"
        )
    return (
        f"{check.rule}: {check.measured}, {check.breach * 100:.2f} pp beyond the "
        f"{check.limit} limit"
    )


MINIMUM_RULES = {"Home-market bias"}


def suitability_table(results: Mapping[str, Sequence[Check] | Mapping[str, Check]]) -> pd.DataFrame:
    """Rule x portfolio table of 'Pass/Fail/Review: measured (limit)'."""
    out = {}
    for name, checks in results.items():
        items = checks.values() if isinstance(checks, Mapping) else checks
        out[name] = {c.rule: f"{STATUS[c.status]}: {c.measured} (limit {c.limit})" for c in items}
    return pd.DataFrame(out)
