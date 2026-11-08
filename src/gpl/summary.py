"""Plain-English summary of an Optimise or Build result, written by code from the report.

Pure functions: no I/O, no printing, no plotting. Every sentence is put together from facts
already in the report (counts, the rule's status and reasons, the weights), so nothing here
calculates a new result or depends on the AI.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from gpl.metrics import no_negative_zero
from gpl.recommend import BENCHMARK, CURRENT, KEEP, TRACKER

# How to read the results: one plain sentence each
GLOSSARY = {
    "Sharpe ratio": "Return above cash for each unit of risk taken; higher is better, and "
    "differences under about 0.05 are within noise.",
    "Information ratio": "Return above the benchmark for each unit of tracking error; 0.10 or "
    "more is needed before we call a portfolio better than the benchmark.",
    "Tracking error": "How far the portfolio's return strays from the benchmark's in a year "
    "(the volatility of the difference).",
    "Held up out of sample": "In the month-by-month test, the portfolio kept at least half of "
    "the Sharpe ratio it showed when fitted to all the data, so it wasn't just a lucky fit.",
    "Turnover": "How much is bought and sold, as a share of the portfolio: 1.00× a year means "
    "trading the whole portfolio's value once a year (costs rise with it).",
}


@dataclass
class ResultSummary:
    did: list[str]  # what we did: 3 bullets
    found: str  # what we found: 1-2 sentences
    recommend: str  # what we recommend and why
    changes: str  # what changes: the trades in words, or "No trades needed."


SWAPS = "Re-weight + swaps"
IPS_PASS = "Your current portfolio passes every IPS rule."


def main_reason(reasons: list[str]) -> str:
    """The reason that explains a 'keep': the last one, skipping the IPS-pass note."""
    rest = [r for r in reasons if r != IPS_PASS]
    return rest[-1] if rest else (reasons[-1] if reasons else "")


def _pct(x: float) -> str:
    return f"{no_negative_zero(x, 4):.2%}"


def change_text(
    current: Mapping[str, float], proposed: Mapping[str, float] | None, max_moves: int = 4
) -> str:
    """The move from ``current`` to ``proposed`` in words, largest changes first, e.g.
    "Sell 15.00% SPY; buy 10.00% AGG and 5.00% GLD (turnover 30.00%)."
    Changes under 0.05 percentage points are ignored; no change gives "No trades needed."
    """
    if proposed is None:
        return "No trades needed."
    now, new = pd.Series(current, dtype=float), pd.Series(proposed, dtype=float)
    idx = now.index.union(new.index)
    every = new.reindex(idx, fill_value=0.0) - now.reindex(idx, fill_value=0.0)
    traded = float(every.abs().sum())  # all changes, so it matches the turnover elsewhere
    diff = every[every.abs() >= 0.0005]
    if diff.empty:
        return "No trades needed."
    order = diff.abs().sort_values(ascending=False).index
    shown, rest = list(order[:max_moves]), len(order) - max_moves

    def words(names: list[str]) -> str:
        items = [f"{_pct(abs(diff[t]))} {t}" for t in names]
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    sells = [t for t in shown if diff[t] < 0]
    buys = [t for t in shown if diff[t] > 0]
    parts = []
    if sells:
        parts.append("Sell " + words(sells))
    if buys:
        parts.append(("buy " if parts else "Buy ") + words(buys))
    text = "; ".join(parts)
    if rest > 0:
        text += f"; plus {rest} smaller change{'s' if rest > 1 else ''}"
    return f"{text} (turnover {_pct(traded)})."


def _candidates(report: Mapping) -> list[str]:
    rows = report["walk_forward_after_costs"]
    return [n for n in rows if n not in (CURRENT, BENCHMARK)]


def _period(report: Mapping) -> str:
    start, _, end = report["backtest_period"].partition(" to ")
    return f"from {start} to {end}"


def result_summary(
    report: Mapping,
    current: Mapping[str, float],
    proposed: Mapping[str, float] | None,
) -> ResultSummary:
    """The four parts of the summary card for a Build or Improve ``report``.

    ``proposed``: the weights the adviser would move to (None, or the current weights, when
    the advice is to keep the portfolio).
    """
    rec = report["recommendation"]
    status, choice = rec["status"], rec["choice"]
    reasons = rec["reasons"]
    cands = _candidates(report)
    n = len(cands)
    bench = (report.get("benchmark") or {}).get("name", "the benchmark")
    build = report.get("mode") == "build"
    after = "after trading costs"
    if build:
        uni = report["universe"]
        screened = uni["eligible"] + len(uni["excluded"])
        did = [
            f"Screened {screened} securities against the client's IPS: {uni['eligible']} eligible.",
            f"Built {n - (TRACKER in cands)} candidate portfolios"
            + (" (plus a benchmark tracker)" if TRACKER in cands else "")
            + ", re-choosing the securities each quarter from data available at the time"
            + (
                f" (turnover limit {report['turnover_limit_per_rebalance']})."
                if report.get("turnover_limit_per_rebalance", "none") != "none"
                else "."
            ),
            f"Tested each one month by month {_period(report)}, {after}, against {bench}.",
        ]
    else:
        limit = report.get("max_turnover")
        swaps = SWAPS in cands
        did = [
            f"Checked your current portfolio against your goal: {report['goal']}.",
            f"Re-optimised {n - swaps} methods"
            + (" (plus a re-weight with swaps)" if swaps else "")
            + " within your limits"
            + (f", moving at most {limit} of the portfolio a month." if limit else "."),
            f"Tested each one month by month {_period(report)}, {after}.",
        ]

    if status == "closest_fails":
        found = "No portfolio passes every rule of the client's IPS."
    elif build and status == "recommended":
        beat = len(rec.get("ranking_under_the_rule") or [])
        found = (
            f"{beat} of the {n} portfolios beat {bench} reliably after costs "
            "(an information ratio of at least 0.10) while passing every test."
        )
    elif build:
        found = f"None of the {n} portfolios beat {bench} reliably after costs within every test."
    elif (
        status == "recommended"
        and (report.get("suitability_gate") or {}).get("current_passes_ips") is False
    ):
        found = (
            "Your current portfolio breaks the client's IPS, so a change is needed; "
            f"{choice} is the best of the methods that pass every rule."
        )
    elif status == "recommended":
        found = f"{choice} did better than your current portfolio under the rule for your goal."
    else:
        found = "None of the methods did enough better than your current portfolio."

    if status == "closest_fails":
        who = "your current portfolio" if choice == KEEP else choice
        recommend = (
            f"Nothing is suitable as it stands; the closest is {who}. "
            "Review the holdings or the IPS with the client."
        )
    elif choice == KEEP:
        recommend = "Keep your current portfolio. " + main_reason(reasons)
    elif status == "next_best":
        recommend = reasons[1] if len(reasons) > 1 else f"{choice} is the next best option."
    else:
        recommend = f"Move to {choice}. " + (reasons[0] if reasons else "")

    keep = choice == KEEP or proposed is None
    changes = "No trades needed." if keep else change_text(current, proposed)
    return ResultSummary(did, found, recommend.strip(), changes)


KPIS = (  # report key, label, higher is better
    ("cagr", "Return (CAGR)", True),
    ("annual_volatility", "Volatility", False),
    ("sharpe", "Sharpe ratio", True),
    ("max_drawdown", "Max drawdown", True),  # negative: closer to zero is better
)


def _number(text: str) -> float:
    text = str(text).strip()
    return float(text.rstrip("%")) / 100 if text.endswith("%") else float(text)


def kpi_comparison(report: Mapping, choice: str) -> list[dict]:
    """Current vs ``choice`` on four walk-forward figures, from the report's own table.

    Each item: label, current and proposed (as shown in the table), change (signed, in the same
    units) and better (True/False/None), e.g. volatility falling is better.
    """
    rows = report["walk_forward_after_costs"]
    now = rows.get(CURRENT, {})
    new = now if choice == KEEP else rows.get(choice, {})
    out = []
    for key, label, up in KPIS:
        if key not in now or key not in new:
            continue
        a, b = _number(now[key]), _number(new[key])
        is_pct = str(now[key]).endswith("%")
        diff = no_negative_zero(b - a, 4 if is_pct else 2)
        change = f"{diff * 100:+.2f} pp" if is_pct else f"{diff:+.2f}"
        better = None if diff == 0 else (diff > 0) == up
        out.append(
            {
                "label": label,
                "current": now[key],
                "proposed": new[key],
                "change": change,
                "better": better,
            }
        )
    return out


def window_note(where: str, period: str) -> str:
    """Why the current portfolio's CAGR and Sharpe differ between steps: each uses its dates."""
    dates = {
        "analyse": "Analyse uses your holdings' own backtest window",
        "improve": "Improve uses the dates every method shares",
        "build": "Build uses the investable universe's dates and trading calendar",
    }
    others = "; ".join(v for k, v in dates.items() if k != where)
    return (
        f"Figures here cover {period}. Your current portfolio's CAGR and Sharpe ratio can differ "
        f"slightly between Analyse, Improve and Build because each uses its own dates: "
        f"{dates[where].split(' uses ', 1)[1]} here; {others[0].lower() + others[1:]}."
    )
