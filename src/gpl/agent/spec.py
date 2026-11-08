"""The Optimise form's goal and limits as an ``OptimisationSpec``, and their descriptions.

No AI: the adviser picks the goal and types any limits; ``spec_from_draft`` turns the form
(percentages) into the optimiser's spec (fractions) and checks it.
"""

from collections.abc import Mapping
from typing import Any

from gpl.optimise import OptimisationSpec
from gpl.recommend import GOALS

TARGET_FIELD = {"target_return": "min_return_pct", "target_volatility": "max_volatility_pct"}


def spec_from_draft(
    draft: Mapping[str, Any], groups_by_ticker: Mapping[str, str]
) -> tuple[str, OptimisationSpec, float | None]:
    """Percent-unit draft (from the model or the edited form) -> (goal, limits, target).

    For the target goals, the target (return or volatility) is taken out of the limits: it
    picks among the methods rather than constraining every one of them. Otherwise a stated
    volatility cap or return floor is a limit for every method. Raises ValueError with a
    readable message for an invalid draft (a target goal without its number, or a limit on
    a ticker that isn't held).
    """

    def frac(v: Any) -> float | None:
        return None if v is None else float(v) / 100

    goal = draft.get("goal") or "max_risk_adjusted"
    if goal not in GOALS:
        raise ValueError(f"Unknown goal {goal!r}.")
    unknown = [
        r["ticker"] for r in draft.get("asset_limits", []) if r["ticker"] not in groups_by_ticker
    ]
    if unknown:
        raise ValueError(f"Limits refer to assets you don't hold: {', '.join(unknown)}.")
    target = None
    if goal in TARGET_FIELD:
        target = frac(draft.get(TARGET_FIELD[goal]))
        if target is None:
            what = "return" if goal == "target_return" else "volatility"
            raise ValueError(f"Enter the target {what} for this goal.")
    max_weight = frac(draft.get("max_weight_pct"))
    limits = OptimisationSpec(
        objective="max_sharpe",  # replaced by each candidate method
        max_weight=1.0 if max_weight is None else max_weight,
        asset_bounds={
            r["ticker"]: (frac(r.get("min_pct")), frac(r.get("max_pct")))
            for r in draft.get("asset_limits", [])
            if r.get("min_pct") is not None or r.get("max_pct") is not None
        },
        groups=dict(groups_by_ticker),
        group_bounds={
            r["group"]: (frac(r.get("min_pct")), frac(r.get("max_pct")))
            for r in draft.get("group_limits", [])
            if r.get("min_pct") is not None or r.get("max_pct") is not None
        },
        max_volatility=None
        if goal == "target_volatility"
        else frac(draft.get("max_volatility_pct")),
        min_return=None if goal == "target_return" else frac(draft.get("min_return_pct")),
    )
    return goal, limits, target


def describe_goal(goal: str, target: float | None) -> str:
    """The goal in plain words, with its target."""
    text = GOALS[goal]
    if goal == "target_return" and target is not None:
        text += f": at least {target:.2%} a year"
    elif goal == "target_volatility" and target is not None:
        text += f": about {target:.2%} a year"
    return text


def describe_limits(spec: OptimisationSpec) -> list[str]:
    """The limits every candidate method must respect, in plain words."""
    return describe_spec(spec)[1:]


def describe_spec(spec: OptimisationSpec) -> list[str]:
    """Plain-English lines describing a spec (for the confirmation view and the memo)."""
    names = {
        "max_sharpe": "Maximise the Sharpe ratio (return per unit of risk)",
        "min_variance": "Minimise volatility",
        "max_return": "Maximise expected return",
        "target_volatility": "Maximise return with volatility at most the cap",
        "target_return": "Minimise volatility with return at least the floor",
        "risk_parity": "Equal risk contributions, or as close as the limits allow",
        "max_diversification": "Maximise the diversification ratio",
    }
    lines = [names[spec.objective]]
    if spec.max_weight < 1:
        lines.append(f"No single asset above {spec.max_weight:.2%}")
    for t, (lo, hi) in spec.asset_bounds.items():
        lines.append(
            f"{t}: "
            + " and ".join(
                p
                for p in [
                    lo is not None and f"at least {lo:.2%}",
                    hi is not None and f"at most {hi:.2%}",
                ]
                if p
            )
        )
    for g, (lo, hi) in spec.group_bounds.items():
        lines.append(
            f"Group {g}: "
            + " and ".join(
                p
                for p in [
                    lo is not None and f"at least {lo:.2%}",
                    hi is not None and f"at most {hi:.2%}",
                ]
                if p
            )
        )
    for name, (members, lo, hi) in spec.sets.items():
        held = ", ".join(sorted(members)) or "no holding"
        lines.append(
            f"{name} ({held}): "
            + " and ".join(
                p
                for p in [
                    lo is not None and f"at least {lo:.2%}",
                    hi is not None and f"at most {hi:.2%}",
                ]
                if p
            )
        )
    if spec.max_volatility is not None:
        lines.append(f"Volatility at most {spec.max_volatility:.2%} a year")
    if spec.min_return is not None:
        lines.append(f"Expected return at least {spec.min_return:.2%} a year (historical)")
    lines.append("Long-only, fully invested")
    return lines


def _bounds_text(lo: float | None, hi: float | None) -> str:
    parts = [
        f"at least {lo:.2%}" if lo is not None else "",
        f"at most {hi:.2%}" if hi is not None else "",
    ]
    return " and ".join(p for p in parts if p)


def _limit_items(spec: OptimisationSpec | None) -> dict[str, str]:
    """Each limit in ``spec`` as name -> value text (no entry for 'no limit')."""
    if spec is None:
        return {}
    items: dict[str, str] = {}
    if spec.max_weight < 1:
        items["Max weight per holding"] = f"{spec.max_weight:.2%}"
    if spec.max_volatility is not None:
        items["Volatility cap (a year)"] = f"{spec.max_volatility:.2%}"
    if spec.min_return is not None:
        items["Return floor (a year, historical)"] = f"{spec.min_return:.2%}"
    for g, (lo, hi) in spec.group_bounds.items():
        if lo is not None or hi is not None:
            items[f"Group {g}"] = _bounds_text(lo, hi)
    for t, (lo, hi) in spec.asset_bounds.items():
        if lo is not None or hi is not None:
            items[f"Holding {t}"] = _bounds_text(lo, hi)
    for name, (members, lo, hi) in spec.sets.items():
        held = ", ".join(sorted(members)) or "no holding"
        items[f"{name} ({held})"] = _bounds_text(lo, hi)
    return items


def active_limits(
    form: OptimisationSpec, ips: OptimisationSpec | None, combined: OptimisationSpec
) -> list[dict[str, str]]:
    """The limits every method runs under, and where each comes from.

    Rows: ``limit``, ``value`` (as applied, i.e. from ``combined``) and ``source``: "Form",
    "Client IPS" or "Form and IPS (stricter applies)". Empty when nothing is limited.
    """
    form_items, ips_items = _limit_items(form), _limit_items(ips)
    rows = []
    for name, value in _limit_items(combined).items():
        in_form, in_ips = name in form_items, name in ips_items
        source = (
            "Form and IPS (stricter applies)"
            if in_form and in_ips
            else "Client IPS"
            if in_ips
            else "Form"
        )
        rows.append({"limit": name, "value": value, "source": source})
    return rows
