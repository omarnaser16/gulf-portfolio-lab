"""Turn an investor's plain-English goals into an optimisation spec, and nothing more.

The model fills in one tool, ``propose_spec``, whose input *is* the spec: an objective and
limits copied from the user's words (percentages exactly as written). It cannot propose
weights (there is no field for them), and every number it writes is checked against the
user's text. The user confirms or edits the spec; only then does the app call the optimiser.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from gpl.optimise import OBJECTIVES, OptimisationSpec

SPEC_SYSTEM_PROMPT = """\
You translate an investor's goals into an optimisation specification for their portfolio. \
You do not choose weights and you never calculate anything.

Call the propose_spec tool exactly once:
- objective: "max_sharpe" for best risk-adjusted return ("maximise Sharpe", "best return for \
the risk"); "min_variance" for lowest risk; "max_return" for highest return; \
"target_volatility" when they want the highest return for a stated volatility; \
"target_return" when they want the lowest risk for a stated return. If they state no \
objective, use "max_sharpe" and say so in the summary.
- Write percentages exactly as the user wrote them, as numbers of percent: "12% vol" -> 12, \
"no more than a quarter in one stock" cannot be written exactly, so put it in \
unmapped_requests rather than converting it.
- max_weight_pct: a limit that applies to every single asset ("no single asset over 25%").
- asset_limits: limits on named holdings (use the ticker). group_limits: limits on groups \
(GCC, Global equity, Bonds, Gold, Other) such as "at least 20% GCC".
- max_volatility_pct / min_return_pct: annual volatility cap / return floor.
- Use null for anything the user did not state. Never add limits they did not ask for.
- unmapped_requests: anything you could not express with these fields (e.g. "avoid oil \
companies", "keep dividends high"), in the user's words.
- summary: one or two sentences restating the spec in plain English.
"""


def spec_tool(tickers: Sequence[str], groups: Sequence[str]) -> dict:
    """Strict tool definition whose input is the spec (no field for weights)."""
    pct = {"type": ["number", "null"]}
    return {
        "name": "propose_spec",
        "description": "Record the optimisation objective and the limits the user stated. "
        "Percentages exactly as written by the user (12 means 12%).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "objective": {"type": "string", "enum": list(OBJECTIVES)},
                "max_weight_pct": pct,
                "asset_limits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string", "enum": list(tickers)},
                            "min_pct": pct,
                            "max_pct": pct,
                        },
                        "required": ["ticker", "min_pct", "max_pct"],
                        "additionalProperties": False,
                    },
                },
                "group_limits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "group": {"type": "string", "enum": list(groups)},
                            "min_pct": pct,
                            "max_pct": pct,
                        },
                        "required": ["group", "min_pct", "max_pct"],
                        "additionalProperties": False,
                    },
                },
                "max_volatility_pct": pct,
                "min_return_pct": pct,
                "unmapped_requests": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
            },
            "required": [
                "objective",
                "max_weight_pct",
                "asset_limits",
                "group_limits",
                "max_volatility_pct",
                "min_return_pct",
                "unmapped_requests",
                "summary",
            ],  # fmt: skip
            "additionalProperties": False,
        },
    }


@dataclass
class SpecProposal:
    """What the model proposed, in the user's percent units, plus checks for the user."""

    draft: dict[str, Any]  # the tool input (percent units), editable by the user
    unmapped: list[str]
    summary: str
    numbers_not_in_goals: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def _pct_numbers(draft: Mapping[str, Any]) -> list[float]:
    values = [
        draft.get("max_weight_pct"),
        draft.get("max_volatility_pct"),
        draft.get("min_return_pct"),
    ]
    for row in [*draft.get("asset_limits", []), *draft.get("group_limits", [])]:
        values += [row.get("min_pct"), row.get("max_pct")]
    return [float(v) for v in values if v is not None]


def numbers_not_in_goals(draft: Mapping[str, Any], goals: str) -> list[str]:
    """Spec numbers that don't appear in the user's text (the model must not invent limits)."""
    written = {round(float(n), 6) for n in re.findall(r"\d+(?:\.\d+)?", goals or "")}
    return [f"{v:g}%" for v in _pct_numbers(draft) if round(v, 6) not in written]


def propose_spec(
    client: Any,
    model: str,
    goals: str,
    groups_by_ticker: Mapping[str, str],
    max_tokens: int = 4_000,
    effort: str = "low",
) -> SpecProposal:
    """Ask the model to transcribe ``goals`` into the spec tool. Raises ValueError if it doesn't."""
    tickers = list(groups_by_ticker)
    groups = sorted(
        set(groups_by_ticker.values()) | {"GCC", "Global equity", "Bonds", "Gold", "Other"}
    )
    holdings = ", ".join(f"{t} ({g})" for t, g in groups_by_ticker.items())
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SPEC_SYSTEM_PROMPT,
        tools=[spec_tool(tickers, groups)],
        output_config={"effort": effort},
        messages=[{"role": "user", "content": f"My holdings: {holdings}.\n\nMy goals: {goals}"}],
    )
    calls = [b for b in response.content if b.type == "tool_use" and b.name == "propose_spec"]
    if response.stop_reason == "refusal" or not calls:
        raise ValueError("The model did not return a specification. Try rephrasing your goals.")
    draft = dict(calls[0].input)
    return SpecProposal(
        draft=draft,
        unmapped=list(draft.get("unmapped_requests") or []),
        summary=str(draft.get("summary") or ""),
        numbers_not_in_goals=numbers_not_in_goals(draft, goals),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


def spec_from_draft(
    draft: Mapping[str, Any], groups_by_ticker: Mapping[str, str]
) -> OptimisationSpec:
    """Percent-unit draft (from the model or the edited form) -> ``OptimisationSpec``.

    Raises ValueError with a readable message for an invalid draft (e.g. a target objective
    without its target, or a limit on a ticker that isn't held).
    """

    def frac(v: Any) -> float | None:
        return None if v is None else float(v) / 100

    unknown = [
        r["ticker"] for r in draft.get("asset_limits", []) if r["ticker"] not in groups_by_ticker
    ]
    if unknown:
        raise ValueError(f"Limits refer to assets you don't hold: {', '.join(unknown)}.")
    max_weight = frac(draft.get("max_weight_pct"))
    return OptimisationSpec(
        objective=draft.get("objective") or "max_sharpe",
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
        max_volatility=frac(draft.get("max_volatility_pct")),
        min_return=frac(draft.get("min_return_pct")),
    )


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
    if spec.max_volatility is not None:
        lines.append(f"Volatility at most {spec.max_volatility:.2%} a year")
    if spec.min_return is not None:
        lines.append(f"Expected return at least {spec.min_return:.2%} a year (historical)")
    lines.append("Long-only, fully invested")
    return lines
