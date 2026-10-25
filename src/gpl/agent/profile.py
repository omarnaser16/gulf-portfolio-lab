"""Fill the client form from the adviser's plain-English notes, and nothing more.

The model fills one strict tool, ``propose_profile``, with facts the adviser wrote (horizon,
amounts, residency, exclusions...). It cannot answer the risk questionnaire or set a risk
score (there are no fields for them): those come from the client's own answers and the tested
scoring rule. Every number it writes is checked against the notes. The adviser reviews the
form before it is used.
"""

import re
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from gpl.client import CLASSIFICATIONS, ClientProfile, month_after

PROFILE_SYSTEM_PROMPT = """\
You fill in a wealth manager's client form from their notes. You never calculate, estimate or \
guess anything, and you never score the client's risk tolerance.

Call the propose_profile tool exactly once:
- Copy facts exactly as written. Amounts in US dollars as plain numbers ("500k" -> 500000, \
"1.5 million" -> 1500000). Percentages as numbers of percent ("20%" -> 20).
- Use null (or an empty list) for anything the notes don't state. Don't infer residency from \
a city unless the country is obvious from it (Dubai -> UAE, Riyadh -> Saudi Arabia, Doha -> \
Qatar).
- goals: each goal the client states, with an amount, a multiple of today's value, or years, \
as written.
- max_volatility_pct / max_drawdown_pct: only if the notes state a limit ("can't lose more \
than 20%" is a drawdown limit of 20).
- unmapped_requests: anything that doesn't fit any field, in the notes' words. Don't list \
anything you recorded in a field.
- summary: one or two sentences restating what you recorded.
"""


def _num(nullable: bool = True) -> dict:
    return {"type": ["number", "null"]} if nullable else {"type": "number"}


def profile_tool() -> dict:
    """Strict tool whose input is the factual part of the client form (no scores, no weights)."""
    props = {
        "horizon_years": _num(),
        "return_target_pct": _num(),
        "portfolio_value_usd": _num(),
        "liquidity_need_usd": _num(),
        "liquidity_months": _num(),
        "residency": {"type": ["string", "null"]},
        "us_person": {"type": ["boolean", "null"]},
        # strict mode: an enum can't share a type list with null, so offer either explicitly
        "classification": {
            "anyOf": [{"type": "string", "enum": list(CLASSIFICATIONS)}, {"type": "null"}]
        },
        "sharia_only": {"type": ["boolean", "null"]},
        "excluded_sectors": {"type": "array", "items": {"type": "string"}},
        "home_bias_gcc_pct": _num(),
        "max_volatility_pct": _num(),
        "max_drawdown_pct": _num(),
        "goals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "amount_usd": _num(),
                    "multiple": _num(),
                    "years": _num(),
                },
                "required": ["description", "amount_usd", "multiple", "years"],
                "additionalProperties": False,
            },
        },
        "unmapped_requests": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    }
    return {
        "name": "propose_profile",
        "description": "Record the client facts stated in the adviser's notes.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": props,
            "required": list(props),
            "additionalProperties": False,
        },
    }


_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}


def numbers_in_text(text: str) -> set[float]:
    """Numbers written in the notes, with 'k', 'm', 'million' etc. expanded."""
    found = set()
    # the suffix needs a word boundary ("12 months" is not 12 million); a bare number doesn't
    # ("2x" is the number 2)
    pattern = r"(\d+(?:[.,]\d+)*)(?:\s*(k|thousand|mn|million|m|bn|billion)\b)?"
    for number, suffix in re.findall(pattern, (text or "").lower()):
        value = float(number.replace(",", ""))
        found.add(round(value, 6))
        if suffix:
            found.add(round(value * _SCALE[suffix], 6))
    return found


def _draft_numbers(draft: dict) -> list[float]:
    keys = [
        "horizon_years",
        "return_target_pct",
        "portfolio_value_usd",
        "liquidity_need_usd",
        "liquidity_months",
        "home_bias_gcc_pct",
        "max_volatility_pct",
        "max_drawdown_pct",
    ]
    values = [draft.get(k) for k in keys]
    for g in draft.get("goals") or []:
        values += [g.get("amount_usd"), g.get("multiple"), g.get("years")]
    return [float(v) for v in values if v is not None]


def numbers_not_in_notes(draft: dict, notes: str) -> list[str]:
    """Numbers the model wrote that don't appear in the adviser's notes."""
    written = numbers_in_text(notes)
    return [f"{v:g}" for v in _draft_numbers(draft) if round(v, 6) not in written]


@dataclass
class ProfileProposal:
    draft: dict[str, Any]
    unmapped: list[str]
    summary: str
    numbers_not_in_notes: list[str] = field(default_factory=list)


def propose_profile(client: Any, model: str, notes: str, effort: str = "low") -> ProfileProposal:
    """Ask the model to transcribe ``notes`` into the profile tool. Raises ValueError if not."""
    response = client.messages.create(
        model=model,
        max_tokens=3000,
        system=PROFILE_SYSTEM_PROMPT,
        tools=[profile_tool()],
        output_config={"effort": effort},
        messages=[{"role": "user", "content": f"Adviser's notes:\n{notes}"}],
    )
    calls = [b for b in response.content if b.type == "tool_use" and b.name == "propose_profile"]
    if response.stop_reason == "refusal" or not calls:
        raise ValueError("The model did not fill the form. Try rephrasing the notes.")
    draft = dict(calls[0].input)
    return ProfileProposal(
        draft,
        list(draft.get("unmapped_requests") or []),
        str(draft.get("summary") or ""),
        numbers_not_in_notes(draft, notes),
    )


def apply_proposal(profile: ClientProfile, draft: dict) -> ClientProfile:
    """Copy the stated facts onto the profile; questionnaire answers are never touched."""
    simple = [
        "horizon_years",
        "return_target_pct",
        "portfolio_value_usd",
        "residency",
        "us_person",
        "classification",
        "sharia_only",
        "home_bias_gcc_pct",
    ]
    updates = {k: draft[k] for k in simple if draft.get(k) is not None}
    if draft.get("liquidity_need_usd"):
        months = int(draft.get("liquidity_months") or 12)
        updates["liquidity_needs"] = [
            {
                "description": "Cash need",
                "amount_usd": float(draft["liquidity_need_usd"]),
                "date": month_after(date.today(), months),
            }
        ]
    if draft.get("excluded_sectors"):
        updates["excluded_sectors"] = [s[:1].upper() + s[1:] for s in draft["excluded_sectors"]]
    if draft.get("goals"):
        updates["goals"] = [{k: v for k, v in g.items() if v is not None} for g in draft["goals"]]
    overrides = dict(profile.overrides)
    for key in ("max_volatility_pct", "max_drawdown_pct"):
        if draft.get(key) is not None:
            overrides[key] = float(draft[key])
    return replace(profile, **updates, overrides=overrides)
