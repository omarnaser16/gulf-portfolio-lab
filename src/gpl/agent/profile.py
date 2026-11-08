"""Fill the client form from the adviser's plain-English notes, and nothing more.

The model fills one tool, ``propose_profile``, with facts the adviser wrote (horizon,
amounts, residency, exclusions...). It cannot answer the risk questionnaire or set a risk
score (there are no fields for them): those come from the client's own answers and the tested
scoring rule. The tool is not strict (its schema is too large for the API's strict-mode
grammar), so ``clean_draft`` checks every field in code before the form is touched: types,
ranges and allowed values; anything else, including any field the form doesn't have, is
dropped and reported. Every number it writes is checked against the notes. The adviser reviews
the form before it is used.
"""

import re
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from gpl.agent.api import create_message
from gpl.client import (
    CLASSIFICATIONS,
    JURISDICTIONS,
    REGIONS,
    ClientProfile,
    month_after,
    parse_month,
    regulator_for,
)
from gpl.factfind import CURRENCIES, EMPLOYMENT, MARITAL, OBJECTIVES, PRODUCTS

PROFILE_SYSTEM_PROMPT = """\
You fill in a wealth manager's client form from their notes. You never calculate, estimate or \
guess anything, and you never score the client's risk tolerance.

Call the propose_profile tool exactly once:
- Copy facts exactly as written. Amounts in US dollars as plain numbers ("500k" -> 500000, \
"1.5 million" -> 1500000). Percentages as numbers of percent ("20%" -> 20).
- Use null (or an empty list) for anything the notes don't state. Don't infer residency from \
a city unless the country is obvious from it (Dubai -> UAE, Riyadh -> Saudi Arabia, Doha -> \
Qatar).
- name: the client's name if written. age: if written.
- jurisdiction: where the client is booked, from the notes: Abu Dhabi or ADGM -> "ADGM (Abu \
Dhabi)"; DIFC -> "DIFC (Dubai)"; elsewhere in the UAE without DIFC or ADGM (e.g. Dubai, \
Sharjah) -> "UAE onshore"; Saudi Arabia -> "Saudi Arabia"; Qatar or QFC -> "QFC (Qatar)"; \
Bahrain -> "Bahrain"; Kuwait -> "Kuwait"; Oman -> "Oman"; null if the notes don't say.
- annual_income_need_usd: yearly spending the client draws from the portfolio, if stated \
("needs 8k a month" -> leave null and list it in unmapped_requests; never multiply).
- tax_notes: anything about taxes, tax residence or citizenship, in the notes' words.
- name, tax_notes, residency: "" when the notes don't say. residency is the country only \
("UAE"); anything more about taxes goes in tax_notes.
- liquidity_needs: each cash withdrawal the client needs. Copy when, never convert it: a \
month or date as YYYY-MM ("Oct 2028" -> "2028-10"; a year alone -> that year's January, \
"2028" -> "2028-01"), or a relative time as written ("in 18 months", "in 2 years"); "" if \
the notes give no time. Never turn a date into months or years yourself.
- region_minimums: minimum allocations to a country or region ("at least 10% in Saudi" -> \
Saudi Arabia, 10). A minimum for the Gulf or GCC as a whole goes in home_bias_gcc_pct.
- goals: each investment goal the client states (cash withdrawals go only in \
liquidity_needs, not here), with an amount, a multiple of today's value, or years, \
as written.
- max_volatility_pct / max_drawdown_pct: only if the notes state a limit ("can't lose more \
than 20%" is a drawdown limit of 20).
- Household, wealth and experience facts, only if written: nationality, tax_residencies \
(countries), marital_status, dependants (number) and dependant_ages, occupation, employer, \
employment_type, retirement_age (planned), reporting_currency, source_of_wealth and \
source_of_funds (in the notes' words), primary_objective (growth, income or capital \
preservation, only if stated), annual_contribution_usd (yearly additions to the portfolio), \
years_investing, products_used. When not stated: "" for text and choices, 0 for \
retirement_age and annual_contribution_usd, -1 for years_investing. Never record whether \
the client is a politically exposed person: the adviser checks that.
- unmapped_requests: anything that doesn't fit any field, in the notes' words. Don't list \
anything you recorded in a field.
- summary: one or two sentences restating what you recorded.
"""


def _num(nullable: bool = True) -> dict:
    return {"type": ["number", "null"]} if nullable else {"type": "number"}


def _choice(options) -> dict:
    """A string enum where "" means not stated (no null: strict tools allow few unions)."""
    return {"type": "string", "enum": ["", *options]}


def profile_tool() -> dict:
    """Tool whose input is the factual part of the client form (no scores, no weights).

    Not strict: with 40 fields the compiled grammar exceeded the API's limit (400 "The compiled
    grammar is too large"). The schema still guides the model; ``clean_draft`` enforces it.
    """
    props = {
        "horizon_years": _num(),
        "return_target_pct": _num(),
        "portfolio_value_usd": _num(),
        "annual_income_need_usd": _num(),
        "name": {"type": "string"},
        "age": _num(),
        "jurisdiction": {
            "anyOf": [{"type": "string", "enum": list(JURISDICTIONS)}, {"type": "null"}]
        },
        "tax_notes": {"type": "string"},
        "liquidity_needs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "amount_usd": {"type": "number"},
                    "when": {"type": "string"},
                },
                "required": ["description", "amount_usd", "when"],
                "additionalProperties": False,
            },
        },
        "region_minimums": {
            "type": "array",
            "description": "Minimum % in one named country or region (e.g. Saudi Arabia). A "
            "minimum for the Gulf or GCC as a whole goes in home_bias_gcc_pct instead.",
            "items": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "enum": list(REGIONS)},
                    "min_pct": {"type": "number"},
                },
                "required": ["region", "min_pct"],
                "additionalProperties": False,
            },
        },
        "residency": {"type": "string"},
        "us_person": {"type": ["boolean", "null"]},
        # strict mode: an enum can't share a type list with null, so offer either explicitly
        "classification": {
            "anyOf": [{"type": "string", "enum": list(CLASSIFICATIONS)}, {"type": "null"}]
        },
        "sharia_only": {"type": ["boolean", "null"]},
        "excluded_sectors": {"type": "array", "items": {"type": "string"}},
        "home_bias_gcc_pct": {
            **_num(),
            "description": "Minimum % in the Gulf / GCC as a whole, e.g. 'at least 15% in the "
            "Gulf' or 'keep 20% in GCC markets' -> 15 or 20. Not a region_minimums entry and "
            "not an unmapped request.",
        },
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
        # fact-find facts (never the questionnaire, PEP status, classification reason or scores)
        "nationality": {"type": "string"},
        "tax_residencies": {"type": "array", "items": {"type": "string"}},
        "marital_status": _choice(MARITAL),
        "dependants": _num(),
        "dependant_ages": {"type": "array", "items": {"type": "number"}},
        "occupation": {"type": "string"},
        "employer": {"type": "string"},
        "employment_type": _choice(EMPLOYMENT),
        "retirement_age": _num(False),  # 0: not stated (strict tools allow few nullable fields)
        "reporting_currency": _choice(CURRENCIES),
        "source_of_wealth": {"type": "string"},
        "source_of_funds": {"type": "string"},
        "primary_objective": _choice(OBJECTIVES),
        "annual_contribution_usd": _num(False),  # 0: none stated
        "years_investing": _num(False),  # -1: not stated
        "products_used": {"type": "array", "items": {"type": "string", "enum": PRODUCTS}},
        "unmapped_requests": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Only requests that fit no other field (check home_bias_gcc_pct, "
            "region_minimums, excluded_sectors and goals first).",
        },
        "summary": {"type": "string"},
    }
    return {
        "name": "propose_profile",
        "description": "Record the client facts stated in the adviser's notes.",
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


SENTINEL = {"retirement_age": 0, "annual_contribution_usd": 0, "years_investing": -1}


def _draft_numbers(draft: dict) -> list[float]:
    keys = [
        "horizon_years",
        "return_target_pct",
        "portfolio_value_usd",
        "annual_income_need_usd",
        "age",
        "home_bias_gcc_pct",
        "max_volatility_pct",
        "max_drawdown_pct",
        "dependants",
        "retirement_age",
        "annual_contribution_usd",
        "years_investing",
    ]
    # sentinels for "not stated" (0 or -1) are not numbers from the notes
    values = [
        None if k in SENTINEL and draft.get(k) == SENTINEL[k] else draft.get(k) for k in keys
    ] + list(draft.get("dependant_ages") or [])
    for g in draft.get("goals") or []:
        values += [g.get("amount_usd"), g.get("multiple"), g.get("years")]
    for need in draft.get("liquidity_needs") or []:
        values.append(need.get("amount_usd"))
        when = str(need.get("when") or "").strip()
        if re.match(r"^\d{4}-\d{2}", when):  # the year must be in the notes (month may be a name)
            values.append(when[:4])
        else:
            values += re.findall(r"\d+(?:\.\d+)?", when)
    for item in draft.get("region_minimums") or []:
        values.append(item.get("min_pct"))
    return [float(v) for v in values if v is not None]


def numbers_not_in_notes(draft: dict, notes: str) -> list[str]:
    """Numbers the model wrote that don't appear in the adviser's notes."""
    written = numbers_in_text(notes)
    return [f"{v:g}" for v in _draft_numbers(draft) if round(v, 6) not in written]


# ------------------------------------------------------------------ checking the model's output

# field -> (lowest, highest, whether 0 is a real value)
NUMBERS = {
    "horizon_years": (0, 100, False),
    "return_target_pct": (-50, 100, True),
    "portfolio_value_usd": (0, 1e12, False),
    "annual_income_need_usd": (0, 1e10, True),
    "age": (16, 120, False),
    "home_bias_gcc_pct": (0, 100, True),
    "max_volatility_pct": (0, 100, False),
    "max_drawdown_pct": (0, 100, False),
    "dependants": (0, 30, True),
    "retirement_age": (30, 100, False),
    "annual_contribution_usd": (0, 1e10, False),
    "years_investing": (0, 90, True),
}
INTEGERS = {"age", "dependants", "retirement_age"}
TEXT = ["name", "tax_notes", "residency", "nationality", "occupation", "employer",
        "source_of_wealth", "source_of_funds", "summary"]  # fmt: skip
CHOICES = {
    "jurisdiction": tuple(JURISDICTIONS),
    "classification": tuple(CLASSIFICATIONS),
    "marital_status": tuple(MARITAL),
    "employment_type": tuple(EMPLOYMENT),
    "reporting_currency": tuple(CURRENCIES),
    "primary_objective": tuple(OBJECTIVES),
}
NULL_CHOICES = {"jurisdiction", "classification"}  # null when not stated (others: "")
BOOLS = ["us_person", "sharia_only"]
TEXT_LISTS = ["excluded_sectors", "tax_residencies", "unmapped_requests"]
MAX_TEXT, MAX_ITEMS = 600, 30


def _number(value: Any) -> float | None:
    """A number as the model wrote it: int, float or a plain numeric string ("500,000")."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return None
    return None


def clean_draft(raw: Any) -> tuple[dict[str, Any], list[str]]:
    """The model's tool input made safe for the form, and what was dropped and why.

    Every field the form has is returned (None, "" or [] when not stated); numbers outside a
    sensible range, unknown choices and fields the form doesn't have (a risk score, say) are
    dropped and listed. The "not stated" sentinels (0 or -1) become None.
    """
    raw = raw if isinstance(raw, dict) else {}
    problems: list[str] = []
    out: dict[str, Any] = {}
    known = set(profile_tool()["input_schema"]["properties"])
    extra = sorted(set(raw) - known)
    if extra:
        risky = [k for k in extra if re.search(r"answer|score|risk|capacity|willing|toler", k)]
        problems.append(
            "Ignored fields the form doesn't have: "
            + ", ".join(extra)
            + (" (the AI never sets risk answers or scores)" if risky else "")
            + "."
        )

    def drop(key: str, value: Any, why: str) -> None:
        problems.append(f"Ignored {key.replace('_', ' ')} = {value!r}: {why}.")

    for key, (lo, hi, zero_ok) in NUMBERS.items():
        value = raw.get(key)
        out[key] = None
        if value is None or value == "" or (key in SENTINEL and value == SENTINEL[key]):
            continue
        x = _number(value)
        if x is None:
            drop(key, value, "not a number")
        elif not (lo <= x <= hi) or (x == 0 and not zero_ok):
            drop(key, value, f"outside {lo:g}-{hi:g}")
        else:
            out[key] = int(round(x)) if key in INTEGERS else x
    for key in TEXT:
        value = raw.get(key)
        out[key] = str(value).strip()[:MAX_TEXT] if isinstance(value, str | int | float) else ""
    for key, options in CHOICES.items():
        value = raw.get(key)
        empty = None if key in NULL_CHOICES else ""
        if value in (None, ""):
            out[key] = empty
        elif value in options:
            out[key] = value
        else:
            out[key] = empty
            drop(key, value, "not one of the form's options")
    for key in BOOLS:
        value = raw.get(key)
        if isinstance(value, str) and value.lower() in ("true", "false"):
            value = value.lower() == "true"
        out[key] = value if isinstance(value, bool) else None
        if value is not None and not isinstance(value, bool):
            drop(key, value, "not yes or no")
    for key in TEXT_LISTS:
        items = raw.get(key) if isinstance(raw.get(key), list) else []
        out[key] = [str(i).strip()[:MAX_TEXT] for i in items if str(i).strip()][:MAX_ITEMS]
    products = raw.get("products_used") if isinstance(raw.get("products_used"), list) else []
    out["products_used"] = [p for p in products if p in PRODUCTS]
    for p in products:
        if p not in PRODUCTS:
            drop("products_used", p, "not one of the form's options")
    ages = raw.get("dependant_ages") if isinstance(raw.get("dependant_ages"), list) else []
    out["dependant_ages"] = [
        int(round(a)) for a in (_number(x) for x in ages) if a is not None and 0 <= a <= 120
    ]
    needs = []
    for n in raw.get("liquidity_needs") or []:
        if not isinstance(n, dict):
            continue
        amount = _number(n.get("amount_usd"))
        if amount is None or not 0 < amount <= 1e11:
            drop("cash need", n.get("amount_usd"), "no usable amount")
            continue
        needs.append(
            {
                "description": str(n.get("description") or "Cash need").strip()[:MAX_TEXT],
                "amount_usd": amount,
                "when": str(n.get("when") or "").strip()[:60],
            }
        )
    out["liquidity_needs"] = needs[:MAX_ITEMS]
    minimums = []
    for r in raw.get("region_minimums") or []:
        if not isinstance(r, dict):
            continue
        pct = _number(r.get("min_pct"))
        if r.get("region") not in REGIONS:
            drop("region minimum", r.get("region"), "not a region the form knows")
        elif pct is None or not 0 < pct <= 100:
            drop("region minimum", r.get("min_pct"), "not a percentage")
        else:
            minimums.append({"region": r["region"], "min_pct": pct})
    out["region_minimums"] = minimums
    goals = []
    for g in raw.get("goals") or []:
        if not isinstance(g, dict):
            continue
        goal = {"description": str(g.get("description") or "").strip()[:MAX_TEXT]}
        for key, (lo, hi) in {"amount_usd": (0, 1e11), "multiple": (0, 100),
                              "years": (0, 100)}.items():  # fmt: skip
            x = _number(g.get(key))
            goal[key] = x if x is not None and lo < x <= hi else None
        if goal["description"] or any(goal[k] is not None for k in ("amount_usd", "multiple")):
            goals.append(goal)
    out["goals"] = goals[:MAX_ITEMS]
    return out, problems


@dataclass
class ProfileProposal:
    draft: dict[str, Any]
    unmapped: list[str]
    summary: str
    numbers_not_in_notes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)  # e.g. a cash need with no date


def propose_profile(client: Any, model: str, notes: str, effort: str = "low") -> ProfileProposal:
    """Ask the model to transcribe ``notes`` into the profile tool. Raises ValueError if not."""
    response, _ = create_message(
        client,
        model=model,
        max_tokens=8000,
        system=PROFILE_SYSTEM_PROMPT,
        tools=[profile_tool()],
        output_config={"effort": effort},
        messages=[{"role": "user", "content": f"Adviser's notes:\n{notes}"}],
    )
    calls = [b for b in response.content if b.type == "tool_use" and b.name == "propose_profile"]
    if response.stop_reason == "max_tokens":
        raise ValueError(
            "The notes were too long to fill the form in one go. Try shorter notes, or split "
            "them and fill the form in two steps."
        )
    if response.stop_reason == "refusal" or not calls:
        raise ValueError("The model did not fill the form. Try rephrasing the notes.")
    draft, cleaned = clean_draft(calls[0].input)
    problems = cleaned + [
        f"Cash need '{n.get('description')}' (USD {n.get('amount_usd') or 0:,.0f}) has no "
        "usable date in the notes: add it on the form."
        for n in draft.get("liquidity_needs") or []
        if need_date(n, date.today()) is None
    ]
    return ProfileProposal(
        draft,
        list(draft.get("unmapped_requests") or []),
        str(draft.get("summary") or ""),
        numbers_not_in_notes(draft, notes),
        problems,
    )


def need_date(need: dict, today: date) -> str | None:
    """The month a need is due, 'YYYY-MM', from the model's copy of the notes' ``when``:
    a month as written ('2028-10'), or a relative time ('in 18 months', 'in 2 years') added
    to ``today`` here, in code, never by the model. None if neither can be read."""
    when = str(need.get("when") or "").strip().lower()
    if re.match(r"^\d{4}-\d{2}", when):
        try:
            parse_month(when)
            return when[:7]
        except ValueError:
            return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(year|yr|month|mo)", when)
    if not match:
        return None
    n = float(match.group(1))
    return month_after(today, round(n * 12 if match.group(2).startswith("y") else n))


def apply_proposal(profile: ClientProfile, draft: dict, today: date | None = None) -> ClientProfile:
    """Copy the stated facts onto the profile; questionnaire answers are never touched.

    The regulator comes from the jurisdiction by a fixed rule (``regulator_for``); dated needs
    get their month from code (``need_date``).
    """
    today = today or date.today()
    simple = [
        "nationality",
        "marital_status",
        "occupation",
        "employer",
        "employment_type",
        "reporting_currency",
        "source_of_wealth",
        "source_of_funds",
        "primary_objective",
        "horizon_years",
        "return_target_pct",
        "portfolio_value_usd",
        "annual_income_need_usd",
        "name",
        "tax_notes",
        "residency",
        "us_person",
        "classification",
        "sharia_only",
        "home_bias_gcc_pct",
    ]
    # null (or "" for text) means "not in the notes": the form keeps what it had
    updates = {k: draft[k] for k in simple if draft.get(k) not in (None, "")}
    for key in ("age", "dependants", "retirement_age"):
        if draft.get(key) is not None and draft.get(key) != SENTINEL.get(key):
            updates[key] = int(draft[key])
    for key in ("annual_contribution_usd", "years_investing"):
        if draft.get(key) is not None and draft[key] != SENTINEL[key]:
            updates[key] = float(draft[key])
    for key in ("tax_residencies", "products_used"):
        if draft.get(key):
            updates[key] = list(draft[key])
    if draft.get("dependant_ages"):
        updates["dependant_ages"] = [int(a) for a in draft["dependant_ages"]]
    if regulator_for(draft.get("jurisdiction")):
        updates["booking_location"] = draft["jurisdiction"]
        updates["regulator"] = regulator_for(draft["jurisdiction"])
    needs = []
    for need in draft.get("liquidity_needs") or []:
        when = need_date(need, today)
        if when and need.get("amount_usd"):
            needs.append(
                {
                    "description": need.get("description") or "Cash need",
                    "amount_usd": float(need["amount_usd"]),
                    "date": when,
                }
            )
    if needs:
        updates["liquidity_needs"] = needs
    if draft.get("region_minimums"):
        updates["region_minimums"] = {
            r["region"]: float(r["min_pct"]) for r in draft["region_minimums"]
        }
    if draft.get("excluded_sectors"):
        updates["excluded_sectors"] = [s[:1].upper() + s[1:] for s in draft["excluded_sectors"]]
    if draft.get("goals"):
        updates["goals"] = [{k: v for k, v in g.items() if v is not None} for g in draft["goals"]]
    overrides = dict(profile.overrides)
    if draft.get("max_volatility_pct") is not None:
        overrides["max_volatility_pct"] = float(draft["max_volatility_pct"])
    # "can't lose more than 20%" is the client's capacity for loss: it can only tighten the
    # band's drawdown limit (an adviser override could loosen it)
    if draft.get("max_drawdown_pct") is not None:
        updates["max_acceptable_drawdown_pct"] = float(draft["max_drawdown_pct"])
    return replace(profile, **updates, overrides=overrides)
