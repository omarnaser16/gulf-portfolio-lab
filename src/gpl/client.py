"""Client profile, risk questionnaire, Investment Policy Statement (IPS) and IPS -> limits.

Pure functions: no I/O, no printing, no plotting.

The IPS follows the CFA RRTTLLU checklist: Return and Risk (objectives); Time horizon, Taxes,
Liquidity, Legal and regulatory, Unique circumstances (constraints). Risk tolerance is scored
by a fixed, tested rule: capacity (can the client afford losses?) and willingness (are they
comfortable with them?) are each the rounded mean of their answers, and the **lower** of the
two sets the risk band. A gap of two or more is flagged for a conversation with the adviser.
This is decision support for a licensed adviser, not a suitability decision in itself.
"""

from dataclasses import asdict, dataclass, field, fields, replace
from datetime import date
from typing import Any

from gpl.optimise import OptimisationSpec

QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "income_stability",
        "dimension": "capacity",
        "text": "How stable is the client's income?",
        "options": ["Very unstable", "Unstable", "Average", "Stable", "Very stable"],
    },
    {
        "id": "wealth_share",
        "dimension": "capacity",
        "text": "What share of total wealth is this portfolio?",
        "options": ["Over 75%", "50-75%", "25-50%", "10-25%", "Under 10%"],
    },
    {
        "id": "dependence",
        "dimension": "capacity",
        "text": "How much does the client rely on this portfolio for living expenses?",
        "options": ["Entirely", "Mostly", "Partly", "A little", "Not at all"],
    },
    {
        "id": "liquidity_need",
        "dimension": "capacity",
        "text": "How large are cash needs from this portfolio in the next two years?",
        "options": ["Most of it", "A large part", "Some", "A little", "None"],
    },
    {
        "id": "loss_reaction",
        "dimension": "willingness",
        "text": "If the portfolio fell 20% in a few months, the client would...",
        "options": ["Sell everything", "Sell some", "Hold", "Buy a little more", "Buy more"],
    },
    {
        "id": "tradeoff",
        "dimension": "willingness",
        "text": "Which matters more to the client?",
        "options": [
            "Protecting capital",
            "Mostly protecting capital",
            "Balance",
            "Mostly growth",
            "Maximum growth",
        ],
    },
    {
        "id": "experience",
        "dimension": "knowledge",
        "text": "How much investing experience does the client have?",
        "options": ["None", "Little", "Some", "Good", "Extensive"],
    },
]
# only capacity and willingness answers set the risk score; knowledge & experience is recorded
# separately (it bears on which products are appropriate, not on how much risk to take)
SCORED = ("capacity", "willingness")
REGULATORS = [
    "DFSA (DIFC)",
    "FSRA (ADGM)",
    "SCA (UAE)",
    "CMA (Saudi Arabia)",
    "QFCRA (Qatar)",
    "QFMA (Qatar)",
    "CBB (Bahrain)",
    "CMA (Kuwait)",
    "FSA (Oman, formerly CMA)",
    "Other",
]
CLASSIFICATIONS = ["Retail", "Professional"]
# where the client is booked -> who regulates the advice (the adviser confirms)
JURISDICTIONS: dict[str, str] = {
    "ADGM (Abu Dhabi)": "FSRA (ADGM)",
    "DIFC (Dubai)": "DFSA (DIFC)",
    "UAE onshore": "SCA (UAE)",
    "Saudi Arabia": "CMA (Saudi Arabia)",
    "QFC (Qatar)": "QFCRA (Qatar)",
    "Qatar onshore": "QFMA (Qatar)",
    "Bahrain": "CBB (Bahrain)",
    "Kuwait": "CMA (Kuwait)",
    "Oman": "FSA (Oman, formerly CMA)",
    "Other": "Other",
}
# residency (country) -> the usual booking location when none is recorded
RESIDENCY_BOOKING: dict[str, str] = {
    "uae": "UAE onshore",
    "united arab emirates": "UAE onshore",
    "dubai": "UAE onshore",
    "abu dhabi": "UAE onshore",
    "saudi arabia": "Saudi Arabia",
    "ksa": "Saudi Arabia",
    "qatar": "QFC (Qatar)",
    "bahrain": "Bahrain",
    "kuwait": "Kuwait",
    "oman": "Oman",
}
# regions for minimum allocations; the names gpl.scenarios.classify_holdings gives holdings
REGIONS = [
    "Saudi Arabia",
    "UAE",
    "Qatar",
    "Kuwait",
    "Bahrain",
    "Oman",
    "United States",
    "United Kingdom",
    "Developed ex-US",
    "Emerging markets",
    "Global",
]
PEGGED = {"USD", "SAR", "AED", "QAR", "BHD", "OMR"}


def regulator_for(jurisdiction: str | None) -> str | None:
    """The regulator for a booking jurisdiction (``JURISDICTIONS``); None if unknown.

    ADGM (Abu Dhabi Global Market) -> FSRA; DIFC (Dubai) -> DFSA; onshore UAE -> SCA;
    Saudi Arabia -> CMA; QFC (Qatar) -> QFCRA; onshore Qatar -> QFMA; Bahrain -> CBB;
    Kuwait -> CMA Kuwait; Oman -> FSA (the Capital Market Authority until 2024).
    """
    return JURISDICTIONS.get(jurisdiction or "")


def default_booking(residency: str | None) -> str | None:
    """The usual booking location for a residency ('UAE' -> 'UAE onshore'); None if unknown.

    Onshore is the default: a client counts as DIFC, ADGM or QFC only when booked there, except
    Qatar, where advisers for this kind of client are usually QFC firms.
    """
    return RESIDENCY_BOOKING.get((residency or "").strip().lower())


def default_regulator(booking_location: str | None, residency: str | None) -> str:
    """The regulator from the booking location, else from the residency; 'Other' if neither maps."""
    return regulator_for(booking_location) or regulator_for(default_booking(residency)) or "Other"


def effective_regulator(profile: ClientProfile) -> str:
    """The adviser's choice if recorded, otherwise the default from booking location/residency."""
    return profile.regulator or default_regulator(profile.booking_location, profile.residency)


def in_region(holding_region: str, region: str) -> bool:
    """'United States (listing)' counts for 'United States'; otherwise names must match."""
    return holding_region == region or holding_region.startswith(region + " (")


def horizon_score(years: float) -> int:
    """Capacity score from the time horizon: under 3 years 1, 3-5 2, 5-10 3, 10-15 4, 15+ 5."""
    for limit, score in ((3, 1), (5, 2), (10, 3), (15, 4)):
        if years < limit:
            return score
    return 5


def parse_month(text: str) -> date:
    """'2028-10' (or '2028-10-15') -> 1 Oct 2028. Raises ValueError otherwise."""
    parts = str(text).strip().split("-")
    if len(parts) < 2:
        raise ValueError(f"expected a month as YYYY-MM, got {text!r}")
    year, month = int(parts[0]), int(parts[1])
    return date(year, month, 1)


def month_after(today: date, months: int) -> str:
    """The month ``months`` after ``today``, as 'YYYY-MM'."""
    index = today.year * 12 + today.month - 1 + months
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def months_until(when: str, today: date) -> int:
    """Whole months from ``today``'s month to ``when`` ('YYYY-MM'); 0 if it has passed.

    Oct 2028 seen from Oct 2026 is 24 months (about two years), not 12.
    """
    d = parse_month(when)
    return max(0, (d.year - today.year) * 12 + d.month - today.month)


def month_text(when: str) -> str:
    """'2028-10' -> 'Oct 2028'."""
    return f"{parse_month(when):%b %Y}"


def cash_needs(
    profile: ClientProfile, today: date, window_months: int = 36
) -> tuple[list[dict], list[dict]]:
    """Split the dated cash needs into those due within ``window_months`` and later ones.

    Needs due within the window must already be held in cash-like holdings (a hard limit for
    the optimiser and the trade list); later ones are recorded and start to bind when they come
    within the window. Each need gets ``months`` (months away). A date that has passed counts
    as due now, so the adviser sees it and updates the profile.
    """
    soon, later = [], []
    for need in profile.liquidity_needs:
        item = {**need, "months": months_until(need["date"], today)}
        (soon if item["months"] <= window_months else later).append(item)
    return soon, later


def cash_like_holdings(
    tickers: list[str],
    groups: dict[str, str],
    listed: list[str] | tuple[str, ...] = (),
    annual_vols: dict[str, float] | None = None,
    max_volatility: float = 0.03,
) -> list[str]:
    """Holdings that can meet a near-term cash need: cash, short-term bonds or sukuk.

    A holding counts if the adviser put it in the "Cash & short-term" group, it is on the
    configured list (T-bill, short Treasury and sukuk ETFs), or its annual volatility is at
    most ``max_volatility`` (3%: short-dated bonds move about 0.2-2% a year; a broad bond fund
    like AGG, around 5-6%, does not count).
    """
    vols = annual_vols or {}
    return [
        t
        for t in tickers
        if groups.get(t) == CASH_GROUP or t in listed or (t in vols and vols[t] <= max_volatility)
    ]


CASH_GROUP = "Cash & short-term"


def _round_half_up(x: float) -> int:
    return int(x + 0.5)


@dataclass
class ClientProfile:
    """Everything the adviser records about the client (percentages as written, e.g. 20)."""

    reference: str = ""
    name: str = ""
    age: int | None = None
    classification: str = "Retail"
    regulator: str = ""  # "": from the booking location / residency (``effective_regulator``)
    residency: str = "UAE"
    booking_location: str = ""  # a key of JURISDICTIONS
    us_person: bool = False
    horizon_years: float = 10
    goals: list[dict] = field(default_factory=list)  # {description, amount_usd|multiple, years}
    return_target_pct: float | None = None
    portfolio_value_usd: float | None = None
    annual_income_need_usd: float = 0.0  # spending drawn from the portfolio each year
    inflation_pct: float | None = None  # None: the config's assumption
    # dated cash needs: {description, amount_usd, date "YYYY-MM"}
    liquidity_needs: list[dict] = field(default_factory=list)
    reserve_pct: float = 0.0
    tax_notes: str = ""
    legal_notes: str = ""
    sharia_only: bool = False
    excluded_sectors: list[str] = field(default_factory=list)
    home_bias_gcc_pct: float | None = None
    region_minimums: dict[str, float] = field(default_factory=dict)  # region -> minimum %
    max_non_pegged_currency_pct: float = 20.0
    unique_notes: str = ""
    answers: dict[str, int] = field(default_factory=dict)
    overrides: dict[str, float] = field(default_factory=dict)  # max_volatility_pct, ...
    adviser_notes: str = ""
    version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict, today: date | None = None) -> ClientProfile:
        """From saved JSON; unknown keys are ignored and missing ones take their defaults.

        Profiles saved before cash needs were dated ("USD X within N months") are converted to
        one dated need, N months after ``today`` (default: the day it is loaded).
        """
        known = {f.name: f for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in known}
        if "liquidity_needs" not in data and float(data.get("liquidity_need_usd") or 0) > 0:
            months = int(data.get("liquidity_months") or 12)
            values["liquidity_needs"] = [
                {
                    "description": "Cash need",
                    "amount_usd": float(data["liquidity_need_usd"]),
                    "date": month_after(today or date.today(), months),
                }
            ]
        out = cls(**values)
        for name in (
            "horizon_years",
            "reserve_pct",
            "max_non_pegged_currency_pct",
            "annual_income_need_usd",
        ):
            try:
                setattr(out, name, float(getattr(out, name)))
            except (TypeError, ValueError) as e:
                raise ValueError(f"{name} must be a number, got {getattr(out, name)!r}") from e
        try:
            out.region_minimums = {str(r): float(v) for r, v in out.region_minimums.items()}
            out.age = None if out.age in (None, "") else int(out.age)
        except (TypeError, ValueError, AttributeError) as e:
            raise ValueError(f"bad age or region minimums: {e}") from e
        needs = []
        for need in out.liquidity_needs:
            try:
                amount = float(need["amount_usd"])
                parse_month(need["date"])
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(
                    f"each cash need needs an amount and a date (YYYY-MM), got {need!r}"
                ) from e
            needs.append({**need, "amount_usd": amount})
        out.liquidity_needs = needs
        return out


LIQUIDITY_SCORE_BANDS = ((0.05, 4), (0.10, 3), (0.25, 2))  # share due < limit -> score


def liquidity_need_score(
    profile: ClientProfile, today: date, months: int = 24
) -> tuple[int | None, float | None]:
    """Answer "cash needs in the next two years" from the dated Cash needs table.

    The share of the portfolio due within ``months`` (24) sets the score: none 5, under 5% 4,
    under 10% 3, under 25% 2, 25% or more 1. Returns (score, share); (None, None) when the
    portfolio value is unknown and needs are recorded (the adviser answers by hand then).
    E.g. USD 50,000 due in 11 months from USD 500,000: 10.00% -> 2.
    """
    due = sum(
        n["amount_usd"] for n in profile.liquidity_needs if months_until(n["date"], today) <= months
    )
    if due <= 0:
        return 5, 0.0
    value = profile.portfolio_value_usd
    if not value:
        return None, None
    share = due / value
    for limit, score in LIQUIDITY_SCORE_BANDS:
        if share < limit:
            return score, share
    return 1, share


def with_derived_answers(profile: ClientProfile, today: date) -> ClientProfile:
    """The profile with answers set by rule from recorded facts (cash needs), where possible."""
    score, _ = liquidity_need_score(profile, today)
    if score is None:
        return profile
    return replace(profile, answers={**profile.answers, "liquidity_need": score})


IPS_REQUIRED = {
    "reference": "client reference",
    "age": "age",
    "residency": "residency",
    "portfolio_value_usd": "portfolio value",
}


def ips_missing(profile: ClientProfile, today: date | None = None) -> list[str]:
    """What must be filled in before the final IPS can be issued (empty list: ready).

    Required facts (``IPS_REQUIRED``), a known regulator, and every capacity and willingness
    answer. Until then only a DRAFT IPS is produced.
    """
    p = with_derived_answers(profile, today or date.today())
    missing = [label for name, label in IPS_REQUIRED.items() if getattr(p, name) in (None, "", 0)]
    if effective_regulator(p) == "Other":
        missing.append("regulator (booking location)")
    missing += [
        f"questionnaire: {q['text']}"
        for q in QUESTIONS
        if q["dimension"] in SCORED and q["id"] not in p.answers
    ]
    return missing


@dataclass
class RiskProfile:
    capacity: int | None
    willingness: int | None
    score: int | None
    mismatch: bool
    missing: list[str]
    explanation: str


def risk_profile(profile: ClientProfile) -> RiskProfile:
    """Score capacity and willingness (rounded means, half up); the lower sets the band."""
    answers = dict(profile.answers)
    missing = [q["id"] for q in QUESTIONS if q["dimension"] in SCORED and q["id"] not in answers]
    if missing:
        return RiskProfile(
            None,
            None,
            None,
            False,
            missing,
            "The questionnaire is incomplete: " + ", ".join(missing) + ".",
        )
    cap_items = [horizon_score(profile.horizon_years)] + [
        int(answers[q["id"]]) for q in QUESTIONS if q["dimension"] == "capacity"
    ]
    will_items = [int(answers[q["id"]]) for q in QUESTIONS if q["dimension"] == "willingness"]
    capacity = _round_half_up(sum(cap_items) / len(cap_items))
    willingness = _round_half_up(sum(will_items) / len(will_items))
    score = min(capacity, willingness)
    mismatch = abs(capacity - willingness) >= 2
    explanation = (
        f"Risk score {score} of 5: the lower of capacity ({capacity}) and willingness "
        f"({willingness}). Capacity averages the horizon ({profile.horizon_years:g} years scores "
        f"{cap_items[0]}) and {len(cap_items) - 1} answers; willingness averages "
        f"{len(will_items)} answers."
    )
    if mismatch:
        explanation += (
            " Capacity and willingness differ by two or more: discuss with the client before "
            "relying on this score."
        )
    return RiskProfile(capacity, willingness, score, mismatch, [], explanation)


@dataclass
class IPS:
    """Investment Policy Statement: RRTTLLU sections in words, and machine-readable limits."""

    client: ClientProfile
    risk: RiskProfile
    sections: dict[str, str]
    limits: dict[str, Any]
    created: str


def _pct(x: float | None) -> str:
    return "not set" if x is None else f"{x:.2%}"


@dataclass
class RequiredReturn:
    income_usd: float
    investable_usd: float
    real: float
    inflation: float
    nominal: float

    def text(self) -> str:
        return (
            f"Required return {self.nominal:.2%} a year: USD {self.income_usd:,.0f} income ÷ "
            f"USD {self.investable_usd:,.0f} investable (after cash set aside) = {self.real:.2%} "
            f"real, with {self.inflation:.2%} inflation: (1 + {self.real:.2%}) × "
            f"(1 + {self.inflation:.2%}) - 1."
        )


def required_return(
    income_usd: float, value_usd: float, reserved_usd: float, inflation: float
) -> RequiredReturn:
    """The return the portfolio must earn to pay the income and keep pace with inflation.

    Investable assets are the value minus the cash set aside for near-term needs (that cash is
    not expected to earn the return). Real required return = income ÷ investable; nominal =
    (1 + real) × (1 + inflation) - 1. E.g. USD 96,000 a year from USD 3m with USD 600,000
    reserved and 2.5% inflation: 4.00% real, 6.60% nominal.
    """
    investable = value_usd - reserved_usd
    if investable <= 0:
        raise ValueError(
            f"the cash set aside (USD {reserved_usd:,.0f}) is at least the portfolio value "
            f"(USD {value_usd:,.0f}): nothing is left to invest"
        )
    real = income_usd / investable
    return RequiredReturn(income_usd, investable, real, inflation, (1 + real) * (1 + inflation) - 1)


def _goal_text(goal: dict) -> str:
    """'University (USD 200,000 in 8 years)' from a goal entry."""
    details = []
    if goal.get("amount_usd"):
        details.append(f"USD {goal['amount_usd']:,.0f}")
    if goal.get("multiple"):
        details.append(f"{goal['multiple']:g}x the starting value")
    if goal.get("years"):
        details.append(f"in {goal['years']:g} year{'' if goal['years'] == 1 else 's'}")
    text = goal.get("description") or "Goal"
    return f"{text} ({' '.join(details)})" if details else text


def build_ips(profile: ClientProfile, cfg: dict, today: date | None = None) -> IPS:
    """Build the IPS. Risk limits come from the risk band unless the adviser overrides them.

    The "cash needs in the next two years" answer is set from the dated cash needs whenever the
    portfolio value is known (``liquidity_need_score``).
    """
    today = today or date.today()
    profile = with_derived_answers(profile, today)
    risk = risk_profile(profile)
    band = cfg["risk_bands"].get(risk.score) if risk.score else None
    limits: dict[str, Any] = {"source": {}}
    for key in ("max_volatility", "max_drawdown", "max_weight"):
        override = profile.overrides.get(f"{key}_pct")
        if override is not None:
            limits[key], limits["source"][key] = float(override) / 100, "adviser override"
        elif band is not None:
            limits[key], limits["source"][key] = band[key], f"band {risk.score}"
        else:
            limits[key], limits["source"][key] = None, "not set"
    limits["home_bias_gcc"] = (
        None if profile.home_bias_gcc_pct is None else profile.home_bias_gcc_pct / 100
    )
    limits["max_non_pegged_currency"] = profile.max_non_pegged_currency_pct / 100
    value = profile.portfolio_value_usd
    window = int(cfg.get("cash_need_window_months", 36))
    soon, later = cash_needs(profile, today, window)
    need_usd = sum(n["amount_usd"] for n in soon)
    reserve_usd = need_usd + (value or 0) * profile.reserve_pct / 100
    limits["liquidity_reserve_usd"] = reserve_usd
    limits["liquidity_reserve"] = reserve_usd / value if value else None
    limits["cash_needs_due"] = soon
    limits["cash_needs_later"] = later
    limits["cash_need_window_months"] = window
    limits["region_minimums"] = {r: v / 100 for r, v in profile.region_minimums.items() if v}
    limits["sharia_only"] = profile.sharia_only
    limits["excluded_sectors"] = list(profile.excluded_sectors)

    goals = "; ".join(_goal_text(g) for g in profile.goals) or "No specific goals recorded."
    target = (
        f" Return target: {profile.return_target_pct:.2f}% a year (historical estimates only)."
        if profile.return_target_pct is not None
        else ""
    )
    limits["required_return"] = None
    if profile.annual_income_need_usd > 0:
        inflation = (
            profile.inflation_pct / 100
            if profile.inflation_pct is not None
            else float(cfg.get("montecarlo", {}).get("inflation", 0.025))
        )
        if not value:
            target += " Required return: enter the portfolio value to calculate it."
        else:
            try:
                req = required_return(profile.annual_income_need_usd, value, reserve_usd, inflation)
            except ValueError as e:
                target += f" Required return can't be calculated: {e}."
            else:
                limits["required_return"] = req.nominal
                target += " " + req.text()
                if profile.return_target_pct is None:
                    target += " No return target is set: discuss with the client."
                elif req.nominal > profile.return_target_pct / 100 + 1e-12:
                    target += (
                        f" This is above the {profile.return_target_pct:.2f}% target: discuss "
                        "the income, the target or the risk the client accepts."
                    )
    risk_text = (
        risk.explanation
        + (
            f" Limits: volatility at most {_pct(limits['max_volatility'])} a year, drawdown at "
            f"most {_pct(limits['max_drawdown'])}, at most {_pct(limits['max_weight'])} in any "
            "one holding."
            if risk.score
            else ""
        )
        if risk.score
        else risk.explanation + " Risk limits can't be set until it is complete."
    )
    sections = {
        "Return": f"Goals: {goals}{target}",
        "Risk": risk_text,
        "Time horizon": f"{profile.horizon_years:g} years."
        + (f" Client age: {profile.age}." if profile.age else ""),
        "Taxes": (
            f"Residency: {profile.residency}. US person: {'yes' if profile.us_person else 'no'}."
            + (f" {profile.tax_notes}" if profile.tax_notes else "")
        ),
        "Liquidity": _liquidity_text(profile, soon, later, window, reserve_usd, limits, value),
        "Legal and regulatory": (
            f"Client classification: {profile.classification}. "
            f"Regulator: {effective_regulator(profile)}."
            + (f" {profile.legal_notes}" if profile.legal_notes else "")
        ),
        "Unique circumstances": " ".join(
            sentence
            for sentence in [
                "Sharia-compliant holdings only." if profile.sharia_only else "",
                f"Excluded sectors: {', '.join(profile.excluded_sectors)}."
                if profile.excluded_sectors
                else "",
                f"At least {profile.home_bias_gcc_pct:g}% in GCC holdings (home-market bias)."
                if profile.home_bias_gcc_pct
                else "",
                " ".join(
                    f"At least {v:g}% in {r}." for r, v in profile.region_minimums.items() if v
                ),
                f"At most {profile.max_non_pegged_currency_pct:g}% in currencies not pegged to "
                "the US dollar.",
                profile.unique_notes,
            ]
            if sentence
        ),
    }
    created = today.isoformat()
    return IPS(profile, risk, sections, limits, created)


def _need_text(n: dict) -> str:
    label = n.get("description") or "Cash need"
    return f"{label}: USD {n['amount_usd']:,.0f} by {month_text(n['date'])} ({n['months']} months)"


def _liquidity_text(profile, soon, later, window, reserve_usd, limits, value) -> str:
    if not (soon or later or profile.reserve_pct):
        return "No cash needs recorded."
    parts = []
    if soon:
        parts.append(
            f"Due within {window} months, held in cash, short-term bonds or sukuk from now: "
            + "; ".join(_need_text(n) for n in soon)
            + "."
        )
    if profile.reserve_pct:
        parts.append(f"Plus a {profile.reserve_pct:g}% emergency reserve.")
    if soon or profile.reserve_pct:
        parts.append(
            f"Total to keep in cash-like holdings: USD {reserve_usd:,.0f}"
            + (f" ({_pct(limits['liquidity_reserve'])} of the portfolio)." if value else ".")
        )
    if later:
        parts.append(
            "Later needs (recorded; they become binding when due within "
            f"{window} months): " + "; ".join(_need_text(n) for n in later) + "."
        )
    return " ".join(parts)


def excluded_holdings(
    ips: IPS, holdings: list[str], attributes: dict[str, dict]
) -> tuple[dict[str, str], dict[str, str]]:
    """Holdings the IPS excludes (ticker -> reason) and those needing review (unknown status)."""
    excluded, review = {}, {}
    for t in holdings:
        attrs = attributes.get(t, {})
        sector = attrs.get("sector")
        if sector and sector.lower() in {s.lower() for s in ips.limits["excluded_sectors"]}:
            excluded[t] = f"sector {sector} is excluded"
        elif ips.limits["sharia_only"]:
            status = attrs.get("sharia", "unknown")
            if status == "not_compliant":
                excluded[t] = "not Sharia-compliant"
            elif status != "compliant":
                review[t] = "Sharia status unknown: check with the client's Sharia screen"
    return excluded, review


def ips_constraints(
    ips: IPS,
    holdings: list[str],
    groups: dict[str, str],
    attributes: dict[str, dict],
    cash_like: list[str] | None = None,
    regions: dict[str, str] | None = None,
) -> tuple[OptimisationSpec, list[str]]:
    """IPS limits as optimiser limits, plus notes for the adviser.

    Excluded holdings get a 0% maximum. Holdings with unknown Sharia status are not excluded
    automatically but noted for review. The drawdown limit can't be an optimiser constraint
    (it isn't convex); the suitability check enforces it. Cash needs due soon (plus any
    reserve) become a hard minimum on the ``cash_like`` holdings (see ``cash_like_holdings``);
    with none of them held, the limit can't be met and the note says what to add. Region
    minimums (e.g. Saudi Arabia >= 10%) become minimums on the holdings in that region
    (``regions``: ticker -> region, from ``gpl.scenarios.classify_holdings``).
    """
    excluded, review = excluded_holdings(ips, holdings, attributes)
    lim = ips.limits
    sets = {}
    notes = []
    need_usd, need = lim["liquidity_reserve_usd"], lim["liquidity_reserve"]
    if need_usd > 0 and need is None:
        notes.append(f"Cash need of USD {need_usd:,.0f}: enter the portfolio value to enforce it.")
    elif need_usd > 0:
        cash = [t for t in (cash_like or []) if t in holdings and t not in excluded]
        sets["Cash need"] = (frozenset(cash), need, None)
        if cash:
            notes.append(
                f"Cash need: at least {need:.2%} (USD {need_usd:,.0f}) in cash-like holdings "
                f"({', '.join(cash)})."
            )
        else:
            notes.append(
                f"Cash need of USD {need_usd:,.0f} ({need:.2%}) can't be met: no holding is cash, "
                "short-term bonds or sukuk. Add one (e.g. BIL, T-bills) or put a holding in the "
                f"'{CASH_GROUP}' group."
            )
    for region, minimum in lim["region_minimums"].items():
        members = [
            t
            for t in holdings
            if t not in excluded and in_region((regions or {}).get(t, ""), region)
        ]
        sets[f"Region {region}"] = (frozenset(members), minimum, None)
        notes.append(
            f"At least {minimum:.2%} in {region} ({', '.join(members)})."
            if members
            else f"At least {minimum:.2%} in {region} can't be met: no holding is in {region}. "
            "Add one (e.g. a country ETF or local stock)."
        )
    spec = OptimisationSpec(
        objective="max_sharpe",
        max_weight=lim["max_weight"] or 1.0,
        asset_bounds={t: (None, 0.0) for t in excluded},
        groups=dict(groups),
        group_bounds={"GCC": (lim["home_bias_gcc"], None)} if lim["home_bias_gcc"] else {},
        max_volatility=lim["max_volatility"],
        sets=sets,
    )
    notes = [f"{t} excluded: {why}." for t, why in excluded.items()] + notes
    notes += [
        f"{t}: {why} (Sharia status unknown, not excluded automatically)."
        for t, why in review.items()
    ]
    if lim["max_drawdown"] is not None:
        notes.append(
            f"The {lim['max_drawdown']:.2%} drawdown limit is checked by the suitability check, "
            "not imposed by the optimiser."
        )
    return spec, notes


def _stricter(lo_a, hi_a, lo_b, hi_b) -> tuple:
    lo = max((v for v in (lo_a, lo_b) if v is not None), default=None)
    hi = min((v for v in (hi_a, hi_b) if v is not None), default=None)
    return lo, hi


def tighter(a: OptimisationSpec, b: OptimisationSpec) -> OptimisationSpec:
    """Combine two sets of limits, keeping the stricter one wherever both set a limit."""
    vols = [v for v in (a.max_volatility, b.max_volatility) if v is not None]
    rets = [v for v in (a.min_return, b.min_return) if v is not None]
    groups = {}
    for g in {**a.group_bounds, **b.group_bounds}:
        groups[g] = _stricter(
            *a.group_bounds.get(g, (None, None)), *b.group_bounds.get(g, (None, None))
        )
    assets = {}
    for t in {**a.asset_bounds, **b.asset_bounds}:
        assets[t] = _stricter(
            *a.asset_bounds.get(t, (None, None)), *b.asset_bounds.get(t, (None, None))
        )
    sets = {}
    for name in {**a.sets, **b.sets}:
        if name in a.sets and name in b.sets:
            lo, hi = _stricter(*a.sets[name][1:], *b.sets[name][1:])
            sets[name] = (a.sets[name][0] & b.sets[name][0], lo, hi)
        else:
            sets[name] = a.sets.get(name) or b.sets[name]
    return replace(
        a,
        sets=sets,
        max_weight=min(a.max_weight, b.max_weight),
        max_volatility=min(vols) if vols else None,
        min_return=max(rets) if rets else None,
        groups={**b.groups, **a.groups},
        group_bounds=groups,
        asset_bounds=assets,
    )


def _example(ref, horizon, value, needs, answers, **kw) -> dict:
    keys = [q["id"] for q in QUESTIONS]
    return ClientProfile(
        reference=ref,
        horizon_years=horizon,
        portfolio_value_usd=value,
        liquidity_needs=needs,
        answers=dict(zip(keys, answers, strict=True)),
        **kw,
    ).to_dict()


EXAMPLE_CLIENTS: dict[str, dict] = {
    "Dubai, moderate, 10 years": _example(
        "EX-DXB",
        10,
        500_000,
        [{"description": "Property deposit", "amount_usd": 50_000, "date": "2027-10"}],
        [4, 3, 4, 4, 3, 3, 3],
        residency="UAE",
        booking_location="DIFC (Dubai)",
        age=45,
        home_bias_gcc_pct=10,
        goals=[{"description": "Grow capital", "multiple": 1.5, "years": 10}],
    ),
    "Riyadh, conservative, 5 years, Sharia": _example(
        "EX-RUH",
        5,
        1_000_000,
        [{"description": "School fees", "amount_usd": 100_000, "date": "2027-04"}],
        [3, 2, 2, 3, 2, 1, 3],
        residency="Saudi Arabia",
        booking_location="Saudi Arabia",
        age=58,
        sharia_only=True,
        home_bias_gcc_pct=30,
        goals=[{"description": "Preserve capital", "years": 5}],
    ),
    "Doha, aggressive, 20 years": _example(
        "EX-DOH",
        20,
        250_000,
        [],
        [5, 4, 5, 5, 4, 5, 4],
        residency="Qatar",
        booking_location="QFC (Qatar)",
        age=34,
        goals=[{"description": "Long-term growth", "multiple": 3, "years": 20}],
    ),
}
