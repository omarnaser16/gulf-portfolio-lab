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
        "dimension": "willingness",
        "text": "How much investing experience does the client have?",
        "options": ["None", "Little", "Some", "Good", "Extensive"],
    },
]
REGULATORS = [
    "DFSA (DIFC)",
    "FSRA (ADGM)",
    "SCA (UAE)",
    "CMA (Saudi Arabia)",
    "QFCRA (Qatar)",
    "Other",
]
CLASSIFICATIONS = ["Retail", "Professional"]
PEGGED = {"USD", "SAR", "AED", "QAR", "BHD", "OMR"}


def horizon_score(years: float) -> int:
    """Capacity score from the time horizon: under 3 years 1, 3-5 2, 5-10 3, 10-15 4, 15+ 5."""
    for limit, score in ((3, 1), (5, 2), (10, 3), (15, 4)):
        if years < limit:
            return score
    return 5


def _round_half_up(x: float) -> int:
    return int(x + 0.5)


@dataclass
class ClientProfile:
    """Everything the adviser records about the client (percentages as written, e.g. 20)."""

    reference: str = ""
    name: str = ""
    classification: str = "Retail"
    regulator: str = "DFSA (DIFC)"
    residency: str = "UAE"
    us_person: bool = False
    horizon_years: float = 10
    goals: list[dict] = field(default_factory=list)  # {description, amount_usd|multiple, years}
    return_target_pct: float | None = None
    portfolio_value_usd: float | None = None
    liquidity_need_usd: float = 0.0
    liquidity_months: int = 12
    reserve_pct: float = 0.0
    tax_notes: str = ""
    legal_notes: str = ""
    sharia_only: bool = False
    excluded_sectors: list[str] = field(default_factory=list)
    home_bias_gcc_pct: float | None = None
    max_non_pegged_currency_pct: float = 20.0
    unique_notes: str = ""
    answers: dict[str, int] = field(default_factory=dict)
    overrides: dict[str, float] = field(default_factory=dict)  # max_volatility_pct, ...
    adviser_notes: str = ""
    version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ClientProfile:
        """From saved JSON; unknown keys are ignored and missing ones take their defaults."""
        known = {f.name: f for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in known}
        out = cls(**values)
        for name in (
            "horizon_years",
            "liquidity_need_usd",
            "reserve_pct",
            "max_non_pegged_currency_pct",
        ):
            try:
                setattr(out, name, float(getattr(out, name)))
            except (TypeError, ValueError) as e:
                raise ValueError(f"{name} must be a number, got {getattr(out, name)!r}") from e
        return out


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
    missing = [q["id"] for q in QUESTIONS if q["id"] not in answers]
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
        f"{cap_items[0]}) and four answers; willingness averages three answers."
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
    """Build the IPS. Risk limits come from the risk band unless the adviser overrides them."""
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
    reserve_usd = profile.liquidity_need_usd + (value or 0) * profile.reserve_pct / 100
    limits["liquidity_reserve_usd"] = reserve_usd
    limits["liquidity_reserve"] = reserve_usd / value if value else None
    limits["sharia_only"] = profile.sharia_only
    limits["excluded_sectors"] = list(profile.excluded_sectors)

    goals = "; ".join(_goal_text(g) for g in profile.goals) or "No specific goals recorded."
    target = (
        f" Return target: {profile.return_target_pct:.2f}% a year (historical estimates only)."
        if profile.return_target_pct is not None
        else ""
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
        "Time horizon": f"{profile.horizon_years:g} years.",
        "Taxes": (
            f"Residency: {profile.residency}. US person: {'yes' if profile.us_person else 'no'}."
            + (f" {profile.tax_notes}" if profile.tax_notes else "")
        ),
        "Liquidity": (
            f"Cash need of USD {profile.liquidity_need_usd:,.0f} within "
            f"{profile.liquidity_months} months, plus a {profile.reserve_pct:g}% reserve: USD "
            f"{reserve_usd:,.0f} to keep in low-volatility holdings"
            + (f" ({_pct(limits['liquidity_reserve'])} of the portfolio)." if value else ".")
        ),
        "Legal and regulatory": (
            f"Client classification: {profile.classification}. Regulator: {profile.regulator}."
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
                f"At most {profile.max_non_pegged_currency_pct:g}% in currencies not pegged to "
                "the US dollar.",
                profile.unique_notes,
            ]
            if sentence
        ),
    }
    created = (today or date.today()).isoformat()
    return IPS(profile, risk, sections, limits, created)


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
    ips: IPS, holdings: list[str], groups: dict[str, str], attributes: dict[str, dict]
) -> tuple[OptimisationSpec, list[str]]:
    """IPS limits as optimiser limits, plus notes for the adviser.

    Excluded holdings get a 0% maximum. Holdings with unknown Sharia status are not excluded
    automatically but noted for review. The drawdown limit can't be an optimiser constraint
    (it isn't convex); the suitability check enforces it.
    """
    excluded, review = excluded_holdings(ips, holdings, attributes)
    lim = ips.limits
    spec = OptimisationSpec(
        objective="max_sharpe",
        max_weight=lim["max_weight"] or 1.0,
        asset_bounds={t: (None, 0.0) for t in excluded},
        groups=dict(groups),
        group_bounds={"GCC": (lim["home_bias_gcc"], None)} if lim["home_bias_gcc"] else {},
        max_volatility=lim["max_volatility"],
    )
    notes = [f"{t} excluded: {why}." for t, why in excluded.items()]
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
    return replace(
        a,
        max_weight=min(a.max_weight, b.max_weight),
        max_volatility=min(vols) if vols else None,
        min_return=max(rets) if rets else None,
        groups={**b.groups, **a.groups},
        group_bounds=groups,
        asset_bounds=assets,
    )


def _example(ref, horizon, value, need, months, answers, **kw) -> dict:
    keys = [q["id"] for q in QUESTIONS]
    return ClientProfile(
        reference=ref,
        horizon_years=horizon,
        portfolio_value_usd=value,
        liquidity_need_usd=need,
        liquidity_months=months,
        answers=dict(zip(keys, answers, strict=True)),
        **kw,
    ).to_dict()


EXAMPLE_CLIENTS: dict[str, dict] = {
    "Dubai, moderate, 10 years": _example(
        "EX-DXB",
        10,
        500_000,
        50_000,
        12,
        [4, 3, 4, 4, 3, 3, 3],
        residency="UAE",
        regulator="DFSA (DIFC)",
        home_bias_gcc_pct=10,
        goals=[{"description": "Grow capital", "multiple": 1.5, "years": 10}],
    ),
    "Riyadh, conservative, 5 years, Sharia": _example(
        "EX-RUH",
        5,
        1_000_000,
        100_000,
        6,
        [3, 2, 2, 3, 2, 1, 3],
        residency="Saudi Arabia",
        regulator="CMA (Saudi Arabia)",
        sharia_only=True,
        home_bias_gcc_pct=30,
        goals=[{"description": "Preserve capital", "years": 5}],
    ),
    "Doha, aggressive, 20 years": _example(
        "EX-DOH",
        20,
        250_000,
        0,
        12,
        [5, 4, 5, 5, 4, 5, 4],
        residency="Qatar",
        regulator="QFCRA (Qatar)",
        goals=[{"description": "Long-term growth", "multiple": 3, "years": 20}],
    ),
}
