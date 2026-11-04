"""Professional fact-find: the sections an adviser completes, completeness, and calculations.

Pure functions: no I/O, no printing, no plotting.

``SECTIONS`` is the single registry of the fact-find: each section lists its fields with a
label, a short helper text, whether it is required, and the kind of input. The client form,
the completeness bar, the IPS PDF, the client report and the demo generator all read it, so a
new field is added in one place. The calculations (net worth, investable assets, savings rate,
emergency-fund months, the return needed to meet the goals) are small, tested formulas.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from scipy.optimize import brentq

from gpl.client import (
    CLASSIFICATIONS,
    QUESTIONS,
    SCORED,
    ClientProfile,
    age_on,
    effective_regulator,
    months_until,
    with_derived_answers,
)

MARITAL = ["Single", "Married", "Divorced", "Widowed"]
EMPLOYMENT = ["Employed", "Self-employed", "Business owner", "Retired", "Not working"]
CURRENCIES = ["USD", "AED", "SAR", "QAR", "KWD", "BHD", "OMR", "EUR", "GBP"]
ACTIVITY = ["Low (a few trades a year)", "Medium (about monthly)", "High (weekly or more)"]
OBJECTIVES = ["Growth", "Income", "Capital preservation"]
PRODUCTS = [
    "Equities",
    "Bonds / sukuk",
    "Funds",
    "ETFs",
    "Derivatives",
    "Alternatives",
    "Private markets",
]
# products whose risks are hard to understand without experience (leverage, illiquidity,
# valuation): recommending one the client hasn't used is flagged for the adviser
COMPLEX_PRODUCTS = ["Derivatives", "Alternatives", "Private markets"]
TRADING = ["Never", "Yearly", "Quarterly", "Monthly", "Weekly or more"]
WILLS = [
    "No will",
    "UAE onshore will",
    "DIFC Wills Service Centre will",
    "ADGM will",
    "Will in another country",
]
STRUCTURES = ["Trust", "Foundation", "Holding company", "Family office"]
SERVICES = ["Advisory", "Discretionary"]
REVIEWS = {"Quarterly": 3, "Semi-annual": 6, "Annual": 12}
INCOME_TYPES = {
    "salary": "Salary",
    "bonus": "Bonus",
    "business": "Business",
    "rental": "Rental",
    "other": "Other",
}
ASSET_TYPES = {
    "cash": "Cash and deposits",
    "investments": "Investments",
    "property": "Property",
    "business": "Business interests",
    "eosg": "End-of-service gratuity",
    "pensions": "Pensions",
}
LIABILITY_TYPES = {"mortgage": "Mortgage", "loans": "Loans", "other": "Other"}
# investable: can be put in the portfolio now (property, a business, gratuity and pensions can't)
INVESTABLE = ("cash", "investments")
SPECIALIST_REMINDER = (
    "Estate and succession planning needs specialist legal and tax advice: involve a lawyer "
    "(e.g. for a DIFC or ADGM will, or a foundation) before acting on these notes."
)
DEMO_LABEL = "DEMO DATA: a fictional client generated for demonstration"
PUBLIC_NOTICE = "Demo tool: do not enter real client data in the public version."


# Yahoo quote types -> fact-find product types (bond and sukuk ETFs are still ETFs)
QUOTE_PRODUCTS = {"ETF": "ETFs", "EQUITY": "Equities", "MUTUALFUND": "Funds"}


def product_type(quote_type: str | None) -> str | None:
    """'ETF' -> 'ETFs', 'EQUITY' -> 'Equities', 'MUTUALFUND' -> 'Funds'; None if unknown."""
    return QUOTE_PRODUCTS.get((quote_type or "").upper())


@dataclass(frozen=True)
class Field:
    """One fact-find input: ``name`` is the ``ClientProfile`` attribute."""

    name: str
    label: str
    kind: str  # text | textarea | int | float | money | bool | yesno | select | multi | ...
    required: bool = False
    help: str = ""
    options: Sequence[str] = ()


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    helper: str
    fields: tuple[Field, ...] = field(default_factory=tuple)


SECTIONS: tuple[Section, ...] = (
    Section(
        "household",
        "Client & household",
        "Who the client is, where they live and pay tax, and who depends on them.",
        (
            Field("reference", "Client reference", "text", True),
            Field("name", "Name", "text", True),
            Field("dob", "Date of birth", "date", False, "Sets the age when given."),
            Field("age", "Age", "int", True),
            Field("nationality", "Nationality", "text", True),
            Field("residency", "Residency (country)", "text", True),
            Field(
                "tax_residencies",
                "Tax residencies",
                "tags",
                True,
                "Every country where the client is tax resident.",
            ),
            Field(
                "us_person", "US person", "bool", False, "Citizen, green card or US tax resident."
            ),
            Field("marital_status", "Marital status", "select", True, options=MARITAL),
            Field("dependants", "Dependants (number)", "int", True),
            Field("dependant_ages", "Dependants' ages", "ints", False, "e.g. 8, 12"),
            Field("occupation", "Occupation", "text", True),
            Field("employer", "Employer", "text"),
            Field("employment_type", "Employment type", "select", True, options=EMPLOYMENT),
            Field("retirement_age", "Planned retirement age", "int", True),
            Field("reporting_currency", "Reporting currency", "select", True, options=CURRENCIES),
        ),
    ),
    Section(
        "kyc",
        "KYC / AML",
        "Where the money comes from, politically exposed status, and the client classification.",
        (
            Field(
                "source_of_wealth",
                "Source of wealth",
                "textarea",
                True,
                "How the client built their total wealth (career, business sale, inheritance).",
            ),
            Field(
                "source_of_funds",
                "Source of funds",
                "textarea",
                True,
                "Where the money for this account comes from (e.g. bank transfer from savings).",
            ),
            Field("pep", "Politically exposed person (PEP)", "yesno", True),
            Field("expected_activity", "Expected activity", "select", True, options=ACTIVITY),
            Field(
                "classification",
                "Client classification",
                "select",
                True,
                options=CLASSIFICATIONS,
            ),
            Field(
                "classification_reason",
                "Reason for the classification",
                "textarea",
                True,
                "e.g. net assets above the professional-client threshold, with evidence.",
            ),
        ),
    ),
    Section(
        "financial",
        "Financial situation",
        "Income, spending, assets and debts: net worth, investable assets and resilience.",
        (
            Field("income", "Annual income by source (USD)", "money_map", True),
            Field("annual_expenses_usd", "Annual expenses (USD)", "money", True),
            Field("assets", "Assets by type (USD)", "money_map", True),
            Field("liabilities", "Liabilities (USD)", "money_map"),
        ),
    ),
    Section(
        "objectives",
        "Objectives & goals",
        "What the money is for, when it is needed, and the return that requires.",
        (
            Field("primary_objective", "Primary objective", "select", True, options=OBJECTIVES),
            Field("horizon_years", "Time horizon (years)", "float", True),
            Field("portfolio_value_usd", "Portfolio value (USD)", "money", True),
            Field("goals", "Goals", "custom"),
            Field("liquidity_needs", "Cash needs", "custom"),
            Field("annual_income_need_usd", "Yearly income from the portfolio (USD)", "money"),
            Field("annual_contribution_usd", "Yearly addition to the portfolio (USD)", "money"),
            Field("return_target_pct", "Return target (% a year)", "float"),
        ),
    ),
    Section(
        "risk",
        "Risk profile",
        "Capacity (can the client afford losses?), willingness (are they comfortable with them?) "
        "and the largest fall they could accept.",
        (
            Field("answers", "Risk questionnaire", "custom", True),
            Field(
                "max_acceptable_drawdown_pct",
                "Capacity for loss: largest acceptable fall (%)",
                "float",
                True,
                "The IPS drawdown limit is the tighter of this and the risk band's.",
            ),
            Field("override_score", "Adviser override of the risk score", "custom"),
        ),
    ),
    Section(
        "knowledge",
        "Knowledge & experience",
        "What the client has invested in before; flags products they may not understand. "
        "Not part of the risk score.",
        (
            Field("years_investing", "Years investing", "float", True),
            Field("products_used", "Products used", "multi", False, options=PRODUCTS),
            Field("trading_frequency", "Trading frequency", "select", True, options=TRADING),
            Field(
                "background",
                "Relevant professional or education background",
                "textarea",
                False,
                "e.g. CFA charterholder, banker, finance degree.",
            ),
        ),
    ),
    Section(
        "constraints",
        "Constraints & preferences",
        "Limits the portfolio must respect: liquidity, tax, legal, Sharia, ESG and holdings.",
        (
            Field("sharia_only", "Sharia-compliant only", "custom"),
            Field("excluded_sectors", "Excluded sectors (ESG)", "custom"),
            Field("region_minimums", "Region minimums", "custom"),
            Field("legacy_holdings", "Concentrated or legacy holdings", "textarea"),
            Field("restricted_securities", "Restricted securities (tickers)", "tags"),
            Field(
                "home_currency_preference",
                "Home-currency preference",
                "select",
                False,
                options=["", *CURRENCIES],
            ),
            Field("tax_notes", "Tax notes", "textarea"),
            Field("legal_notes", "Legal / regulatory notes", "textarea"),
        ),
    ),
    Section(
        "estate",
        "Estate & succession",
        "Flags only: wills, beneficiaries and structures, for the specialists to act on.",
        (
            Field("will_status", "Will", "select", True, options=WILLS),
            Field("beneficiaries_noted", "Beneficiaries noted", "bool"),
            Field("structures", "Existing structures", "multi", False, options=STRUCTURES),
            Field("estate_notes", "Estate notes", "textarea"),
        ),
    ),
    Section(
        "service",
        "Service & review",
        "How the relationship runs: mandate, reviews, fees and the responsible adviser.",
        (
            Field("service_type", "Service", "select", True, options=SERVICES),
            Field("review_frequency", "Review frequency", "select", True, options=list(REVIEWS)),
            Field("next_review", "Next review date", "date"),
            Field("fee_basis", "Fee basis", "text", True, "e.g. 1.00% a year of assets."),
            Field("adviser_name", "Adviser name", "text", True),
            Field("adviser_licence", "Adviser licence number", "text", True),
        ),
    ),
)
SECTION_BY_KEY = {s.key: s for s in SECTIONS}
# the final IPS needs these sections complete (estate and constraints are optional flags)
IPS_SECTIONS = ("household", "kyc", "financial", "objectives", "risk", "knowledge", "service")


def is_filled(profile: ClientProfile, f: Field) -> bool:
    """Whether a field has an answer. Money maps need one positive amount; the questionnaire
    needs every capacity and willingness answer; booleans are always answered."""
    if f.name == "answers":
        return all(q["id"] in profile.answers for q in QUESTIONS if q["dimension"] in SCORED)
    value = getattr(profile, f.name)
    if f.kind == "bool":
        return True
    if f.name == "age":
        return value not in (None, "") or bool(profile.dob)
    if f.kind == "money_map":
        return any((v or 0) > 0 for v in (value or {}).values())
    if f.name == "dependants":
        return value is not None  # 0 is an answer
    return value not in (None, "", [], {}) and not (f.kind == "money" and value == 0)


@dataclass
class Completeness:
    done: int
    total: int
    missing: list[str]

    @property
    def share(self) -> float:
        return self.done / self.total if self.total else 1.0


def completeness(profile: ClientProfile, today: date | None = None) -> dict[str, Completeness]:
    """Required fields answered per section (cash-need answers derived first)."""
    p = with_derived_answers(profile, today or date.today())
    out = {}
    for s in SECTIONS:
        required = [f for f in s.fields if f.required]
        missing = [f.label for f in required if not is_filled(p, f)]
        out[s.key] = Completeness(len(required) - len(missing), len(required), missing)
    return out


def overall_completeness(parts: dict[str, Completeness]) -> float:
    """Share of all required fields answered (each field counts once)."""
    total = sum(c.total for c in parts.values())
    return sum(c.done for c in parts.values()) / total if total else 1.0


def ips_missing(profile: ClientProfile, today: date | None = None) -> list[str]:
    """What must be filled in before the final IPS can be issued (empty list: ready).

    Every required field of the sections in ``IPS_SECTIONS``, a known regulator, and a reason
    for any adviser override of the risk score. Until then only a DRAFT IPS is produced.
    """
    parts = completeness(profile, today)
    missing = [
        f"{SECTION_BY_KEY[key].title}: {label}"
        for key in IPS_SECTIONS
        for label in parts[key].missing
    ]
    if effective_regulator(profile) == "Other":
        missing.append("Client & household: regulator (booking location)")
    if profile.override_score is not None and not profile.override_reason.strip():
        missing.append("Risk profile: reason for the adviser override")
    return missing


@dataclass
class Finances:
    income: float
    expenses: float | None
    assets: float
    liabilities: float
    net_worth: float
    investable: float
    savings_rate: float | None
    emergency_months: float | None


def finances(profile: ClientProfile) -> Finances:
    """Net worth, investable assets, savings rate and emergency-fund months.

    - Net worth = total assets - total liabilities.
    - Investable assets = cash + investments (property, business interests, end-of-service
      gratuity and pensions can't be put in a portfolio now).
    - Savings rate = (income - expenses) / income.
    - Emergency-fund months = cash / (expenses / 12): how long cash covers spending.

    E.g. income 300,000, expenses 180,000, cash 90,000: savings rate 40.00%, 6.0 months.
    """
    income = sum(v or 0 for v in profile.income.values())
    assets = sum(v or 0 for v in profile.assets.values())
    liabilities = sum(v or 0 for v in profile.liabilities.values())
    expenses = profile.annual_expenses_usd
    cash = profile.assets.get("cash") or 0
    return Finances(
        income=income,
        expenses=expenses,
        assets=assets,
        liabilities=liabilities,
        net_worth=assets - liabilities,
        investable=sum(profile.assets.get(k) or 0 for k in INVESTABLE),
        savings_rate=(income - expenses) / income if income and expenses is not None else None,
        emergency_months=cash / (expenses / 12) if expenses else None,
    )


@dataclass
class GoalsReturn:
    rate: float | None
    flows: list[tuple[str, float, float]]  # (label, years, USD)
    reason: str


def goals_required_return(
    value: float | None,
    goals: Sequence[dict],
    cash_needs: Sequence[dict] = (),
    annual_contribution: float = 0.0,
    today: date | None = None,
    bounds: tuple[float, float] = (-0.5, 0.5),
) -> GoalsReturn:
    """The yearly return r at which today's value plus future additions pays every goal.

    Solves value + PV(additions at r) = PV(goals and dated cash needs at r), with additions
    made at the end of each year until the last goal. Goals stated as a multiple of today's
    value count as that multiple × value. Goals without an amount or a year are left out (and
    named). With one goal and no additions this is (goal / value) ^ (1 / years) - 1: USD
    100,000 growing to USD 200,000 in 10 years needs 7.18% a year.
    """
    today = today or date.today()
    if not value:
        return GoalsReturn(None, [], "Enter the portfolio value to calculate it.")
    flows, skipped = [], []
    for g in goals:
        years = g.get("years")
        amount = g.get("amount_usd") or (g["multiple"] * value if g.get("multiple") else None)
        if not years or not amount:
            skipped.append(g.get("description") or "a goal")
            continue
        flows.append((g.get("description") or "Goal", float(years), float(amount)))
    for n in cash_needs:
        years = max(months_until(n["date"], today), 1) / 12
        flows.append((n.get("description") or "Cash need", years, float(n["amount_usd"])))
    if not flows:
        return GoalsReturn(None, [], "No goals with an amount and a year are recorded.")
    last = max(y for _, y, _ in flows)

    def surplus(r: float) -> float:
        adds = sum(annual_contribution / (1 + r) ** t for t in range(1, int(last) + 1))
        return value + adds - sum(a / (1 + r) ** y for _, y, a in flows)

    note = f" Left out (no amount or year): {', '.join(skipped)}." if skipped else ""
    lo, hi = bounds
    if surplus(hi) < 0:
        return GoalsReturn(
            None, flows, f"The goals can't be met even at {hi:.0%} a year: revisit them.{note}"
        )
    if surplus(lo) > 0:
        return GoalsReturn(
            lo, flows, f"The goals are covered even if the portfolio loses {-lo:.0%} a year.{note}"
        )
    rate = brentq(surplus, lo, hi, xtol=1e-10)
    return GoalsReturn(
        rate,
        flows,
        f"{rate:.2%} a year pays every goal and dated cash need on time "
        f"({len(flows)} payment{'s' if len(flows) != 1 else ''}, the last in {last:g} years)."
        + note,
    )


def complex_product_flags(
    products_used: Sequence[str], years_investing: float | None, held: Sequence[str] = ()
) -> list[str]:
    """Warnings about products the client may not understand.

    ``held``: product types in the portfolio or recommendation. Any complex product (``COMPLEX_
    PRODUCTS``) the client hasn't used is flagged; so is any product type held that they
    haven't used when they have under two years' experience.
    """
    flags = []
    used = set(products_used)
    for p in dict.fromkeys(held):
        if p in COMPLEX_PRODUCTS and p not in used:
            flags.append(
                f"{p}: complex, and the client hasn't used them. Explain the risks and record "
                "that the client understands them, or leave them out."
            )
        elif p not in used and (years_investing or 0) < 2:
            flags.append(
                f"{p}: new to a client with under two years' experience. Explain how they work."
            )
    if not held and (years_investing or 0) < 2 and not used:
        flags.append("Limited experience: explain every product recommended.")
    return flags


def consistency_warnings(profile: ClientProfile, today: date | None = None) -> list[str]:
    """Facts that contradict each other, for the adviser to fix (nothing is changed)."""
    today = today or date.today()
    out = []
    p = profile
    if p.dob:
        try:
            age = age_on(p.dob, today)
        except ValueError:
            out.append(f"Date of birth {p.dob!r} isn't a date like 1980-05-31.")
        else:
            if p.age is not None and abs(age - p.age) > 1:
                out.append(f"Age {p.age} doesn't match the date of birth (age {age}).")
    if p.age and p.retirement_age and p.retirement_age <= p.age and p.employment_type != "Retired":
        out.append(f"Planned retirement age {p.retirement_age} is not after age {p.age}.")
    if p.dependants is not None and p.dependant_ages and len(p.dependant_ages) != p.dependants:
        out.append(f"{p.dependants} dependants but {len(p.dependant_ages)} ages are recorded.")
    f = finances(p)
    if p.portfolio_value_usd and f.investable and p.portfolio_value_usd > f.investable + 1:
        out.append(
            f"The portfolio value (USD {p.portfolio_value_usd:,.0f}) is more than the investable "
            f"assets recorded (USD {f.investable:,.0f})."
        )
    if f.savings_rate is not None and f.savings_rate < 0:
        out.append("Expenses exceed income: the client is drawing on savings.")
    long = [g for g in p.goals if (g.get("years") or 0) > p.horizon_years]
    if long:
        out.append(
            "Goals beyond the time horizon: "
            + ", ".join(g.get("description", "goal") for g in long)
        )
    if p.override_score is not None and not p.override_reason.strip():
        out.append("The adviser override of the risk score has no reason: it is ignored.")
    return out


def next_review(frequency: str, today: date) -> str:
    """The default next review date: ``today`` plus the review interval, as 'YYYY-MM-DD'."""
    months = REVIEWS.get(frequency, 12)
    index = today.year * 12 + today.month - 1 + months
    year, month = index // 12, index % 12 + 1
    day = min(today.day, 28)
    return date(year, month, day).isoformat()
