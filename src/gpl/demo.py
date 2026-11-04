"""Seeded demo clients: a realistic, internally consistent GCC client and portfolio.

Pure functions: no I/O, no printing, no plotting. No AI: every value comes from a seeded
random generator (``numpy.random.default_rng(seed)``), so a demo can be reproduced exactly by
re-using its seed. Every profile is marked ``demo=True`` (shown as DEMO DATA everywhere).

Consistency rules (tested over many seeds): age matches the date of birth; retirement comes
after today's age; the horizon runs to retirement (or 10-15 years once retired); goals fall
within the horizon; cash needs are in the future; the portfolio is at most the investable
assets, which are at most the net worth; expenses are below income; the questionnaire answers
follow the client's age, horizon and objective; and every required fact-find field is filled.
The names are invented combinations, not real people.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from gpl.client import ClientProfile, age_on, month_after
from gpl.factfind import next_review

FIRST = ["Omar", "Layla", "Khalid", "Noura", "Faisal", "Mariam", "Yousef", "Huda", "Tariq",
         "Aisha", "Rashid", "Salma", "Hamad", "Reem", "Karim", "Dana", "Sami", "Lina"]  # fmt: skip
LAST = ["Al Mansouri", "Al Harbi", "Al Thani", "Al Sabah", "Al Khalifa", "Al Busaidi",
        "Haddad", "Nasser", "Rahman", "Farouk", "Saleh", "Kareem", "Aziz", "Hamdan"]  # fmt: skip
# residency -> (booking locations, local nationality, home currency)
PLACES = {
    "UAE": (["DIFC (Dubai)", "ADGM (Abu Dhabi)", "UAE onshore"], "Emirati", "AED"),
    "Saudi Arabia": (["Saudi Arabia"], "Saudi", "SAR"),
    "Qatar": (["QFC (Qatar)"], "Qatari", "QAR"),
    "Kuwait": (["Kuwait"], "Kuwaiti", "KWD"),
    "Bahrain": (["Bahrain"], "Bahraini", "BHD"),
    "Oman": (["Oman"], "Omani", "OMR"),
}
EXPATS = ["British", "Indian", "Egyptian", "Lebanese", "Jordanian", "Pakistani", "French"]
JOBS = {
    "Employed": [
        ("Petroleum engineer", "a national oil company"),
        ("Bank director", "a regional bank"),
        ("Surgeon", "a private hospital"),
        ("Management consultant", "a consulting firm"),
        ("Government adviser", "a government ministry"),
    ],
    "Business owner": [
        ("Founder, logistics company", "own company"),
        ("Owner, retail group", "own company"),
        ("Real estate developer", "own company"),
    ],
    "Self-employed": [("Architect", "own practice"), ("Lawyer", "own practice")],
}
WEALTH = {
    "Employed": "Salary and bonuses saved over a {years}-year career as {job}.",
    "Business owner": "Profits from {job_lower}, built over {years} years.",
    "Self-employed": "Fees from {years} years of private practice as {job_lower}.",
    "Retired": "Career savings and end-of-service gratuity; retired from a {years}-year career.",
}
ADVISERS = ["S. Haddad", "R. Menon", "L. Fischer", "A. Qureshi", "M. Darwish"]

# real, liquid tickers by role; Sharia-compliant alternatives are used for Sharia clients
POOL = {
    "equity": ["SPY", "EFA", "EEM", "URTH"],
    "gcc": ["KSA", "UAE", "QAT", "2222.SR", "1120.SR", "EMAAR.AE", "QNBK.QA", "NBK.KW"],
    "bonds": ["AGG", "TLT"],
    "gold": ["GLD"],
    "cash": ["BIL", "SGOV"],
}
SHARIA_POOL = {
    "equity": ["SPUS", "HLAL"],  # UMMA left out: it starts in 2022 and would cut the sample
    "gcc": ["2222.SR", "1120.SR"],
    "bonds": ["SPSK"],
    "gold": ["GLD"],
    "cash": ["SPSK"],
}
GROUP = {
    "equity": "Global equity",
    "gcc": "GCC",
    "bonds": "Bonds",
    "gold": "Gold",
    "cash": "Cash & short-term",
}
# target weights by role for each risk score (rows sum to 1)
MIX = {
    1: {"equity": 0.15, "gcc": 0.10, "bonds": 0.45, "gold": 0.10, "cash": 0.20},
    2: {"equity": 0.25, "gcc": 0.15, "bonds": 0.35, "gold": 0.10, "cash": 0.15},
    3: {"equity": 0.35, "gcc": 0.20, "bonds": 0.25, "gold": 0.10, "cash": 0.10},
    4: {"equity": 0.45, "gcc": 0.25, "bonds": 0.15, "gold": 0.08, "cash": 0.07},
    5: {"equity": 0.55, "gcc": 0.28, "bonds": 0.07, "gold": 0.05, "cash": 0.05},
}


@dataclass
class DemoClient:
    profile: ClientProfile
    holdings: list[dict]  # {"ticker", "group", "value"}: USD amounts
    seed: int


def _round(x: float, step: float = 1000) -> float:
    return float(round(x / step) * step)


def demo_client(seed: int, today: date | None = None) -> DemoClient:
    """A random but realistic and internally consistent GCC client, reproducible by ``seed``."""
    today = today or date.today()
    rng = np.random.default_rng(seed)

    def pick(items: Sequence):
        return items[int(rng.integers(len(items)))]

    residency = pick(list(PLACES))
    bookings, local, home_ccy = PLACES[residency]
    nationality = local if rng.random() < 0.55 else pick(EXPATS)
    age = int(rng.integers(32, 69))
    dob = (today - timedelta(days=int(age * 365.25 + rng.integers(15, 350)))).isoformat()
    age = age_on(dob, today)
    retired = age >= 62 and rng.random() < 0.6
    employment = "Retired" if retired else pick(["Employed", "Employed", "Business owner",
                                                 "Self-employed"])  # fmt: skip
    if retired:
        retirement_age = int(rng.integers(55, age + 1))
        horizon = float(rng.integers(10, 16))
        job, employer = "Retired", ""
    else:
        retirement_age = int(rng.integers(max(age + 3, 55), max(age + 4, 66)))
        horizon = float(min(max(retirement_age - age, 3), 30))
        job, employer = pick(JOBS[employment])
    married = rng.random() < 0.75
    dependants = int(rng.integers(0, 4)) if married else 0
    child_max = max(min(age - 24, 25), 1)
    ages = sorted(int(rng.integers(0, child_max)) for _ in range(dependants))

    # wealth: the portfolio first, then the rest of the balance sheet around it
    value = _round(float(np.exp(rng.uniform(np.log(300_000), np.log(8_000_000)))), 10_000)
    scale = value / 1_000_000
    salary = 0.0 if retired else _round(rng.uniform(120_000, 450_000) * max(scale, 0.5) ** 0.5)
    income = {"salary": salary} if salary else {}
    if salary and employment == "Employed":
        income["bonus"] = _round(salary * rng.uniform(0.0, 0.4))
    if employment == "Business owner":
        income["business"] = _round(rng.uniform(100_000, 600_000) * max(scale, 0.5) ** 0.5)
    if rng.random() < 0.4:
        income["rental"] = _round(rng.uniform(20_000, 120_000))
    if retired:
        income["other"] = _round(value * rng.uniform(0.02, 0.04))  # pension and dividends
    total_income = sum(income.values())
    expenses = _round(total_income * rng.uniform(0.45, 0.8))
    cash = _round(expenses / 12 * rng.uniform(4, 12))
    investments = _round(value * rng.uniform(1.0, 1.5))
    assets = {"cash": cash, "investments": investments}
    if rng.random() < 0.8:
        assets["property"] = _round(value * rng.uniform(0.3, 1.5), 10_000)
    if employment == "Business owner":
        assets["business"] = _round(value * rng.uniform(0.5, 2.0), 10_000)
    if employment == "Employed" and residency != "Saudi Arabia":
        assets["eosg"] = _round(salary / 12 * rng.uniform(5, 20))
    other = sum(v for k, v in assets.items() if k not in ("cash", "investments"))
    liabilities = {}
    if assets.get("property") and rng.random() < 0.5:
        liabilities["mortgage"] = _round(assets["property"] * rng.uniform(0.1, 0.5))
    if rng.random() < 0.2:
        liabilities["loans"] = _round(rng.uniform(10_000, 80_000))
    if sum(liabilities.values()) > other:  # net worth never below the investable assets
        liabilities = {k: _round(v * other / sum(liabilities.values())) for k, v in
                       liabilities.items()}  # fmt: skip
        liabilities = {k: v for k, v in liabilities.items() if v}

    # objective and the answers that go with it
    objective = (
        pick(["Income", "Capital preservation"]) if retired or horizon < 6
        else pick(["Growth", "Growth", "Income"])
    )  # fmt: skip
    base = {"Growth": 4, "Income": 3, "Capital preservation": 2}[objective]
    base = base - (1 if age >= 60 else 0) + (1 if horizon >= 15 and base < 5 else 0)

    def answer(shift: int = 0) -> int:
        return int(np.clip(base + shift + rng.integers(-1, 2), 1, 5))

    years_investing = float(rng.integers(1, max(age - 22, 2)))
    answers = {
        "income_stability": answer(1 if salary and employment == "Employed" else 0),
        "wealth_share": answer(),
        "dependence": answer(-1 if retired else 1),
        "liquidity_need": 4,  # replaced by the Cash needs rule when the value is known
        "loss_reaction": answer(),
        "tradeoff": answer(),
        "experience": int(np.clip(1 + years_investing // 5, 1, 5)),
    }
    loss = {1: 10, 2: 15, 3: 20, 4: 30, 5: 40}[int(np.clip(base, 1, 5))]
    products = ["Equities", "Funds"]
    if years_investing >= 3:
        products.append("ETFs")
    if years_investing >= 6:
        products.append("Bonds / sukuk")
    if years_investing >= 12 and rng.random() < 0.4:
        products.append("Alternatives")

    # goals within the horizon; dated cash needs in the future
    goals = []
    if not retired:
        goals.append(
            {
                "description": "Retirement fund",
                # what 3-8% a year would grow the portfolio to by retirement
                "amount_usd": _round(value * (1 + rng.uniform(0.03, 0.08)) ** horizon, 10_000),
                "years": horizon,
            }
        )
    else:
        goals.append({"description": "Preserve capital for heirs", "multiple": 1.0,
                      "years": horizon})  # fmt: skip
    young = [a for a in ages if a < 17]
    if young:
        years = float(min(18 - min(young), horizon))
        goals.append(
            {
                "description": "University fees",
                "amount_usd": _round(rng.uniform(120_000, 300_000), 10_000),
                "years": years,
            }
        )
    needs = []
    for _ in range(int(rng.integers(0, 3))):
        months = int(rng.integers(3, 48))
        needs.append(
            {
                "description": pick(
                    ["Property deposit", "Wedding", "Car", "Home renovation", "Business investment"]
                ),  # fmt: skip
                "amount_usd": _round(value * rng.uniform(0.02, 0.12)),
                "date": month_after(today, months),
            }
        )
    needs.sort(key=lambda n: n["date"])

    sharia = rng.random() < (0.6 if residency == "Saudi Arabia" else 0.3)
    region_minimums = {}
    if rng.random() < 0.3:
        region_minimums = {residency if residency != "UAE" else "UAE": 5.0}
    net_assets = sum(assets.values()) - sum(liabilities.values())
    professional = net_assets >= 1_000_000 and years_investing >= 5 and rng.random() < 0.5
    review = pick(["Quarterly", "Semi-annual", "Annual"])
    name = f"{pick(FIRST)} {pick(LAST)}"
    wills = (
        ["DIFC Wills Service Centre will", "ADGM will", "No will"]
        if residency == "UAE" and nationality not in ("Emirati",)
        else ["UAE onshore will", "No will"]
        if residency == "UAE"
        else ["Will in another country", "No will"]
    )
    profile = ClientProfile(
        reference=f"DEMO-{seed}",
        name=name,
        age=age,
        dob=dob,
        classification="Professional" if professional else "Retail",
        classification_reason=(
            f"Net assets of USD {net_assets:,.0f} (over USD 1m) and {years_investing:g} years' "
            "experience; the client asked to be treated as professional, in writing."
            if professional
            else "Retail by default: full retail protections apply."
        ),
        residency=residency,
        booking_location=pick(bookings),
        nationality=nationality,
        tax_residencies=[residency],
        us_person=False,
        marital_status="Married" if married else pick(["Single", "Divorced"]),
        dependants=dependants,
        dependant_ages=ages,
        occupation=job,
        employer=employer,
        employment_type=employment,
        retirement_age=retirement_age,
        reporting_currency=pick(["USD", home_ccy]),
        source_of_wealth=WEALTH[employment].format(
            years=max(age - 24, 5), job=job.lower(), job_lower=job.lower()
        ),
        source_of_funds="Transfer from the client's own bank account in "
        f"{residency}, from the savings described above.",
        pep=False,
        expected_activity=pick(["Low (a few trades a year)", "Medium (about monthly)"]),
        income=income,
        annual_expenses_usd=expenses,
        assets=assets,
        liabilities=liabilities,
        primary_objective=objective,
        horizon_years=horizon,
        goals=goals,
        portfolio_value_usd=value,
        liquidity_needs=needs,
        annual_contribution_usd=0.0 if retired else _round((total_income - expenses) * 0.3),
        annual_income_need_usd=_round(value * rng.uniform(0.02, 0.04))
        if objective == "Income"
        else 0.0,  # fmt: skip
        answers=answers,
        max_acceptable_drawdown_pct=float(loss),
        years_investing=years_investing,
        products_used=products,
        trading_frequency=pick(["Yearly", "Quarterly", "Monthly"]),
        background=pick(["", "Finance degree", "Former banker", "CFA Level I"]),
        sharia_only=sharia,
        home_bias_gcc_pct=float(pick([0, 10, 20])) or None,
        region_minimums=region_minimums,
        excluded_sectors=["Tobacco"] if rng.random() < 0.3 else [],
        home_currency_preference=home_ccy if rng.random() < 0.4 else "",
        will_status=pick(wills),
        beneficiaries_noted=bool(rng.random() < 0.7),
        structures=["Foundation"] if value > 4_000_000 and rng.random() < 0.4 else [],
        service_type=pick(["Advisory", "Discretionary"]),
        review_frequency=review,
        next_review=next_review(review, today),
        fee_basis=f"{pick([0.6, 0.75, 0.9, 1.0]):.2f}% a year of assets under advice",
        adviser_name=pick(ADVISERS),
        adviser_licence=f"DEMO-{int(rng.integers(10_000, 99_999))}",
        demo=True,
    )
    return DemoClient(profile, demo_holdings(rng, value, int(np.clip(base, 1, 5)), sharia), seed)


def demo_holdings(
    rng: np.random.Generator,
    value: float,
    score: int,
    sharia: bool,
    pools: Mapping[str, Sequence[str]] | None = None,
) -> list[dict]:
    """5-8 real tickers with USD amounts adding up to ``value``, tilted by the risk score.

    Each role (global equity, GCC, bonds/sukuk, gold, cash) gets its target share for the
    score (``MIX``), split over one to three tickers with random tilts; Sharia clients use
    Sharia-compliant funds.
    """
    pools = pools or (SHARIA_POOL if sharia else POOL)
    count = int(rng.integers(5, 9))
    roles = list(MIX[score])
    picks: dict[str, list[str]] = {r: [] for r in roles}
    used: set[str] = set()
    for role in roles:  # one ticker per role first
        options = [t for t in pools[role] if t not in used]
        if options:
            t = options[int(rng.integers(len(options)))]
            picks[role].append(t)
            used.add(t)
    while sum(len(v) for v in picks.values()) < count:
        role = roles[int(rng.choice(len(roles), p=[MIX[score][r] for r in roles]))]
        options = [t for t in pools[role] if t not in used]
        if not options:
            if all(not [t for t in pools[r] if t not in used] for r in roles):
                break
            continue
        t = options[int(rng.integers(len(options)))]
        picks[role].append(t)
        used.add(t)
    weights: dict[str, tuple[str, float]] = {}
    for role, tickers in picks.items():
        if not tickers:
            continue
        split = rng.dirichlet(np.ones(len(tickers)) * 3) * MIX[score][role]
        for t, w in zip(tickers, split, strict=True):
            weights[t] = (GROUP[role], float(w))
    total = sum(w for _, w in weights.values())
    rows = [
        {"ticker": t, "group": g, "value": _round(value * w / total, 100)}
        for t, (g, w) in weights.items()
    ]
    rows[0]["value"] += value - sum(r["value"] for r in rows)  # amounts add up exactly
    return rows
