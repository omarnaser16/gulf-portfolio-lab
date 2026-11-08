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

The portfolio follows the client's IPS by construction (tested over many seeds): 5-10 real
tickers from the verified investable universe; the mix of equity and defensive holdings is set
by the client's own computed risk score; no holding is above the score's per-holding cap;
GCC holdings meet any home-market minimum, and a region minimum is set only when a holding
in that market covers it; cash needs due within 36 months are held in cash-like funds; and a
Sharia-only client holds only screened compliant funds. The curated universe has one sukuk
fund and no Sharia cash fund, so a cautious Sharia portfolio can't meet the per-holding cap
(see the Riyadh example client): Sharia demo clients are therefore growth investors (score 4+).
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np

from gpl.client import (
    ClientProfile,
    age_on,
    month_after,
    months_until,
    risk_profile,
    with_derived_answers,
)
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

# real, liquid tickers by role, all in the verified investable universe
# (config/investable_universe.yaml); Sharia clients use screened compliant funds only
POOL = {
    "equity": ["SPY", "EFA", "EEM", "URTH", "ACWI"],
    "gcc": ["KSA", "UAE", "QAT", "2222.SR", "1120.SR", "EMAAR.AE", "QNBK.QA", "NBK.KW"],
    "bonds": ["AGG", "IEF", "BNDX", "TLT"],
    "gold": ["GLD"],
    "cash": ["BIL", "SHV", "SHY"],
}
SHARIA_POOL = {
    "equity": ["SPUS", "HLAL", "ISWD.L", "ISDE.L"],  # UMMA left out: it starts in 2022
    "gcc": ["2222.SR", "1120.SR"],
    "sukuk": ["SPSK"],  # sukuk, and the cash-like holding for cash needs
}
LONG_BONDS = {"TLT"}  # 20+ year Treasuries (fell ~45% in 2022): growth clients only (4+)
GCC_MARKET = {  # the market each GCC holding gives exposure to
    "KSA": "Saudi Arabia", "2222.SR": "Saudi Arabia", "1120.SR": "Saudi Arabia",
    "UAE": "UAE", "EMAAR.AE": "UAE", "QAT": "Qatar", "QNBK.QA": "Qatar", "NBK.KW": "Kuwait",
}  # fmt: skip
NON_PEGGED = {"NBK.KW"}  # KWD follows a basket, not the dollar: kept small (IPS max 20%)
NON_PEGGED_CAP = 0.15
GROUP = {
    "equity": "Global equity",
    "gcc": "GCC",
    "bonds": "Bonds",
    "sukuk": "Bonds",
    "gold": "Gold",
    "cash": "Cash & short-term",
}
# target weights by role for each risk score (rows sum to 1)
MIX = {
    1: {"equity": 0.12, "gcc": 0.08, "bonds": 0.30, "gold": 0.10, "cash": 0.40},
    2: {"equity": 0.20, "gcc": 0.12, "bonds": 0.33, "gold": 0.10, "cash": 0.25},
    3: {"equity": 0.30, "gcc": 0.17, "bonds": 0.30, "gold": 0.10, "cash": 0.13},
    4: {"equity": 0.45, "gcc": 0.25, "bonds": 0.15, "gold": 0.08, "cash": 0.07},
    5: {"equity": 0.55, "gcc": 0.28, "bonds": 0.07, "gold": 0.05, "cash": 0.05},
}
# equity share (global + GCC) that fits each risk score: the test range for the mix
EQUITY_RANGE = {1: (0.10, 0.30), 2: (0.25, 0.45), 3: (0.40, 0.60), 4: (0.60, 0.80),
                5: (0.70, 0.95)}  # fmt: skip
# per-holding cap and largest acceptable fall by risk score (config risk_bands; tested equal)
CAP = {1: 0.20, 2: 0.25, 3: 0.30, 4: 0.35, 5: 0.40}
LOSS = {1: 10, 2: 15, 3: 20, 4: 30, 5: 40}
SHARIA_MIN_SCORE = 4
CASH_WINDOW_MONTHS = 36  # config cash_need_window_months
MARGIN = 0.005  # headroom under each cap for rounding amounts to USD 100


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
                "amount_usd": _round(value * rng.uniform(0.02, 0.06)),
                "date": month_after(today, months),
            }
        )
    needs.sort(key=lambda n: n["date"])

    wants_sharia = rng.random() < (0.6 if residency == "Saudi Arabia" else 0.3)
    wants_region_minimum = rng.random() < 0.3
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
        years_investing=years_investing,
        products_used=products,
        trading_frequency=pick(["Yearly", "Quarterly", "Monthly"]),
        background=pick(["", "Finance degree", "Former banker", "CFA Level I"]),
        home_bias_gcc_pct=float(pick([0, 10, 20])) or None,  # fitted to the holdings below
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
    # the portfolio follows the client's own risk score and IPS
    score = risk_profile(with_derived_answers(profile, today)).score
    sharia = wants_sharia and score >= SHARIA_MIN_SCORE
    soon = sum(
        n["amount_usd"] for n in needs if months_until(n["date"], today) <= CASH_WINDOW_MONTHS
    )
    holdings = demo_holdings(rng, value, score, sharia, residency, soon / value)
    weights = {h["ticker"]: h["value"] / value for h in holdings}
    gcc = sum(w for t, w in weights.items() if t in GCC_MARKET)
    home_bias = profile.home_bias_gcc_pct
    if home_bias and home_bias / 100 > gcc - MARGIN:
        home_bias = 10.0 if gcc - MARGIN >= 0.10 else None
    home = sum(w for t, w in weights.items() if GCC_MARKET.get(t) == residency)
    profile = replace(
        profile,
        max_acceptable_drawdown_pct=float(LOSS[score]),
        sharia_only=sharia,
        home_bias_gcc_pct=home_bias,
        region_minimums={residency: 5.0} if wants_region_minimum and home >= 0.06 else {},
    )
    return DemoClient(profile, holdings, seed)


def _capped_split(share: float, caps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Split ``share`` over holdings at random, none above its cap (needs caps.sum() >= share)."""
    w = rng.dirichlet(np.ones(len(caps)) * 3) * share
    for _ in range(len(caps)):
        over = w > caps
        if not over.any():
            break
        excess = float((w[over] - caps[over]).sum())
        w[over] = caps[over]
        room = w < caps
        w[room] += excess * w[room] / w[room].sum()
    return w


def role_mix(score: int, sharia: bool, cash_need: float) -> dict[str, float]:
    """Target weight per role: ``MIX`` for the score, with cash needs due soon held in cash.

    The extra cash comes from bonds, then gold, so the equity share still fits the score.
    Sharia: bonds, gold and cash become sukuk (no gold or cash fund is screened compliant).
    """
    mix = dict(MIX[score])
    need = cash_need + MARGIN if cash_need > 0 else 0.0
    if sharia:
        sukuk = max(mix.pop("bonds") + mix.pop("gold") + mix.pop("cash"), need)
        mix["equity"] -= sukuk - (1 - mix["equity"] - mix["gcc"])
        return {**mix, "sukuk": sukuk}
    extra = max(need - mix["cash"], 0.0)
    for role in ("bonds", "gold"):
        take = min(extra, mix[role])
        mix[role] -= take
        mix["cash"] += take
        extra -= take
    return mix


def demo_holdings(
    rng: np.random.Generator,
    value: float,
    score: int,
    sharia: bool,
    residency: str | None = None,
    cash_need: float = 0.0,
) -> list[dict]:
    """5-10 real tickers with USD amounts adding up to ``value``, following the IPS.

    Each role (global equity, GCC, bonds or sukuk, gold, cash) gets its share for the score
    (``role_mix``), split over enough tickers that none is above the score's cap (``CAP``,
    less a small margin); a GCC client's own market comes first among the GCC holdings.
    """
    pools = SHARIA_POOL if sharia else POOL
    mix = {r: w for r, w in role_mix(score, sharia, cash_need).items() if w > 1e-9}
    cap = CAP[score] - MARGIN

    def ticker_cap(t: str) -> float:
        return min(cap, NON_PEGGED_CAP) if t in NON_PEGGED else cap

    options = {}
    for role in mix:
        pool = [t for t in pools[role] if not (score <= 3 and t in LONG_BONDS)]
        order = [pool[i] for i in rng.permutation(len(pool))]
        if role == "gcc":  # home market first
            order.sort(key=lambda t: GCC_MARKET.get(t) != residency)
        options[role] = order
    picks: dict[str, list[str]] = {r: [] for r in mix}
    for role, share in mix.items():  # enough tickers for the cap
        while not picks[role] or sum(map(ticker_cap, picks[role])) < share:
            picks[role].append(options[role].pop(0))
    count = int(rng.integers(5, 11))
    while sum(map(len, picks.values())) < count:
        open_roles = [r for r in mix if options[r]]
        if not open_roles:
            break
        p = np.array([mix[r] for r in open_roles])
        role = open_roles[int(rng.choice(len(open_roles), p=p / p.sum()))]
        picks[role].append(options[role].pop(0))
    weights: dict[str, tuple[str, float]] = {}
    for role, tickers in picks.items():
        caps = np.array([ticker_cap(t) for t in tickers])
        for t, w in zip(tickers, _capped_split(mix[role], caps, rng), strict=True):
            weights[t] = (GROUP[role], float(w))
    rows = [{"ticker": t, "group": g, "value": _round(value * w, 100)} for t, (g, w) in
            weights.items()]  # fmt: skip
    rows[0]["value"] += value - sum(r["value"] for r in rows)  # amounts add up exactly
    return rows
