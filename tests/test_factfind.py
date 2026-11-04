"""Tests for the fact-find registry, completeness and the financial calculations."""

from datetime import date

import pytest

from gpl.client import ClientProfile, age_on
from gpl.factfind import (
    IPS_SECTIONS,
    SECTIONS,
    completeness,
    complex_product_flags,
    consistency_warnings,
    finances,
    goals_required_return,
    ips_missing,
    next_review,
    overall_completeness,
)

TODAY = date(2026, 11, 1)
ANSWERS = {
    "income_stability": 4,
    "wealth_share": 3,
    "dependence": 4,
    "liquidity_need": 4,
    "loss_reaction": 3,
    "tradeoff": 3,
    "experience": 3,
}


def full_profile(**kw) -> ClientProfile:
    """Every required field answered."""
    base = dict(
        reference="C-1",
        name="Test Client",
        age=45,
        nationality="Emirati",
        residency="UAE",
        tax_residencies=["UAE"],
        marital_status="Married",
        dependants=2,
        dependant_ages=[8, 12],
        occupation="Engineer",
        employment_type="Employed",
        retirement_age=60,
        reporting_currency="USD",
        source_of_wealth="Salary savings",
        source_of_funds="Transfer from savings account",
        pep=False,
        expected_activity="Low (a few trades a year)",
        classification="Retail",
        classification_reason="Default: retail protections apply.",
        income={"salary": 300_000},
        annual_expenses_usd=180_000,
        assets={"cash": 90_000, "investments": 1_000_000, "property": 800_000},
        liabilities={"mortgage": 300_000},
        primary_objective="Growth",
        horizon_years=15,
        portfolio_value_usd=1_000_000,
        answers=ANSWERS,
        max_acceptable_drawdown_pct=25,
        years_investing=10,
        products_used=["Equities", "ETFs"],
        trading_frequency="Quarterly",
        will_status="DIFC Wills Service Centre will",
        service_type="Advisory",
        review_frequency="Annual",
        fee_basis="1.00% a year",
        adviser_name="A. Adviser",
        adviser_licence="L-123",
    )
    return ClientProfile(**{**base, **kw})


def test_registry_names_are_profile_fields_and_unique() -> None:
    names = [f.name for s in SECTIONS for f in s.fields]
    assert len(names) == len(set(names))
    profile_fields = set(ClientProfile.__dataclass_fields__)
    assert set(names) <= profile_fields
    assert [s.title for s in SECTIONS][:2] == ["Client & household", "KYC / AML"]
    assert set(IPS_SECTIONS) <= {s.key for s in SECTIONS}


def test_completeness_counts_required_fields() -> None:
    full = completeness(full_profile(), TODAY)
    assert all(c.missing == [] for c in full.values()), {k: c.missing for k, c in full.items()}
    assert overall_completeness(full) == 1.0
    assert ips_missing(full_profile(), TODAY) == []
    blank = completeness(ClientProfile(), TODAY)
    assert "Name" in blank["household"].missing
    assert "Politically exposed person (PEP)" in blank["kyc"].missing  # None: not answered
    assert completeness(full_profile(pep=True), TODAY)["kyc"].missing == []  # yes is an answer
    assert completeness(full_profile(dependants=0), TODAY)["household"].missing == []
    assert 0 < overall_completeness(blank) < 1
    assert blank["constraints"].total == 0 and blank["constraints"].share == 1.0


def test_ips_missing_names_sections_regulator_and_override_reason() -> None:
    missing = ips_missing(full_profile(residency="France", override_score=2), TODAY)
    assert "Client & household: regulator (booking location)" in missing
    assert "Risk profile: reason for the adviser override" in missing
    assert ips_missing(full_profile(will_status=""), TODAY) == []  # estate: optional for IPS
    assert "Service & review: Adviser name" in ips_missing(full_profile(adviser_name=""), TODAY)


def test_finances_by_hand() -> None:
    f = finances(full_profile())
    assert f.income == 300_000 and f.assets == 1_890_000 and f.liabilities == 300_000
    assert f.net_worth == 1_590_000
    assert f.investable == 1_090_000  # cash + investments
    assert f.savings_rate == pytest.approx(0.40)  # (300k - 180k) / 300k
    assert f.emergency_months == pytest.approx(6.0)  # 90k / (180k / 12)
    empty = finances(ClientProfile())
    assert empty.savings_rate is None and empty.emergency_months is None


def test_goals_required_return_one_goal_closed_form() -> None:
    r = goals_required_return(
        100_000, [{"description": "Double", "amount_usd": 200_000, "years": 10}]
    )
    assert r.rate == pytest.approx(2**0.1 - 1)  # 7.18%
    assert "7.18%" in r.reason
    multiple = goals_required_return(100_000, [{"multiple": 2, "years": 10}])
    assert multiple.rate == pytest.approx(r.rate)


def test_goals_required_return_with_additions_and_cash_needs() -> None:
    # 100k now + 10k at the end of years 1 and 2 pays 121k at year 2:
    # 100(1+r)^2 + 10(1+r) + 10 = 121
    r = goals_required_return(
        100_000, [{"amount_usd": 121_000, "years": 2}], annual_contribution=10_000
    )
    g = r.rate
    assert 100_000 * (1 + g) ** 2 + 10_000 * (1 + g) + 10_000 == pytest.approx(121_000)
    # a dated cash need counts as a payment: 110k due in 12 months from 100k -> 10%
    need = [{"description": "Deposit", "amount_usd": 110_000, "date": "2027-11"}]
    assert goals_required_return(100_000, [], need, today=TODAY).rate == pytest.approx(0.10)


def test_goals_required_return_edge_cases() -> None:
    assert goals_required_return(None, [{"amount_usd": 1, "years": 1}]).rate is None
    none = goals_required_return(100_000, [{"description": "Vague"}])
    assert none.rate is None and "No goals" in none.reason
    huge = goals_required_return(100_000, [{"amount_usd": 10_000_000, "years": 2}])
    assert huge.rate is None and "can't be met" in huge.reason
    easy = goals_required_return(100_000, [{"amount_usd": 10_000, "years": 2}])
    assert easy.rate == -0.5 and "covered" in easy.reason
    skipped = goals_required_return(
        100_000, [{"amount_usd": 200_000, "years": 10}, {"description": "Yacht"}]
    )
    assert "Left out (no amount or year): Yacht" in skipped.reason


def test_complex_product_flags() -> None:
    flags = complex_product_flags(["Equities"], 10, ["Equities", "Derivatives"])
    assert len(flags) == 1 and flags[0].startswith("Derivatives: complex")
    novice = complex_product_flags([], 1, ["ETFs"])
    assert novice == [
        "ETFs: new to a client with under two years' experience. Explain how they work."
    ]
    assert complex_product_flags(["ETFs"], 5, ["ETFs"]) == []
    assert complex_product_flags([], 0) == [
        "Limited experience: explain every product recommended."
    ]


def test_consistency_warnings() -> None:
    assert consistency_warnings(full_profile(), TODAY) == []
    w = consistency_warnings(
        full_profile(
            dob="1990-01-01",
            age=45,
            retirement_age=40,
            dependants=3,
            portfolio_value_usd=5_000_000,
            annual_expenses_usd=400_000,
            goals=[{"description": "Late goal", "years": 30}],
            override_score=2,
        ),
        TODAY,
    )
    text = " ".join(w)
    assert "doesn't match the date of birth (age 36)" in text
    assert "retirement age 40 is not after age 45" in text
    assert "3 dependants but 2 ages" in text
    assert "more than the investable assets" in text
    assert "Expenses exceed income" in text
    assert "Late goal" in text and "has no reason" in text


def test_age_and_next_review() -> None:
    assert age_on("1980-11-02", TODAY) == 45 and age_on("1980-11-01", TODAY) == 46
    assert next_review("Quarterly", date(2026, 11, 30)) == "2027-02-28"
    assert next_review("Annual", TODAY) == "2027-11-01"


def test_goals_required_return_rounds_a_zero_rate_to_zero() -> None:
    # 500k + 30k a year for 10 years = 800k pays 750k + 50k at about 0%
    r = goals_required_return(
        500_000,
        [{"multiple": 1.5, "years": 10}, {"amount_usd": 50_000, "years": 10}],
        annual_contribution=30_000,
    )
    assert r.rate == 0.0 and r.reason.startswith("0.00% a year")
