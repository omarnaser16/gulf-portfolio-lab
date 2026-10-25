"""Tests for the GCC domicile check (information, not tax advice), with hand-worked tax."""

import pandas as pd
import pytest

from gpl.data import load_universe
from gpl.domicile import domicile_report, illustrative_us_estate_tax, is_us_situs, tentative_tax

CFG = load_universe()
DOM = CFG["domicile"]


@pytest.mark.parametrize(
    ("estate", "tax"),
    [
        (0, 0),
        (30_000, 0),  # tentative 6,000, below the 13,000 credit
        (60_000, 0),  # tentative 13,000 = the credit: the 60,000 exemption
        (100_000, 10_800),  # 23,800 - 13,000
        (600_000, 179_800),  # 155,800 + 37% x 100,000 - 13,000
        (1_000_000, 332_800),  # 345,800 - 13,000
        (2_000_000, 732_800),  # 345,800 + 40% x 1,000,000 - 13,000
    ],
)
def test_us_estate_tax_schedule(estate, tax) -> None:
    assert illustrative_us_estate_tax(estate, DOM["nra_unified_credit_usd"]) == pytest.approx(tax)


def test_tentative_tax_brackets() -> None:
    assert tentative_tax(10_000) == pytest.approx(1_800)
    assert tentative_tax(250_000) == pytest.approx(70_800)


def test_us_situs_by_listing_and_override() -> None:
    assert is_us_situs("SPY", {}) and is_us_situs("NVDA", {}) and is_us_situs("KSA", {})
    assert not is_us_situs("2222.SR", {}) and not is_us_situs("CSPX.L", {})
    assert not is_us_situs("SPY", {"SPY": "Ireland"})


def test_domicile_report_for_a_gcc_client() -> None:
    weights = pd.Series({"SPY": 0.5, "KSA": 0.1, "2222.SR": 0.3, "NVDA": 0.1})
    r = domicile_report(weights, 1_000_000, CFG, us_person=False)
    assert r["applies"] and r["us_situs_value_usd"] == pytest.approx(700_000)
    # 155,800 + 37% x 200,000 = 229,800 tentative, minus 13,000
    assert r["illustrative_estate_tax_usd"] == pytest.approx(216_800)
    rows = {h["ticker"]: h for h in r["holdings"]}
    assert rows["SPY"]["alternative"]["ticker"] == "CSPX.L"
    assert rows["KSA"]["alternative"]["match"].startswith("similar exposure")
    assert rows["NVDA"]["alternative"] is None and rows["NVDA"]["us_situs"]
    assert not rows["2222.SR"]["us_situs"]
    assert "not tax advice" in r["label"].lower()


def test_not_applicable_to_us_persons_or_without_value() -> None:
    w = pd.Series({"SPY": 1.0})
    assert not domicile_report(w, 1_000_000, CFG, us_person=True)["applies"]
    r = domicile_report(w, None, CFG, us_person=False)
    assert r["applies"] and r["illustrative_estate_tax_usd"] is None
    assert r["us_situs_share"] == pytest.approx(1.0)


def test_format_domicile() -> None:
    from gpl.domicile import format_domicile

    r = domicile_report(pd.Series({"SPY": 0.5, "NVDA": 0.2, "2222.SR": 0.3}), 1_000_000, CFG, False)
    f = format_domicile(r)
    spy = next(row for row in f["rows"] if row["Ticker"] == "SPY")
    assert spy["Irish-domiciled alternative"].startswith("CSPX.L: iShares Core S&P 500")
    assert spy["Value (USD)"] == "500,000" and spy["US-situs"] == "yes"
    # 700,000 US-situs: 155,800 + 37% x 200,000 - 13,000 = 216,800
    assert any("USD 216,800" in s for s in f["summary"])
    assert any("NVDA" in s and "No verified" in s for s in f["summary"])
    assert format_domicile(domicile_report(pd.Series({"SPY": 1.0}), 1, CFG, True))["rows"] == []
