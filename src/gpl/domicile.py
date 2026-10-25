"""GCC domicile check: US-situs holdings for non-US clients and Irish-domiciled alternatives.

Pure functions: no I/O, no printing, no plotting. INFORMATION, NOT TAX ADVICE.

A non-US person who holds US-situs assets (US-listed funds and stocks, e.g. SPY or NVDA) can
owe US estate tax on them above a 60,000 USD exemption, at graduated rates of 18% to 40%, and
pays 30% US withholding on their dividends. Irish-domiciled UCITS funds holding the same
exposure are generally not US-situs, and pay 15% on US dividends at fund level under the
US-Ireland treaty. The estimate below applies the statutory rate schedule and the non-resident
unified credit (13,000 USD, equivalent to the 60,000 exemption) to the US-situs value, as if
the client died today with no treaty, deductions or other assets: an illustration of the
exposure only. Rates and thresholds change; a tax adviser must confirm.
"""

from collections.abc import Mapping

import pandas as pd

# US estate and gift tax rate schedule (taxable amount over -> base tax, marginal rate).
US_ESTATE_SCHEDULE = [
    (0, 0, 0.18),
    (10_000, 1_800, 0.20),
    (20_000, 3_800, 0.22),
    (40_000, 8_200, 0.24),
    (60_000, 13_000, 0.26),
    (80_000, 18_200, 0.28),
    (100_000, 23_800, 0.30),
    (150_000, 38_800, 0.32),
    (250_000, 70_800, 0.34),
    (500_000, 155_800, 0.37),
    (750_000, 248_300, 0.39),
    (1_000_000, 345_800, 0.40),
]
LABEL = (
    "Information, not tax advice. Illustrative US estate tax and withholding for a non-US "
    "person, using {year} rates and thresholds; treaties, deductions and other US assets can "
    "change the result. Confirm with a tax adviser."
)


def tentative_tax(amount: float) -> float:
    """Tax on ``amount`` under the graduated schedule (before any credit)."""
    floor, base, rate = next(b for b in reversed(US_ESTATE_SCHEDULE) if amount >= b[0])
    return base + rate * (amount - floor)


def illustrative_us_estate_tax(us_situs_value: float, unified_credit: float = 13_000) -> float:
    """Tentative tax on the US-situs value minus the non-resident unified credit (min 0)."""
    return max(0.0, tentative_tax(max(us_situs_value, 0.0)) - unified_credit)


def is_us_situs(ticker: str, overrides: Mapping[str, str]) -> bool:
    """US-listed tickers (no exchange suffix) are treated as US-situs unless overridden."""
    if ticker in overrides:
        return overrides[ticker] == "US"
    return "." not in ticker and "=" not in ticker and not ticker.startswith("^")


def domicile_report(weights: pd.Series, value: float | None, cfg: Mapping, us_person: bool) -> dict:
    """US-situs holdings, illustrative estate-tax exposure and Irish alternatives."""
    dom, mapping = cfg["domicile"], cfg.get("ucits_equivalents", {})
    label = LABEL.format(year=dom["rates_as_of"])
    if us_person:
        return {
            "applies": False,
            "label": label,
            "reason": "The client is a US person: US estate tax applies to worldwide assets "
            "under different rules, so this non-resident check does not apply.",
        }
    w = weights[weights > 0]
    rows = []
    for t, wt in w.items():
        us = is_us_situs(t, dom.get("overrides", {}))
        rows.append(
            {
                "ticker": t,
                "weight": float(wt),
                "value_usd": None if value is None else float(wt * value),
                "us_situs": us,
                "alternative": dict(mapping[t]) if us and t in mapping else None,
            }
        )
    share = float(sum(r["weight"] for r in rows if r["us_situs"]))
    us_value = None if value is None else share * value
    tax = (
        None
        if us_value is None
        else illustrative_us_estate_tax(us_value, dom["nra_unified_credit_usd"])
    )
    return {
        "applies": True,
        "label": label,
        "holdings": rows,
        "us_situs_share": share,
        "us_situs_value_usd": us_value,
        "exemption_usd": dom["us_estate_exemption_usd"],
        "illustrative_estate_tax_usd": tax,
        "withholding": {
            "us_domiciled_non_us_person": dom["withholding_us_fund_non_us_person"],
            "irish_fund_on_us_dividends": dom["withholding_irish_fund_us_dividends"],
        },
        "no_alternative": [r["ticker"] for r in rows if r["us_situs"] and not r["alternative"]],
    }


def format_domicile(report: dict) -> dict:
    """Formatted table rows and summary sentences (shared by the app, the AI tool and the PDF)."""
    if not report["applies"]:
        return {"label": report["label"], "summary": [report["reason"]], "rows": []}
    rows = []
    for h in report["holdings"]:
        alt = h["alternative"]
        rows.append(
            {
                "Ticker": h["ticker"],
                "Weight": f"{h['weight']:.2%}",
                "Value (USD)": "–" if h["value_usd"] is None else f"{h['value_usd']:,.0f}",
                "US-situs": "yes" if h["us_situs"] else "no",
                "Irish-domiciled alternative": (
                    f"{alt['ticker']}: {alt['name']} ({alt['match']})"
                    if alt
                    else ("no verified alternative" if h["us_situs"] else "–")
                ),
            }
        )
    wh = report["withholding"]
    summary = [
        f"US-situs holdings are {report['us_situs_share']:.2%} of the portfolio"
        + (
            ""
            if report["us_situs_value_usd"] is None
            else f" (USD {report['us_situs_value_usd']:,.0f})"
        )
        + f"; the US estate-tax exemption for non-residents is USD {report['exemption_usd']:,.0f}.",
        "Enter the portfolio value to estimate the estate-tax exposure."
        if report["illustrative_estate_tax_usd"] is None
        else f"Illustrative US estate tax if the client died today: USD "
        f"{report['illustrative_estate_tax_usd']:,.0f} (graduated 18%-40% schedule minus the "
        "non-resident credit).",
        f"Dividends from US-domiciled holdings: {wh['us_domiciled_non_us_person']:.0%} US "
        f"withholding for a non-US person; Irish funds pay {wh['irish_fund_on_us_dividends']:.0%} "
        "on US dividends at fund level.",
    ]
    if report["no_alternative"]:
        summary.append(
            "No verified Irish-domiciled alternative for: "
            + ", ".join(report["no_alternative"])
            + " (single US stocks have none; holding them directly stays US-situs)."
        )
    summary.append("Before switching, compare costs, liquidity, tracking and share class.")
    return {"label": report["label"], "summary": summary, "rows": rows}
