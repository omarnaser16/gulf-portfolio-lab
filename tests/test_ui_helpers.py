"""Pure presentation helpers of the redesigned app: one-line summaries, IPS rows, KPI cards."""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import client_ui  # noqa: E402
import layout  # noqa: E402
import workstation as ws  # noqa: E402

from gpl.client import build_ips  # noqa: E402
from gpl.data import load_universe  # noqa: E402
from gpl.demo import demo_client  # noqa: E402

TODAY = date(2026, 11, 4)
CFG = load_universe()


def test_section_summaries_are_one_line_and_built_from_the_facts() -> None:
    p = demo_client(2, TODAY).profile
    lines = {s: client_ui.section_summary(s, p) for s in (
        "household", "kyc", "financial", "objectives", "risk", "knowledge", "constraints",
        "estate", "service",
    )}  # fmt: skip
    assert all(lines.values()) and all("\n" not in v for v in lines.values())
    assert lines["household"].startswith(f"{p.name} · age {p.age}")
    assert lines["kyc"].startswith(f"{p.classification} client · not a PEP")
    assert "Net worth USD" in lines["financial"] and "savings rate" in lines["financial"]
    assert lines["risk"].startswith("Questionnaire complete")
    assert lines["constraints"].startswith("Optional")
    assert f"adviser {p.adviser_name}" in lines["service"]
    assert client_ui.section_summary("unknown", p) == ""


def test_ips_rows_follow_the_ips_limits() -> None:
    p = demo_client(2, TODAY).profile
    ips = build_ips(p, CFG, TODAY)
    rows = dict(client_ui.ips_rows(ips))
    assert list(rows) == ["Return", "Risk", "Time", "Taxes", "Liquidity", "Legal", "Unique"]
    lim = ips.limits
    assert rows["Risk"] == (
        f"Volatility ≤ {lim['max_volatility'] * 100:g}%, drawdown ≤ "
        f"{lim['max_drawdown'] * 100:g}%, ≤ {lim['max_weight'] * 100:g}% in one holding"
    )
    assert rows["Time"].startswith(f"{p.horizon_years:g} years")
    assert "non-USD-pegged currencies ≤ 20%" in rows["Unique"]
    assert client_ui._usd(1_310_000) == "USD 1.31m" and client_ui._usd(21_000) == "USD 21k"


def test_key_finding_names_the_holding_that_dominates_risk() -> None:
    w = pd.Series({"EEM": 0.2647, "AGG": 0.2269, "BIL": 0.13})
    rc = pd.Series({"EEM": 0.575, "AGG": 0.0485, "BIL": 0.0})
    assert ws.key_finding(w, rc) == "EEM is 26% of the money but 57% of the risk."
    even = pd.Series({"A": 0.5, "B": 0.5})
    assert ws.key_finding(even, even) == (
        "Each holding's share of risk is close to its share of money."
    )


def test_kpi_card_and_pill_html() -> None:
    card = layout.kpi_card("Sharpe ratio", "0.52", delta="↑ 0.19", tone="pass", before="0.33")
    assert '<span class="gpl-before">0.33</span>' in card and ">0.52</div>" in card
    assert '<span class="gpl-pill pass">↑ 0.19</span>' in card
    plain = layout.kpi_card("Portfolio value", "$360.0k", sub="7 holdings")
    assert '<div class="gpl-kpi-sub">7 holdings</div>' in plain and "gpl-pill" not in plain
    assert layout.pill("<b>", "fail") == '<span class="gpl-pill fail">&lt;b&gt;</span>'


def test_stylesheet_uses_the_design_tokens() -> None:
    from gpl import theme

    for token in (theme.BRAND, theme.BRAND_BUTTON, theme.ACCENT_BUTTON, theme.TEXT, theme.BG):
        assert token in layout.APP_CSS
    assert f"max-width: {theme.MAX_WIDTH_PX + 48}px" in layout.APP_CSS
    assert '[class*="st-key-cta_"] button' in layout.APP_CSS  # the one pink action


def test_answer_sentence_switch_keep_and_closest() -> None:
    rows = {
        "Current": {"cagr": "5.04%", "annual_volatility": "8.59%"},
        "Max Sharpe": {"cagr": "6.67%", "annual_volatility": "8.33%"},
    }

    def report(choice, status):
        return {"recommendation": {"choice": choice, "status": status},
                "walk_forward_after_costs": rows}  # fmt: skip

    assert ws.answer_sentence(report("Max Sharpe", "recommended")) == (
        "Switch to Max Sharpe: return 5.04% to 6.67%, volatility 8.59% to 8.33%."
    )
    assert ws.answer_sentence(report("Keep your current portfolio", "keep")) == (
        "Keep your current portfolio: no method beat it after costs."
    )
    assert ws.answer_sentence(report("Max Sharpe", "next_best")).startswith("Next best: switch")
    part = report("Max Sharpe", "compliant_move")
    part["max_turnover"] = "30.00%"
    part["recommendation"]["full_move_not_recommended_because"] = [
        "the full portfolio fails the IPS over the full history (Volatility)"
    ]
    assert ws.answer_sentence(part) == (
        "Move part-way towards Max Sharpe (within the 30.00% turnover limit): the full Max "
        "Sharpe portfolio isn't recommended because the full portfolio fails the IPS over the "
        "full history (Volatility)."
    )
    assert ws.answer_sentence(report("Keep your current portfolio", "closest_fails")).startswith(
        "No portfolio meets every IPS rule: your current portfolio is the closest"
    )


def test_money_has_separators_and_never_a_negative_zero() -> None:
    from formatting import money

    assert money(1_269_900) == "USD 1,269,900"
    assert money(-0.4) == "USD 0" and money(-0.0) == "USD 0"  # never "USD -0"
    assert money(107.981, 2) == "USD 107.98"
    assert money(None) == "–" and money(float("nan")) == "–"


def test_finance_cards_are_compact_with_the_exact_value_in_the_tooltip() -> None:
    from gpl.factfind import Finances

    fin = Finances(300_000, 180_000, 18_000_000, 52_000, 17_948_000, 7_483_000, 0.4, 6.0)
    assert client_ui.finance_cards(fin) == [
        ("Net worth", "$17.95m", "USD 17,948,000"),
        ("Investable assets", "$7.48m", "USD 7,483,000"),
        ("Savings rate", "40.00%", None),
        ("Emergency fund", "6.0 months", None),
    ]
    empty = Finances(0, None, 0, 0, 0, 0, None, None)
    assert [v for _, v, _ in client_ui.finance_cards(empty)] == ["$0", "$0", "–", "–"]
    assert all(len(v) <= 10 for _, v, _ in client_ui.finance_cards(fin))  # fits a card


def test_money_echo_adds_thousands_separators_under_an_input() -> None:
    shown = []

    class Col:
        def caption(self, text: str) -> None:
            shown.append(text)

    client_ui.money_echo(Col(), 7_393_000.0)
    client_ui.money_echo(Col(), None)
    client_ui.money_echo(Col(), 0.0)
    assert shown == ["$7,393,000"]
