"""Tests for the PDF documents (uncompressed in tests so their text can be searched)."""

from datetime import date

from gpl.client import EXAMPLE_CLIENTS, ClientProfile, build_ips
from gpl.data import load_universe
from gpl.report import FOOTER, esc, ips_pdf

CFG = load_universe()


def _text(pdf: bytes) -> str:
    return pdf.decode("latin-1")


def test_ips_pdf_contains_rrttllu_limits_and_footer() -> None:
    p = ClientProfile.from_dict(EXAMPLE_CLIENTS["Riyadh, conservative, 5 years, Sharia"])
    ips = build_ips(p, CFG, today=date(2026, 10, 25))
    pdf = ips_pdf(ips, compress=False)
    assert pdf.startswith(b"%PDF")
    text = _text(pdf)
    for heading in [
        "Return",
        "Risk",
        "Time horizon",
        "Taxes",
        "Liquidity",
        "Legal and regulatory",
        "Unique circumstances",
    ]:
        assert f"({heading})" in text, heading
    assert "(9.00%)" in text and "(15.00%)" in text  # band 2 limits
    assert "2026-10-25" in text and "EX-RUH" in text
    assert FOOTER[:40] in text
    assert len(ips_pdf(ips)) < len(pdf)  # compressed by default


def test_markup_is_escaped() -> None:
    assert esc("A & B <c>") == "A &amp; B &lt;c&gt;"
    p = ClientProfile(reference="R&D <1>", horizon_years=5)
    assert ips_pdf(build_ips(p, CFG)).startswith(b"%PDF")  # no markup error


# ------------------------------------------------------------------------- client report

import pandas as pd  # noqa: E402

from gpl.agent.charts import chart_data  # noqa: E402
from gpl.agent.tools import AnalystContext  # noqa: E402
from gpl.data import BenchmarkData  # noqa: E402
from gpl.domicile import domicile_report, format_domicile  # noqa: E402
from gpl.metrics import kpi_table  # noqa: E402
from gpl.report import ReportInputs, client_report, kpi_rows, markdown_flowables  # noqa: E402
from gpl.suitability import check_suitability  # noqa: E402

CURRENT = {"SPY": 0.5, "AGG": 0.3, "GLD": 0.1, "KSA": 0.1}


def _inputs(
    ctx, memo: str | None = "### Summary\n- **SPY** dominates risk [tool]."
) -> ReportInputs:
    rets = ctx.all_returns
    bench = BenchmarkData("60/40", "test", 0.6 * rets["SPY"] + 0.4 * rets["AGG"], False, [])
    c = AnalystContext(
        prices=ctx.prices,
        rf=ctx.rf,
        groups=ctx.groups,
        benchmarks=ctx.benchmarks,
        regime_features=ctx.regime_features,
        regime_labels=ctx.regime_labels,
        settings={**ctx.settings, "stress_episodes": CFG["stress_episodes"]},
        user_portfolio=CURRENT,
        benchmark=bench,
        holding_groups={"SPY": "Global equity", "AGG": "Bonds", "GLD": "Gold", "KSA": "GCC"},
    )
    profile = ClientProfile.from_dict(EXAMPLE_CLIENTS["Dubai, moderate, 10 years"])
    ips = build_ips(profile, CFG, today=date(2026, 10, 25))
    growth = chart_data(c, "growth_of_1")
    port = growth.frames["values"]["Current"].pct_change().dropna()
    checks = check_suitability(
        pd.Series(CURRENT),
        rets,
        ips,
        CFG["security_attributes"],
        {},
        c.holding_groups,
        252,
        CFG["stress_episodes"],
        500_000,
    )
    return ReportInputs(
        report_date=date(2026, 10, 25),
        data_as_of="22 Oct 2026",
        benchmark="60/40",
        holdings=pd.DataFrame(
            {
                "Value (USD)": ["250,000", "150,000", "50,000", "50,000"],
                "Weight": ["50.00%", "30.00%", "10.00%", "10.00%"],
            },
            index=list(CURRENT),
        ),
        value=500_000,
        allocation=pd.Series(CURRENT),
        client=profile,
        ips=ips,
        kpis=kpi_table(port, bench.returns, ctx.rf, 252),
        charts=[growth, chart_data(c, "drawdown_comparison"), chart_data(c, "stress_test")],
        suitability={"Current": checks},
        memo=memo,
        memo_unverified=["12.34%"],
        domicile=format_domicile(domicile_report(pd.Series(CURRENT), 500_000, CFG, False)),
    )


def test_client_report_sections_figures_and_date(ctx) -> None:
    inputs = _inputs(ctx)
    pdf = client_report(inputs, compress=False)
    assert pdf.startswith(b"%PDF")
    text = _text(pdf)
    for section in [
        "Your portfolio review",
        "In brief",
        "Your goals and limits",
        "What you hold today",
        "How it has done",
        "What we recommend",
        "Proposed trades",
        "Does it fit your investment policy?",
        "Your adviser's commentary",
        "Sign-off",
        "Important information",
        "Appendix: Investment Policy Statement",
    ]:
        assert f"({section})" in text, section
    assert "25 October 2026" in text and "EX-DXB" in text
    assert "(250,000)" in text and "Total: USD 500,000." in text  # holdings in USD
    # every KPI figure is the tested kpi_table output, formatted
    for row in kpi_rows(inputs.kpis, "60/40")[1:]:
        for cell in row[1:]:
            assert f"({cell})" in text, cell
    # every suitability result appears with its measured value
    for check in inputs.suitability["Current"]:  # narrow cells may wrap: check word by word
        words = [w for w in check.measured.split() if w != "–"]  # the dash is byte-escaped
        assert all(word in text for word in words), check.rule
    assert "[tool]" not in text  # tool citations are stripped from the memo
    assert text.count("/Subtype /Image") >= 4  # allocation, growth, drawdown, stress
    assert "domiciled \\(information, not tax advice\\))" in text and "CSPX.L" in text


def test_report_without_memo_or_client_says_so(ctx) -> None:
    inputs = _inputs(ctx, memo=None)
    inputs.client, inputs.ips, inputs.suitability = None, None, None
    text = _text(client_report(inputs, compress=False))
    assert "(No AI memo was generated for this report.)" in text
    assert "No client profile was entered" in text
    assert "No optimisation was run for this review" in text


def test_memo_markdown() -> None:
    flow = markdown_flowables("### Risk\n- **SPY** is 79.35% of risk [tool]\nPlain <text> & more")
    assert [type(f).__name__ for f in flow] == ["Paragraph"] * 3
    assert flow[1].bulletText == "•" and "<b>SPY</b>" in flow[1].text
    assert "&lt;text&gt; &amp; more" in flow[2].text
    md = (
        "# Summary\n* star bullet\n• dot bullet\n  - nested\n1. first\n2) second\n"
        "*italic* and _also_ and `code`\n\n| Metric | Value |\n|---|---:|\n| Sharpe | 0.70 |"
    )
    flow = markdown_flowables(md)
    kinds = [type(f).__name__ for f in flow]
    assert kinds == ["Paragraph"] * 7 + ["Table", "Spacer"]
    assert [f.bulletText for f in flow[1:6]] == ["•", "•", "•", "1.", "2."]
    assert flow[3].style.name == "bullet2"  # indented bullet
    assert flow[6].text == "<i>italic</i> and <i>also</i> and code"
    assert len(flow[7]._cellvalues) == 2  # header + one row; the separator is dropped


def test_clean_memo_strips_tool_tags_and_the_app_disclaimer() -> None:
    from gpl.report import clean_memo

    memo = (
        "Volatility 9.55% [diagnose_portfolio]. Fell 21.67% [stress_test, run_candidates] "
        "and [show_chart; what_if_portfolio]. See [the guide](https://x.y).\n\n---\n"
        "*Not investment advice: AI-generated.*"
    )
    assert clean_memo(memo) == ("Volatility 9.55%. Fell 21.67% and. See [the guide](https://x.y).")


def test_recommendation_and_trade_sections() -> None:
    from gpl.report import recommendation_section, trade_section

    latest = pd.DataFrame(
        {"SPY": [0.5, 0.4], "BIL": [0.0, 0.2], "AGG": [0.5, 0.4]},
        index=["Current", "Min variance"],
    )
    row = {"cagr": "6.00%", "annual_volatility": "8.00%", "sharpe": "0.50", "max_drawdown": "-12%"}
    report = {
        "goal": "Reduce risk",
        "backtest_period": "2016-01-01 to 2026-09-30",
        "walk_forward_after_costs": {"Current": row, "Min variance": row, "Benchmark": row},
        "recommendation": {
            "choice": "Min variance",
            "status": "recommended",
            "reasons": ["Min variance ranks first."],
        },
        "suitability_gate": {"failed_rules": {"Current": ["Liquidity: ..."], "Min variance": []}},
    }
    rec = recommendation_section(report, latest)
    assert rec["headline"] == "We recommend moving to the Min variance portfolio."
    assert rec["weights"][2] == ["BIL", "0.00%", "20.00%", "+20.00 pp"]
    assert rec["metrics"][0] == ["", "Today", "Min variance", "Benchmark"] and rec["failed"] == []
    closest = {
        **report,
        "recommendation": {**report["recommendation"], "status": "closest_fails"},
        "suitability_gate": {"failed_rules": {"Min variance": ["Volatility: 13%, 1 pp beyond"]}},
    }
    rec = recommendation_section(closest, latest)
    assert "not as a recommendation" in rec["headline"]
    assert rec["failed"] == ["Volatility: 13%, 1 pp beyond"]
    trades = pd.DataFrame(
        {
            "trade_usd": [-10_000.0, 9_990.0, 0.0],
            "target_usd": [40_000.0, 19_990.0, 40_000.0],
            "cost_usd": [10.0, 9.99, 0.0],
            "action": ["Sell", "Buy", "Hold"],
        },
        index=["SPY", "BIL", "AGG"],
    )
    section = trade_section(
        trades, 10, {"held": 19_990.0, "need": 25_000.0, "shortfall": 5_010.0, "ok": False}
    )
    assert [r[0] for r in section["rows"]] == ["Holding", "SPY", "BIL"]  # holds are not listed
    assert section["summary"].startswith("Sales of USD 10,000 pay for purchases of USD 9,990")
    assert "USD 5,010 short of the USD 25,000" in section["cash"]


def test_report_shows_the_recommendation_trades_bullets_and_sign_off(ctx) -> None:
    from gpl.report import recommendation_section, trade_section

    inputs = _inputs(ctx)
    latest = pd.DataFrame({t: [w, w] for t, w in CURRENT.items()}, index=["Current", "Risk parity"])
    row = {"cagr": "6.00%", "annual_volatility": "8.00%", "sharpe": "0.50", "max_drawdown": "-12%"}
    report = {
        "goal": "Reduce risk",
        "backtest_period": "2016 to 2026",
        "walk_forward_after_costs": {"Current": row, "Risk parity": row},
        "recommendation": {
            "choice": "Risk parity",
            "status": "recommended",
            "reasons": ["Risk parity ranks first under the rule."],
        },
    }
    inputs.recommendation = recommendation_section(report, latest)
    trades = pd.DataFrame(
        {"trade_usd": [-1_000.0], "target_usd": [249_000.0], "cost_usd": [1.0], "action": ["Sell"]},
        index=["SPY"],
    )
    inputs.trades = trade_section(trades, 10)
    inputs.adviser, inputs.adviser_licence = "A. Adviser", "F001234"
    text = _text(client_report(inputs, compress=False))
    assert "(We recommend moving to the Risk parity portfolio.)" in text
    assert "Risk parity ranks first under the rule." in text
    assert "(Proposed trades)" in text and "(1,000)" in text
    assert "(F001234)" in text and "(Client acknowledgement and date)" in text
    assert text.count("(\\177)") >= 5  # bullets drawn (reportlab's WinAnsi bullet, octal 177)


def test_percent_axis_ticks_never_round() -> None:
    import matplotlib.pyplot as plt

    from gpl.report import _style

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0.0, -0.11, -0.22])  # matplotlib would pick 2.5% steps here
    _style(ax, "Drawdown", percent=True)
    fig.canvas.draw()
    ticks = [t for t in ax.get_yticks() if -0.25 <= t <= 0.01]
    steps = {round(abs(b - a), 6) for a, b in zip(ticks, ticks[1:], strict=False)}
    assert steps <= {0.01, 0.02, 0.05, 0.1}  # 1-2-5 steps
    labels = [ax.yaxis.get_major_formatter()(t) for t in ticks]
    assert all(lbl.rstrip("%").lstrip("−-").replace(".", "").isdigit() for lbl in labels)
    plt.close(fig)


def test_draft_ips_is_watermarked() -> None:
    ips = build_ips(ClientProfile.from_dict(EXAMPLE_CLIENTS["Dubai, moderate, 10 years"]), CFG)
    final, draft = ips_pdf(ips, compress=False), ips_pdf(ips, compress=False, draft=True)
    assert b"DRAFT" in draft and b"DRAFT" not in final


def test_ips_pdf_has_fact_find_record_and_demo_marks() -> None:
    from dataclasses import replace

    base = ClientProfile.from_dict(EXAMPLE_CLIENTS["Dubai, moderate, 10 years"])
    p = replace(
        base,
        nationality="Emirati",
        income={"salary": 300_000},
        annual_expenses_usd=180_000,
        assets={"cash": 90_000, "investments": 600_000},
        will_status="DIFC Wills Service Centre will",
        suitability_record="- Client: risk score 3.\n- Conclusion: suitable.",
        demo=True,
    )
    text = _text(ips_pdf(build_ips(p, CFG, today=date(2026, 11, 1)), compress=False))
    for snippet in (
        "Client fact-find",
        "Emirati",
        "Savings rate",
        "40.00%",
        "DIFC Wills Service Centre will",
        "specialist legal and tax advice",
        "Suitability record",
        "Conclusion: suitable.",
        "Edited by the adviser.",
        "DEMO DATA",
        "do not enter real client data",
    ):
        assert snippet in text, snippet


def test_client_report_has_record_fact_find_and_demo_marks(ctx) -> None:
    from dataclasses import replace

    from gpl.report import client_report

    base = _inputs(ctx, memo=None)
    demo = replace(base.client, demo=True, nationality="Saudi")
    inputs = replace(
        base,
        client=demo,
        ips=build_ips(demo, CFG, today=date(2026, 10, 25)),
        suitability_record="- Client: risk score 3.\n- Conclusion: Current is not suitable.",
        record_edited=False,
    )
    text = _text(client_report(inputs, compress=False))
    for snippet in (
        "Your portfolio review \\(DEMO DATA\\)",  # PDF strings escape brackets
        "Suitability record",
        "Conclusion: Current is not suitable.",
        "Generated from the IPS rules and the suitability checks.",
        "Client fact-find",  # in the IPS appendix
        "Saudi",
        "do not enter real client data",
    ):
        assert snippet in text, snippet


def test_clean_memo_drops_chart_markers_and_keeps_citation_notes() -> None:
    from gpl.report import clean_memo

    memo = (
        "Sharpe 0.73 [get_portfolio_diagnosis; the windows differ].\n"
        "SPY dominates risk [get_portfolio_diagnosis].\n[show_chart: money_vs_risk]\nEnd."
    )
    assert clean_memo(memo) == ("Sharpe 0.73 (the windows differ).\nSPY dominates risk.\nEnd.")


def test_build_recommendation_in_the_report_has_securities_and_relative_metrics() -> None:
    from gpl.report import recommendation_section

    report = {
        "mode": "build",
        "recommendation": {
            "choice": "Max IR",
            "status": "recommended",
            "rule": "r",
            "reasons": ["Max IR ranks first."],
        },  # fmt: skip
        "walk_forward_after_costs": {
            "Current": {"cagr": "8.00%", "sharpe": "0.60"},
            "Max IR": {
                "cagr": "9.00%",
                "sharpe": "0.70",
                "excess_return_vs_benchmark": "1.00%",
                "information_ratio": "0.30",
                "hit_rate_months": "55.00%",
                "worst_relative_drawdown": "-6.00%",
            },  # fmt: skip
            "Benchmark": {"cagr": "8.50%", "sharpe": "0.65"},
        },
        "securities_chosen": {
            "SPY": {
                "weight": "30.00%",
                "name": "SPDR S&P 500 ETF",
                "asset_class": "Equity",
                "region": "United States",
                "why": "in the benchmark",
            },  # fmt: skip
        },
        "swaps": [{"why": "Sell X: duplicate of Y."}],
        "domicile_suggestions_information_not_tax_advice": [{"why": "SPY -> CSPX.L."}],
    }
    latest = pd.DataFrame({"SPY": [0.5, 0.3], "AGG": [0.5, 0.0], "QUAL": [0.0, 0.7]},
                          index=["Current", "Max IR"])  # fmt: skip
    rec = recommendation_section(report, latest)
    labels = [row[0] for row in rec["metrics"]]
    assert "Months ahead of the benchmark" in labels
    assert rec["securities"][1][:3] == ["SPY", "SPDR S&P 500 ETF (Equity, United States)",
                                         "30.00%"]  # fmt: skip
    assert rec["swaps"] == ["Sell X: duplicate of Y."] and rec["domicile"] == ["SPY -> CSPX.L."]
    next_best = {**report, "recommendation": {**report["recommendation"], "status": "next_best"}}
    assert "does not aim to beat" in recommendation_section(next_best, latest)["headline"]
