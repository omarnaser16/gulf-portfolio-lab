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
            {"Weight": ["50.00%", "30.00%", "10.00%", "10.00%"]}, index=list(CURRENT)
        ),
        allocation=pd.Series(CURRENT),
        client=profile,
        ips=ips,
        kpis=kpi_table(port, bench.returns, ctx.rf, 252),
        charts=[growth, chart_data(c, "drawdown_comparison"), chart_data(c, "stress_test")],
        suitability={"Current": checks},
        memo=memo,
        memo_unverified=["12.34%"],
    )


def test_client_report_sections_figures_and_date(ctx) -> None:
    inputs = _inputs(ctx)
    pdf = client_report(inputs, compress=False)
    assert pdf.startswith(b"%PDF")
    text = _text(pdf)
    for section in [
        "Portfolio review",
        "Client summary",
        "Holdings",
        "Performance vs 60/40",
        "Suitability against the IPS",
        "Analyst memo",
        "Methodology and limitations",
        "Important information",
    ]:
        assert f"({section})" in text, section
    assert "2026-10-25" in text and "EX-DXB" in text
    # every KPI figure is the tested kpi_table output, formatted
    for row in kpi_rows(inputs.kpis, "60/40")[1:]:
        for cell in row[1:]:
            assert f"({cell})" in text, cell
    # every suitability result appears with its measured value
    for check in inputs.suitability["Current"]:  # narrow cells may wrap: check word by word
        words = [w for w in check.measured.split() if w != "–"]  # the dash is byte-escaped
        assert all(word in text for word in words), check.rule
    assert text.count("/Subtype /Image") >= 4  # allocation, growth, drawdown, stress


def test_report_without_memo_or_client_says_so(ctx) -> None:
    inputs = _inputs(ctx, memo=None)
    inputs.client, inputs.ips, inputs.suitability = None, None, None
    text = _text(client_report(inputs, compress=False))
    assert "(No AI memo was generated for this report.)" in text
    assert "No client profile was entered" in text


def test_memo_markdown() -> None:
    flow = markdown_flowables("### Risk\n- **SPY** is 79.35% of risk [tool]\nPlain <text> & more")
    assert [type(f).__name__ for f in flow] == ["Paragraph"] * 3
    assert flow[1].bulletText == "•" and "<b>SPY</b>" in flow[1].text
    assert "&lt;text&gt; &amp; more" in flow[2].text
