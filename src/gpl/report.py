"""PDF documents for the adviser: the Investment Policy Statement and the client report.

Functions return the PDF as bytes (the app offers them as downloads); nothing is written to
disk. Every figure in a document is passed in from tested code; nothing is calculated here
beyond formatting. Every page carries the decision-support disclaimer.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from io import BytesIO

import matplotlib

matplotlib.use("Agg")  # no display: render charts to PNG in memory
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from gpl.client import IPS, QUESTIONS, ClientProfile

FOOTER = (
    "Decision support for a licensed adviser. Not investment, tax or legal advice. "
    "Figures are historical and come from tested code; they are not forecasts."
)
INK, MUTED, RULE, ACCENT = "#1a1a19", "#5f5d57", "#d9d7d0", "#b36f00"

_base = getSampleStyleSheet()
STYLES = {
    "title": ParagraphStyle(
        "title",
        parent=_base["Title"],
        fontSize=18,
        leading=22,
        textColor=INK,
        alignment=0,
        spaceAfter=4,
    ),
    "subtitle": ParagraphStyle(
        "subtitle", parent=_base["Normal"], fontSize=9, textColor=MUTED, spaceAfter=10
    ),
    "h2": ParagraphStyle(
        "h2",
        parent=_base["Heading2"],
        fontSize=12.5,
        textColor=ACCENT,
        spaceBefore=10,
        spaceAfter=4,
    ),
    "body": ParagraphStyle("body", parent=_base["Normal"], fontSize=9.5, leading=13, textColor=INK),
    "small": ParagraphStyle(
        "small", parent=_base["Normal"], fontSize=8, leading=10.5, textColor=MUTED
    ),
    "cell": ParagraphStyle("cell", parent=_base["Normal"], fontSize=8.5, leading=11, textColor=INK),
}


def esc(text: str) -> str:
    """Escape text for reportlab's mini-markup."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def para(text: str, style: str = "body") -> Paragraph:
    return Paragraph(esc(text), STYLES[style])


def table(
    rows: Sequence[Sequence[str]],
    widths: Sequence[float] | None = None,
    header: bool = True,
    row_heights: Sequence[float | None] | None = None,
) -> Table:
    """A ruled table; cells are wrapped paragraphs so long text never overflows."""
    data = [[Paragraph(esc(c), STYLES["cell"]) for c in row] for row in rows]
    t = Table(
        data,
        colWidths=widths,
        rowHeights=row_heights,
        repeatRows=1 if header else 0,
        hAlign="LEFT",
    )
    style = [
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(RULE)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1efe8")),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(MUTED)),
        ]
    t.setStyle(TableStyle(style))
    return t


def build_pdf(story: list, title: str, compress: bool = True) -> bytes:
    """Lay out ``story`` on A4 with the footer disclaimer and page numbers on every page."""
    buffer = BytesIO()

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(18 * mm, 10 * mm, FOOTER)
        canvas.drawRightString(A4[0] - 18 * mm, 14 * mm, f"{title} · page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title=title,
        author="Gulf Portfolio Lab",
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        pageCompression=1 if compress else 0,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


def ips_story(ips: IPS) -> list:
    """The IPS as flowables (reused inside the client report)."""
    lim = ips.limits

    def pct(x) -> str:
        return "not set" if x is None else f"{x:.2%}"

    story = [para("Objectives and constraints (RRTTLLU)", "h2")]
    story.append(
        table(
            [["Element", "Policy"], *[[k, v] for k, v in ips.sections.items()]],
            widths=[38 * mm, 136 * mm],
        )
    )
    story.append(para("Limits used by the optimiser and the suitability check", "h2"))
    rows = [
        ["Limit", "Value", "Source"],
        [
            "Maximum volatility (annual)",
            pct(lim["max_volatility"]),
            lim["source"]["max_volatility"],
        ],
        ["Maximum drawdown", pct(lim["max_drawdown"]), lim["source"]["max_drawdown"]],
        ["Maximum in one holding", pct(lim["max_weight"]), lim["source"]["max_weight"]],
        ["Minimum in GCC holdings", pct(lim["home_bias_gcc"]), "client"],
        ["Maximum in non-USD-pegged currencies", pct(lim["max_non_pegged_currency"]), "client"],
        [
            "Liquidity reserve",
            f"USD {lim['liquidity_reserve_usd']:,.0f}"
            + (f" ({pct(lim['liquidity_reserve'])})" if lim["liquidity_reserve"] else ""),
            "client",
        ],
        ["Sharia-compliant only", "yes" if lim["sharia_only"] else "no", "client"],
        ["Excluded sectors", ", ".join(lim["excluded_sectors"]) or "none", "client"],
    ]
    story.append(table(rows, widths=[70 * mm, 50 * mm, 54 * mm]))
    story.append(para("Risk questionnaire", "h2"))
    answers = ips.client.answers
    q_rows = [["Question", "Answer", "Score"]]
    for q in QUESTIONS:
        score = answers.get(q["id"])
        q_rows.append(
            [
                q["text"],
                q["options"][score - 1] if score else "not answered",
                str(score) if score else "–",
            ]
        )
    story.append(table(q_rows, widths=[100 * mm, 52 * mm, 22 * mm]))
    story.append(Spacer(1, 4))
    story.append(para(ips.risk.explanation, "small"))
    return story


def ips_pdf(ips: IPS, compress: bool = True) -> bytes:
    """The IPS for the adviser and client to review and sign."""
    c = ips.client
    story = [
        para("Investment Policy Statement", "title"),
        para(
            f"Client {c.reference or c.name or '(no reference)'} · prepared {ips.created} · "
            "Gulf Portfolio Lab",
            "subtitle",
        ),
        *ips_story(ips),
    ]
    if c.adviser_notes:
        story += [para("Adviser notes", "h2"), para(c.adviser_notes)]
    story.append(
        KeepTogether(
            [
                para("Agreement", "h2"),
                para(
                    "We have reviewed this policy statement and agree that it reflects the "
                    "client's objectives and constraints. It should be reviewed at least once a "
                    "year and whenever the client's circumstances change."
                ),
                Spacer(1, 18),
                table(
                    [["Client signature and date", "Adviser signature and date"], ["", ""]],
                    widths=[87 * mm, 87 * mm],
                    row_heights=[None, 22 * mm],
                ),
            ]
        )
    )
    return build_pdf(story, f"IPS {c.reference or ''} {ips.created}".strip(), compress)


# ------------------------------------------------------------------------- client report

PRINT_COLORS = ["#c98500", "#2a9fb5", "#d55181", "#008300", "#9085e9", "#e66767", "#3987e5"]
REFERENCE = "#8e8e8a"
LIMITATIONS = [
    "All figures are historical, from the dashboard's tested code, in USD; they are not "
    "forecasts and past performance does not predict future results.",
    "Performance holds today's weights over the past (rebalanced monthly after costs), so it "
    "shows how this mix would have behaved, not what the client earned.",
    "Stress tests hold today's weights through each episode, rebalanced daily, without costs, "
    "and need the holdings' common history to cover it.",
    "Suitability checks follow the IPS rules; 'review' means the data can't decide and the "
    "adviser must check. Sharia and sector data are partial.",
    "Performance figures cover the backtest period (after the estimation window, after "
    "costs); suitability risk figures use the holdings' full common history, which can include "
    "earlier crises, so their drawdowns can be larger.",
    "The AI memo explains results; every figure in it was copied from tested code and checked, "
    "but its wording is not reviewed by a compliance officer.",
]


@dataclass
class ReportInputs:
    """Everything the client report shows, computed beforehand by tested code."""

    report_date: date
    data_as_of: str
    benchmark: str
    holdings: pd.DataFrame  # formatted table, index = ticker
    allocation: pd.Series  # weights, for the allocation chart
    client: ClientProfile | None = None
    ips: IPS | None = None
    kpis: pd.DataFrame | None = None  # gpl.metrics.kpi_table output
    charts: list = field(default_factory=list)  # gpl.agent.charts.ChartData
    suitability: dict | None = None  # portfolio name -> list[gpl.suitability.Check]
    memo: str | None = None
    memo_unverified: list[str] = field(default_factory=list)
    adviser: str = ""
    domicile: dict | None = None  # gpl.domicile.format_domicile output


def _png(fig) -> Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    width = 174 * mm
    w, h = fig.get_size_inches()
    return Image(buffer, width=width, height=width * h / w)


def _style(ax, title: str, percent: bool = False) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    ax.tick_params(labelsize=8, colors=MUTED)
    if percent:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def _line_chart(frame: pd.DataFrame, title: str, reference: str | None, percent: bool) -> Image:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    slot = 0
    for name in frame.columns:
        if name == reference:
            ax.plot(frame.index, frame[name], color=REFERENCE, ls="--", lw=1.4, label=name)
        else:
            ax.plot(frame.index, frame[name], color=PRINT_COLORS[slot], lw=1.6, label=name)
            slot += 1
    _style(ax, title, percent)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=18)  # room for the legend
    return _png(fig)


def chart_image(data) -> Image | None:
    """Print version of an Analyst chart (``ChartData``) for the report; None if unsupported."""
    f = data.frames
    if data.chart == "growth_of_1":
        return _line_chart(f["values"], data.title, data.reference, percent=False)
    if data.chart == "drawdown_comparison":
        return _line_chart(f["values"], data.title, data.reference, percent=True)
    if data.chart == "stress_test":
        t = f["table"]
        episodes = list(dict.fromkeys(t["episode"]))
        series = list(dict.fromkeys(t["series"]))
        fig, ax = plt.subplots(figsize=(8, 3.2))
        width = 0.8 / len(series)
        slot = 0
        for i, name in enumerate(series):
            rows = t[t.series == name].set_index("episode").reindex(episodes)
            color = REFERENCE if name == data.reference else PRINT_COLORS[slot]
            slot += name != data.reference
            ax.bar(
                [j + i * width for j in range(len(episodes))],
                rows["total_return"].fillna(0),
                width=width,
                color=color,
                label=name,
                hatch="//" if name == data.reference else None,
                edgecolor="white",
            )
        labels = [
            f"{e}\n{t.loc[t.episode == e, 'start'].iloc[0]:%d %b %Y} - "
            f"{t.loc[t.episode == e, 'end'].iloc[0]:%d %b %Y}"
            for e in episodes
        ]
        ax.set_xticks([j + 0.4 - width / 2 for j in range(len(episodes))], labels)
        ax.axhline(0, color=MUTED, lw=0.6)
        _style(ax, data.title, percent=True)
        ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
        ax.set_title(data.title, loc="left", fontsize=10, color=INK, pad=18)
        return _png(fig)
    return None


def allocation_image(weights: pd.Series) -> Image:
    w = weights[weights > 0].sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.4 + 0.32 * len(w)))
    ax.barh([str(t) for t in w.index], w.to_numpy(), color=PRINT_COLORS[0])
    for y, v in enumerate(w):
        ax.text(v, y, f" {v:.2%}", va="center", fontsize=7.5, color=INK)
    ax.set_xlim(0, float(w.max()) * 1.18)  # room for the largest label
    _style(ax, "Allocation by holding")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    return _png(fig)


def _inline(text: str) -> str:
    """Escape, then turn **bold** into <b>bold</b>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc(text))


def markdown_flowables(text: str) -> list:
    """The memo's simple markdown (### headings, - bullets, **bold**, tables as text)."""
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "---":
            continue
        if stripped.startswith("#"):
            out.append(Paragraph(_inline(stripped.lstrip("#").strip()), STYLES["h2"]))
        elif stripped.startswith(("- ", "* ")):
            out.append(Paragraph(_inline(stripped[2:]), STYLES["body"], bulletText="•"))
        else:
            out.append(Paragraph(_inline(stripped), STYLES["body"]))
    return out


KPI_LABELS = {
    "cagr": "Return (CAGR)",
    "volatility": "Volatility (annual)",
    "sharpe": "Sharpe",
    "max_drawdown": "Max drawdown",
}


def kpi_rows(kpis: pd.DataFrame, benchmark: str) -> list[list[str]]:
    def fmt(metric: str, x: float, signed: bool = False) -> str:
        if pd.isna(x):
            return "n/a"
        if metric == "sharpe":
            return f"{x:+.2f}" if signed else f"{x:.2f}"
        return f"{x * 100:+.2f} pp" if signed else f"{x:.2%}"

    rows = [["", "Portfolio", benchmark, "Difference"]]
    for m, label in KPI_LABELS.items():
        rows.append(
            [
                label,
                fmt(m, kpis.loc[m, "portfolio"]),
                fmt(m, kpis.loc[m, "benchmark"]),
                fmt(m, kpis.loc[m, "delta"], signed=True),
            ]
        )
    return rows


def client_report(inputs: ReportInputs, compress: bool = True) -> bytes:
    """One-click client report: summary, IPS, holdings, performance, charts, suitability,
    AI memo, methodology and disclaimers. Every figure is passed in from tested code."""
    c = inputs.client
    ref = (c.reference or c.name) if c else ""
    stamp = inputs.report_date.isoformat()
    story = [
        para("Portfolio review", "title"),
        para(
            f"{'Client ' + ref + ' · ' if ref else ''}report date {stamp} · data as of "
            f"{inputs.data_as_of} · benchmark {inputs.benchmark}"
            + (f" · adviser {inputs.adviser}" if inputs.adviser else ""),
            "subtitle",
        ),
        para(
            "Prepared as decision support for the client's licensed adviser. It is not "
            "investment, tax or legal advice.",
            "small",
        ),
    ]
    # client summary and IPS
    story.append(para("Client summary", "h2"))
    if c is None:
        story.append(
            para(
                "No client profile was entered, so there is no IPS and no "
                "suitability check in this report."
            )
        )
    else:
        r = inputs.ips.risk
        story.append(
            table(
                [
                    ["Client", ref or "(no reference)"],
                    ["Classification and regulator", f"{c.classification}, {c.regulator}"],
                    ["Residency", f"{c.residency}; US person: {'yes' if c.us_person else 'no'}"],
                    ["Time horizon", f"{c.horizon_years:g} years"],
                    [
                        "Risk score",
                        "incomplete questionnaire"
                        if r.score is None
                        else f"{r.score} of 5 (capacity {r.capacity}, willingness {r.willingness})",
                    ],
                ],
                widths=[60 * mm, 114 * mm],
                header=False,
            )
        )
        story += ips_story(inputs.ips)
    # holdings and allocation
    story.append(para("Holdings", "h2"))
    h = inputs.holdings
    story.append(
        table(
            [
                ["Ticker", *h.columns],
                *[[t, *map(str, row)] for t, row in zip(h.index, h.to_numpy(), strict=True)],
            ]
        )
    )
    story.append(Spacer(1, 6))
    story.append(allocation_image(inputs.allocation))
    # performance
    story.append(para(f"Performance vs {inputs.benchmark}", "h2"))
    if inputs.kpis is None:
        story.append(para("Not available: choose a benchmark with data and enough history."))
    else:
        story.append(
            table(
                kpi_rows(inputs.kpis, inputs.benchmark), widths=[52 * mm, 40 * mm, 46 * mm, 36 * mm]
            )
        )
        story.append(
            para(
                "Current weights rebalanced monthly after costs, over the backtest "
                "period; differences are portfolio minus benchmark.",
                "small",
            )
        )
    for data in inputs.charts:
        image = chart_image(data)
        if image is not None:
            story += [Spacer(1, 6), image]
            notes = [n for n in data.notes if not n.startswith("No proposed portfolio")]
            if notes:
                story.append(para(" ".join(notes), "small"))
    # suitability
    story.append(para("Suitability against the IPS", "h2"))
    if not inputs.suitability:
        story.append(para("No suitability check: no client profile was entered."))
    else:
        story.append(
            para(
                "Risk rules use today's weights over the holdings' full common "
                "history (rebalanced daily, no costs), which can start before the "
                "backtest above; each reason states its period.",
                "small",
            )
        )
        for name, checks in inputs.suitability.items():
            story.append(para(name, "body"))
            story.append(
                table(
                    [["Rule", "Result", "Measured", "Limit", "Reason"]]
                    + [
                        [ch.rule, ch.status.capitalize(), ch.measured, ch.limit, ch.reason]
                        for ch in checks
                    ],
                    widths=[28 * mm, 15 * mm, 22 * mm, 22 * mm, 87 * mm],
                )
            )
            story.append(Spacer(1, 6))
    # domicile
    if inputs.domicile is not None:
        story.append(para("Fund domicile (information, not tax advice)", "h2"))
        story.append(para(inputs.domicile["label"], "small"))
        rows = inputs.domicile["rows"]
        if rows:
            cols = list(rows[0])
            story.append(
                table(
                    [cols, *[[r[c] for c in cols] for r in rows]],
                    widths=[20 * mm, 18 * mm, 24 * mm, 18 * mm, 94 * mm],
                )
            )
        story += [
            Paragraph(esc(s), STYLES["body"], bulletText="•") for s in inputs.domicile["summary"]
        ]
    # memo
    story.append(para("Analyst memo", "h2"))
    story.append(
        para(
            "AI-generated explanation. Every figure was copied from tested code and "
            "checked; the AI did not calculate anything or choose any weights.",
            "small",
        )
    )
    if inputs.memo:
        story += markdown_flowables(inputs.memo)
        if inputs.memo_unverified:
            story.append(
                para(
                    "Figures that did not match any tool result: "
                    + ", ".join(inputs.memo_unverified),
                    "small",
                )
            )
    else:
        story.append(para("No AI memo was generated for this report."))
    # methodology and disclaimers
    story.append(para("Methodology and limitations", "h2"))
    story += [Paragraph(esc(text), STYLES["body"], bulletText="•") for text in LIMITATIONS]
    story.append(para("Important information", "h2"))
    story.append(
        para(
            "This report is decision support for a licensed adviser and does not constitute "
            "investment, tax or legal advice, an offer, or a recommendation to buy or sell any "
            "security. Results are historical and risk-adjusted; they do not guarantee future "
            "returns. Data from Yahoo Finance; GCC currencies converted at their official USD pegs."
        )
    )
    return build_pdf(story, f"Client report {ref} {stamp}".strip(), compress)
