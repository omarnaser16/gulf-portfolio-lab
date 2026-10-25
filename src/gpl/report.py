"""PDF documents for the adviser: the Investment Policy Statement and the client report.

Functions return the PDF as bytes (the app offers them as downloads); nothing is written to
disk. Every figure in a document is passed in from tested code; nothing is calculated here
beyond formatting. Every page carries the decision-support disclaimer.
"""

from collections.abc import Sequence
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from gpl.client import IPS, QUESTIONS

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
