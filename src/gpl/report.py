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
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from gpl.client import IPS, QUESTIONS, ClientProfile, effective_regulator
from gpl.factfind import (
    ASSET_TYPES,
    DEMO_LABEL,
    INCOME_TYPES,
    LIABILITY_TYPES,
    PUBLIC_NOTICE,
    SECTIONS,
    SPECIALIST_REMINDER,
    finances,
    goals_required_return,
)
from gpl.plots import REFERENCE_COLOR, SERIES_COLORS

FOOTER = (
    "Decision support for a licensed adviser. Not investment, tax or legal advice. "
    "Figures are historical and come from tested code; they are not forecasts."
)
# the app's light private-bank palette (app/formatting.py THEME): navy headings, grey rules
INK, MUTED, RULE, ACCENT = "#2B2B2B", "#6B6B6B", "#E3E1DA", "#1B2A4A"
HEADER_FILL = "#F1F0EB"

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
STYLES["h3"] = ParagraphStyle(
    "h3", parent=STYLES["body"], fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=2
)
for _heading in ("h2", "h3"):  # a heading never ends a page: it moves with what follows
    STYLES[_heading].keepWithNext = 1
STYLES["bullet"] = ParagraphStyle(
    "bullet", parent=STYLES["body"], leftIndent=12, bulletIndent=2, spaceAfter=1
)
STYLES["bullet2"] = ParagraphStyle(
    "bullet2", parent=STYLES["bullet"], leftIndent=24, bulletIndent=14
)


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
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(HEADER_FILL)),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(MUTED)),
        ]
    t.setStyle(TableStyle(style))
    return t


def build_pdf(
    story: list, title: str, compress: bool = True, watermark: str | None = None
) -> bytes:
    """Lay out ``story`` on A4 with the footer disclaimer and page numbers on every page.

    ``watermark`` (e.g. "DRAFT") is printed diagonally in light grey across every page.
    """
    buffer = BytesIO()

    def footer(canvas, doc) -> None:
        canvas.saveState()
        if watermark:
            # as large as fits along the page diagonal (at most 96 pt)
            width = canvas.stringWidth(watermark, "Helvetica-Bold", 96)
            size = min(96.0, 96.0 * 0.68 * (A4[0] ** 2 + A4[1] ** 2) ** 0.5 / width)
            canvas.setFont("Helvetica-Bold", size)
            canvas.setFillColor(colors.HexColor(RULE))
            canvas.translate(A4[0] / 2, A4[1] / 2)
            canvas.rotate(45)
            canvas.drawCentredString(0, -size / 3, watermark)
            canvas.restoreState()
            canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(18 * mm, 10 * mm, FOOTER)
        canvas.drawString(18 * mm, 6.5 * mm, PUBLIC_NOTICE)
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


MAPS = {"income": INCOME_TYPES, "assets": ASSET_TYPES, "liabilities": LIABILITY_TYPES}


def _fact_value(c: ClientProfile, name: str, kind: str) -> str:
    """A fact-find answer as text ('' when not answered)."""
    value = getattr(c, name)
    if kind == "money_map":
        return "; ".join(f"{MAPS[name][k]} USD {v:,.0f}" for k, v in value.items() if v)
    if kind == "money":
        return f"USD {value:,.0f}" if value else ""
    if kind in ("bool", "yesno"):
        return "" if value is None else ("yes" if value else "no")
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}%" if name.endswith("_pct") else f"{value:g}"
    return "" if value is None else str(value)


def factfind_story(c: ClientProfile) -> list:
    """The fact-find by section (answered fields only), with the calculated figures."""
    story = [para("Client fact-find", "h2")]
    if c.demo:
        story.append(para(DEMO_LABEL + ".", "small"))
    fin = finances(c)
    extra = {
        "financial": [
            ["Net worth", f"USD {fin.net_worth:,.0f}"],
            ["Investable assets", f"USD {fin.investable:,.0f}"],
            [
                "Savings rate",
                "–" if fin.savings_rate is None else f"{fin.savings_rate:.2%}",
            ],
            [
                "Emergency fund",
                "–" if fin.emergency_months is None else f"{fin.emergency_months:.1f} months",
            ],
        ],
        "estate": [["Reminder", SPECIALIST_REMINDER]],
    }
    goals = goals_required_return(
        c.portfolio_value_usd, c.goals, c.liquidity_needs, c.annual_contribution_usd
    )
    extra["objectives"] = [
        [
            "Return needed for the goals",
            goals.reason,
        ]
    ]
    for s in SECTIONS:
        rows = [
            [f.label, text]
            for f in s.fields
            if f.kind != "custom" and (text := _fact_value(c, f.name, f.kind))
        ]
        rows += extra.get(s.key, [])
        if rows:
            story.append(para(s.title, "h3"))
            story.append(table(rows, widths=[62 * mm, 112 * mm], header=False))
    return story


def record_story(record: str, edited: bool) -> list:
    """The suitability record (rule-generated, or as the adviser edited it)."""
    lines = [ln.strip().removeprefix("- ").removeprefix("• ") for ln in record.splitlines()]
    return [
        para("Suitability record", "h2"),
        *_bullets([ln for ln in lines if ln]),
        para(
            "Edited by the adviser."
            if edited
            else "Generated from the IPS rules and the suitability checks.",
            "small",
        ),
    ]


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
    story += factfind_story(ips.client)
    return story


def ips_pdf(ips: IPS, compress: bool = True, draft: bool = False) -> bytes:
    """The IPS for the adviser and client to review and sign.

    ``draft``: watermarked "DRAFT" and titled as a draft, for an incomplete profile.
    """
    c = ips.client
    marks = [m for m, on in (("DRAFT", draft), ("DEMO DATA", c.demo)) if on]
    story = [
        para("Investment Policy Statement" + (f" ({', '.join(marks)})" if marks else ""), "title"),
        para(
            f"Client {c.reference or c.name or '(no reference)'} · prepared {ips.created} · "
            "Gulf Portfolio Lab",
            "subtitle",
        ),
        *ips_story(ips),
    ]
    if c.suitability_record:
        story += record_story(c.suitability_record, True)
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
    title = f"IPS {c.reference or ''} {ips.created}".strip()
    return build_pdf(story, title, compress, watermark=" · ".join(marks) or None)


# ------------------------------------------------------------------------- client report

PRINT_COLORS = SERIES_COLORS[:7]  # the same categorical order as the dashboard
REFERENCE = REFERENCE_COLOR
LIMITATIONS = [
    "All figures are historical, in US dollars, and come from tested code. They are not "
    "forecasts: past performance does not predict future results.",
    "Past performance shows how today's mix would have behaved (rebalanced monthly, after "
    "costs), not what you actually earned.",
    "Stress tests hold today's mix through each past crisis.",
    "The policy checks follow the rules in your Investment Policy Statement. 'Adviser to "
    "check' means the data cannot decide (for example, an unknown Sharia status).",
    "The recommendation compares several standard portfolio methods on past data, each "
    "re-estimated at each rebalance using only information available at the time, after costs. "
    "It is risk-adjusted and historical, not a promise of higher returns.",
    "The commentary was drafted with an AI assistant from the figures above and reviewed by "
    "your adviser; the AI did not calculate any figure or choose any weights.",
]
STATUS_WORDS = {"pass": "Meets", "fail": "Does not meet", "review": "Adviser to check"}


@dataclass
class ReportInputs:
    """Everything the client report shows, computed beforehand by tested code."""

    report_date: date
    data_as_of: str
    benchmark: str
    holdings: pd.DataFrame  # formatted table, index = ticker (e.g. Name, Value (USD), Weight)
    allocation: pd.Series  # weights, for the allocation chart
    client: ClientProfile | None = None
    ips: IPS | None = None
    kpis: pd.DataFrame | None = None  # gpl.metrics.kpi_table output
    charts: list = field(default_factory=list)  # gpl.agent.charts.ChartData
    suitability: dict | None = None  # portfolio name -> list[gpl.suitability.Check]
    memo: str | None = None
    memo_unverified: list[str] = field(default_factory=list)  # shown in the app, not printed
    adviser: str = ""
    domicile: dict | None = None  # gpl.domicile.format_domicile output
    value: float | None = None  # the portfolio value (USD)
    recommendation: dict | None = None  # recommendation_section output
    trades: dict | None = None  # trade_section output
    adviser_licence: str = ""
    suitability_record: str = ""  # rule-generated or adviser-edited rationale
    record_edited: bool = False


def recommendation_section(report: dict, latest: pd.DataFrame) -> dict:
    """The Optimise run's recommendation, formatted for the report.

    ``report`` and ``latest`` are ``gpl.agent.tools.run_candidates`` outputs (the report dict
    and ``CandidateRun.latest_weights``). Returns ``status``, ``choice`` (a method, or None to
    keep the current portfolio), ``headline``, ``reasons``, ``weights`` (rows: holding, today,
    recommended, change), ``metrics`` (rows) and ``failed`` (rules the shown portfolio fails).
    """
    rec = report["recommendation"]
    keep = rec["choice"] == "Keep your current portfolio"
    choice = None if keep else rec["choice"]
    gate = report.get("suitability_gate") or {}
    failed = gate.get("failed_rules", {}).get("Current" if keep else choice, [])
    if rec["status"] == "closest_fails":
        headline = (
            "No portfolio we tested meets every rule of your investment policy. "
            + (
                "Your current portfolio is the closest"
                if keep
                else f"The closest is the {choice} portfolio"
            )
            + "; it is shown for discussion with your adviser, not as a recommendation."
        )
    elif rec["status"] == "next_best":
        headline = (
            "No portfolio we built beat the benchmark after costs within your investment "
            f"policy, so the next best option is the {choice}. It does not aim to beat the "
            "benchmark."
        )
    elif keep:
        headline = "We recommend keeping your current portfolio."
    else:
        headline = f"We recommend moving to the {choice} portfolio."
    rows = report["walk_forward_after_costs"]
    shown = [
        *(["Current"] if "Current" in rows else []),
        *([] if keep else [choice]),
        *(["Benchmark"] if "Benchmark" in rows else []),
    ]
    metrics = [["", *[("Today" if n == "Current" else n) for n in shown]]]
    for key, label in [
        ("cagr", "Average yearly growth (CAGR)"),
        ("annual_volatility", "Typical yearly swing (volatility)"),
        ("sharpe", "Return per unit of risk (Sharpe)"),
        ("max_drawdown", "Largest fall from a peak"),
        ("excess_return_vs_benchmark", "Ahead of (+) or behind (-) the benchmark, a year"),
        ("information_ratio", "Extra return per unit of difference (information ratio)"),
        ("hit_rate_months", "Months ahead of the benchmark"),
        ("worst_relative_drawdown", "Largest fall behind the benchmark"),
    ]:
        if any(key in rows[n] for n in shown):
            metrics.append([label, *[str(rows[n].get(key, "–")) for n in shown]])
    weights = [["Holding", "Today", "Recommended" if not keep else "Today (kept)", "Change"]]
    target = latest.loc["Current"] if keep else latest.loc[choice]
    for t in latest.columns:
        now, new = float(latest.loc["Current", t]), float(target[t])
        if now > 1e-9 or new > 1e-9:
            weights.append([str(t), f"{now:.2%}", f"{new:.2%}", f"{(new - now) * 100:+.2f} pp"])
    securities = [["Security", "What it is", "Weight", "Why it was chosen"]] + [
        [t, f"{v['name']} ({v['asset_class']}, {v['region']})", v["weight"], v["why"]]
        for t, v in report.get("securities_chosen", {}).items()
    ]
    return {
        "status": rec["status"],
        "choice": choice,
        "headline": headline,
        "goal": report.get("goal", ""),
        "period": report.get("backtest_period", ""),
        "reasons": list(rec["reasons"]),
        "weights": weights,
        "metrics": metrics,
        "failed": list(failed),
        "securities": securities if len(securities) > 1 else None,
        "swaps": [sw["why"] for sw in report.get("swaps", [])],
        "domicile": [
            d["why"] for d in report.get("domicile_suggestions_information_not_tax_advice", [])
        ],
        "relative_drawdown": None,  # set by the app from the walk-forward equity curves
    }


def trade_section(trades: pd.DataFrame, cost_bps: float, cash: dict | None = None) -> dict:
    """``gpl.scenarios.trade_list`` (and ``cash_need_check``) output, formatted."""
    moved = trades[trades["action"] != "Hold"]
    rows = [["Holding", "Action", "Amount (USD)", "Est. cost (USD)", "After (USD)"]]
    for t, r in moved.iterrows():
        rows.append(
            [
                str(t),
                r["action"],
                f"{abs(r['trade_usd']):,.0f}",
                f"{r['cost_usd']:,.0f}",
                f"{r['target_usd']:,.0f}",
            ]
        )
    sells = -trades.loc[trades.trade_usd < 0, "trade_usd"].sum()
    buys = trades.loc[trades.trade_usd > 0, "trade_usd"].sum()
    costs = trades["cost_usd"].sum()
    out = {
        "rows": rows,
        "summary": f"Sales of USD {sells:,.0f} pay for purchases of USD {buys:,.0f} and estimated "
        f"trading costs of USD {costs:,.0f} ({cost_bps:g} basis points); no new money is needed.",
        "cash": None,
    }
    if cash is not None:
        out["cash"] = (
            f"After these trades, cash and short-term holdings are USD {cash['held']:,.0f}, "
            + (
                f"covering the USD {cash['need']:,.0f} you need soon."
                if cash["ok"]
                else f"USD {cash['shortfall']:,.0f} short of the USD {cash['need']:,.0f} you "
                "need soon: your adviser will adjust the trades."
            )
        )
    return out


_TOOL_TAG = re.compile(r"\s*\[[a-z][a-z0-9_]*(?:\s*(?:,|;|/|&|and)\s*[a-z][a-z0-9_]*)*\](?!\()")
_CHART_TAG = re.compile(r"\s*\[show_chart:\s*[a-z0-9_]+\]")  # where the app drew a chart
# "[tool; a note in words]" (a note has a space; "[tool_a; tool_b]" is two citations)
_TOOL_NOTE = re.compile(r"\s*\[[a-z][a-z0-9_]*\s*;\s*([^\s\[\]][^\[\]]*\s[^\[\]]*)\]")


def clean_memo(text: str) -> str:
    """The memo for a client: tool citations ('[stress_test]'), chart markers
    ('[show_chart: stress_test]') and the app's appended disclaimer removed (the report has
    its own). A citation carrying a note ('[tool; the windows differ]') keeps the note in
    brackets."""
    cut = text.rfind("\n---")
    if cut >= 0 and "not investment advice" in text[cut:].lower():
        text = text[:cut]
    text = _CHART_TAG.sub("", text)
    text = _TOOL_NOTE.sub(lambda m: f" ({m.group(1).strip()})", text)
    return _TOOL_TAG.sub("", text).strip()


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
    ax.grid(axis="y", color=RULE, linewidth=0.6)
    ax.tick_params(labelsize=8, colors=MUTED)
    if percent:
        # ticks on 1-2-5 steps, so a label never rounds (2.5% steps printed as "-2%, -5%, -7%")
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=None))


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
    ax.grid(axis="x", color=RULE, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    return _png(fig)


def _inline(text: str) -> str:
    """Escape, then **bold** -> <b>, *italic* or _italic_ -> <i>, `code` -> plain."""
    out = esc(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", out)
    out = re.sub(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", r"<i>\1</i>", out)
    return out.replace("`", "")


def _md_table(lines: list[str]) -> Table:
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue  # the |---|---| separator
        rows.append(cells)
    width = 174 * mm / max(len(r) for r in rows)
    data = [[Paragraph(_inline(c), STYLES["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=[width] * len(data[0]), repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(RULE)),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(HEADER_FILL)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return t


def markdown_flowables(text: str) -> list:
    """The memo's markdown as flowables: # to ### headings, bullets (-, *, •), numbered
    lists, **bold**, *italics* and pipe tables."""
    out: list = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out += [_md_table(block), Spacer(1, 4)]
            continue
        i += 1
        if not stripped or re.fullmatch(r"-{3,}|\*{3,}", stripped):
            continue
        if stripped.startswith("#"):
            out.append(Paragraph(_inline(stripped.lstrip("#").strip()), STYLES["h3"]))
        elif re.match(r"^[-*•]\s+", stripped):
            body = re.sub(r"^[-*•]\s+", "", stripped)
            indent = len(lines[i - 1]) - len(lines[i - 1].lstrip())
            style = STYLES["bullet2"] if indent >= 2 else STYLES["bullet"]
            out.append(Paragraph(_inline(body), style, bulletText="•"))
        elif m := re.match(r"^(\d+)[.)]\s+(.*)", stripped):
            out.append(
                Paragraph(_inline(m.group(2)), STYLES["bullet"], bulletText=f"{m.group(1)}.")
            )
        else:
            out.append(Paragraph(_inline(stripped), STYLES["body"]))
    return out


KPI_LABELS = {
    "cagr": "Average yearly growth (CAGR)",
    "volatility": "Typical yearly swing (volatility)",
    "sharpe": "Return per unit of risk (Sharpe)",
    "max_drawdown": "Largest fall from a peak",
}


def kpi_rows(kpis: pd.DataFrame, benchmark: str) -> list[list[str]]:
    def fmt(metric: str, x: float, signed: bool = False) -> str:
        if pd.isna(x):
            return "n/a"
        if metric == "sharpe":
            return f"{x:+.2f}" if signed else f"{x:.2f}"
        return f"{x * 100:+.2f} pp" if signed else f"{x:.2%}"

    rows = [["", "Your portfolio", benchmark, "Difference"]]
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


def _goals_and_limits(c: ClientProfile, ips: IPS) -> Table:
    lim, r = ips.limits, ips.risk

    def pct(x) -> str:
        return "not set" if x is None else f"{x:.2%}"

    rows = [
        ["Your goals", ips.sections["Return"].removeprefix("Goals: ")],
        ["Time horizon", ips.sections["Time horizon"]],
        [
            "Your risk profile",
            "Questionnaire incomplete"
            if r.score is None
            else f"{r.score} of 5 (ability to take risk {r.capacity}, willingness {r.willingness}; "
            "the lower of the two sets your limits)",
        ],
        [
            "Risk limits",
            f"Typical yearly swing at most {pct(lim['max_volatility'])}; largest fall at most "
            f"{pct(lim['max_drawdown'])}; at most {pct(lim['max_weight'])} in any one holding.",
        ],
        ["Cash you need", ips.sections["Liquidity"]],
        ["Preferences", ips.sections["Unique circumstances"] or "None recorded."],
        ["Taxes", ips.sections["Taxes"]],
    ]
    return table(rows, widths=[38 * mm, 136 * mm], header=False)


def _in_brief(inputs: ReportInputs) -> list[str]:
    lines = []
    if inputs.value:
        lines.append(
            f"Your portfolio is worth USD {inputs.value:,.0f} (prices as of {inputs.data_as_of})."
        )
    rec = inputs.recommendation
    if rec is None:
        lines.append("No optimisation was run for this review, so there is no recommendation.")
    else:
        lines.append(rec["headline"])
    if inputs.suitability and "Current" in inputs.suitability:
        failed = [ch.rule for ch in inputs.suitability["Current"] if ch.status == "fail"]
        lines.append(
            "Your current portfolio meets every rule of your investment policy."
            if not failed
            else "Your current portfolio does not meet these rules of your investment policy: "
            + ", ".join(failed).lower()
            + "."
        )
    if inputs.trades:
        lines.append(inputs.trades["summary"])
        if inputs.trades["cash"]:
            lines.append(inputs.trades["cash"])
    return lines


def _bullets(lines: Sequence[str]) -> list:
    return [Paragraph(_inline(t), STYLES["bullet"], bulletText="•") for t in lines]


def _periods(results: dict) -> str:
    """The full-history period the risk checks used ('' if not recorded)."""
    for checks in results.values():
        for ch in checks:
            if ch.period.startswith("full history: "):
                return ch.period.removeprefix("full history: ")
    return ""


def _suitability_table(results: dict) -> Table:
    names = list(results)
    by_rule: dict[str, dict] = {}
    for name in names:
        for ch in results[name]:
            by_rule.setdefault(ch.rule, {})[name] = ch
    header = ["Policy rule", *[("Today" if n == "Current" else n) for n in names], "What it means"]
    rows = [header]
    for rule, checks in by_rule.items():
        last = checks[names[-1]] if names[-1] in checks else next(iter(checks.values()))
        rows.append(
            [
                rule,
                *[
                    f"{STATUS_WORDS[checks[n].status]} ({checks[n].measured})"
                    if n in checks
                    else "–"
                    for n in names
                ],
                last.reason,
            ]
        )
    status_w = 30 * mm
    widths = [28 * mm, *[status_w] * len(names), 174 * mm - 28 * mm - status_w * len(names)]
    return table(rows, widths=widths)


def client_report(inputs: ReportInputs, compress: bool = True) -> bytes:
    """One-click client report in plain language: summary, goals and limits, holdings in USD,
    past performance, the Optimise recommendation, the trades, the policy checks, the
    adviser's commentary, sign-off and disclaimers. Every figure is passed in from tested code.
    """
    c = inputs.client
    who = (c.name or c.reference) if c else ""
    stamp = inputs.report_date.isoformat()
    story = [
        para("Your portfolio review" + (" (DEMO DATA)" if c and c.demo else ""), "title"),
        para(
            (f"Prepared for {who} · " if who else "")
            + f"{inputs.report_date:%d %B %Y}"
            + (f" · by {inputs.adviser}" if inputs.adviser else "")
            + f" · prices as of {inputs.data_as_of} · compared with {inputs.benchmark}",
            "subtitle",
        ),
        para("In brief", "h2"),
        *_bullets(_in_brief(inputs)),
    ]
    # goals and limits
    story.append(para("Your goals and limits", "h2"))
    if c is None or inputs.ips is None:
        story.append(
            para("No client profile was entered, so there is no investment policy to check.")
        )
    else:
        story.append(_goals_and_limits(c, inputs.ips))
        story.append(para("The full Investment Policy Statement is in the appendix.", "small"))
    # holdings
    story.append(para("What you hold today", "h2"))
    h = inputs.holdings
    story.append(
        table(
            [
                ["Holding", *h.columns],
                *[[t, *map(str, row)] for t, row in zip(h.index, h.to_numpy(), strict=True)],
            ]
        )
    )
    if inputs.value:
        story.append(para(f"Total: USD {inputs.value:,.0f}.", "small"))
    story += [Spacer(1, 6), allocation_image(inputs.allocation)]
    # performance
    story.append(para("How it has done", "h2"))
    if inputs.kpis is None:
        story.append(para("Not available: choose a benchmark with data and enough history."))
    else:
        story.append(
            table(
                kpi_rows(inputs.kpis, inputs.benchmark), widths=[62 * mm, 36 * mm, 42 * mm, 34 * mm]
            )
        )
        story.append(
            para(
                f"Period: {inputs.kpis.attrs.get('period', 'the backtest period')}. "
                "How today's mix would have done in the past, rebalanced monthly after costs. "
                "The difference is your portfolio minus the benchmark. Past results do not "
                "predict future results.",
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
    # recommendation
    story.append(para("What we recommend", "h2"))
    rec = inputs.recommendation
    if rec is None:
        story.append(
            para(
                "No optimisation was run for this review. Your adviser can run one on the "
                "Optimise page and prepare this report again."
            )
        )
    else:
        story.append(Paragraph(f"<b>{esc(rec['headline'])}</b>", STYLES["body"]))
        if rec["goal"]:
            story.append(para(f"Goal: {rec['goal']}.", "small"))
        if rec["failed"]:
            story.append(para("It still falls short of these policy rules:"))
            story += _bullets(rec["failed"])
        story.append(para("Why", "h3"))
        story += _bullets(rec["reasons"])
        if rec.get("swaps"):
            story.append(para("Changes to your holdings", "h3"))
            story += _bullets(rec["swaps"])
        if rec.get("securities"):
            story.append(para("Securities chosen and why", "h3"))
            story.append(table(rec["securities"], widths=[20 * mm, 52 * mm, 18 * mm, 84 * mm]))
        story += [Spacer(1, 4), table(rec["weights"], widths=[44 * mm, 40 * mm, 50 * mm, 40 * mm])]
        story += [Spacer(1, 6), table(rec["metrics"])]
        rel = rec.get("relative_drawdown")
        if rel is not None and len(rel):
            story += [
                Spacer(1, 6),
                _line_chart(rel.to_frame(), "Falls behind the benchmark", None, percent=True),
            ]
        if rec.get("domicile"):
            story.append(para("Fund domicile ideas (information, not tax advice)", "h3"))
            story += _bullets(rec["domicile"])
        story.append(
            para(
                f"Tested on past data ({rec['period']}), re-estimated at each rebalance with "
                "only the data available then, after costs. "
                "Risk-adjusted and historical, not a promise of higher returns.",
                "small",
            )
        )
    # trades
    story.append(para("Proposed trades", "h2"))
    if inputs.trades is None:
        story.append(
            para(
                "No trades are proposed"
                + (
                    ": no portfolio meets your investment policy yet."
                    if rec and rec["status"] == "closest_fails"
                    else "."
                    if rec
                    else " until an optimisation is run."
                )
            )
        )
    else:
        story.append(
            table(inputs.trades["rows"], widths=[30 * mm, 22 * mm, 40 * mm, 40 * mm, 42 * mm])
        )
        story += _bullets(
            [inputs.trades["summary"], *([inputs.trades["cash"]] if inputs.trades["cash"] else [])]
        )
        story.append(
            para(
                "Amounts are estimates at the latest prices; the final trades are placed by "
                "your adviser.",
                "small",
            )
        )
    # suitability
    if not inputs.suitability:
        story.append(para("Does it fit your investment policy?", "h2"))
        story.append(para("No policy check: no client profile was entered."))
    else:  # the heading stays with its table
        story.append(
            KeepTogether(
                [para("Does it fit your investment policy?", "h2"),
                 _suitability_table(inputs.suitability)]
            )
        )  # fmt: skip
        story.append(
            para(
                "Risk figures (volatility, drawdown) use each mix over the holdings' full shared "
                "history"
                + (f" ({_periods(inputs.suitability)})" if _periods(inputs.suitability) else "")
                + ", which can include earlier crises than the performance period above; the "
                "reasons also give the performance-period figures.",
                "small",
            )
        )
    if inputs.suitability_record:
        story += record_story(inputs.suitability_record, inputs.record_edited)
    # domicile
    if inputs.domicile is not None:
        story.append(para("Where your funds are domiciled (information, not tax advice)", "h2"))
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
        story += _bullets(inputs.domicile["summary"])
    # commentary
    story.append(para("Your adviser's commentary", "h2"))
    if inputs.memo:
        story += markdown_flowables(clean_memo(inputs.memo))
    else:
        story.append(para("No AI memo was generated for this report."))
    # sign-off
    story.append(
        KeepTogether(
            [
                para("Sign-off", "h2"),
                table(
                    [
                        ["Adviser", inputs.adviser or ""],
                        ["Licence / registration no.", inputs.adviser_licence or ""],
                        ["Regulator", effective_regulator(c) if c else ""],
                        ["Date", f"{inputs.report_date:%d %B %Y}"],
                    ],
                    widths=[55 * mm, 119 * mm],
                    header=False,
                ),
                Spacer(1, 10),
                table(
                    [["Adviser signature", "Client acknowledgement and date"], ["", ""]],
                    widths=[87 * mm, 87 * mm],
                    row_heights=[None, 22 * mm],
                ),
                para(
                    "By signing, the client confirms they have received and discussed this "
                    "review; it does not oblige them to trade.",
                    "small",
                ),
            ]
        )
    )
    # disclaimers
    story.append(
        KeepTogether(
            [
                para("Important information", "h2"),
                para(
                    "This report is decision support for a licensed adviser and does not "
                    "constitute investment, tax or legal advice, an offer, or a recommendation to "
                    "buy or sell any security on its own. Results are historical and "
                    "risk-adjusted; they do not guarantee future returns. Data from Yahoo "
                    "Finance; GCC currencies converted at their official USD pegs."
                ),
            ]
        )
    )
    story += _bullets(LIMITATIONS)
    if inputs.ips is not None:
        story += [PageBreak(), para("Appendix: Investment Policy Statement", "title")]
        story += ips_story(inputs.ips)
    return build_pdf(
        story,
        f"Portfolio review {who} {stamp}".strip(),
        compress,
        watermark="DEMO DATA" if c and c.demo else None,
    )
