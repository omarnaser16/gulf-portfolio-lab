"""Workflow layout and design system: header band, step cards, page headers, Back/Next bar.

Pure presentation: every figure shown is passed in (computed by tested code elsewhere). Colours,
type and spacing come from the design tokens in ``gpl.theme`` (one place); ``APP_CSS`` turns
them into the page's stylesheet.

Steps are the top-level tabs (``st.tabs(key=STEP_KEY)``), so a button can move to a step by
setting ``st.session_state[STEP_KEY]`` in a callback. The tab row itself is hidden (CSS): the
step cards are the navigation, each with a full-size button over it. Sub-views are tabs too,
restyled as segmented pills.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from html import escape

import streamlit as st

from gpl import theme as t

STEP_KEY = "main_tab"
STEPS = [
    "1 Client",
    "2 Portfolio",
    "3 Analyse",
    "4 Optimise",
    "5 Suitability",
    "6 Report & trades",
]
DISCLAIMER = (
    "Educational project, not investment advice. Past performance does not predict future results."
)


@dataclass
class StepStatus:
    done: bool
    note: str  # one short line: what is done or what is missing
    missing: list[str] = field(default_factory=list)


@dataclass
class HeaderInfo:
    client: str | None  # name or reference; None: no client loaded
    risk_score: int | None
    value: str  # formatted portfolio value, or "not set"
    ips_status: str  # "Ready", "Draft" or "–"
    demo: bool = False
    seed: int | None = None  # the demo's seed, to reproduce it


def go_to(step: str) -> None:
    """Callback: open ``step``."""
    st.session_state[STEP_KEY] = step


# ---------------------------------------------------------------------------------- HTML bits


def pill(text: str, tone: str = "") -> str:
    """A small rounded label; tone: "", "pass", "fail", "warn", "brand", "accent", "dark"."""
    return f'<span class="gpl-pill {tone}">{escape(text)}</span>'


def kpi_card(
    label: str,
    value: str,
    sub: str = "",
    delta: str = "",
    tone: str = "",
    before: str = "",
) -> str:
    """A KPI card: label, big value (optionally "before → value"), a delta pill or a sub line.

    ``tone`` colours the delta: "pass" (better), "fail" (worse) or "" (neutral).
    """
    lead = (
        f'<span class="gpl-before">{escape(before)}</span><span class="gpl-arrow">→</span>'
        if before
        else ""
    )
    extra = (
        f'<div class="gpl-kpi-delta">{pill(delta, tone)}</div>'
        if delta
        else (f'<div class="gpl-kpi-sub">{escape(sub)}</div>' if sub else "")
    )
    return (
        f'<div class="gpl-kpi"><div class="gpl-kpi-label">{escape(label)}</div>'
        f'<div class="gpl-kpi-value">{lead}{escape(value)}</div>{extra}</div>'
    )


def kpi_row(cards: Sequence[str]) -> None:
    """Cards side by side in one row (equal widths)."""
    st.markdown(
        f'<div class="gpl-kpis" style="--n:{len(cards)}">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------- header + steps


def render_header(container, info: HeaderInfo) -> None:
    """Client, risk score, portfolio value and IPS status as white chips in the header band."""
    ips_cls = {"Ready": "ok", "Draft": "warn"}.get(info.ips_status, "")
    chips = [
        ("Client", info.client or "none loaded", ""),
        ("Risk score", f"{info.risk_score} / 5" if info.risk_score else "–", ""),
        ("Portfolio", info.value, ""),
    ]
    html = "".join(
        f'<span class="gpl-chip {cls}">{escape(label)} <b>{escape(value)}</b></span>'
        for label, value, cls in chips
    )
    dot = '<i class="gpl-dot-ok"></i>' if ips_cls == "ok" else ""
    html += f'<span class="gpl-chip {ips_cls}">IPS {dot}<b>{escape(info.ips_status)}</b></span>'
    if info.demo:
        seed = f" · seed {info.seed}" if info.seed is not None else ""
        html += f'<span class="gpl-chip demo">DEMO DATA{seed}</span>'
    container.markdown(f'<div class="gpl-chips">{html}</div>', unsafe_allow_html=True)


CHECK_SVG = (
    '<svg viewBox="0 0 16 16" width="12" height="12"><path d="M3.5 8.5l3 3 6-7" fill="none" '
    'stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


def _card_html(step: str, status: StepStatus | None, active: str | None) -> str:
    number, title = step.split(" ", 1)
    done = bool(status and status.done)
    cls = ("done" if done else "todo") + (" active" if step == active else "")
    note = status.note if status else ""
    mark = CHECK_SVG if done else number
    return (
        f'<div class="gpl-step {cls}"><span class="gpl-dot">{mark}</span>'
        f'<span class="gpl-step-text"><b>{escape(title)}</b>'
        f"<small>{escape(note) or '&nbsp;'}</small></span></div>"
    )


def render_stepper(statuses: Sequence[StepStatus] | None, active: str | None) -> list:
    """The six step cards, drawn at the top of the run; click a card to open its step.

    The step is set in the button's callback, before the rerun, so the highlight here and the
    page below change in the same render. ``statuses`` are the previous run's (None on the
    first run); ``update_stepper`` refreshes the text in place once this run knows them.
    Returns the placeholders for ``update_stepper``.
    """
    slots = []
    with st.container(key="gpl_steps"):
        columns = st.columns(len(STEPS), gap="small")
    for i, (column, step) in enumerate(zip(columns, STEPS, strict=True)):
        status = statuses[i] if statuses else None
        with column.container(key=f"step_card_{i}"):
            slot = st.empty()
            slot.markdown(_card_html(step, status, active), unsafe_allow_html=True)
            slots.append(slot)
            st.button(
                f"Open step {step}",
                key=f"step_nav_{i}",
                on_click=go_to,
                args=(step,),
                help=f"Open {step.split(' ', 1)[1]}",
            )
    return slots


def update_stepper(slots: list, statuses: Sequence[StepStatus], active: str | None) -> None:
    """Refresh the cards' marks and notes with this run's statuses."""
    for slot, step, status in zip(slots, STEPS, statuses, strict=True):
        slot.markdown(_card_html(step, status, active), unsafe_allow_html=True)


# --------------------------------------------------------------------------- page frame


def page_header(title: str, guidance: str, help: str | None = None, right_width: float = 0.45):
    """Step title and its one guidance line (the rest in the (i) tooltip).

    Returns the right-hand column for meta text or a switch (e.g. "Data as of …", pills).
    """
    left, right = st.columns([1 - right_width, right_width], vertical_alignment="bottom")
    with left:
        st.markdown(f'<h2 class="gpl-title">{escape(title)}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p class="gpl-guidance">{escape(guidance)}</p>', unsafe_allow_html=True,
                    help=help, width="content")  # fmt: skip
    return right


def meta(text: str, help: str | None = None, container=None) -> None:
    """Small right-aligned meta text (dates, sources) with an optional (i) tooltip."""
    (container or st).markdown(
        f'<p class="gpl-meta">{escape(text)}</p>', unsafe_allow_html=True, help=help
    )


def step_footer(step: str, status: StepStatus) -> None:
    """The sticky bar: the step's status and the disclaimer; Back and Next buttons."""
    index = STEPS.index(step)
    with st.container(key=f"footer_{index}"):
        left, mid, right = st.columns([5, 1.1, 1.6], vertical_alignment="center")
        name = step.split(" ", 1)[1]
        line = (
            "<b>Still to do:</b> " + escape("; ".join(status.missing))
            if status.missing
            else f"<b>{escape(name)}:</b> {escape(status.note)}"
        )
        left.markdown(
            f'<div class="gpl-foot">{line}<small>{escape(DISCLAIMER)}</small></div>',
            unsafe_allow_html=True,
        )
        if index > 0:
            mid.button(
                "← Back",
                key=f"back_{index}",
                on_click=go_to,
                args=(STEPS[index - 1],),
                width="stretch",
            )
        if index + 1 < len(STEPS):
            nxt = STEPS[index + 1]
            right.button(
                f"Next: {nxt.split(' ', 1)[1]} →",
                key=f"next_{index}",
                on_click=go_to,
                args=(nxt,),
                type="primary",
                width="stretch",
            )


# ------------------------------------------------------------------------------- stylesheet

APP_CSS = f"""
<style>
:root {{
  --gpl-bg: {t.BG}; --gpl-card: {t.CARD}; --gpl-border: {t.BORDER}; --gpl-subtle: {t.SUBTLE};
  --gpl-text: {t.TEXT}; --gpl-muted: {t.MUTED}; --gpl-brand: {t.BRAND};
  --gpl-brand-btn: {t.BRAND_BUTTON}; --gpl-brand-tint: {t.BRAND_TINT};
  --gpl-accent: {t.ACCENT}; --gpl-accent-btn: {t.ACCENT_BUTTON}; --gpl-accent-tint: {t.ACCENT_TINT};
  --gpl-pass: {t.GAIN}; --gpl-pass-tint: {t.GAIN_TINT}; --gpl-fail: {t.LOSS};
  --gpl-fail-tint: {t.LOSS_TINT}; --gpl-warn: {t.WARN}; --gpl-warn-tint: {t.WARN_TINT};
  --gpl-radius: {t.RADIUS_PX}px;
}}
html, body, [data-testid="stAppViewContainer"] {{
  font-family: {t.FONT_STACK}; font-variant-numeric: tabular-nums; color: var(--gpl-text);
}}
/* Streamlit's own top bar (toolbar, and on Cloud its menu) is fixed over the first 3.75rem:
   our header starts below it and sticks below it, so nothing of ours sits under its buttons */
[data-testid="stHeader"] {{ background: var(--gpl-bg); }}
.block-container, [data-testid="stMainBlockContainer"] {{
  max-width: {t.MAX_WIDTH_PX + 2 * 3 * t.SPACE_PX}px; padding-top: 3.75rem !important;
  padding-bottom: {14 * t.SPACE_PX}px !important;
}}
h1, h2, h3, h4 {{ color: var(--gpl-text) !important; letter-spacing: -0.01em; }}

/* ---- header band (full-bleed brand blue, sticky) */
.st-key-gpl_header {{
  position: sticky; top: 3.75rem; z-index: 990; background: var(--gpl-brand);
  box-shadow: 0 0 0 100vmax var(--gpl-brand); clip-path: inset(0 -100vmax);
  padding: {t.space(1.5)} 0 {t.space(2.5)} 0; margin-bottom: {t.space(2)};
}}
.st-key-gpl_header .gpl-brandmark {{ color: #fff !important; }}
.st-key-gpl_header button p {{ color: var(--gpl-text) !important; }}
.gpl-brandmark {{ display: flex; align-items: center; gap: 10px; margin: 0; white-space: nowrap;
  font-size: 1.3rem; font-weight: 700; color: #fff; }}
.gpl-brandmark i {{ display: inline-flex; width: 28px; height: 28px; border-radius: 7px;
  background: #fff; align-items: flex-end; justify-content: center; gap: 2px; padding: 6px; }}
.gpl-brandmark i b {{ display: block; width: 3px; background: var(--gpl-brand);
  border-radius: 1px; }}
.st-key-gpl_header button {{ background: #fff !important; color: var(--gpl-text) !important;
  border: 1px solid rgba(255,255,255,.6) !important; font-weight: 600; }}
.st-key-gpl_header .st-key-fill_demo button p {{ color: var(--gpl-brand-btn) !important; }}
.st-key-gpl_header [data-baseweb="select"] > div {{ background: #fff; }}
.gpl-chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: {t.space(0.5)}; }}
.gpl-brandmark, .gpl-chips, .gpl-step, .gpl-kpis, .gpl-summary {{ pointer-events: none; }}
.gpl-chip {{ font-size: 0.76rem; padding: 2px 9px; border-radius: 999px; background: #fff;
  color: var(--gpl-muted); white-space: nowrap; }}
.gpl-chip b {{ color: var(--gpl-text); font-weight: 600; }}
.gpl-chip.ok b {{ color: var(--gpl-pass); }}
.gpl-chip.warn b {{ color: var(--gpl-warn); }}
.gpl-dot-ok {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: var(--gpl-pass); margin: 0 4px 1px 2px; }}
.gpl-chip.demo {{ background: var(--gpl-text); color: #fff; font-weight: 700;
  letter-spacing: .03em; }}

/* ---- step cards (the navigation; the top-level tab row is hidden) */
.st-key-main_tab > div > [role="tablist"], .st-key-main_tab > div > div > [role="tablist"] {{
  display: none; }}
[class*="st-key-step_card_"] {{ position: relative; }}
[class*="st-key-step_nav_"] {{ position: absolute; inset: 0; z-index: 5; margin: 0; }}
[class*="st-key-step_nav_"] button {{ width: 100%; height: 100%; opacity: 0; cursor: pointer; }}
[class*="st-key-step_card_"]:hover .gpl-step {{ border-color: var(--gpl-brand); }}
.gpl-step {{ height: 100%; display: flex; gap: 10px; align-items: center; padding: 9px 12px;
  border: 1px solid var(--gpl-border); border-radius: var(--gpl-radius);
  background: var(--gpl-card); min-width: 0; }}
.gpl-step.active {{ border: 2px solid var(--gpl-brand); background: var(--gpl-brand-tint);
  padding: 8px 11px; }}
.gpl-dot {{ flex: none; width: 22px; height: 22px; border-radius: 50%; display: flex;
  align-items: center; justify-content: center; font-weight: 600; font-size: 0.72rem;
  border: 1.5px solid var(--gpl-border); color: var(--gpl-muted); background: #fff; }}
.gpl-step.done .gpl-dot {{ background: var(--gpl-brand); border-color: var(--gpl-brand); }}
.gpl-step-text {{ display: flex; flex-direction: column; min-width: 0; line-height: 1.25; }}
.gpl-step-text b {{ font-size: 0.82rem; font-weight: 600; }}
.gpl-step-text small {{ font-size: 0.68rem; color: var(--gpl-muted); white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }}

/* ---- page header */
.gpl-title {{ font-size: 1.9rem !important; font-weight: 700 !important;
  margin: {t.space(3)} 0 2px 0 !important;
  padding: 0 !important; }}
.gpl-guidance {{ color: var(--gpl-muted); margin: 0; font-size: 0.95rem; }}
.gpl-meta {{ color: var(--gpl-muted); font-size: 0.78rem; text-align: right; margin: 0; }}
.gpl-eyebrow {{ color: var(--gpl-brand-btn); font-size: .72rem; font-weight: 700;
  letter-spacing: .12em; text-transform: uppercase; }}
.gpl-section {{ font-size: 1.05rem; font-weight: 600; margin: {t.space(1)} 0; }}

/* ---- buttons: brand primary; one pink call to action per screen (keys "cta_*") */
button[kind="primary"], [data-testid="stBaseButton-primary"] {{
  background: var(--gpl-brand-btn) !important; border-color: var(--gpl-brand-btn) !important;
  color: #fff !important; font-weight: 600; }}
[class*="st-key-cta_"] button {{ background: var(--gpl-accent-btn) !important;
  border-color: var(--gpl-accent-btn) !important; color: #fff !important; font-weight: 600;
  box-shadow: 0 6px 16px rgba(196, 44, 130, .25); }}
[class*="st-key-cta_"] button p, button[kind="primary"] p,
[data-testid="stBaseButton-primary"] p {{ color: #fff !important; }}
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondary"]:hover {{
  border-radius: 10px; }}
[data-testid="stBaseButton-tertiary"] {{ color: var(--gpl-brand-btn) !important; }}
[data-testid="stChatInputSubmitButton"] {{ background: var(--gpl-accent-btn) !important;
  color: #fff !important; border-radius: 8px; }}
[data-testid="stChatInputSubmitButton"] svg {{ fill: #fff !important; color: #fff !important; }}

/* ---- cards: bordered containers, metrics, expanders, tables */
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stVerticalBlock"][data-border="true"] {{
  background: var(--gpl-card); border-color: var(--gpl-border) !important;
  border-radius: var(--gpl-radius) !important; }}
[data-testid="stMetric"] {{ background: var(--gpl-card); border: 1px solid var(--gpl-border);
  border-radius: var(--gpl-radius); padding: {t.space(2)}; }}
[data-testid="stMetricValue"] {{ font-weight: 700; }}
[data-testid="stExpander"] details {{ background: var(--gpl-card);
  border: 1px solid var(--gpl-border); border-radius: var(--gpl-radius); }}
[data-testid="stExpander"] summary p {{ font-weight: 600; }}
[data-testid="stTable"] table {{ background: var(--gpl-card); }}
[data-testid="stTable"] th {{ background: var(--gpl-subtle); color: var(--gpl-muted);
  font-weight: 600; }}
/* (?) icons sit right after their label, not at the far end of the row */
[data-testid="stWidgetLabel"] {{ justify-content: flex-start !important; gap: 6px; }}
[data-testid="stWidgetLabel"] > div {{ flex: 0 1 auto !important; width: auto !important; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
  color: var(--gpl-muted); }}

/* ---- sub-view tabs as segmented pills (Analyse keeps an underline row, as in the mockup) */
[data-testid="stTabs"]:not(.st-key-main_tab):not(.st-key-analyse_tab) [role="tablist"] {{
  background: var(--gpl-subtle); border-radius: 10px; padding: 4px; gap: 2px;
  width: fit-content; }}
[data-testid="stTabs"]:not(.st-key-main_tab):not(.st-key-analyse_tab) [role="tab"] {{
  border-radius: 8px; padding: 6px 14px; height: auto; }}
[data-testid="stTabs"]:not(.st-key-main_tab):not(.st-key-analyse_tab) [aria-selected="true"] {{
  background: #fff; box-shadow: 0 1px 3px rgba(26,37,48,.12); }}
[data-testid="stTabs"]:not(.st-key-main_tab):not(.st-key-analyse_tab)
  [data-baseweb="tab-highlight"],
[data-testid="stTabs"]:not(.st-key-main_tab):not(.st-key-analyse_tab) [data-baseweb="tab-border"] {{
  display: none; }}
[data-testid="stTabs"] [aria-selected="true"] p {{ font-weight: 600; }}
.st-key-analyse_tab [aria-selected="true"] p {{ color: var(--gpl-brand-btn); }}
.st-key-analyse_tab [data-baseweb="tab-highlight"] {{ background: var(--gpl-brand-btn); }}
.st-key-optimise_tab > div > [role="tablist"],
.st-key-optimise_tab > div > div > [role="tablist"] {{
  display: none; }}

/* ---- KPI cards and pills */
.gpl-kpis {{ display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr));
  gap: {t.space(2)}; margin: {t.space(1)} 0 {t.space(2)} 0; }}
.gpl-kpi {{ background: var(--gpl-card); border: 1px solid var(--gpl-border);
  border-radius: var(--gpl-radius); padding: {t.space(2)}; min-width: 0; }}
.gpl-kpi.warn {{ background: var(--gpl-warn-tint); border-color: #F3D9A8; }}
.gpl-kpi-label {{ font-size: .8rem; color: var(--gpl-muted); }}
.gpl-kpi.warn .gpl-kpi-label {{ color: var(--gpl-warn); font-weight: 600; }}
.gpl-kpi-value {{ font-size: 1.7rem; font-weight: 700; margin-top: 6px; line-height: 1.15;
  white-space: nowrap; }}
.gpl-kpi.warn .gpl-kpi-value {{ font-size: .9rem; font-weight: 400; white-space: normal; }}
.gpl-before {{ font-size: 1rem; font-weight: 400; color: var(--gpl-muted); }}
.gpl-arrow {{ font-size: .9rem; color: var(--gpl-muted); margin: 0 8px; }}
.gpl-kpi-sub {{ font-size: .78rem; color: var(--gpl-muted); margin-top: 6px; }}
.gpl-kpi-delta {{ margin-top: 8px; }}
.gpl-pill {{ display: inline-block; font-size: .74rem; font-weight: 600; padding: 2px 8px;
  border-radius: 999px; background: var(--gpl-subtle); color: var(--gpl-muted);
  white-space: nowrap; }}
.gpl-pill.pass {{ background: var(--gpl-pass-tint); color: var(--gpl-pass); }}
.gpl-pill.fail {{ background: var(--gpl-fail-tint); color: var(--gpl-fail); }}
.gpl-pill.warn {{ background: var(--gpl-warn-tint); color: var(--gpl-warn); }}
.gpl-pill.brand {{ background: var(--gpl-brand-tint); color: var(--gpl-brand-btn); }}
.gpl-pill.accent {{ background: var(--gpl-accent-btn); color: #fff; letter-spacing: .04em; }}
.gpl-pill.dark {{ background: var(--gpl-text); color: #fff; }}

/* ---- HTML tables (suitability, candidates) and list rows */
.gpl-table {{ width: 100%; border-collapse: collapse; font-size: .86rem; }}
.gpl-table th {{ text-align: left; font-weight: 600; color: var(--gpl-muted); font-size: .76rem;
  padding: 10px 12px; border-bottom: 1px solid var(--gpl-border); }}
.gpl-table td {{ padding: 10px 12px; border-bottom: 1px solid var(--gpl-border);
  vertical-align: top; }}
.gpl-table td small {{ display: block; color: var(--gpl-muted); font-size: .74rem;
  margin-top: 2px; }}
.gpl-table .num {{ text-align: right; white-space: nowrap; }}
.gpl-table .rec {{ background: var(--gpl-accent-tint); font-weight: 600; }}
.gpl-table th.rec {{ color: var(--gpl-accent-btn); }}
.gpl-table tr:last-child td {{ border-bottom: none; }}
.gpl-rows {{ display: flex; flex-direction: column; gap: 8px; }}
.gpl-row {{ display: flex; justify-content: space-between; align-items: center; gap: 12px;
  border: 1px solid var(--gpl-border); border-radius: 10px; padding: 10px 14px; }}
.gpl-row.rec {{ border: 2px solid var(--gpl-accent); background: var(--gpl-accent-tint); }}
.gpl-row b {{ font-weight: 600; }}
.gpl-row small {{ display: block; color: var(--gpl-muted); font-size: .76rem; }}
.gpl-row small.gpl-why {{ color: var(--gpl-text); margin-top: 2px; }}
.gpl-row .gpl-pill {{ white-space: normal; text-align: right; max-width: 45%; }}
.gpl-kv {{ display: grid; grid-template-columns: 34% 66%; font-size: .84rem; }}
.gpl-kv div {{ padding: 8px 0; border-bottom: 1px solid var(--gpl-border); }}
.gpl-kv div:nth-child(odd) {{ color: var(--gpl-muted); }}
.gpl-bar {{ display: flex; gap: 3px; }}
.gpl-bar i {{ width: 22px; height: 6px; border-radius: 3px; background: var(--gpl-border); }}
.gpl-bar i.on {{ background: var(--gpl-brand); }}

/* ---- four-part summary (Optimise) */
.gpl-summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  border: 1px solid var(--gpl-border); border-radius: var(--gpl-radius); overflow: hidden;
  background: var(--gpl-card); margin: {t.space(1)} 0; }}
.gpl-summary > div {{ padding: {t.space(2.5)}; border-left: 1px solid var(--gpl-border);
  font-size: .9rem; line-height: 1.5; }}
.gpl-summary > div:first-child {{ border-left: none; }}
.gpl-summary > div.rec {{ background: var(--gpl-accent-tint); }}
.gpl-summary h5 {{ font-size: .72rem !important; letter-spacing: .1em; text-transform: uppercase;
  color: var(--gpl-muted) !important; margin: 0 0 8px 0 !important; padding: 0 !important; }}
.gpl-summary > div.rec h5 {{ color: var(--gpl-accent-btn) !important; }}
.gpl-answer {{ font-size: 1.25rem; font-weight: 650; line-height: 1.4; }}
.gpl-explain {{ color: var(--gpl-text); background: var(--gpl-subtle); border-radius: 10px;
  padding: 12px 14px; font-size: .9rem; margin: 8px 0; }}
.gpl-runwith {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: .82rem; }}

/* ---- the result card: scrolled into view below the sticky header, briefly highlighted */
.st-key-result_summary {{ scroll-margin-top: 9rem; border-radius: var(--gpl-radius);
  animation: gpl-highlight 2.4s ease-out 1; }}
@keyframes gpl-highlight {{
  0% {{ box-shadow: 0 0 0 4px rgba(5,147,198,.45); }}
  100% {{ box-shadow: 0 0 0 0 rgba(5,147,198,0); }}
}}

/* ---- welcome hero */
.st-key-welcome {{ padding: {t.space(5)} {t.space(3)} !important; }}
.st-key-welcome [data-testid="stMarkdownContainer"] {{ text-align: center; }}
.st-key-welcome h1 {{ font-size: 2.2rem !important; font-weight: 700 !important; line-height: 1.2;
  max-width: 560px; margin: 8px auto !important; text-align: center; }}
.st-key-welcome .gpl-lead {{ color: var(--gpl-muted); max-width: 500px; margin: 0 auto 8px auto;
  text-align: center; }}
.st-key-cta_welcome_demo, .st-key-welcome_blank {{ width: 100% !important; display: flex;
  justify-content: center; }}
.st-key-cta_welcome_demo button {{ font-size: 1.05rem; min-height: 3rem; padding: 0 2rem; }}

/* ---- sticky Back/Next bar */
[class*="st-key-footer_"] {{ position: fixed; bottom: 0; left: 0; right: 0; z-index: 980;
  background: #fff; border-top: 1px solid var(--gpl-border);
  padding: 10px max(180px, calc((100vw - {t.MAX_WIDTH_PX}px) / 2)) 10px
    max(24px, calc((100vw - {t.MAX_WIDTH_PX}px) / 2)); }}
/* the right padding keeps the buttons clear of Streamlit Cloud's "Manage app" badge */
.gpl-foot {{ font-size: .82rem; line-height: 1.3; white-space: normal; overflow-wrap: anywhere; }}
.gpl-foot small {{ display: block; color: var(--gpl-muted); font-size: .72rem; }}
</style>
"""
