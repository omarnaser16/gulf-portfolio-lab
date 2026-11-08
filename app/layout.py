"""Workflow layout: the six-step stepper, the sticky client header and "Next" buttons.

Pure presentation: every figure shown is passed in (computed by tested code elsewhere).
Steps are the top-level tabs (``st.tabs(key=STEP_KEY)``), so a button can move to a step by
setting ``st.session_state[STEP_KEY]`` in a callback. The tab row itself is hidden (CSS): the
step cards are the navigation, each with a full-size button over it.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from html import escape

import streamlit as st

STEP_KEY = "main_tab"
STEPS = [
    "1 Client",
    "2 Portfolio",
    "3 Analyse",
    "4 Optimise",
    "5 Suitability",
    "6 Report & trades",
]
REPORT_KEY = "report_tab"
REPORT_TABS = ["Trades", "Reports", "AI analyst"]


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


def go_to(step: str, report_tab: str | None = None) -> None:
    """Callback: open ``step`` (and a Report & trades sub-tab)."""
    st.session_state[STEP_KEY] = step
    if report_tab:
        st.session_state[REPORT_KEY] = report_tab


def render_header(container, info: HeaderInfo) -> None:
    """Client, risk score, portfolio value and IPS status as compact chips."""
    chips = [
        ("Client", info.client or "none loaded", "" if info.client else "muted"),
        ("Risk score", f"{info.risk_score} / 5" if info.risk_score else "–", ""),
        ("Portfolio", info.value, "" if info.value != "not set" else "muted"),
        (
            "IPS",
            info.ips_status,
            {"Ready": "ok", "Draft": "warn"}.get(info.ips_status, "muted"),
        ),
    ]
    html = "".join(
        f'<span class="gpl-chip {cls}"><b>{escape(label)}</b> {escape(value)}</span>'
        for label, value, cls in chips
    )
    if info.demo:
        seed = f" · seed {info.seed}" if info.seed is not None else ""
        html += f'<span class="gpl-chip demo">DEMO DATA{seed}</span>'
    container.markdown(f'<div class="gpl-chips">{html}</div>', unsafe_allow_html=True)


def render_stepper(statuses: Sequence[StepStatus], active: str | None) -> None:
    """The six steps with done / to-do marks; click a card to open its step."""
    columns = st.columns(len(STEPS), gap="small")
    for i, (column, step, status) in enumerate(zip(columns, STEPS, statuses, strict=True)):
        number, title = step.split(" ", 1)
        cls = "done" if status.done else "todo"
        if step == active:
            cls += " active"
        mark = "✓" if status.done else number
        with column.container(key=f"step_card_{i}"):
            st.markdown(
                f'<div class="gpl-step {cls}"><span class="gpl-dot">{mark}</span>'
                f'<span class="gpl-step-text"><b>{escape(title)}</b>'
                f"<small>{escape(status.note)}</small></span></div>",
                unsafe_allow_html=True,
            )
            st.button(
                f"Open step {step}",
                key=f"step_nav_{i}",
                on_click=go_to,
                args=(step,),
                help=f"Open {title}: {status.note}",
            )


def step_footer(step: str, status: StepStatus) -> None:
    """What is still missing in this step, and a button to the next one."""
    st.divider()
    left, right = st.columns([3, 1], vertical_alignment="center")
    if status.missing:
        left.caption("**Still to do:** " + "; ".join(status.missing))
    else:
        left.caption(f"**{step.split(' ', 1)[1]}:** {status.note}")
    index = STEPS.index(step)
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


# Sticky header, chips and stepper. Colours come from the theme tokens in formatting.py.
LAYOUT_CSS = """
<style>
/* the result summary: scrolled into view below the sticky header, with a brief highlight */
.st-key-result_summary {
  scroll-margin-top: 6rem; border-radius: 0.6rem;
  animation: gpl-highlight 2.4s ease-out 1;
}
@keyframes gpl-highlight {
  0% { box-shadow: 0 0 0 4px rgba(176, 141, 87, 0.75); background: rgba(176, 141, 87, 0.12); }
  100% { box-shadow: 0 0 0 0 rgba(176, 141, 87, 0); background: transparent; }
}
.st-key-gpl_header {
  position: sticky; top: 0; z-index: 990; padding: 0.35rem 0 0.25rem 0;
  background: var(--gpl-bg); border-bottom: 1px solid var(--gpl-border);
}
/* the demo button: the primary call to action, slightly larger than the others */
.st-key-fill_demo button, .st-key-welcome_demo button, .st-key-client_try_demo button {
  font-weight: 650; font-size: 1.02rem; min-height: 2.75rem;
}
.st-key-welcome_demo button, .st-key-client_try_demo button {
  font-size: 1.1rem; min-height: 3.1rem; padding: 0 1.6rem;
}
.st-key-welcome { background: var(--gpl-card); }
.st-key-gpl_header h3 { margin: 0; padding: 0; white-space: nowrap; color: var(--gpl-navy); }
.gpl-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; }
.gpl-chip {
  font-size: 0.8rem; padding: 0.2rem 0.6rem; border-radius: 999px;
  border: 1px solid var(--gpl-border); background: var(--gpl-card);
  white-space: nowrap;
}
.gpl-chip b { font-weight: 600; opacity: 0.7; margin-right: 0.15rem; }
.gpl-chip.ok { border-color: var(--gpl-pass); color: var(--gpl-pass); }
.gpl-chip.warn { border-color: var(--gpl-warn); color: var(--gpl-warn); }
.gpl-chip.muted { opacity: 0.65; }
.gpl-chip.demo {
  background: var(--gpl-accent); color: #fff; border-color: transparent;
  font-weight: 700; letter-spacing: 0.04em;
}
/* the step cards are the navigation: the top-level tab row is hidden */
.st-key-main_tab > div > [role="tablist"] { display: none; }
[class*="st-key-step_card_"] { position: relative; }
[class*="st-key-step_nav_"] { position: absolute; inset: 0; z-index: 5; margin: 0; }
[class*="st-key-step_nav_"] button {
  width: 100%; height: 100%; opacity: 0; cursor: pointer;
}
[class*="st-key-step_card_"]:hover .gpl-step { border-color: var(--gpl-accent); }
.gpl-step {
  height: 100%; display: flex; gap: 0.5rem; align-items: center; padding: 0.45rem 0.6rem;
  border: 1px solid var(--gpl-border); border-radius: 0.5rem;
  background: var(--gpl-card); min-width: 0;
}
.gpl-step.active {
  border-color: var(--gpl-accent); box-shadow: 0 0 0 1px var(--gpl-accent);
}
.gpl-dot {
  flex: none; width: 1.6rem; height: 1.6rem; border-radius: 50%; display: flex;
  align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem;
  border: 1.5px solid var(--gpl-muted); color: var(--gpl-muted);
}
.gpl-step.done .gpl-dot {
  background: var(--gpl-pass); border-color: var(--gpl-pass); color: #fff;
}
.gpl-step.active .gpl-dot {
  border-color: var(--gpl-accent); color: var(--gpl-accent);
}
.gpl-step.active.done .gpl-dot { color: #fff; }
.gpl-step-text { display: flex; flex-direction: column; min-width: 0; line-height: 1.2; }
.gpl-step-text b { font-size: 0.86rem; }
.gpl-step-text small {
  font-size: 0.72rem; opacity: 0.75; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
</style>
"""
