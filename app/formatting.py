"""Display formatting for the dashboard: column labels, number formats, table styling.

Presentation only. Percentages are always shown with exactly two decimals (``9.00%``).
"""

import pandas as pd
import streamlit as st
from pandas.io.formats.style import Styler

from gpl.metrics import no_negative_zero


def pct(x: float) -> str:
    """Fraction -> percentage string with exactly two decimals, e.g. 0.09 -> '9.00%'.
    A value that rounds to zero shows as '0.00%', never '-0.00%'."""
    return f"{no_negative_zero(x, 4):.2%}"


def two_dp(x: float) -> str:
    """Two decimals, e.g. a Sharpe ratio; '0.00', never '-0.00'."""
    return f"{no_negative_zero(x, 2):.2f}"


PCT = pct  # callable format, used by pandas Styler


def metric_columns(var_level: float) -> dict[str, tuple[str, str]]:
    """``summary_table`` column -> (header, value format). Dict order = screen order."""
    level = f"{var_level:.0%}"  # a confidence-level label, not a data value
    return {
        "ann_return": ("Return (ann.)", PCT),
        "cagr": ("CAGR", PCT),
        "ann_vol": ("Volatility (ann.)", PCT),
        "sharpe": ("Sharpe", two_dp),
        "sortino": ("Sortino", two_dp),
        "max_drawdown": ("Max drawdown", PCT),
        "annual_turnover": ("Turnover / yr", "{:.2f}×"),
        "hist_var": (f"VaR {level} (historical)", PCT),
        "param_var": (f"VaR {level} (normal)", PCT),
        "expected_shortfall": (f"Expected shortfall {level}", PCT),
    }


def style_metrics(table: pd.DataFrame, var_level: float) -> Styler:
    """Readable headers, 2-decimal percentages; drops start/end/n_obs (captioned instead)."""
    spec = metric_columns(var_level)
    columns = [c for c in spec if c in table.columns]
    shown = table[columns].rename(columns={c: spec[c][0] for c in columns})
    return shown.style.format({spec[c][0]: spec[c][1] for c in columns})


def show_metrics_table(table: pd.DataFrame, var_level: float) -> None:
    """Static table that fits the page width and wraps long headers (no horizontal scroll).

    ``st.table`` instead of ``st.dataframe``: the interactive grid cannot wrap headers, so
    ten metric columns overflowed. Rows arrive pre-sorted, so click-to-sort isn't needed.
    """
    st.table(style_metrics(table, var_level))


def style_percent(table: pd.DataFrame) -> Styler:
    """Every numeric cell is a fraction -> 2-decimal percentage."""
    numeric = list(table.select_dtypes("number").columns)
    return table.style.format(PCT, subset=numeric)


def style_columns(table: pd.DataFrame, formats: dict[str, str] | None = None) -> Styler:
    """Numeric cells as 2-decimal percentages, except columns given their own format."""
    formats = formats or {}
    numeric = table.select_dtypes("number").columns
    return table.style.format({c: formats.get(c, PCT) for c in numeric})


def plot(fig, **kwargs) -> None:
    """Show a plotly figure with the project's ``gpl_light`` template (not Streamlit's theme)."""
    kwargs.setdefault("width", "stretch")
    st.plotly_chart(fig, theme=None, **kwargs)


# Light private-bank palette (also in .streamlit/config.toml and the PDF colours in
# gpl/report.py). Green and red are reserved for gains, losses and pass/fail.
THEME = {
    "bg": "#F7F6F2",  # off-white page
    "card": "#FFFFFF",  # cards, tables, charts
    "border": "#E3E1DA",  # subtle card borders
    "navy": "#1B2A4A",  # primary: headings, header
    "accent": "#2563EB",  # azure: buttons, highlights, active tabs
    "text": "#2B2B2B",
    "muted": "#6B6B6B",
    "pass": "#2E7D4F",  # gains, pass
    "fail": "#B23B3B",  # losses, fail
    "warn": "#A66A00",  # review, draft
}

# Minimal, robust CSS: tabular monospaced figures in metric cards and table cells (labels and
# headers keep the normal font so they don't wrap); white bordered cards for metrics, tables
# and expanders; navy headings; the theme colours as CSS variables for app/layout.py.
NUMBERS_CSS = f"""
<style>
:root {{
  --gpl-bg: {THEME["bg"]}; --gpl-card: {THEME["card"]}; --gpl-border: {THEME["border"]};
  --gpl-navy: {THEME["navy"]}; --gpl-accent: {THEME["accent"]}; --gpl-muted: {THEME["muted"]};
  --gpl-pass: {THEME["pass"]}; --gpl-fail: {THEME["fail"]}; --gpl-warn: {THEME["warn"]};
}}
h1, h2, h3, h4 {{ color: var(--gpl-navy) !important; }}
[data-testid="stMetric"] {{
  background: var(--gpl-card); border: 1px solid var(--gpl-border); border-radius: 0.5rem;
  padding: 0.6rem 0.8rem;
}}
[data-testid="stExpander"] details {{
  background: var(--gpl-card); border: 1px solid var(--gpl-border); border-radius: 0.5rem;
}}
[data-testid="stTable"] table {{ background: var(--gpl-card); }}
[data-testid="stTable"] th {{ background: #F1F0EB; color: var(--gpl-navy); }}
[data-testid="stTabs"] [aria-selected="true"] p {{ font-weight: 600; }}
</style>
<style>
[data-testid="stMetricValue"], [data-testid="stMetricDelta"], [data-testid="stTable"] td {{
  font-family: SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-variant-numeric: tabular-nums;
}}
</style>
"""


def compact_usd(x: float | None) -> str:
    """'$3.00m', '$250.0k', '$1.25bn', '$950' (fits a KPI card); '–' when unknown."""
    if x is None:
        return "–"
    for size, suffix, decimals in ((1e9, "bn", 2), (1e6, "m", 2), (1e3, "k", 1)):
        if abs(x) >= size:
            return f"${x / size:,.{decimals}f}{suffix}"
    return f"${x:,.0f}"
