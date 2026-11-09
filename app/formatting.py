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


def money(x: float | None, decimals: int = 0, prefix: str = "USD ") -> str:
    """One money format everywhere: 'USD 1,269,900' (thousands separators); never a negative
    zero ('USD -0' reads 'USD 0'); '–' when unknown."""
    if x is None or pd.isna(x):
        return "–"
    return f"{prefix}{no_negative_zero(float(x), decimals):,.{decimals}f}"


def compact_usd(x: float | None) -> str:
    """'$3.00m', '$250.0k', '$1.25bn', '$950' (fits a KPI card); '–' when unknown."""
    if x is None:
        return "–"
    sign, x = ("-" if x < 0 else ""), abs(x)  # '-$1.20m', not '$-1.20m'
    for size, suffix, decimals in ((1e9, "bn", 2), (1e6, "m", 2), (1e3, "k", 1)):
        if x >= size:
            return f"{sign}${x / size:,.{decimals}f}{suffix}"
    return f"{sign}${x:,.0f}" if round(x) else "$0"
