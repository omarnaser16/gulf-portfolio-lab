"""Display formatting for the dashboard: column labels, number formats, table styling.

Presentation only. Percentages are always shown with exactly two decimals (``9.00%``).
"""

import pandas as pd
import streamlit as st
from pandas.io.formats.style import Styler

PCT = "{:.2%}"


def pct(x: float) -> str:
    """Fraction -> percentage string with exactly two decimals, e.g. 0.09 -> '9.00%'."""
    return PCT.format(x)


def metric_columns(var_level: float) -> dict[str, tuple[str, str]]:
    """``summary_table`` column -> (header, value format). Dict order = screen order."""
    level = f"{var_level:.0%}"  # a confidence-level label, not a data value
    return {
        "ann_return": ("Return (ann.)", PCT),
        "cagr": ("CAGR", PCT),
        "ann_vol": ("Volatility (ann.)", PCT),
        "sharpe": ("Sharpe", "{:.2f}"),
        "sortino": ("Sortino", "{:.2f}"),
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
