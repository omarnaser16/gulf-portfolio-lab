"""Tests for the dashboard's display formatting (app/formatting.py)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from formatting import pct, style_metrics, style_percent  # noqa: E402


@pytest.mark.parametrize(
    ("value", "text"),
    [(0.09, "9.00%"), (0.018, "1.80%"), (0.007, "0.70%"), (-0.3372, "-33.72%"), (1.0, "100.00%")],
)
def test_pct_always_two_decimals(value: float, text: str) -> None:
    assert pct(value) == text


@pytest.fixture
def summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ann_return": [0.09],
            "cagr": [0.018],
            "ann_vol": [0.1],
            "sharpe": [0.5],
            "sortino": [0.7],
            "max_drawdown": [-0.2],
            "hist_var": [0.007],
            "param_var": [0.008],
            "expected_shortfall": [0.01],
            "start": [pd.Timestamp("2020-01-01")],
            "end": [pd.Timestamp("2021-01-01")],
            "n_obs": [252],
            "annual_turnover": [0.3],
        },
        index=["X"],
    )


def test_metrics_table_headers_order_and_formats(summary: pd.DataFrame) -> None:
    styled = style_metrics(summary, 0.95)
    assert list(styled.data.columns) == [
        "Return (ann.)", "CAGR", "Volatility (ann.)", "Sharpe", "Sortino", "Max drawdown",
        "Turnover / yr", "VaR 95% (historical)", "VaR 95% (normal)", "Expected shortfall 95%",
    ]  # fmt: skip
    html = styled.to_html()
    for text in ["9.00%", "1.80%", "0.70%", "-20.00%", "0.50", "0.30×"]:
        assert text in html
    assert "2020" not in html and "252" not in html  # start/end/n_obs dropped


def test_style_percent_formats_numbers_only() -> None:
    table = pd.DataFrame({"when": ["Jan 2020"], "share": [0.2]})
    html = style_percent(table).to_html()
    assert "20.00%" in html and "Jan 2020" in html
