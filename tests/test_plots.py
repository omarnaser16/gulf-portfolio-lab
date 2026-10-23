"""Smoke tests for gpl.plots: figures build with the right traces and colours."""

import numpy as np
import pandas as pd
import pytest

from gpl.plots import (
    SERIES_COLORS,
    drawdown_chart,
    equity_chart,
    fan_chart,
    frontier_chart,
    regime_chart,
    risk_contribution_chart,
    weights_chart,
)

IDX = pd.bdate_range("2024-01-01", periods=5)


def test_equity_chart_one_line_per_strategy() -> None:
    eq = pd.DataFrame({"EW": np.linspace(1, 1.1, 5), "SPY": np.linspace(1, 1.2, 5)}, index=IDX)
    fig = equity_chart(eq)
    assert [t.name for t in fig.data] == ["EW", "SPY"]
    assert [t.line.color for t in fig.data] == SERIES_COLORS[:2]


def test_drawdown_chart() -> None:
    fig = drawdown_chart(pd.DataFrame({"EW": [0, -0.1, -0.05, 0, 0]}, index=IDX))
    assert len(fig.data) == 1


def test_too_many_series_raises() -> None:
    with pytest.raises(ValueError):
        equity_chart(pd.DataFrame(np.ones((5, 9)), index=IDX))


def test_weights_chart_groups_assets() -> None:
    w = pd.DataFrame({"SPY": [0.5, 0.4], "AGG": [0.3, 0.4], "TLT": [0.2, 0.2]}, index=IDX[:2])
    fig = weights_chart(w, groups={"Equity": ["SPY"], "Bonds": ["AGG", "TLT"]})
    assert [t.name for t in fig.data] == ["Equity", "Bonds"]
    np.testing.assert_allclose(fig.data[1].y, [0.5, 0.6])


def test_frontier_chart_marks_portfolios() -> None:
    front = pd.DataFrame({"ret": [0.04, 0.06], "vol": [0.05, 0.08]})
    fig = frontier_chart(front, {"Max Sharpe": (0.07, 0.055)})
    assert [t.name for t in fig.data] == ["Frontier", "Max Sharpe"]


def test_fan_chart_has_bands_and_median() -> None:
    paths = np.vstack([np.ones(100), np.linspace(0.8, 1.4, 100)])
    fig = fan_chart(paths)
    assert fig.data[-1].name == "Median"
    assert len(fig.data) == 5


def test_regime_chart_shades_realised_stress() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    p = pd.Series([0.1, 0.6, 0.7, 0.2, 0.1, 0.8], index=idx)
    realised = pd.Series([0, 1, 1, 0, 0, 1], index=idx)
    fig = regime_chart(p, 0.5, realised)
    rects = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(rects) == 2  # two stressed stretches
    assert fig.data[0].name == "P(stressed)"


def test_frontier_labels_point_away_from_the_line() -> None:
    # Frontier from (5%, 4%) to (15%, 10%). Max Sharpe sits on it; equal weight is inside.
    front = pd.DataFrame({"vol": [0.05, 0.08, 0.15], "ret": [0.04, 0.07, 0.10]})
    fig = frontier_chart(front, {"Max Sharpe": (0.08, 0.07), "Equal weight": (0.12, 0.06)})
    labels = {a.text: a for a in fig.layout.annotations}
    on, inside = labels["Max Sharpe"], labels["Equal weight"]
    assert on.ax < 0 and on.ay < 0  # up-left: the empty side of the frontier
    assert inside.ax > 0 and inside.ay > 0  # down-right: away from the curve
    assert all(t.mode == "markers" for t in fig.data[1:])  # no text glued to markers


def test_percent_axes_use_two_decimals() -> None:
    dd = drawdown_chart(pd.DataFrame({"EW": [0, -0.1, -0.05, 0, 0]}, index=IDX))
    assert dd.layout.yaxis.tickformat == ".2%"
    front = frontier_chart(pd.DataFrame({"ret": [0.04, 0.06], "vol": [0.05, 0.08]}))
    assert front.layout.xaxis.tickformat == ".2%" and front.layout.yaxis.tickformat == ".2%"


def test_regime_threshold_is_in_the_legend_not_on_the_data() -> None:
    idx = pd.bdate_range("2024-01-01", periods=4)
    fig = regime_chart(pd.Series([0.1, 0.6, 0.7, 0.2], index=idx), 0.5)
    assert "Threshold 50.00%" in [t.name for t in fig.data]
    assert not [a for a in fig.layout.annotations if "hreshold" in (a.text or "")]


def test_risk_contribution_chart_pairs_weight_and_risk() -> None:
    w = pd.Series({"SPY": 0.6, "AGG": 0.4})
    rc = pd.Series({"SPY": 0.9, "AGG": 0.1})
    fig = risk_contribution_chart(w, rc)
    assert [t.name for t in fig.data] == ["Weight", "Share of risk"]
    assert list(fig.data[1].y) == ["AGG", "SPY"]  # sorted by risk share
    np.testing.assert_allclose(fig.data[0].x, [0.4, 0.6])
