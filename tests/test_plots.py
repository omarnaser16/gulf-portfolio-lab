"""Smoke tests for gpl.plots: figures build with the right traces and colours."""

import numpy as np
import pandas as pd
import pytest

from gpl import plots
from gpl.plots import (
    SERIES_COLORS,
    allocation_chart,
    drawdown_chart,
    equity_chart,
    fan_chart,
    frontier_chart,
    paired_bars_chart,
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
    assert [t.name for t in fig.data] == ["Share of money", "Share of risk"]
    assert [t.marker.color for t in fig.data] == ["#7CC4E4", "#1F3A5F"]  # light blue, navy
    assert list(fig.data[1].y) == ["AGG", "SPY"]  # sorted by risk share
    np.testing.assert_allclose(fig.data[0].x, [0.4, 0.6])


def test_allocation_chart_groups_small_slices() -> None:
    w = pd.Series(
        {f"T{i}": v for i, v in enumerate([0.3, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03])}
    )
    fig = allocation_chart(w)
    labels = list(fig.data[0].labels)
    assert len(labels) == 8 and labels[-1] == "Others (3)"
    assert fig.data[0].values[-1] == pytest.approx(0.10)
    assert list(allocation_chart(pd.Series({"A": 0.6, "B": 0.4})).data[0].labels) == ["A", "B"]


def test_paired_bars_chart_current_vs_optimised() -> None:
    now = pd.Series({"SPY": 0.5, "AGG": 0.5})
    new = pd.Series({"SPY": 0.25, "AGG": 0.6, "KSA": 0.15})
    fig = paired_bars_chart(now, new, ("Current", "Optimised"), "Weights")
    assert [t.name for t in fig.data] == ["Current", "Optimised"]
    assert list(fig.data[0].y) == ["KSA", "SPY", "AGG"]  # ordered by optimised weight
    np.testing.assert_allclose(fig.data[0].x, [0.0, 0.5, 0.5])  # KSA not held now: 0


def test_allocation_chart_sends_plain_lists() -> None:
    # numpy values serialise as base64 typed arrays, which some plotly.js builds don't draw
    import json

    data = json.loads(allocation_chart(pd.Series({"A": 0.6, "B": 0.4})).to_json())["data"][0]
    assert data["values"] == [0.6, 0.4] and data["labels"] == ["A", "B"]


def test_dark_template_and_reference_series() -> None:
    from gpl.plots import REFERENCE_COLOR, SURFACE, series_styles

    styles = series_styles(["Current", "Benchmark", "Proposed"], reference="Benchmark")
    assert styles["Benchmark"] == {"color": REFERENCE_COLOR, "width": 2, "dash": "dash"}
    assert styles["Current"]["color"] == SERIES_COLORS[0]  # amber
    assert styles["Proposed"]["color"] == SERIES_COLORS[1]  # cyan: the reference takes no slot
    idx = pd.bdate_range("2024-01-01", periods=3)
    fig = equity_chart(
        pd.DataFrame({"A": [1, 1.1, 1.2], "Benchmark": [1, 1, 1]}, index=idx), reference="Benchmark"
    )
    assert fig.layout.paper_bgcolor == SURFACE and fig.data[1].line.dash == "dash"
    # set on each figure, not only as plotly's default (importing Streamlit changes that)
    assert fig.layout.template.layout.colorway == tuple(SERIES_COLORS)


def test_several_benchmarks_are_grey_with_different_dashes_and_take_no_slot() -> None:
    import pandas as pd

    from gpl.plots import (
        REFERENCE_COLOR,
        SERIES_COLORS,
        drawdown_chart,
        equity_chart,
        series_styles,
    )

    styles = series_styles(["Max Sharpe", "60/40", "Risk parity", "SPY"], ["60/40", "SPY"])
    assert styles["60/40"] == {"color": REFERENCE_COLOR, "width": 2, "dash": "dash"}
    assert styles["SPY"] == {"color": REFERENCE_COLOR, "width": 2, "dash": "dot"}
    assert styles["Risk parity"]["color"] == SERIES_COLORS[1]  # benchmarks skip the slots
    frame = pd.DataFrame(
        {n: [1.0, 1.1] for n in [*(f"S{i}" for i in range(8)), "60/40", "SPY"]},
        index=pd.bdate_range("2024-01-01", periods=2),
    )
    fig = equity_chart(frame, reference=["60/40", "SPY", "not shown"])  # 8 slots + 2 grey
    dashes = {t.name: t.line.dash for t in fig.data}
    assert dashes["60/40"] == "dash" and dashes["SPY"] == "dot" and dashes["S0"] is None
    assert drawdown_chart(frame - 1, reference=["60/40", "SPY"]).data[8].line.dash == "dash"


def test_allocation_donut_centre_text_and_hidden_labels() -> None:
    fig = allocation_chart(pd.Series({"A": 0.6, "B": 0.4}), "", center="Total<br>$360K<br>2",
                           labels=False)  # fmt: skip
    texts = [a.text for a in fig.layout.annotations]
    ys = [a.y for a in fig.layout.annotations]
    assert texts == ["Total", "$360K", "2"] and ys[0] > ys[1] > ys[2]  # separate lines
    assert ys[0] - ys[1] >= 0.1  # far enough apart not to overlap
    assert fig.data[0].texttemplate == "" and fig.data[0].textposition == "none"
    assert list(fig.data[0].marker.colors) == ["#1F3A5F", "#0593C6"]  # the composition palette


def test_risk_return_chart_highlights_the_recommendation() -> None:
    from gpl.plots import risk_return_chart

    pts = pd.DataFrame(
        {"volatility": [0.12, 0.11, 0.09, 0.05], "cagr": [0.08, 0.09, 0.085, 0.05]},
        index=["Current", "Benchmark", "Max Sharpe", "Min variance"],
    )
    fig = risk_return_chart(pts, highlight="Max Sharpe")
    by = {t.name: t for t in fig.data}
    assert by["Max Sharpe"].marker.color == "#D83995" and by["Max Sharpe"].marker.size == 16
    assert by["Current"].marker.color == "#0593C6"
    assert by["Benchmark"].marker.symbol == "diamond"
    assert list(by["Min variance"].x) == [0.05] and list(by["Min variance"].y) == [0.05]
    assert fig.layout.xaxis.title.text.startswith("Volatility")


def test_equity_chart_can_hide_series_in_the_legend() -> None:
    eq = pd.DataFrame({"Current": [1, 1.1], "Max Sharpe": [1, 1.2], "Min variance": [1, 1.05]})
    fig = equity_chart(eq, visible=["Current", "Max Sharpe"])
    assert [t.visible for t in fig.data] == [True, True, "legendonly"]


def test_risk_return_labels_cover_no_dot_and_no_other_label() -> None:
    from gpl.plots import label_boxes, risk_return_chart

    # the seed-2 demo's crowded chart, where labels used to sit on neighbouring dots
    pts = pd.DataFrame(
        {
            "volatility": [0.0833, 0.0902, 0.0859, 0.0863, 0.0391, 0.0524, 0.0468, 0.1128],
            "cagr": [0.0667, 0.0690, 0.0504, 0.0494, 0.0260, 0.0437, 0.0391, 0.0943],
        },
        index=["Max Sharpe", "Target volatility", "Current", "Re-weight + swaps",
               "Min variance", "Max diversification", "Risk parity", "Benchmark"],
    )  # fmt: skip
    fig = risk_return_chart(pts, "Max Sharpe")
    sides = {}
    for a in fig.layout.annotations:  # labels are annotations, drawn above every dot
        sides[a.name] = next(
            k for k, v in plots._LABEL_SIDES.items()
            if (v[0], v[1], v[2], v[3]) == (a.xanchor, a.yanchor, a.xshift, a.yshift)
        )  # fmt: skip
    assert set(sides) == set(pts.index)
    xr, yr = tuple(fig.layout.xaxis.range), tuple(fig.layout.yaxis.range)
    boxes = label_boxes(pts, sides, xr, yr)
    for name, (x0, x1, y0, y1) in boxes.items():
        assert xr[0] <= x0 and x1 <= xr[1] and yr[0] <= y0 and y1 <= yr[1], name
        for other, (vol, ret) in pts.iterrows():
            if other != name:
                assert not (x0 < vol < x1 and y0 < ret < y1), (name, other)
        for other, (q0, q1, r0, r1) in boxes.items():
            if other != name:
                assert not (x0 < q1 and q0 < x1 and y0 < r1 and r0 < y1), (name, other)
