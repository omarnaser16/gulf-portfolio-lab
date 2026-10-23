"""Plotly chart helpers shared by notebooks and the Streamlit dashboard.

Percentages always show exactly two decimals (axis ticks, hovers, labels).

Presentation only: these functions take results from the finance modules and return
``plotly.graph_objects.Figure`` objects. They never compute finance numbers themselves
beyond reshaping (e.g. summing weights within an asset group).

Colour follows the entity, not its rank: series get categorical slots in a fixed order.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Validated categorical palette (light surface), assigned in this order, never cycled.
SERIES_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
PCT = ".2%"  # d3 format: 0.09 -> 9.00%
SEQUENTIAL_BLUE = ["#cfe0f6", "#8fb8ea", "#2a78d6"]  # light -> dark, one hue


def _base_layout(fig: go.Figure, title: str, y_title: str, y_format: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"color": TEXT_PRIMARY, "size": 16}},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": TEXT_SECONDARY, "size": 12},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 60, "r": 20, "t": 80, "b": 40},
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(title=y_title, tickformat=y_format, gridcolor=GRID, zerolinecolor=GRID)
    return fig


def _check_series_count(n: int) -> None:
    if n > len(SERIES_COLORS):
        raise ValueError(f"{n} series exceed {len(SERIES_COLORS)} colour slots; group or facet")


def equity_chart(equity: pd.DataFrame, title: str = "Growth of $1") -> go.Figure:
    """Line chart of wealth curves, one column per strategy."""
    _check_series_count(equity.shape[1])
    fig = go.Figure()
    for color, name in zip(SERIES_COLORS, equity.columns, strict=False):
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity[name],
                name=str(name),
                mode="lines",
                line={"color": color, "width": 2},
                hovertemplate="%{y:.2f}",
            )
        )
    return _base_layout(fig, title, "Wealth (nominal USD)", ".2f")


def drawdown_chart(drawdowns: pd.DataFrame, title: str = "Drawdown from peak") -> go.Figure:
    """Line chart of drawdown series (values <= 0), one column per strategy."""
    _check_series_count(drawdowns.shape[1])
    fig = go.Figure()
    for color, name in zip(SERIES_COLORS, drawdowns.columns, strict=False):
        fig.add_trace(
            go.Scatter(
                x=drawdowns.index,
                y=drawdowns[name],
                name=str(name),
                mode="lines",
                line={"color": color, "width": 2},
                hovertemplate="%{y:.2%}",
            )
        )
    return _base_layout(fig, title, "Drawdown", PCT)


def weights_chart(
    weights: pd.DataFrame,
    groups: Mapping[str, Sequence[str]] | None = None,
    title: str = "Target weights",
) -> go.Figure:
    """Stacked area of target weights at each rebalance, optionally summed by asset group."""
    if groups is not None:
        weights = pd.DataFrame(
            {
                g: weights.reindex(columns=list(t), fill_value=0.0).sum(axis=1)
                for g, t in groups.items()
            }
        )
    _check_series_count(weights.shape[1])
    fig = go.Figure()
    for color, name in zip(SERIES_COLORS, weights.columns, strict=False):
        fig.add_trace(
            go.Scatter(
                x=weights.index,
                y=weights[name],
                name=str(name),
                mode="lines",
                stackgroup="weights",
                line={"color": SURFACE, "width": 1, "shape": "hv"},  # surface gap between fills
                fillcolor=color,
                hovertemplate="%{y:.2%}",
            )
        )
    fig = _base_layout(fig, title, "Weight", PCT)
    fig.update_yaxes(range=[0, 1])
    return fig


def _on_frontier(frontier: pd.DataFrame, vol: float, ret: float, tol: float = 0.002) -> bool:
    """True if (vol, ret) lies on the frontier line (within ``tol`` of its volatility)."""
    ordered = frontier.sort_values("ret")
    frontier_vol = np.interp(ret, ordered["ret"], ordered["vol"])
    return vol <= frontier_vol + tol


def frontier_chart(
    frontier: pd.DataFrame,
    portfolios: Mapping[str, tuple[float, float]] | None = None,
    title: str = "Efficient frontier",
) -> go.Figure:
    """Frontier line (``vol`` vs ``ret``) with named portfolios as labelled markers.

    ``portfolios`` maps a name to ``(vol, ret)``. Labels never sit on the frontier line:
    points on the frontier are labelled up-left (that region is empty by definition: no
    portfolio has more return for the same risk), points inside it down-right.
    """
    portfolios = dict(portfolios or {})
    _check_series_count(1 + len(portfolios))
    fig = go.Figure(
        go.Scatter(
            x=frontier["vol"],
            y=frontier["ret"],
            name="Frontier",
            mode="lines",
            line={"color": SERIES_COLORS[0], "width": 2},
            hovertemplate="vol %{x:.2%}<br>return %{y:.2%}<extra></extra>",
        )
    )
    for color, (name, (vol, ret)) in zip(SERIES_COLORS[1:], portfolios.items(), strict=False):
        fig.add_trace(
            go.Scatter(
                x=[vol],
                y=[ret],
                name=name,
                mode="markers",
                marker={"color": color, "size": 10, "line": {"color": SURFACE, "width": 2}},
                hovertemplate="vol %{x:.2%}<br>return %{y:.2%}",
            )
        )
        up_left = _on_frontier(frontier, vol, ret)
        fig.add_annotation(
            x=vol,
            y=ret,
            text=name,
            showarrow=True,
            arrowhead=0,
            arrowwidth=1,
            arrowcolor=TEXT_SECONDARY,
            ax=-45 if up_left else 45,  # pixel offsets from the point
            ay=-30 if up_left else 30,
            xanchor="right" if up_left else "left",
            font={"color": TEXT_PRIMARY},
            bgcolor=SURFACE,
            borderpad=2,
        )
    fig = _base_layout(fig, title, "Expected return (annual)", PCT)
    fig.update_xaxes(title="Volatility (annual)", tickformat=PCT)
    fig.update_layout(hovermode="closest")
    return fig


def fan_chart(
    paths: np.ndarray,
    start: pd.Timestamp | None = None,
    title: str = "Simulated wealth",
    y_title: str = "Wealth (nominal USD)",
) -> go.Figure:
    """Monte Carlo fan: 5-95% and 25-75% bands with the median, from ``simulate_paths``.

    ``y_title`` states the units, e.g. nominal vs real (inflation-adjusted) USD.
    """
    x = (
        np.arange(paths.shape[0]) / 252
        if start is None
        else pd.bdate_range(start, periods=paths.shape[0])
    )
    p5, p25, p50, p75, p95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
    fig = go.Figure()
    for low, high, color, label in [
        (p5, p95, SEQUENTIAL_BLUE[0], "5-95%"),
        (p25, p75, SEQUENTIAL_BLUE[1], "25-75%"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=x, y=low, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=high,
                name=label,
                mode="lines",
                fill="tonexty",
                fillcolor=color,
                line={"width": 0},
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p50,
            name="Median",
            mode="lines",
            line={"color": SEQUENTIAL_BLUE[2], "width": 2},
            hovertemplate="%{y:.2f}",
        )
    )
    fig = _base_layout(fig, title, y_title, ".2f")
    if start is None:
        fig.update_xaxes(title="Years")
    return fig


def _runs(flags: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Start and end dates of consecutive True stretches."""
    flags = flags.fillna(False).astype(bool)
    starts = flags & ~flags.shift(1, fill_value=False)
    ends = flags & ~flags.shift(-1, fill_value=False)
    return list(zip(flags.index[starts], flags.index[ends], strict=True))


def regime_chart(
    p_stressed: pd.Series,
    threshold: float = 0.5,
    realised: pd.Series | None = None,
    title: str = "Probability of a stressed month ahead",
) -> go.Figure:
    """P(stressed) line with the decision threshold; realised stress shaded in grey.

    ``realised`` is the 0/1 label series (what actually happened next month).
    """
    fig = go.Figure(
        go.Scatter(
            x=p_stressed.index,
            y=p_stressed,
            name="P(stressed)",
            mode="lines",
            line={"color": SERIES_COLORS[0], "width": 2},
            hovertemplate="%{y:.2%}",
        )
    )
    # Threshold as a dashed line with a legend entry (an in-plot label collided with the data).
    fig.add_hline(y=threshold, line={"color": TEXT_SECONDARY, "width": 1, "dash": "dash"})
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=f"Threshold {threshold:.2%}",
            line={"color": TEXT_SECONDARY, "width": 1, "dash": "dash"},
        )
    )
    if realised is not None:
        for start, end in _runs(realised.reindex(p_stressed.index) == 1):
            fig.add_vrect(
                x0=start, x1=end, fillcolor=GRID, opacity=0.8, line_width=0, layer="below"
            )
        # legend entry for the shading
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Actually stressed",
                marker={"color": GRID, "size": 12, "symbol": "square"},
            )
        )
    fig = _base_layout(fig, title, "P(stressed)", PCT)
    fig.update_yaxes(range=[0, 1])
    return fig


def paired_bars_chart(
    first: pd.Series,
    second: pd.Series,
    names: tuple[str, str],
    title: str,
    sort_by_second: bool = True,
) -> go.Figure:
    """Horizontal bar pairs per item (e.g. weight vs share of risk, current vs optimised)."""
    order = (second if sort_by_second else first).sort_values().index
    fig = go.Figure()
    for color, name, values in zip(SERIES_COLORS, names, (first, second), strict=False):
        fig.add_trace(
            go.Bar(
                y=list(order),
                x=values.reindex(order).fillna(0.0),
                name=name,
                orientation="h",
                marker={"color": color, "line": {"color": SURFACE, "width": 2}},
                hovertemplate="%{y}: %{x:.2%}<extra>" + name + "</extra>",
            )
        )
    fig = _base_layout(fig, title, "", PCT)
    fig.update_layout(barmode="group", hovermode="closest", height=120 + 45 * len(order))
    fig.update_xaxes(tickformat=PCT, gridcolor=GRID)
    fig.update_yaxes(tickformat="")
    return fig


def risk_contribution_chart(
    weights: pd.Series, contributions: pd.Series, title: str = "Share of money vs share of risk"
) -> go.Figure:
    """Horizontal bars per holding: its weight and its share of portfolio risk."""
    return paired_bars_chart(weights, contributions, ("Weight", "Share of risk"), title)


def allocation_chart(weights: pd.Series, title: str = "Allocation") -> go.Figure:
    """Donut of portfolio weights; beyond seven slices the smallest are combined as 'Others'."""
    w = weights[weights > 0].sort_values(ascending=False)
    if len(w) > len(SERIES_COLORS):
        keep = len(SERIES_COLORS) - 1
        w = pd.concat(
            [w.iloc[:keep], pd.Series({f"Others ({len(w) - keep})": w.iloc[keep:].sum()})]
        )
    fig = go.Figure(
        go.Pie(
            labels=list(w.index),
            values=w.to_numpy(),
            hole=0.55,
            sort=False,
            marker={"colors": SERIES_COLORS[: len(w)], "line": {"color": SURFACE, "width": 2}},
            texttemplate="%{label}<br>%{percent:.2%}",
            hovertemplate="%{label}: %{percent:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": title, "font": {"color": TEXT_PRIMARY, "size": 16}},
        paper_bgcolor=SURFACE,
        font={"color": TEXT_SECONDARY, "size": 12},
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=360,
    )
    return fig
