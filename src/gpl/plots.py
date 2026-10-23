"""Plotly chart helpers shared by notebooks and the Streamlit dashboard.

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
    return _base_layout(fig, title, "Wealth ($)", ".2f")


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
                hovertemplate="%{y:.1%}",
            )
        )
    return _base_layout(fig, title, "Drawdown", ".0%")


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
                hovertemplate="%{y:.0%}",
            )
        )
    fig = _base_layout(fig, title, "Weight", ".0%")
    fig.update_yaxes(range=[0, 1])
    return fig


def frontier_chart(
    frontier: pd.DataFrame,
    portfolios: Mapping[str, tuple[float, float]] | None = None,
    title: str = "Efficient frontier",
) -> go.Figure:
    """Frontier line (``vol`` vs ``ret``) with named portfolios as labelled markers.

    ``portfolios`` maps a name to ``(vol, ret)``.
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
            hovertemplate="vol %{x:.1%}<br>return %{y:.1%}<extra></extra>",
        )
    )
    for color, (name, (vol, ret)) in zip(SERIES_COLORS[1:], portfolios.items(), strict=False):
        fig.add_trace(
            go.Scatter(
                x=[vol],
                y=[ret],
                name=name,
                mode="markers+text",
                text=[name],
                textposition="middle right",
                textfont={"color": TEXT_PRIMARY},
                marker={"color": color, "size": 10, "line": {"color": SURFACE, "width": 2}},
                hovertemplate="vol %{x:.1%}<br>return %{y:.1%}",
            )
        )
    fig = _base_layout(fig, title, "Expected return (annual)", ".0%")
    fig.update_xaxes(title="Volatility (annual)", tickformat=".0%")
    fig.update_layout(hovermode="closest")
    return fig


def fan_chart(
    paths: np.ndarray,
    start: pd.Timestamp | None = None,
    title: str = "Simulated wealth",
) -> go.Figure:
    """Monte Carlo fan: 5-95% and 25-75% bands with the median, from ``simulate_paths``."""
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
    fig = _base_layout(fig, title, "Wealth ($)", ".2f")
    if start is None:
        fig.update_xaxes(title="Years")
    return fig
