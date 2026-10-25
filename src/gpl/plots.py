"""Plotly chart helpers shared by notebooks and the Streamlit dashboard.

Percentages always show exactly two decimals (axis ticks, hovers, labels).

Presentation only: these functions take results from the finance modules and return
``plotly.graph_objects.Figure`` objects. They never compute finance numbers themselves
beyond reshaping (e.g. summing weights within an asset group).

Colour follows the entity, not its rank: series get categorical slots in a fixed order.
All charts share one dark plotly template, ``gpl_dark``, set on every figure (not only as
plotly's default, which importing Streamlit replaces), matching the dashboard's terminal-style
theme. A benchmark or other reference series is drawn as a neutral grey dashed line outside the
categorical slots.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Categorical palette for the dark surface, assigned in this order, never cycled. Checked with
# the dataviz validator against #121316: lightness band, chroma, adjacent colour-blind
# separation (worst 8.3) and normal-vision separation (worst 19.7) all pass. Slot 1 (amber) is
# "current", slot 2 (cyan) "proposed".
SERIES_COLORS = [
    "#c98500",  # amber
    "#2a9fb5",  # cyan
    "#d55181",  # magenta
    "#008300",  # green
    "#9085e9",  # violet
    "#e66767",  # red
    "#3987e5",  # blue
    "#199e70",  # aqua
]
REFERENCE_COLOR = "#8e8e8a"  # benchmark / reference series: grey, dashed
SHADE = "#30333a"  # background shading (e.g. stressed months), visible on the dark surface
SURFACE = "#121316"
TEXT_PRIMARY = "#e6e4df"
TEXT_SECONDARY = "#a9a69f"
GRID = "#26282d"
PCT = ".2%"  # d3 format: 0.09 -> 9.00%
SEQUENTIAL_BLUE = ["#16323f", "#1f5d6e", "#2a9fb5"]  # dark -> bright cyan, one hue
MONO = "SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

pio.templates["gpl_dark"] = go.layout.Template(
    layout={
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {"color": TEXT_SECONDARY, "size": 12},
        "title": {"font": {"color": TEXT_PRIMARY, "size": 16}, "x": 0.01, "xanchor": "left"},
        "colorway": SERIES_COLORS,
        "xaxis": {
            "gridcolor": GRID,
            "linecolor": GRID,
            "zerolinecolor": GRID,
            "tickfont": {"family": MONO},
        },
        "yaxis": {
            "gridcolor": GRID,
            "linecolor": GRID,
            "zerolinecolor": GRID,
            "tickfont": {"family": MONO},
        },
        "hoverlabel": {"bgcolor": "#1d2025", "font": {"family": MONO, "color": TEXT_PRIMARY}},
        "legend": {"font": {"color": TEXT_SECONDARY}},
    }
)
pio.templates.default = "gpl_dark"


def series_styles(names: Sequence[str], reference: str | None = None) -> dict[str, dict]:
    """Line style per series: categorical slots in order; ``reference`` grey and dashed."""
    styles, slot = {}, 0
    for name in names:
        if name == reference:
            styles[name] = {"color": REFERENCE_COLOR, "width": 2, "dash": "dash"}
        else:
            styles[name] = {"color": SERIES_COLORS[slot], "width": 2}
            slot += 1
    return styles


def _base_layout(fig: go.Figure, title: str, y_title: str, y_format: str) -> go.Figure:
    # A long legend wraps onto the title, so with many series it goes below the plot instead.
    below = len(fig.data) > 5
    fig.update_layout(
        template="gpl_dark",
        title={"text": title, "font": {"color": TEXT_PRIMARY, "size": 16}},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": TEXT_SECONDARY, "size": 12},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "top", "y": -0.12, "x": 0}
        if below
        else {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 60, "r": 20, "t": 60 if below else 80, "b": 100 if below else 40},
        height=520 if below else None,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(title=y_title, tickformat=y_format, gridcolor=GRID, zerolinecolor=GRID)
    return fig


def _check_series_count(n: int) -> None:
    if n > len(SERIES_COLORS):
        raise ValueError(f"{n} series exceed {len(SERIES_COLORS)} colour slots; group or facet")


def equity_chart(
    equity: pd.DataFrame, title: str = "Growth of $1", reference: str | None = None
) -> go.Figure:
    """Line chart of wealth curves, one column per strategy (``reference`` grey dashed)."""
    _check_series_count(equity.shape[1] - (reference in equity.columns))
    fig = go.Figure()
    styles = series_styles(list(equity.columns), reference)
    for name in equity.columns:
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity[name],
                name=str(name),
                mode="lines",
                line=styles[name],
                hovertemplate="%{y:.2f}",
            )
        )
    return _base_layout(fig, title, "Wealth (nominal USD)", ".2f")


def drawdown_chart(
    drawdowns: pd.DataFrame, title: str = "Drawdown from peak", reference: str | None = None
) -> go.Figure:
    """Line chart of drawdown series (values <= 0), one column per strategy."""
    _check_series_count(drawdowns.shape[1] - (reference in drawdowns.columns))
    fig = go.Figure()
    styles = series_styles(list(drawdowns.columns), reference)
    for name in drawdowns.columns:
        fig.add_trace(
            go.Scatter(
                x=drawdowns.index,
                y=drawdowns[name],
                name=str(name),
                mode="lines",
                line=styles[name],
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
                x0=start, x1=end, fillcolor=SHADE, opacity=0.9, line_width=0, layer="below"
            )
        # legend entry for the shading
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Actually stressed",
                marker={"color": SHADE, "size": 12, "symbol": "square"},
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
            labels=[str(t) for t in w.index],
            values=[float(v) for v in w],  # plain list: some plotly.js builds drop typed arrays
            hole=0.55,
            sort=False,
            marker={"colors": SERIES_COLORS[: len(w)], "line": {"color": SURFACE, "width": 2}},
            texttemplate="%{label}<br>%{percent:.2%}",
            hovertemplate="%{label}: %{percent:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        template="gpl_dark",
        title={"text": title, "font": {"color": TEXT_PRIMARY, "size": 16}},
        paper_bgcolor=SURFACE,
        font={"color": TEXT_SECONDARY, "size": 12},
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=360,
    )
    return fig


DIVERGING = ["#3987e5", "#383835", "#c98500"]  # -1 blue, 0 neutral grey, +1 amber


def lines_chart(
    frame: pd.DataFrame,
    title: str,
    y_title: str,
    y_format: str = PCT,
    reference: str | None = None,
) -> go.Figure:
    """Generic line chart, one column per series (``reference`` grey dashed)."""
    _check_series_count(frame.shape[1] - (reference in frame.columns))
    styles = series_styles(list(frame.columns), reference)
    fig = go.Figure()
    for name in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[name],
                name=str(name),
                mode="lines",
                line=styles[name],
                hovertemplate="%{y:" + y_format + "}",
            )
        )
    return _base_layout(fig, title, y_title, y_format)


def correlation_heatmap(
    corr: pd.DataFrame, title: str = "Correlation of daily returns"
) -> go.Figure:
    """Correlation matrix on a diverging scale (blue negative, grey zero, amber positive)."""
    labels = [str(c) for c in corr.columns]
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale=[[0.0, DIVERGING[0]], [0.5, DIVERGING[1]], [1.0, DIVERGING[2]]],
            text=[[f"{v:.2f}" for v in row] for row in corr.to_numpy()],
            texttemplate="%{text}",
            textfont={"family": MONO, "color": TEXT_PRIMARY},
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    fig = _base_layout(fig, title, "", ".2f")
    fig.update_layout(hovermode="closest", height=140 + 45 * len(labels))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def fan_compare_chart(
    bands: Mapping[str, pd.DataFrame], title: str = "Simulated wealth", y_title: str = "Wealth"
) -> go.Figure:
    """Median and 5-95% band per portfolio. ``bands[name]`` has columns p5, p50, p95 indexed
    by years ahead."""
    _check_series_count(len(bands))
    fig = go.Figure()
    for color, (name, b) in zip(SERIES_COLORS, bands.items(), strict=False):
        x = list(b.index)
        fig.add_trace(
            go.Scatter(
                x=x, y=b["p5"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=b["p95"],
                name=f"{name} 5-95%",
                mode="lines",
                fill="tonexty",
                fillcolor=_rgba(color, 0.18),
                line={"width": 0},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=b["p50"],
                name=f"{name} median",
                mode="lines",
                line={"color": color, "width": 2},
                hovertemplate="%{y:.2f}",
            )
        )
    fig = _base_layout(fig, title, y_title, ".2f")
    fig.update_xaxes(title="Years ahead")
    return fig


def exposure_chart(
    exposures_by_portfolio: Mapping[str, Mapping[str, pd.Series]], title: str = "Exposure"
) -> go.Figure:
    """Bars of weight by region, asset type and currency (one panel each), per portfolio."""
    from plotly.subplots import make_subplots

    panels = [("region", "Region"), ("asset_type", "Asset type"), ("currency", "Currency")]
    fig = make_subplots(
        rows=1, cols=3, subplot_titles=[p[1] for p in panels], horizontal_spacing=0.12
    )
    for color, (name, exp) in zip(SERIES_COLORS, exposures_by_portfolio.items(), strict=False):
        for col, (key, _) in enumerate(panels, start=1):
            s = exp[key]
            fig.add_trace(
                go.Bar(
                    y=[str(i) for i in s.index],
                    x=[float(v) for v in s],
                    name=name,
                    orientation="h",
                    marker={"color": color, "line": {"color": SURFACE, "width": 2}},
                    legendgroup=name,
                    showlegend=col == 1,
                    hovertemplate="%{y}: %{x:.2%}<extra>" + name + "</extra>",
                ),
                row=1,
                col=col,
            )
    fig = _base_layout(fig, title, "", PCT)
    fig.update_layout(barmode="group", hovermode="closest", height=380)
    fig.update_xaxes(tickformat=PCT)
    fig.update_yaxes(tickformat="", autorange="reversed")
    fig.update_annotations(font={"color": TEXT_PRIMARY})
    return fig


def stress_chart(
    table: pd.DataFrame, reference: str | None = None, title: str = "Historical stress tests"
) -> go.Figure:
    """Grouped bars: total return in each episode per series (``episode_performance`` rows).

    Episodes the data doesn't cover are labelled "not covered" instead of a bar.
    """
    episodes = list(dict.fromkeys(table["episode"]))
    series = list(dict.fromkeys(table["series"]))
    styles = series_styles(series, reference)
    labels = {
        e: f"{e}<br>{table.loc[table.episode == e, 'start'].iloc[0]:%d %b %Y} – "
        f"{table.loc[table.episode == e, 'end'].iloc[0]:%d %b %Y}"
        for e in episodes
    }
    fig = go.Figure()
    for name in series:
        rows = table[table.series == name].set_index("episode").reindex(episodes)
        fig.add_trace(
            go.Bar(
                x=[labels[e] for e in episodes],
                y=[float(v) for v in rows["total_return"]],
                name=name,
                marker={
                    "color": styles[name]["color"],
                    "line": {"color": SURFACE, "width": 2},
                    "pattern": {"shape": "/" if name == reference else ""},
                },
                hovertemplate="%{x}: %{y:.2%}<extra>" + name + "</extra>",
            )
        )
    for e in episodes:
        if not table.loc[table.episode == e, "covered"].all():
            fig.add_annotation(
                x=labels[e],
                y=0,
                text="not covered by your data",
                showarrow=False,
                yshift=12,
                font={"color": TEXT_SECONDARY},
            )
    fig = _base_layout(fig, title, "Total return in the episode", PCT)
    fig.update_layout(barmode="group", hovermode="closest")
    return fig
