"""Portfolio workstation UI: search, holdings table, allocation and screen.

Holdings live in ``st.session_state`` as rows of ``{"ticker", "group", "value"}``; names and
prices come from cached live quotes. Every number shown comes from a ``gpl`` function.
"""

from collections.abc import Callable, Mapping, Sequence

import pandas as pd
import streamlit as st
from inputs import (
    DEFAULT_HOLDINGS,
    GROUPS,
    equal_values,
    guess_group,
    normalise_holdings,
    normalise_values,
    parse_holdings,
)
from streamlit_searchbox import st_searchbox

from formatting import pct, plot
from gpl.metrics import (
    benchmark_metrics,
    beta,
    calmar_ratio,
    correlation_to_rest,
    summary_table,
    trailing_return,
)
from gpl.optimise import risk_contributions
from gpl.plots import allocation_chart

ROWS, MODE, VERSION, LAST_PICK, META = "ws_rows", "ws_mode", "ws_version", "ws_last_pick", "ws_meta"
USD, PCT = "USD amount", "Weight (%)"
DEFAULT_TOTAL_USD = 10_000.0
EDIT_LABEL = {USD: "✏️ USD", PCT: "✏️ Input %"}
READ_ONLY = {"background-color": "#1b1d22", "color": "#8f8c85"}  # muted = can't edit
READ_ONLY_COLUMNS = ["Name", "Weight", "Price"]


def init_state(config_groups: Mapping[str, Sequence[str]]) -> None:
    """First visit: start from the example portfolio (weights in %)."""
    if ROWS in st.session_state:
        return
    weights, _, _ = parse_holdings(DEFAULT_HOLDINGS, "weights")
    st.session_state[ROWS] = [
        {"ticker": t, "group": guess_group(t, "", "ETF", "", config_groups), "value": w * 100}
        for t, w in weights.items()
    ]
    st.session_state[MODE] = PCT
    st.session_state["ws_mode_prev"] = PCT
    st.session_state[VERSION] = 0


def current_weights() -> tuple[dict[str, float], list[str], str]:
    """Weights from the holdings rows (readable before any widget is drawn)."""
    mode = "usd" if st.session_state.get(MODE, PCT) == USD else "pct"
    values = {r["ticker"]: r["value"] for r in st.session_state.get(ROWS, [])}
    return normalise_holdings(values, mode)


def groups_of_holdings() -> dict[str, str]:
    return {r["ticker"]: r["group"] for r in st.session_state.get(ROWS, [])}


def _bump() -> None:
    st.session_state[VERSION] = st.session_state.get(VERSION, 0) + 1


def _convert_mode() -> None:
    """When switching USD <-> %, keep the same weights."""
    new, old = st.session_state[MODE], st.session_state.get("ws_mode_prev", PCT)
    if new == old:
        return
    weights, _, _ = normalise_holdings(
        {r["ticker"]: r["value"] for r in st.session_state[ROWS]}, "usd" if old == USD else "pct"
    )
    total = DEFAULT_TOTAL_USD
    for r in st.session_state[ROWS]:
        w = weights.get(r["ticker"], 0.0)
        r["value"] = round(w * (total if new == USD else 100), 2)
    st.session_state["ws_mode_prev"] = new
    _bump()


def _set_equal() -> None:
    rows = st.session_state[ROWS]
    usd = st.session_state[MODE] == USD
    total = sum(r["value"] for r in rows if r["value"] > 0) or DEFAULT_TOTAL_USD
    values = equal_values([r["ticker"] for r in rows], usd, total)
    for r in rows:
        r["value"] = values[r["ticker"]]
    _bump()


def _set_normalised() -> None:
    rows = st.session_state[ROWS]
    values = normalise_values({r["ticker"]: r["value"] for r in rows})
    for r in rows:
        r["value"] = values[r["ticker"]]
    _bump()


def add_holding(symbol: str, meta: Mapping[str, str], config_groups) -> None:
    rows = st.session_state[ROWS]
    if any(r["ticker"] == symbol for r in rows):
        return
    group = guess_group(
        symbol, meta.get("exchange", ""), meta.get("type", ""), meta.get("name", ""), config_groups
    )
    default = 1_000.0 if st.session_state[MODE] == USD else 10.0
    rows.append({"ticker": symbol, "group": group, "value": default})
    _bump()


def render_holdings(
    search: Callable[[str], list[dict]],
    quote: Callable[[str], object],
    config_groups: Mapping[str, Sequence[str]],
    names: Mapping[str, str],
    presets: Mapping[str, Sequence[str]] | None = None,
) -> None:
    """Search box, editable holdings table, import box and allocation chart."""
    presets = presets or {}

    def suggestions(term: str) -> list[tuple[str, str]]:
        results = search(term)
        st.session_state[META] = {r["symbol"]: r for r in results}
        return [
            (f"{r['name']} · {r['symbol']} · {r['exchange']} · {r['type']}", r["symbol"])
            for r in results
        ]

    picked = st_searchbox(
        suggestions,
        placeholder="Search by name or ticker, e.g. Aramco, Emaar, Apple, SPY",
        label="Add a security",
        key="ws_search",
        debounce=300,
        clear_on_submit=True,
    )
    if picked and picked != st.session_state.get(LAST_PICK):
        st.session_state[LAST_PICK] = picked
        add_holding(picked, st.session_state.get(META, {}).get(picked, {}), config_groups)
        st.rerun()

    rows = st.session_state[ROWS]
    st.radio("Enter holdings as", [USD, PCT], key=MODE, horizontal=True, on_change=_convert_mode)
    if not rows:
        st.info("Search above to add your first holding, or import a list below.")
    else:
        weights, _, _ = current_weights()
        quotes = {r["ticker"]: quote(r["ticker"]) for r in rows}

        def price_text(q) -> str:
            if not q:
                return "–"
            local = f"{q.price:,.2f} {q.currency}"
            return local if q.currency == "USD" else f"{local} · {q.price_usd:,.2f} USD"

        table = pd.DataFrame(
            {
                "Name": [
                    getattr(quotes[r["ticker"]], "name", None) or names.get(r["ticker"], "")
                    for r in rows
                ],
                "Group": [r["group"] for r in rows],
                st.session_state[MODE]: [r["value"] for r in rows],
                "Weight": [
                    pct(weights[r["ticker"]]) if r["ticker"] in weights else "–" for r in rows
                ],
                "Remove": [False] * len(rows),
                "Price": [price_text(quotes[r["ticker"]]) for r in rows],
            },
            index=pd.Index([r["ticker"] for r in rows], name="Ticker"),
        )
        mode_col = st.session_state[MODE]
        st.caption(
            f"Click a cell in the **{EDIT_LABEL[mode_col]}** column to change it; weights update "
            "automatically. White columns are editable, grey ones are calculated. Tick Remove "
            "to drop a holding."
        )
        # Streamlit only styles read-only columns, so the editable ones stand out by contrast
        styled = table.style.set_properties(subset=READ_ONLY_COLUMNS, **READ_ONLY)
        edited = st.data_editor(
            styled,
            key=f"ws_table_{st.session_state[VERSION]}",
            width="stretch",
            column_config={
                "_index": st.column_config.TextColumn("Ticker", width=95),
                "Name": st.column_config.TextColumn(width=180),
                "Group": st.column_config.SelectboxColumn(options=GROUPS, required=True, width=110),
                mode_col: st.column_config.NumberColumn(
                    EDIT_LABEL[mode_col],
                    min_value=0.0,
                    format="%.2f",
                    width=100,
                    help="Click to edit: what you hold. Weights are normalised to 100%.",
                ),
                "Weight": st.column_config.TextColumn(width=75),
                "Remove": st.column_config.CheckboxColumn(
                    width=75, help="Tick to remove this holding."
                ),
                "Price": st.column_config.TextColumn(width=165),
            },
            disabled=READ_ONLY_COLUMNS,
        )
        left, right, _ = st.columns([1, 1, 2])
        left.button(
            "Equal weight",
            key="ws_equal",
            on_click=_set_equal,
            help="Give every holding the same weight.",
            width="stretch",
        )
        right.button(
            "Normalise to 100%",
            key="ws_normalise",
            on_click=_set_normalised,
            disabled=mode_col == USD,
            help="Rescale the inputs so they add up to exactly 100%."
            if mode_col == PCT
            else "Amounts are always normalised; switch to Weight (%) to rescale inputs.",
            width="stretch",
        )
        times: dict[str, list[str]] = {}
        for r in rows:
            times.setdefault(getattr(quotes[r["ticker"]], "as_of", "not available"), []).append(
                r["ticker"]
            )
        st.caption(
            "Prices as of: " + "; ".join(f"{when} ({', '.join(ts)})" for when, ts in times.items())
        )
        changed = False
        kept = []
        for r in rows:
            e = edited.loc[r["ticker"]]
            if bool(e["Remove"]):
                changed = True
                if st.session_state.get(LAST_PICK) == r["ticker"]:
                    st.session_state[LAST_PICK] = None
                continue
            value = float(e[st.session_state[MODE]] or 0.0)
            if e["Group"] != r["group"] or value != r["value"]:
                changed = True
            kept.append({"ticker": r["ticker"], "group": e["Group"], "value": value})
        if changed:
            st.session_state[ROWS] = kept
            _bump()
            st.rerun()

    with st.expander("Import a list or start from a preset"):
        if presets:
            st.selectbox("Preset list (equal weights)", list(presets), key="ws_preset")
            st.button(
                "Replace holdings with this preset",
                key="ws_preset_go",
                on_click=_load_preset,
                args=(presets, quote, config_groups),
            )
            st.divider()
        st.text_area("One holding per line, e.g. 'SPY 40' or '2222.SR 30'", key="ws_import")
        st.radio(
            "Values are", ["Weights (%)", "Amounts (USD)"], key="ws_import_kind", horizontal=True
        )
        st.button(
            "Replace holdings",
            key="ws_import_go",
            on_click=_import_list,  # runs before widgets are drawn, so it may set the mode
            args=(quote, config_groups),
        )
        for problem in st.session_state.pop("ws_import_problems", []):
            st.error(problem)


def _load_preset(
    presets: Mapping[str, Sequence[str]], quote: Callable[[str], object], config_groups
) -> None:
    tickers = list(presets[st.session_state.get("ws_preset") or next(iter(presets))])
    st.session_state["ws_import"] = "\n".join(f"{t} {100 / len(tickers):.6f}" for t in tickers)
    st.session_state["ws_import_kind"] = "Weights (%)"
    _import_list(quote, config_groups)


def _import_list(quote: Callable[[str], object], config_groups) -> None:
    text = st.session_state.get("ws_import", "")
    kind = st.session_state.get("ws_import_kind", "Weights (%)")
    weights, problems, _ = parse_holdings(
        text, "weights" if kind.startswith("Weights") else "amounts"
    )
    st.session_state["ws_import_problems"] = problems
    if not weights:
        return
    new_rows = []
    for t, w in weights.items():
        q = quote(t)
        group = guess_group(
            t,
            getattr(q, "exchange", ""),
            getattr(q, "quote_type", ""),
            getattr(q, "name", ""),
            config_groups,
        )
        new_rows.append({"ticker": t, "group": group, "value": round(w * 100, 4)})
    st.session_state[ROWS] = new_rows
    st.session_state[MODE] = PCT
    st.session_state["ws_mode_prev"] = PCT
    _bump()


def render_allocation(weights: pd.Series, groups: Mapping[str, str]) -> None:
    by = st.radio("Show allocation by", ["Holding", "Group"], key="ws_alloc_by", horizontal=True)
    shown = weights if by == "Holding" else weights.groupby(lambda t: groups.get(t, "Other")).sum()
    plot(allocation_chart(shown, title=f"Allocation by {by.lower()}"))


def screen_table(
    returns: pd.DataFrame,
    weights: pd.Series,
    rf,
    level: float,
    periods: int,
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """Per-holding metrics for the Screen (unformatted). Every figure comes from ``gpl``.

    Beta uses the dates both the holdings and ``benchmark`` cover; NaN without a benchmark or
    for a benchmark that doesn't move (cash). 1Y and 3Y returns are NaN with too little history.
    """
    held = returns[weights.index]
    table = summary_table(held, rf=rf, level=level, periods_per_year=periods)
    table["calmar"] = calmar_ratio(held, periods)
    table["ret_1y"] = trailing_return(held, 1, periods)
    table["ret_3y"] = trailing_return(held, 3, periods)
    table["correlation_to_rest"] = correlation_to_rest(held, weights)
    table["risk_share"] = risk_contributions(weights, held.cov())
    if benchmark is not None:
        dates = held.index.intersection(benchmark.index)
        table["beta"] = beta(held.loc[dates], benchmark.loc[dates])
    else:
        table["beta"] = float("nan")
    table["weight"] = weights
    return table


SCREEN_COLUMNS = {  # label: (column, lower is better)
    "Return (ann.)": ("ann_return", False),
    "CAGR": ("cagr", False),
    "1Y return": ("ret_1y", False),
    "3Y return (ann.)": ("ret_3y", False),
    "Volatility (ann.)": ("ann_vol", True),
    "Sharpe": ("sharpe", False),
    "Sortino": ("sortino", False),
    "Calmar": ("calmar", False),
    "Max drawdown": ("max_drawdown", False),
    "VaR 95% (1-day)": ("hist_var", True),
    "Expected shortfall 95% (1-day)": ("expected_shortfall", True),
    "Beta to benchmark": ("beta", True),
    "Share of portfolio risk": ("risk_share", True),
    "Correlation to rest": ("correlation_to_rest", True),
}
_RATIOS = {"sharpe", "sortino", "calmar", "beta", "correlation_to_rest"}


def _fmt(column: str, value: float) -> str:
    if pd.isna(value):
        return "–"
    return f"{value:.2f}" if column in _RATIOS else pct(value)


def rank_screen(table: pd.DataFrame, by: str) -> pd.DataFrame:
    """Rank holdings by the chosen metric (missing values last), formatted for display."""
    column, ascending = SCREEN_COLUMNS[by]
    ranked = table.sort_values(column, ascending=ascending, na_position="last")
    shown = pd.DataFrame(
        {label: [_fmt(col, v) for v in ranked[col]] for label, (col, _) in SCREEN_COLUMNS.items()},
        index=ranked.index,
    )
    shown.insert(0, "Weight", ranked["weight"].map(pct))
    shown.insert(0, by + " ▼", shown.pop(by))  # the ranking metric first
    shown.insert(0, "Rank", range(1, len(shown) + 1))
    return shown


def render_screen(table: pd.DataFrame, names: Mapping[str, str], benchmark_name: str) -> None:
    by = st.selectbox("Rank by", list(SCREEN_COLUMNS), index=5, key="ws_rank_by")
    shown = rank_screen(table, by)
    shown.insert(1, "Name", [names.get(t, "") for t in shown.index])
    st.dataframe(
        shown, width="stretch", column_config={"Name": st.column_config.TextColumn(width=160)}
    )
    st.caption(
        f"Lower ranks higher for volatility, VaR, expected shortfall, beta, share of portfolio "
        f"risk and correlation to rest; higher ranks higher for the others (max drawdown "
        f"closest to 0). Beta is measured against **{benchmark_name}**. VaR and expected "
        "shortfall are 1-day losses at 95%. Calmar = CAGR ÷ |max drawdown|. Correlation to "
        "rest: how closely each holding moves with the rest of your portfolio (low values "
        "diversify). '–' means not enough history (3Y return) or not defined (beta to cash); "
        "these rank last. Scroll sideways for every column."
    )


def benchmark_comparison(
    portfolio: pd.Series, benchmark: pd.Series, rf, periods: int, is_cash: bool
) -> pd.DataFrame:
    """Portfolio vs benchmark on the dates both cover, formatted. Every figure from ``gpl``.

    Relative measures describe the portfolio against the benchmark, so the benchmark column
    shows '–'. Against cash, beta, alpha, correlation and capture are not defined (cash has no
    market risk).
    """
    dates = portfolio.index.intersection(benchmark.index)
    both = pd.DataFrame({"Your portfolio": portfolio.loc[dates], "Benchmark": benchmark.loc[dates]})
    table = summary_table(both, rf=rf, periods_per_year=periods)
    rel = benchmark_metrics(portfolio, benchmark, rf, periods, market_risk=not is_cash)
    na = "n/a (cash has no market risk)"

    def ratio(x: float) -> str:
        return na if is_cash and pd.isna(x) else ("–" if pd.isna(x) else f"{x:.2f}")

    def share(x: float) -> str:
        return na if is_cash and pd.isna(x) else ("–" if pd.isna(x) else pct(x))

    out = pd.DataFrame(
        {
            "Return (CAGR)": table["cagr"].map(pct),
            "Volatility (ann.)": table["ann_vol"].map(pct),
            "Sharpe": table["sharpe"].map("{:.2f}".format),
            "Max drawdown": table["max_drawdown"].map(pct),
        }
    ).T
    relative = {
        "Beta": ratio(rel["beta"]),
        "Alpha (ann., Jensen)": share(rel["alpha"]),
        "Tracking error (ann.)": share(rel["tracking_error"]),
        "Information ratio": ratio(rel["information_ratio"]),
        "Up capture": share(rel["up_capture"]),
        "Down capture": share(rel["down_capture"]),
        "Correlation": ratio(rel["correlation"]),
    }
    for name, value in relative.items():
        out.loc[name] = [value, "–"]
    return out
