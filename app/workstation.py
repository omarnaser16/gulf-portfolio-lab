"""Portfolio workstation UI: search, holdings table, allocation and screen.

Holdings live in ``st.session_state`` as rows of ``{"ticker", "group", "value"}``; names and
prices come from cached live quotes. Every number shown comes from a ``gpl`` function.
"""

from collections.abc import Callable, Mapping, Sequence

import pandas as pd
import streamlit as st
from inputs import DEFAULT_HOLDINGS, GROUPS, guess_group, normalise_holdings, parse_holdings
from streamlit_searchbox import st_searchbox

from formatting import pct
from gpl.metrics import correlation_to_rest, summary_table
from gpl.plots import allocation_chart

ROWS, MODE, VERSION, LAST_PICK, META = "ws_rows", "ws_mode", "ws_version", "ws_last_pick", "ws_meta"
USD, PCT = "USD amount", "Weight (%)"
DEFAULT_TOTAL_USD = 10_000.0


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
) -> None:
    """Search box, editable holdings table, import box and allocation chart."""

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
        edited = st.data_editor(
            table,
            key=f"ws_table_{st.session_state[VERSION]}",
            width="stretch",
            column_config={
                "_index": st.column_config.TextColumn("Ticker", width=95),
                "Name": st.column_config.TextColumn(width=200),
                "Group": st.column_config.SelectboxColumn(options=GROUPS, required=True, width=110),
                st.session_state[MODE]: st.column_config.NumberColumn(
                    "USD" if st.session_state[MODE] == USD else "Input %",
                    min_value=0.0,
                    format="%.2f",
                    width=80,
                    help="What you hold. Weights are normalised to 100%.",
                ),
                "Weight": st.column_config.TextColumn(width=75),
                "Remove": st.column_config.CheckboxColumn(
                    width=75, help="Tick to remove this holding."
                ),
                "Price": st.column_config.TextColumn(width=175),
            },
            disabled=["Name", "Price", "Weight"],
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

    with st.expander("Import a list"):
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
    st.plotly_chart(allocation_chart(shown, title=f"Allocation by {by.lower()}"), width="stretch")


def screen_table(
    returns: pd.DataFrame, weights: pd.Series, rf, level: float, periods: int
) -> pd.DataFrame:
    """Per-holding metrics plus correlation to the rest of the portfolio (unformatted)."""
    table = summary_table(returns[weights.index], rf=rf, level=level, periods_per_year=periods)
    table["correlation_to_rest"] = correlation_to_rest(returns[weights.index], weights)
    table["weight"] = weights
    return table


SCREEN_COLUMNS = {
    "Return (ann.)": ("ann_return", False),
    "Volatility (ann.)": ("ann_vol", True),
    "Sharpe": ("sharpe", False),
    "Max drawdown": ("max_drawdown", False),
    "Correlation to rest": ("correlation_to_rest", True),
}


def render_screen(table: pd.DataFrame, names: Mapping[str, str]) -> None:
    by = st.selectbox("Rank by", list(SCREEN_COLUMNS), index=2, key="ws_rank_by")
    column, ascending = SCREEN_COLUMNS[by]
    ranked = table.sort_values(column, ascending=ascending)
    shown = pd.DataFrame(
        {
            "Name": [names.get(t, "") for t in ranked.index],
            "Weight": ranked["weight"].map(pct),
            "Return (ann.)": ranked["ann_return"].map(pct),
            "Volatility (ann.)": ranked["ann_vol"].map(pct),
            "Sharpe": ranked["sharpe"].map("{:.2f}".format),
            "Max drawdown": ranked["max_drawdown"].map(pct),
            "Correlation to rest": ranked["correlation_to_rest"].map("{:.2f}".format),
        },
        index=ranked.index,
    )
    shown.insert(0, "Rank", range(1, len(shown) + 1))
    st.table(shown)
    st.caption(
        "Correlation to rest: how closely each holding moves with the rest of your portfolio "
        "(0.00 = unrelated, 1.00 = in lockstep). Low or negative values diversify. Lower "
        "volatility and correlation rank higher; for the others, higher is better (max "
        "drawdown closest to 0)."
    )
