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
    parse_amounts,
    parse_holdings,
)
from streamlit_searchbox import st_searchbox

from formatting import compact_usd, money, pct, plot
from gpl.metrics import (
    benchmark_metrics,
    beta,
    calmar_ratio,
    correlation_to_rest,
    no_negative_zero,
    summary_table,
    trailing_return,
)
from gpl.optimise import risk_contributions
from gpl.plots import allocation_chart
from gpl.theme import COMPOSITION

ROWS, MODE, VERSION, META = "ws_rows", "ws_mode", "ws_version", "ws_meta"
SEARCH = "ws_search"
CASH_LIKE = "ws_cash_like"  # configured cash-like funds (grouped as "Cash & short-term")
USD, PCT = "USD amount", "Weight (%)"
STARTER = "ws_starter_rows"  # the first-visit example holdings
PF_VALUE, PF_VERSION = "pf_value", "pf_v"  # the one portfolio value (USD), entered once
EDIT_LABEL = {USD: "✏️ USD", PCT: "✏️ Input %"}
READ_ONLY = {"background-color": "#F1F0EB", "color": "#6B6B6B"}  # muted = can't edit
READ_ONLY_COLUMNS = ["Name", "Weight", "Price (local)", "Ccy", "Price (USD)"]


def init_state(config_groups: Mapping[str, Sequence[str]], cash_like: Sequence[str] = ()) -> None:
    """First visit: start from the example portfolio (weights in %)."""
    st.session_state[CASH_LIKE] = list(cash_like)
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
    st.session_state[STARTER] = [dict(r) for r in st.session_state[ROWS]]


def portfolio_started() -> bool:
    """True once the holdings differ from the starter example shown on a first visit."""
    rows = st.session_state.get(ROWS, [])
    starter = st.session_state.get(STARTER)
    return starter is None or [(r["ticker"], r["value"]) for r in rows] != [
        (r["ticker"], r["value"]) for r in starter
    ]


def current_weights() -> tuple[dict[str, float], list[str], str]:
    """Weights from the holdings rows (readable before any widget is drawn)."""
    mode = "usd" if st.session_state.get(MODE, PCT) == USD else "pct"
    values = {r["ticker"]: r["value"] for r in st.session_state.get(ROWS, [])}
    return normalise_holdings(values, mode)


def portfolio_value() -> float | None:
    """The single portfolio value (USD) used everywhere: the total of the USD amounts when
    holdings are entered in USD, otherwise the value entered once (Holdings or Client tab).
    None when unknown; there is no default."""
    if st.session_state.get(MODE) == USD:
        total = sum(r["value"] for r in st.session_state.get(ROWS, []) if (r["value"] or 0) > 0)
        return total or None
    value = st.session_state.get(PF_VALUE)
    return float(value) if value and value > 0 else None


def set_portfolio_value(value: float | None) -> None:
    """Record the entered value (ignored in USD mode, where the amounts define it)."""
    st.session_state[PF_VALUE] = float(value) if value and value > 0 else None
    st.session_state[PF_VERSION] = st.session_state.get(PF_VERSION, 0) + 1


def _value_entered(key: str) -> None:
    set_portfolio_value(st.session_state.get(key))


def value_input(container, key_prefix: str, label: str = "Portfolio value (USD)") -> None:
    """The portfolio value input; every copy of it edits the same value."""
    usd = st.session_state.get(MODE) == USD
    key = f"{key_prefix}_{st.session_state.get(PF_VERSION, 0)}"
    if usd:  # the total follows every holdings edit: a new key shows the new total
        key += f"_{round((portfolio_value() or 0) * 100)}"
    container.number_input(
        label,
        min_value=0.0,
        value=portfolio_value(),
        step=1000.0,
        format="%.2f",
        key=key,
        placeholder="enter once",
        disabled=usd,
        on_change=_value_entered,
        args=(key,),
        help="The total of the USD amounts on Holdings."
        if usd
        else "Used for the suitability check, rebalancing trades, domicile check and report.",
    )


def groups_of_holdings() -> dict[str, str]:
    return {r["ticker"]: r["group"] for r in st.session_state.get(ROWS, [])}


def _bump() -> None:
    st.session_state[VERSION] = st.session_state.get(VERSION, 0) + 1


def _convert_mode() -> None:
    """When switching USD <-> %, keep the same weights and the same portfolio value.

    % -> USD needs the portfolio value (there is no default); without it the switch is undone.
    """
    new, old = st.session_state[MODE], st.session_state.get("ws_mode_prev", PCT)
    if new == old:
        return
    values = {r["ticker"]: r["value"] for r in st.session_state[ROWS]}
    weights, _, _ = normalise_holdings(values, "usd" if old == USD else "pct")
    if new == USD:
        total = st.session_state.get(PF_VALUE)
        if not total:
            st.session_state[MODE] = old
            st.session_state["ws_mode_error"] = (
                "Enter the portfolio value first, so the weights can be turned into USD amounts."
            )
            return
    else:
        total = sum(v for v in values.values() if v and v > 0)
        set_portfolio_value(total)  # the USD total becomes the entered value
    for r in st.session_state[ROWS]:
        w = weights.get(r["ticker"], 0.0)
        r["value"] = round(w * (total if new == USD else 100), 2)
    st.session_state["ws_mode_prev"] = new
    _bump()


def _set_equal() -> None:
    rows = st.session_state[ROWS]
    usd = st.session_state[MODE] == USD
    total = sum(r["value"] for r in rows if (r["value"] or 0) > 0)  # USD: keep the same total
    if usd and not total:
        return
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


def _cell_value(x) -> float | None:
    """A typed cell: a number, or None when blank (never NaN, which compares unequal to itself
    and used to make every run look like a new edit)."""
    if x is None:
        return None
    try:
        value = float(x)
    except TypeError, ValueError:
        return None
    return None if value != value else value  # NaN -> None


def _apply_edits(key: str, tickers: list[str], mode_col: str) -> None:
    """Editor callback: apply the edited cells and removals to the holdings.

    Runs before the next script run, so every committed edit is kept. The editor's state holds
    every edit since it was drawn, as absolute values, so applying it again is harmless; its key
    therefore stays the same for value and group edits (a new key would drop an edit typed while
    the app reruns). Only a removal, which shifts the rows, redraws the editor.
    """
    state = st.session_state.get(key) or {}
    by_ticker = {r["ticker"]: dict(r) for r in st.session_state[ROWS]}
    removed = set()
    for index, changes in (state.get("edited_rows") or {}).items():
        ticker = tickers[int(index)]
        row = by_ticker.get(ticker)
        if row is None:
            continue
        if changes.get("Remove"):
            removed.add(ticker)
        if mode_col in changes:
            row["value"] = _cell_value(changes[mode_col])
        if changes.get("Group"):
            row["group"] = changes["Group"]
    st.session_state[ROWS] = [
        by_ticker[r["ticker"]] for r in st.session_state[ROWS] if r["ticker"] not in removed
    ]
    if removed:
        _bump()


def add_holding(symbol: str, meta: Mapping[str, str], config_groups) -> None:
    rows = st.session_state[ROWS]
    if any(r["ticker"] == symbol for r in rows):
        return
    group = guess_group(
        symbol,
        meta.get("exchange", ""),
        meta.get("type", ""),
        meta.get("name", ""),
        config_groups,
        st.session_state.get(CASH_LIKE, ()),
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
    has_data: Callable[[str], tuple[bool, str]] | None = None,
) -> str:
    """Search box, entry controls, import popover and the editable holdings table.

    ``has_data``: checks a ticker has prices before it is added (``gpl.data.has_price_data``).
    Returns when each holding's price is from ("prices as of"), for the page header.
    """
    presets = presets or {}

    def suggestions(term: str) -> list[tuple[str, str]]:
        results = search(term)
        st.session_state[META] = {r["symbol"]: r for r in results}
        return [
            (f"{r['name']} · {r['symbol']} · {r['exchange']} · {r['type']}", r["symbol"])
            for r in results
        ]

    def picked(symbol: str) -> None:
        """Add the pick once, then forget it: the searchbox would otherwise keep returning it
        (re-adding a holding just removed) and ignore the same ticker picked again later."""
        if symbol:
            ok, why = has_data(symbol) if has_data else (True, "")
            if ok:
                add_holding(symbol, st.session_state.get(META, {}).get(symbol, {}), config_groups)
            else:
                st.session_state["ws_add_error"] = f"{symbol}: {why}, so it was not added."
        st.session_state[SEARCH]["result"] = None

    st_searchbox(
        suggestions,
        placeholder="Add a security: search Aramco, Emaar, Apple, SPY…",
        label=None,
        key=SEARCH,
        debounce=300,
        clear_on_submit=True,
        submit_function=picked,
    )

    if st.session_state.get("ws_add_error"):
        st.error(st.session_state.pop("ws_add_error"))
    rows = st.session_state[ROWS]
    st.radio("Enter as", [USD, PCT], key=MODE, horizontal=True, on_change=_convert_mode)
    if st.session_state.get(MODE) != USD:
        value_input(st.columns([1, 2])[0], "pf_value_holdings")
    mode_col = st.session_state[MODE]
    bar = st.container(horizontal=True, wrap=True, gap="small", key="holdings_toolbar")
    b1 = b2 = b3 = bar
    b1.button(
        "Equal weight",
        key="ws_equal",
        on_click=_set_equal,
        help="Give every holding the same weight.",
        width="content",
        disabled=not rows,
    )
    b2.button(
        "Normalise to 100%",
        key="ws_normalise",
        on_click=_set_normalised,
        disabled=mode_col == USD or not rows,
        help="Rescale the inputs so they add up to exactly 100%."
        if mode_col == PCT
        else "Amounts are always normalised; switch to Weight (%) to rescale inputs.",
        width="content",
    )
    with b3.popover("Import a list or preset", width="content"):
        if presets:
            st.selectbox("Preset list (equal weights)", list(presets), key="ws_preset")
            st.button(
                "Replace holdings with this preset",
                key="ws_preset_go",
                on_click=_load_preset,
                args=(presets, quote, config_groups, has_data),
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
            args=(quote, config_groups, has_data),
        )
    for problem in st.session_state.pop("ws_import_problems", []):
        st.error(problem)
    if st.session_state.get("ws_mode_error"):
        st.warning(st.session_state.pop("ws_mode_error"))
    as_of = ""
    if not rows:
        st.info("Search above to add your first holding, or import a list.")
    else:
        weights, _, _ = current_weights()
        quotes = {r["ticker"]: quote(r["ticker"]) for r in rows}

        table = pd.DataFrame(
            {
                "Name": [
                    getattr(quotes[r["ticker"]], "name", None) or names.get(r["ticker"], "")
                    for r in rows
                ],
                "Group": [r["group"] for r in rows],
                st.session_state[MODE]: [
                    float("nan") if r["value"] is None else r["value"] for r in rows
                ],
                "Weight": [
                    weights[r["ticker"]] * 100 if r["ticker"] in weights else float("nan")
                    for r in rows
                ],
                "Remove": [False] * len(rows),
                "Price (local)": [
                    getattr(quotes[r["ticker"]], "price", float("nan")) for r in rows
                ],
                "Ccy": [getattr(quotes[r["ticker"]], "currency", "") for r in rows],
                "Price (USD)": [
                    getattr(quotes[r["ticker"]], "price_usd", float("nan")) for r in rows
                ],
            },
            index=pd.Index([r["ticker"] for r in rows], name="Ticker"),
        )
        edit_help = (
            f"Click a cell in the {EDIT_LABEL[mode_col]} column to change it; weights update "
            "automatically. White columns are editable, grey ones are calculated. Tick Remove "
            "to drop a holding."
        )
        # Streamlit only styles read-only columns, so the editable ones stand out by contrast
        styled = table.style.set_properties(subset=READ_ONLY_COLUMNS, **READ_ONLY)
        key = f"ws_table_{st.session_state[VERSION]}"
        st.data_editor(
            styled,
            key=key,
            width="stretch",
            on_change=_apply_edits,
            args=(key, [r["ticker"] for r in rows], mode_col),
            column_config={
                "_index": st.column_config.TextColumn("Ticker", width=76),
                "Name": st.column_config.TextColumn(width=340),  # long fund names in full
                "Group": st.column_config.SelectboxColumn(options=GROUPS, required=True, width=140),
                mode_col: st.column_config.NumberColumn(
                    EDIT_LABEL[mode_col],
                    min_value=0.0,
                    format="localized" if mode_col == USD else "%.2f",
                    width=96,
                    help=edit_help,
                ),
                "Weight": st.column_config.NumberColumn(format="%.2f%%", width=66),
                "Remove": st.column_config.CheckboxColumn(
                    "✕", width=34, help="Tick to remove this holding."
                ),
                "Price (local)": st.column_config.NumberColumn(format="%.2f", width=100),
                "Ccy": st.column_config.TextColumn(width=50),
                "Price (USD)": st.column_config.NumberColumn(format="%.2f", width=100),
            },
            disabled=READ_ONLY_COLUMNS,
        )
        blanks = [r["ticker"] for r in rows if not (r["value"] or 0) > 0]
        if blanks:
            st.warning(
                f"Enter a positive {'amount' if mode_col == USD else 'weight'} for "
                f"{', '.join(blanks)}, or tick Remove; until then it is left out."
            )
        total = sum(r["value"] or 0 for r in rows if (r["value"] or 0) > 0)
        st.markdown(
            f'<div class="gpl-kv" style="grid-template-columns:70% 30%;font-weight:600">'
            f"<div>Total · {len(rows)} holding{'s' if len(rows) != 1 else ''}</div>"
            f'<div style="text-align:right">'
            f"{money(total, prefix='') if mode_col == USD else f'{total:.2f}%'}</div></div>",
            unsafe_allow_html=True,
        )
        times: dict[str, list[str]] = {}
        for r in rows:
            times.setdefault(getattr(quotes[r["ticker"]], "as_of", "not available"), []).append(
                r["ticker"]
            )
        as_of = "; ".join(f"{when} ({', '.join(ts)})" for when, ts in times.items())
    return as_of


def _load_preset(
    presets: Mapping[str, Sequence[str]],
    quote: Callable[[str], object],
    config_groups,
    has_data=None,
) -> None:
    tickers = list(presets[st.session_state.get("ws_preset") or next(iter(presets))])
    st.session_state["ws_import"] = "\n".join(f"{t} {100 / len(tickers):.6f}" for t in tickers)
    st.session_state["ws_import_kind"] = "Weights (%)"
    _import_list(quote, config_groups, has_data)


def remove_holding(ticker: str) -> None:
    """Callback: drop one holding (e.g. a new listing that shortens the common sample)."""
    st.session_state[ROWS] = [r for r in st.session_state[ROWS] if r["ticker"] != ticker]
    _bump()


def _import_list(quote: Callable[[str], object], config_groups, has_data=None) -> None:
    text = st.session_state.get("ws_import", "")
    kind = st.session_state.get("ws_import_kind", "Weights (%)")
    amounts = not kind.startswith("Weights")
    weights, problems, _ = parse_holdings(text, "amounts" if amounts else "weights")
    st.session_state["ws_import_problems"] = problems
    if not weights:
        return
    raw = parse_amounts(text) if amounts else {}
    if has_data:  # tickers without prices are reported, not added
        for t in list(weights):
            ok, why = has_data(t)
            if not ok:
                problems.append(f"{t}: {why}, so it was not added.")
                weights.pop(t)
                raw.pop(t, None)
        st.session_state["ws_import_problems"] = problems
        if not weights:
            return
        if not amounts:  # the remaining weights are rescaled to 100%
            total = sum(weights.values())
            weights = {t: w / total for t, w in weights.items()}
    new_rows = []
    for t, w in weights.items():
        q = quote(t)
        group = guess_group(
            t,
            getattr(q, "exchange", ""),
            getattr(q, "quote_type", ""),
            getattr(q, "name", ""),
            config_groups,
            st.session_state.get(CASH_LIKE, ()),
        )
        new_rows.append(
            {"ticker": t, "group": group, "value": raw[t] if amounts else round(w * 100, 4)}
        )
    st.session_state[ROWS] = new_rows
    mode = USD if amounts else PCT  # amounts stay amounts: they define the portfolio value
    st.session_state[MODE] = mode
    st.session_state["ws_mode_prev"] = mode
    if amounts:
        set_portfolio_value(sum(raw.values()))
    _bump()


def set_holdings(rows: list[dict]) -> None:
    """Replace the holdings with USD amounts (they define the portfolio value)."""
    st.session_state[ROWS] = [dict(r) for r in rows]
    st.session_state[MODE] = USD
    st.session_state["ws_mode_prev"] = USD
    set_portfolio_value(sum(r["value"] for r in rows))
    _bump()


def answer_sentence(report: Mapping) -> str:
    """The Optimise result in one sentence, from the report's own walk-forward figures.

    E.g. "Switch to Max Sharpe: return 5.04% to 6.67%, volatility 8.59% to 8.33%." or "Keep
    your current portfolio: no method beat it after costs."
    """
    rec = report["recommendation"]
    rows = report["walk_forward_after_costs"]
    choice, status = rec["choice"], rec["status"]
    if status == "closest_fails":
        return (
            "No portfolio meets every IPS rule: "
            + ("your current portfolio" if choice not in rows else choice)
            + " is the closest, shown for discussion, not as suitable."
        )
    if choice not in rows:  # keep
        return "Keep your current portfolio: no method beat it after costs."
    now, new = rows["Current"], rows[choice]
    lead = "Next best: switch to" if status == "next_best" else "Switch to"
    return (
        f"{lead} {choice}: return {now['cagr']} to {new['cagr']}, volatility "
        f"{now['annual_volatility']} to {new['annual_volatility']}."
    )


def key_finding(weights: pd.Series, risk_share: pd.Series) -> str:
    """The holding whose share of risk most exceeds its share of money, in one sentence.

    E.g. "EEM is 26% of the money but 58% of the risk." From the risk contributions already
    computed (``gpl.optimise.risk_contributions``); no new figures.
    """
    gap = (risk_share - weights).dropna()
    if gap.empty:
        return ""
    t = gap.idxmax()
    if gap[t] <= 0.005:
        return "Each holding's share of risk is close to its share of money."
    return f"{t} is {weights[t]:.0%} of the money but {risk_share[t]:.0%} of the risk."


def render_allocation(weights: pd.Series, groups: Mapping[str, str]) -> None:
    """The Allocation card: a donut with the total in the middle and a legend beside it."""
    h1, h2 = st.columns([1, 1.2], vertical_alignment="center")
    h1.markdown('<div class="gpl-section">Allocation</div>', unsafe_allow_html=True)
    by = h2.radio(
        "Show allocation by",
        ["Holding", "Group"],
        key="ws_alloc_by",
        horizontal=True,
        label_visibility="collapsed",
    )
    if by == "Holding":
        shown = weights.sort_values(ascending=False)
        names = {t: f"{t} · {groups.get(t, 'Other')}" for t in shown.index}
    else:
        shown = weights.groupby(lambda t: groups.get(t, "Other")).sum().sort_values(ascending=False)
        names = {g: str(g) for g in shown.index}
    value = portfolio_value()
    center = (
        f"Total<br><b>{compact_usd(value)}</b><br>{len(weights)} holdings"
        if value
        else f"{len(weights)} holdings"
    )
    chart_col, legend_col = st.columns([1, 1.2], vertical_alignment="center")
    with chart_col:
        plot(allocation_chart(shown, "", center=center, labels=False))
    colours = COMPOSITION
    rows = "".join(
        f'<div><i style="display:inline-block;width:9px;height:9px;border-radius:2px;'
        f'background:{colours[i % len(colours)]};margin-right:8px"></i>{names[k]}</div>'
        f'<div style="text-align:right;font-weight:600">{v:.2%}</div>'
        for i, (k, v) in enumerate(shown.items())
    )
    legend_col.markdown(
        f'<div class="gpl-kv" style="grid-template-columns:75% 25%">{rows}</div>',
        unsafe_allow_html=True,
    )


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


def render_kpis(
    kpi: pd.DataFrame | None, value_text: str, benchmark_name: str, is_cash: bool
) -> None:
    """Five KPI cards: value, return (CAGR), volatility, Sharpe, max drawdown, with deltas vs the
    benchmark (from ``gpl.metrics.kpi_table``). Lower volatility counts as better."""
    cols = st.columns(5)
    exact = portfolio_value()
    cols[0].metric(
        "Portfolio value",
        value_text,
        help=f"{money(exact, 2)}"
        if exact is not None
        else "Enter the portfolio value on Holdings (or holdings as USD amounts).",
    )
    if kpi is None:
        labels = ["Return (CAGR)", "Volatility", "Sharpe", "Max drawdown"]
        for c, label in zip(cols[1:], labels, strict=True):
            c.metric(label, "–")
        return

    # deltas are vs the benchmark (named in the caption below the cards); kept short so they fit
    def pp(x: float) -> str:
        return f"{no_negative_zero(x * 100, 2):+.2f} pp"

    sharpe_delta = kpi.loc["sharpe", "delta"]
    cols[1].metric("Return (CAGR)", pct(kpi.loc["cagr", "portfolio"]), pp(kpi.loc["cagr", "delta"]))
    cols[2].metric(
        "Volatility",
        pct(kpi.loc["volatility", "portfolio"]),
        pp(kpi.loc["volatility", "delta"]),
        delta_color="inverse",
    )
    cols[3].metric(
        "Sharpe",
        f"{kpi.loc['sharpe', 'portfolio']:.2f}",
        None if pd.isna(sharpe_delta) or is_cash else f"{no_negative_zero(sharpe_delta, 2):+.2f}",
    )
    cols[4].metric(
        "Max drawdown",
        pct(kpi.loc["max_drawdown", "portfolio"]),
        None if is_cash else pp(kpi.loc["max_drawdown", "delta"]),
    )


def portfolio_value_text() -> str:
    """The portfolio value, compact so it fits a KPI card ('$3.00m'); '–' if not entered."""
    return compact_usd(portfolio_value())


def _norm(heading: str) -> str:
    return " ".join(heading.strip().lstrip("#").strip().strip("*").lower().split())


def memo_sections(text: str, charts: Sequence[tuple[str, object]]) -> list[tuple[str, list]]:
    """Split a memo on '###' headings and attach each chart to the section it names.

    Returns ``[(markdown, charts), ...]`` in memo order. Text before the first heading is its
    own block; charts whose section matches no heading are attached to a final empty block,
    so nothing the analyst asked for is lost.
    """
    blocks: list[tuple[str, str]] = []  # (heading key, markdown)
    current: list[str] = []
    key = ""
    for line in text.splitlines():
        if line.startswith("### "):
            if current:
                blocks.append((key, "\n".join(current)))
            key, current = _norm(line), [line]
        else:
            current.append(line)
    if current:
        blocks.append((key, "\n".join(current)))
    out = [(md, []) for _, md in blocks]
    unplaced = []
    keys = [k for k, _ in blocks]
    for section, data in charts:
        target = _norm(section)
        if target and target in keys:
            out[keys.index(target)][1].append(data)
        else:
            unplaced.append(data)
    if unplaced:
        # before the disclaimer the app appends ("---" then "... not investment advice ...")
        last_md = out[-1][0] if out else ""
        cut = last_md.rfind("\n---\n")
        if cut >= 0 and "not investment advice" in last_md[cut:].lower():
            out[-1] = (last_md[:cut], out[-1][1])
            out += [("", unplaced), (last_md[cut:], [])]
        else:
            out.append(("", unplaced))
    return out


TARGET_COL = "Target % (edit)"


def _edited_targets(key: str, base: pd.Series) -> pd.Series:
    """The targets with the adviser's pending edits applied (read before the editor is drawn,
    so the drift, trades and totals shown in the same table follow the edits)."""
    target = base.copy()
    for row, change in ((st.session_state.get(key) or {}).get("edited_rows") or {}).items():
        new = change.get(TARGET_COL)
        if new is not None and int(row) < len(target):
            target.iloc[int(row)] = float(new) / 100
    return target


def render_rebalance(
    current: pd.Series,
    default_target: pd.Series,
    target_source: str,
    usd_values: pd.Series | None,
    value: float | None,
    price_usd: Callable[[str], float | None],
    cost_bps: float,
    cash_need: Mapping | None = None,
    cost_by_suffix: Mapping[str, float] | None = None,
    kpis=None,
) -> dict | None:
    """Trade list card: one table with each holding's action, current and editable target
    weight, drift, amount, cost and value after (``gpl.scenarios``); totals as KPI cards in
    ``kpis``. Returns the formatted result.

    ``cash_need``: ``{"need_usd", "due", "cash_like"}`` from the client's IPS; the trades are
    checked against it (``gpl.scenarios.cash_need_check``).
    """
    from gpl.scenarios import cash_need_check, drift, needs_rebalance, trade_list

    kpis = kpis if kpis is not None else st.container()
    tickers = list(dict.fromkeys([*current.index, *default_target.index]))
    key = "rb_" + "_".join(tickers) + "_" + target_source.replace(" ", "")
    base = default_target.reindex(tickers).fillna(0.0)
    target = _edited_targets(key, base)
    result = None
    with st.container(border=True, key="trades_card"):
        head, band_col, csv_col = st.columns([2.2, 1.3, 1.5], vertical_alignment="bottom")
        head.markdown(
            '<div class="gpl-section">Trade list</div>',
            unsafe_allow_html=True,
            help=f"Targets start from the {target_source}; edit any target. Drift = current minus "
            "target weight. A rebalance brings every holding back to its target, with costs "
            "paid from the sales.",
            width="content",
        )
        band = (
            band_col.number_input(
                "Drift band ± (pp)",
                0.5,
                50.0,
                5.0,
                0.5,
                key="rb_band",
                help="A holding further than this from its target, in percentage points, "
                "triggers a rebalance suggestion.",
            )
            / 100
        )
        ok = value is not None and abs(target.sum() - 1) <= 1e-4
        d = drift(current, target, band) if ok else None
        values = usd_values if usd_values is not None else current * (value or 0)
        prices = pd.Series({t: price_usd(t) for t in tickers}, dtype=float)
        trades = (
            trade_list(
                values,
                target,
                prices,
                cost_bps,
                None if usd_values is not None else value,
                cost_by_suffix,
            )  # fmt: skip
            if ok
            else None
        )

        def num(series):
            return [
                float(series[t]) if series is not None and t in series.index else float("nan")
                for t in tickers
            ]

        frame = pd.DataFrame(
            {
                "Action": [
                    str(trades["action"][t]) if trades is not None and t in trades.index else "–"
                    for t in tickers
                ],
                "Current": [current.get(t, 0.0) * 100 for t in tickers],
                TARGET_COL: base * 100,
                "Drift": [
                    f"{no_negative_zero(d['drift'][t] * 100, 2):+.2f} pp"
                    if d is not None and t in d.index
                    else "–"
                    for t in tickers
                ],
                "Amount (USD)": num(None if trades is None else trades["trade_usd"].abs().round(0)),
                "Est. cost": num(None if trades is None else trades["cost_usd"]),
                "After (USD)": num(None if trades is None else trades["target_usd"].round(0)),
            },
            index=pd.Index(tickers, name="Ticker"),
        )
        outside = set() if d is None else set(d.index[d["outside_band"]])
        styled = frame.style.apply(
            lambda c: [
                "background-color: #FFF4E0; color: #B26B00; font-weight: 600"
                if t in outside else "" for t in frame.index
            ],
            subset=["Drift"],
        )  # fmt: skip
        st.data_editor(
            styled,
            key=key,
            width="stretch",
            disabled=[c for c in frame.columns if c != TARGET_COL],
            column_config={
                "_index": st.column_config.TextColumn("Ticker", width=82),
                "Action": st.column_config.TextColumn(width=55),
                "Current": st.column_config.NumberColumn("Current %", format="%.2f", width=80),
                "Amount (USD)": st.column_config.NumberColumn(format="localized", width=112),
                "After (USD)": st.column_config.NumberColumn(format="localized", width=100),
                TARGET_COL: st.column_config.NumberColumn(
                    "Target %",
                    format="%.2f",
                    min_value=0.0,
                    max_value=100.0,
                    width=92,  # room for the header's edit icon
                    help="Edit any target.",
                ),
                "Drift": st.column_config.TextColumn(width=78),
                "Est. cost": st.column_config.NumberColumn(format="%.2f", width=70),
            },
        )
        st.caption(
            "Units are estimates at the latest price; round to the broker's lot size. Check "
            "taxes and liquidity before trading."
        )
        if value is None:
            st.info("Enter the portfolio value on step 2 (Portfolio) to see the drift and trades.")
            return None
        if not ok:
            st.warning(f"Targets add up to {target.sum():.2%}; they must add up to 100.00%.")
            return None
        flagged = list(d.index[d["outside_band"]])
        shown = pd.DataFrame(
            {
                "Current": d["current"].map(pct),
                "Target": d["target"].map(pct),
                "Drift": d["drift"].map(lambda x: f"{no_negative_zero(x * 100, 2):+.2f} pp"),
                "Outside band": d["outside_band"].map({True: "⚠️ yes", False: "no"}),
            },
            index=d.index,
        )
        result = {
            "band": pct(band),
            "outside_band": flagged,
            "drift": shown.to_dict(orient="index"),
        }
        if cash_need and cash_need["need_usd"] > 0:
            check = cash_need_check(
                trades["target_usd"], cash_need["cash_like"], cash_need["need_usd"]
            )
            held_in = ", ".join(cash_need["cash_like"]) or "none held"
            if check["ok"]:
                st.success(
                    f"✅ After these trades, cash-like holdings ({held_in}) are USD "
                    f"{check['held']:,.0f}, covering the {money(check['need'])} needed "
                    f"({cash_need['due']})."
                )
            else:
                st.warning(
                    f"⚠️ After these trades, cash-like holdings ({held_in}) are USD "
                    f"{check['held']:,.0f}, {money(check['shortfall'])} short of the USD "
                    f"{check['need']:,.0f} the client needs ({cash_need['due']}). Raise the "
                    "target for cash, short-term bonds or sukuk."
                )
            result["cash_need_check"] = {
                "cash_like_after_trades_usd": f"{check['held']:,.0f}",
                "needed_usd": f"{check['need']:,.0f}",
                "shortfall_usd": f"{check['shortfall']:,.0f}",
                "due": cash_need["due"],
            }
        csv_col.download_button(
            "Trade list (CSV)",
            trades.to_csv(),
            file_name="trade_list.csv",
            mime="text/csv",
            key="rb_csv",
            icon=":material/download:",
            width="stretch",
        )
    buys = trades.loc[trades.trade_usd > 0, "trade_usd"].sum()
    sells = -trades.loc[trades.trade_usd < 0, "trade_usd"].sum()
    costs = trades["cost_usd"].sum()
    rebalance = needs_rebalance(d)
    with kpis:
        layout_cards = [
            ("Sells", f"{money(sells)}", ""),
            ("Buys", f"{money(buys)}", ""),
            ("Estimated costs", f"{money(costs, 2)}", ""),
        ]
        from html import escape as _esc

        cards = "".join(
            f'<div class="gpl-kpi"><div class="gpl-kpi-label">{lab}</div>'
            f'<div class="gpl-kpi-value">{val}</div></div>'
            for lab, val, _ in layout_cards
        )
        if rebalance:
            cards += (
                '<div class="gpl-kpi warn"><div class="gpl-kpi-label">Rebalance suggested</div>'
                f'<div class="gpl-kpi-value">{_esc(", ".join(flagged))} '
                f"{'is' if len(flagged) == 1 else 'are'} outside the ±{band * 100:g} pp band."
                "</div></div>"
            )
        else:
            cards += (
                '<div class="gpl-kpi"><div class="gpl-kpi-label">Drift</div>'
                f'<div class="gpl-kpi-value" style="font-size:.9rem;font-weight:400;'
                f'white-space:normal">Every holding is within ±{band * 100:g} pp of its target: '
                "no rebalance needed.</div></div>"
            )
        st.markdown(f'<div class="gpl-kpis" style="--n:4">{cards}</div>', unsafe_allow_html=True,
                    help=f"Estimated costs at {cost_bps:g} bps for US listings and local-market "
                    "rates for the others (buys + costs = sells).")  # fmt: skip
    if rebalance:
        result["trades"] = pd.DataFrame(
            {
                "Action": trades["action"],
                "Amount (USD)": trades["trade_usd"].abs().map("{:,.2f}".format),
                "Est. units": trades["units"].map(
                    lambda u: "–" if pd.isna(u) else f"{abs(u):,.2f}"
                ),
                "Cost rate": trades["cost_bps"].map("{:g} bps".format),
                "Est. cost (USD)": trades["cost_usd"].map("{:,.2f}".format),
                "After (USD)": trades["target_usd"].map("{:,.2f}".format),
            },
            index=trades.index,
        ).to_dict(orient="index")
    result["totals"] = {
        "sells_usd": f"{sells:,.2f}",
        "buys_usd": f"{buys:,.2f}",
        "costs_usd": f"{costs:,.2f}",
    }
    return result
