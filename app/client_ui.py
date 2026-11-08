"""Step 1 (Client): fact-find form, risk questionnaire, IPS preview and downloads.

The profile lives in ``st.session_state["client"]`` (a dict, ``ClientProfile.to_dict()``).
Widgets are keyed with a version number so loading a profile (demo, example or JSON)
redraws the form. Every number shown comes from ``gpl.client`` (tested rules).
"""

import json
import secrets
from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from html import escape

import layout
import pandas as pd
import streamlit as st
import workstation as ws

from formatting import money
from gpl.client import (
    CLOSEST_MIX_EXAMPLES,
    CLOSEST_MIX_LABEL,
    EXAMPLE_CLIENTS,
    JURISDICTIONS,
    QUESTIONS,
    REGIONS,
    REGULATORS,
    SCORED,
    ClientProfile,
    age_on,
    build_ips,
    closest_mix_reason,
    default_regulator,
    effective_regulator,
    example_holdings,
    ips_constraints,
    liquidity_need_score,
    month_text,
    parse_month,
)
from gpl.demo import demo_client
from gpl.factfind import (
    ASSET_TYPES,
    INCOME_TYPES,
    LIABILITY_TYPES,
    PUBLIC_NOTICE,
    SECTION_BY_KEY,
    SPECIALIST_REMINDER,
    Completeness,
    Field,
    completeness,
    complex_product_flags,
    consistency_warnings,
    finances,
    goals_required_return,
    ips_missing,
    next_review,
    overall_completeness,
)
from gpl.report import ips_pdf

CLIENT, VERSION = "client", "client_v"
DEMO_BUTTON = "▶ Try a demo client"
DEMO_TOAST = (
    "Demo client loaded. Review the client and IPS, then use Next to walk through each step."
)
WELCOME_TITLE = "Turn a client's facts into a suitable, tested portfolio"
WELCOME = (
    "Fact-find, IPS, analysis, optimisation, suitability and trades in six steps. Every figure "
    "comes from tested code."
)
SCORED_QUESTIONS = [q["id"] for q in QUESTIONS if q["dimension"] in SCORED]
RISK_WORDS = {1: "Conservative", 2: "Cautious", 3: "Moderate", 4: "Growth", 5: "Aggressive"}
EXAMPLE = "client_example_loaded"  # the example whose own portfolio was loaded
SECTOR_CHOICES = [
    "Tobacco",
    "Alcohol",
    "Gambling",
    "Weapons",
    "Adult entertainment",
    "Conventional banks",
    "Energy",
    "Fossil fuels",
]


def current_profile() -> ClientProfile | None:
    """The client profile, with the app's single portfolio value (``ws.portfolio_value``)."""
    data = st.session_state.get(CLIENT)
    if data is None:
        return None
    profile = ClientProfile.from_dict(data)
    profile.portfolio_value_usd = ws.portfolio_value()
    return profile


def _set_profile(profile: ClientProfile) -> None:
    if profile.portfolio_value_usd and st.session_state.get(ws.MODE) != ws.USD:
        ws.set_portfolio_value(profile.portfolio_value_usd)  # USD holdings define it otherwise
    st.session_state[CLIENT] = profile.to_dict()
    st.session_state[VERSION] = st.session_state.get(VERSION, 0) + 1
    st.session_state.pop("ws_draft", None)  # the Optimise form re-reads the IPS
    # a new form version, so the goal starts from this client's risk score (old widget values
    # would otherwise win)
    st.session_state["ws_draft_v"] = st.session_state.get("ws_draft_v", 0) + 1


def _load_example() -> None:
    name = st.session_state.get("client_example")
    if name in EXAMPLE_CLIENTS:
        profile = ClientProfile.from_dict(EXAMPLE_CLIENTS[name])
        ws.set_holdings(example_holdings(name, profile.portfolio_value_usd))  # its own portfolio
        _set_profile(profile)
        st.session_state[EXAMPLE] = name
        for key in ("ws_opt", "client_report", "client_report_flags", "suit_fails"):
            st.session_state.pop(key, None)


def _load_json() -> None:
    upload = st.session_state.get("client_upload")
    if upload is None:
        return
    try:
        _set_profile(ClientProfile.from_dict(json.loads(upload.getvalue())))
        st.session_state.pop("client_error", None)
        st.session_state.pop(EXAMPLE, None)
    except (ValueError, TypeError) as e:  # bad JSON or bad values
        st.session_state.client_error = f"Couldn't read that profile: {e}"


def fill_demo(seed: int | None = None) -> None:
    """Callback: a seeded demo client and portfolio (no AI), then open step 1 (Client).

    ``seed`` None takes the one typed in ⚙ Settings › Demo seed, else a new random one; the
    seed is shown in the header so the same demo can be reproduced.
    """
    if seed is None:
        typed = st.session_state.get("demo_seed_input")
        seed = int(typed) if typed is not None else secrets.randbelow(100_000)
    demo = demo_client(seed)
    ws.set_holdings(demo.holdings)
    _set_profile(demo.profile)
    st.session_state.pop(EXAMPLE, None)
    st.session_state["demo_seed"] = seed
    st.session_state["demo_tickers"] = sorted(h["ticker"] for h in demo.holdings)
    for key in ("ws_opt", "client_report", "client_report_flags", "suit_fails", "ctx_key"):
        st.session_state.pop(key, None)
    st.session_state["main_tab"] = "1 Client"  # start at the beginning; Next walks the steps
    st.session_state["demo_toast"] = True  # shown once, with a scroll to the top


def _new_client() -> None:
    _set_profile(ClientProfile())
    st.session_state.pop(EXAMPLE, None)


def start_blank() -> None:
    """Callback: a new blank client, then open the Client step to fill it in."""
    _new_client()
    st.session_state["main_tab"] = "1 Client"


def closest_mix_note(ips) -> str | None:
    """The Riyadh-style label and reason while that example's own portfolio is loaded.

    None once another client is loaded or the holdings are changed: then the label would no
    longer describe what is on screen.
    """
    name = st.session_state.get(EXAMPLE)
    if name not in CLOSEST_MIX_EXAMPLES or ips is None:
        return None
    value = ws.portfolio_value()
    rows = {r["ticker"]: round(r["value"], 2) for r in st.session_state.get(ws.ROWS, [])}
    own = {h["ticker"]: round(h["value"], 2) for h in example_holdings(name, value or 0)}
    if rows != own:
        return None
    return closest_mix_reason(ips)


def _example_label(name: str) -> str:
    return f"{name} · closest mix" if name in CLOSEST_MIX_EXAMPLES else name


def _usd(x: float) -> str:
    """'USD 1.31m', 'USD 260k', 'USD 950' (short, for one-line summaries)."""
    if abs(x) >= 1e6:
        return f"USD {x / 1e6:.2f}m"
    if abs(x) >= 1e3:
        return f"{money(x / 1e3)}k"
    return f"{money(x)}"


def _join(parts) -> str:
    return " · ".join(p for p in parts if p)


def section_summary(key: str, p: ClientProfile) -> str:
    """One line summing up a fact-find section, from the recorded facts (no new figures)."""
    if key == "household":
        deps = f"{p.dependants} dependant{'s' if p.dependants != 1 else ''}" if p.dependants else ""
        return _join([p.name, f"age {p.age}" if p.age else "", p.residency,
                      p.marital_status.lower() if p.marital_status else "", deps])  # fmt: skip
    if key == "kyc":
        return _join([
            f"{p.classification} client" if p.classification else "",
            "PEP" if p.pep else "not a PEP",
            p.expected_activity.split(" (")[0].lower() + " activity" if p.expected_activity else "",
        ])  # fmt: skip
    if key == "financial":
        f = finances(p)
        return _join([
            f"Net worth {money(f.net_worth)}",
            f"investable {money(f.investable)}",
            f"savings rate {f.savings_rate:.1%}" if f.savings_rate is not None else "",
            f"emergency fund {f.emergency_months:.1f} months" if f.emergency_months else "",
        ])  # fmt: skip
    if key == "objectives":
        goals = [
            f"{g.get('description', 'goal').lower()} {_usd(g['amount_usd'])}"
            for g in p.goals if g.get("amount_usd")
        ]  # fmt: skip
        needs = [
            f"{n['description'].lower()} {_usd(n['amount_usd'])} ({month_text(n['date'])})"
            for n in p.liquidity_needs
        ]
        return _join([p.primary_objective, f"{p.horizon_years:g} years" if p.horizon_years else "",
                      *goals, *needs])  # fmt: skip
    if key == "risk":
        missing = [q for q in SCORED_QUESTIONS if q not in p.answers]
        return _join([
            "Questionnaire complete" if not missing else f"{len(missing)} answers missing",
            f"capacity for loss {p.max_acceptable_drawdown_pct:g}%"
            if p.max_acceptable_drawdown_pct else "",
            f"adviser override to {p.override_score}"
            if p.override_score
            else "no adviser override",
        ])  # fmt: skip
    if key == "knowledge":
        return _join([
            f"{p.years_investing:g} years investing" if p.years_investing else "",
            ", ".join(x.lower() for x in p.products_used),
            p.background.lower() if p.background else "",
        ])  # fmt: skip
    if key == "constraints":
        regions = [f"{r} at least {v:g}%" for r, v in p.region_minimums.items() if v]
        return _join([
            "Optional",
            "Sharia-compliant only" if p.sharia_only else "",
            f"GCC at least {p.home_bias_gcc_pct:g}%" if p.home_bias_gcc_pct else "",
            *regions,
            f"non-USD-pegged currencies at most {p.max_non_pegged_currency_pct:g}%",
            ("excluding " + ", ".join(x.lower() for x in p.excluded_sectors))
            if p.excluded_sectors else "",
        ])  # fmt: skip
    if key == "estate":
        return _join([p.will_status, "beneficiaries noted" if p.beneficiaries_noted else "",
                      ", ".join(p.structures)])  # fmt: skip
    if key == "service":
        fee = p.fee_basis.replace(" a year of assets under advice", " of assets")
        review = f"{p.review_frequency.lower()} review" if p.review_frequency else ""
        adviser = f"adviser {p.adviser_name}" if p.adviser_name else ""
        return _join([p.service_type, review, fee, adviser])
    return ""


def ips_rows(ips) -> list[tuple[str, str]]:
    """The IPS in short rows (Return, Risk, Time, Taxes, Liquidity, Legal, Unique) for the card.

    Built from the IPS's own limits and facts; the full RRTTLLU text stays in the expander.
    """
    p, lim = ips.client, ips.limits

    def pc(x):
        return "–" if x is None else f"{x * 100:g}%"

    req = lim.get("required_return")
    target = p.return_target_pct
    ret = (
        f"{req:.2%} a year needed" if req is not None else "No income drawn from the portfolio"
    ) + (f"; target {target:g}%" if target is not None else "; no target set")
    risk = (
        f"Volatility ≤ {pc(lim['max_volatility'])}, drawdown ≤ {pc(lim['max_drawdown'])}, "
        f"≤ {pc(lim['max_weight'])} in one holding"
        if lim["max_volatility"] is not None
        else "Limits set once the questionnaire is complete"
    )
    due = lim["cash_needs_due"]
    liquidity = (
        "; ".join(
            f"{n['description']}, {money(n['amount_usd'])} by {month_text(n['date'])}" for n in due
        )  # fmt: skip
        if due
        else f"No cash need within {lim['cash_need_window_months']} months"
    )
    unique = _join([
        "Sharia only" if lim["sharia_only"] else "",
        f"GCC ≥ {pc(lim['home_bias_gcc'])}" if lim["home_bias_gcc"] else "",
        *(f"{r} ≥ {pc(v)}" for r, v in lim["region_minimums"].items()),
        f"non-USD-pegged currencies ≤ {pc(lim['max_non_pegged_currency'])}",
        ("no " + ", ".join(x.lower() for x in lim["excluded_sectors"]))
        if lim["excluded_sectors"] else "",
    ])  # fmt: skip
    return [
        ("Return", ret),
        ("Risk", risk),
        ("Time", _join([f"{p.horizon_years:g} years", f"client aged {p.age}" if p.age else ""])),
        (
            "Taxes",
            _join(
                [
                    f"Resident in {p.residency}" if p.residency else "",
                    "US person" if p.us_person else "not a US person",
                ]
            ),
        ),  # fmt: skip
        ("Liquidity", liquidity),
        ("Legal", _join([f"{p.classification} client", effective_regulator(p)])),
        ("Unique", unique),
    ]


def _default_regulator(k: str) -> None:
    """Booking location or residency changed: reset the regulator to the default for it."""
    st.session_state["cl_reg" + k] = default_regulator(
        st.session_state.get("cl_book" + k), st.session_state.get("cl_res" + k)
    )


def _opt(value):
    return None if value is None else float(value)


# existing widget keys (kept so saved sessions and tests still find them)
KEYS = {
    "reference": "cl_ref",
    "name": "cl_name",
    "age": "cl_age",
    "classification": "cl_class",
    "residency": "cl_res",
    "us_person": "cl_us",
    "horizon_years": "cl_hor",
    "return_target_pct": "cl_ret",
    "annual_income_need_usd": "cl_income",
    "tax_notes": "cl_tax",
    "legal_notes": "cl_legal",
}


def _key(name: str, k: str) -> str:
    return KEYS.get(name, f"cl_f_{name}") + k


def _field(col, f: Field, p: ClientProfile, k: str, **kw):
    """Draw one registry field in ``col`` and return its value (in the profile's form)."""
    value = getattr(p, f.name)
    key = _key(f.name, k)
    label = f.label + ("" if f.required else " (optional)")
    help_ = f.help or None
    if f.kind == "text":
        return col.text_input(label, value, key=key, help=help_, **kw)
    if f.kind == "textarea":
        return col.text_area(label, value, key=key, help=help_, height=80, **kw)
    if f.kind == "int":
        out = col.number_input(label, 0, 120, value, 1, key=key, help=help_, placeholder="–", **kw)
        return None if out is None else int(out)
    if f.kind in ("float", "money"):
        money = f.kind == "money"
        return col.number_input(
            label,
            0.0 if money else -100.0,
            1e12 if money else 100.0,
            _opt(value),
            1000.0 if money else 1.0,
            format="%.0f" if money else "%.2f",
            key=key,
            help=help_,
            placeholder="–",
            **kw,
        )
    if f.kind == "bool":
        return col.checkbox(label, bool(value), key=key, help=help_, **kw)
    if f.kind == "yesno":
        choice = col.radio(
            label,
            ["No", "Yes"],
            index=None if value is None else int(bool(value)),
            key=key,
            horizontal=True,
            help=help_,
        )
        return None if choice is None else choice == "Yes"
    if f.kind == "select":
        options = list(f.options)
        return col.selectbox(
            label,
            options,
            index=options.index(value) if value in options else None,
            key=key,
            help=help_,
            placeholder="choose",
            **kw,
        )
    if f.kind == "multi":
        options = list(f.options) + [x for x in value if x not in f.options]
        return col.multiselect(label, options, default=list(value), key=key, help=help_)
    if f.kind in ("tags", "ints"):
        text = col.text_input(
            label, ", ".join(str(x) for x in value), key=key, help=help_ or "Separate with commas."
        )
        items = [t.strip() for t in text.split(",") if t.strip()]
        if f.kind == "tags":
            return items
        ints = [int(t) for t in items if t.isdigit()]
        if len(ints) != len(items):
            col.warning("Ages must be whole numbers, e.g. 8, 12.")
        return ints
    if f.kind == "date":
        current = date.fromisoformat(value) if value else None
        out = col.date_input(
            label,
            current,
            min_value=date(1920, 1, 1),
            max_value=date(2100, 12, 31),
            key=key,
            help=help_,
            format="YYYY-MM-DD",
        )
        return out.isoformat() if out else ""
    raise ValueError(f"no widget for kind {f.kind!r}")


def _fields(p: ClientProfile, k: str, section: str, skip: tuple[str, ...] = ()) -> dict:
    """Draw a section's simple fields in three columns; returns name -> value."""
    simple = [
        f
        for f in SECTION_BY_KEY[section].fields
        if f.kind not in ("custom", "money_map") and f.name not in skip
    ]
    cols = st.columns(3)
    return {f.name: _field(cols[i % 3], f, p, k) for i, f in enumerate(simple)}


def _money_map(p: ClientProfile, k: str, name: str, types: Mapping[str, str], label: str) -> dict:
    st.markdown(f"**{label}**")
    cols = st.columns(len(types))
    values = getattr(p, name)
    out = {}
    for col, (key, text) in zip(cols, types.items(), strict=True):
        amount = col.number_input(
            text,
            0.0,
            1e12,
            _opt(values.get(key)),
            1000.0,
            format="%.0f",
            key=f"cl_{name}_{key}" + k,
            placeholder="0",
        )
        if amount:
            out[key] = float(amount)
    return out


def _card(section: str, parts: Mapping[str, Completeness], p: ClientProfile | None = None):
    """A collapsible section: its title and a one-line summary; a check when complete."""
    s = SECTION_BY_KEY[section]
    c = parts[section]
    summary = section_summary(section, p) if p is not None else ""
    if c.total and c.missing:
        summary = f"{c.done} of {c.total} done · " + summary if summary else f"{c.done}/{c.total}"
    label = f"**{s.title}**" + (f" &nbsp; :gray[{summary}]" if summary else "")
    icon = (
        ":material/check_circle:"
        if c.total and not c.missing
        else ":material/radio_button_unchecked:"
    )
    first_open = section == next((k for k, v in parts.items() if v.missing), "household")
    box = st.expander(label, expanded=first_open, icon=icon)
    box.caption(s.helper)
    return box


def _form(
    p: ClientProfile, v: int, product_types: Mapping[str, str] | None = None
) -> ClientProfile:
    """Draw the fact-find from ``p`` and return the profile the widgets now describe."""
    k = f"_{v}"
    vals: dict = {}
    parts = completeness(p)

    # 1 Client & household -------------------------------------------------------------
    with _card("household", parts, p):
        c1, c2, c3 = st.columns([1, 1, 1.4])  # regulator names are long: a wider column
        c4 = st
        vals["residency"] = c1.text_input(
            "Residency (country)",
            p.residency,
            key="cl_res" + k,
            on_change=_default_regulator,
            args=(k,),
        )
        bookings = ["", *JURISDICTIONS]
        vals["booking_location"] = c2.selectbox(
            "Booking location",
            bookings,
            index=bookings.index(p.booking_location) if p.booking_location in bookings else 0,
            key="cl_book" + k,
            format_func=lambda b: b or "from residency",
            on_change=_default_regulator,
            args=(k,),
            help="Where the client is booked sets the regulator: ADGM -> FSRA, DIFC -> DFSA, "
            "onshore UAE -> SCA, Saudi Arabia -> CMA, QFC -> QFCRA, Bahrain -> CBB, Kuwait -> "
            "CMA, Oman -> FSA. Change the regulator if it differs.",
        )
        current = effective_regulator(p)
        vals["regulator"] = c3.selectbox(
            "Regulator",
            REGULATORS,
            index=REGULATORS.index(current) if current in REGULATORS else len(REGULATORS) - 1,
            key="cl_reg" + k,
        )
        vals["us_person"] = c4.checkbox(
            "US person (citizen, green card or US tax resident)", p.us_person, key="cl_us" + k
        )
        vals |= _fields(p, k, "household", skip=("residency", "us_person"))
        if vals["dob"]:
            vals["age"] = age_on(vals["dob"], date.today())

    # 2 KYC / AML ------------------------------------------------------------------------
    with _card("kyc", parts, p):
        vals |= _fields(p, k, "kyc")
        if vals["pep"]:
            st.warning(
                "Politically exposed person: enhanced due diligence and senior-management "
                "approval are needed before onboarding."
            )

    # 3 Financial situation --------------------------------------------------------------
    with _card("financial", parts, p):
        vals["income"] = _money_map(p, k, "income", INCOME_TYPES, "Annual income by source (USD)")
        c1, _ = st.columns(2)
        vals["annual_expenses_usd"] = _field(c1, SECTION_BY_KEY["financial"].fields[1], p, k)
        vals["assets"] = _money_map(p, k, "assets", ASSET_TYPES, "Assets by type (USD)")
        vals["liabilities"] = _money_map(
            p, k, "liabilities", LIABILITY_TYPES, "Liabilities (USD, optional)"
        )
        fin = finances(replace(p, **vals))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Net worth (USD)", f"{fin.net_worth:,.0f}")
        m2.metric("Investable assets (USD)", f"{fin.investable:,.0f}")
        m3.metric("Savings rate", "–" if fin.savings_rate is None else f"{fin.savings_rate:.2%}")
        m4.metric(
            "Emergency fund",
            "–" if fin.emergency_months is None else f"{fin.emergency_months:.1f} months",
        )
        st.caption(
            "Net worth = assets - liabilities. Investable = cash + investments. Savings rate = "
            "(income - expenses) ÷ income. Emergency fund = cash ÷ monthly expenses."
        )

    # 4 Objectives & goals ---------------------------------------------------------------
    with _card("objectives", parts, p):
        c1, c2, c3 = st.columns(3)
        vals["primary_objective"] = _field(c1, SECTION_BY_KEY["objectives"].fields[0], p, k)
        vals["horizon_years"] = c2.number_input(
            "Time horizon (years)", 0.5, 60.0, float(p.horizon_years), 0.5, key="cl_hor" + k
        )
        ws.value_input(c3, "pf_value_client")
        c1, c2, c3, c4 = st.columns(4)
        vals["return_target_pct"] = c1.number_input(
            "Return target (% a year, optional)",
            -20.0,
            50.0,
            _opt(p.return_target_pct),
            0.5,
            key="cl_ret" + k,
            placeholder="none",
        )
        vals["annual_income_need_usd"] = c2.number_input(
            "Yearly income drawn from the portfolio (USD)",
            0.0,
            1e9,
            float(p.annual_income_need_usd),
            1000.0,
            key="cl_income" + k,
            help="Sets the required return: income ÷ investable assets (after cash set aside), "
            "plus inflation.",
        )
        vals["annual_contribution_usd"] = c3.number_input(
            "Yearly addition to the portfolio (USD)",
            0.0,
            1e9,
            float(p.annual_contribution_usd),
            1000.0,
            key="cl_f_annual_contribution_usd" + k,
            help="Used in the return needed to meet the goals.",
        )
        vals["inflation_pct"] = c4.number_input(
            "Inflation assumption (% a year)",
            0.0,
            20.0,
            _opt(p.inflation_pct),
            0.25,
            key="cl_infl" + k,
            placeholder="config default (2.5)",
        )
        st.caption(
            "Cash needs: what the client must withdraw, and when (YYYY-MM). Needs due within "
            "the IPS window must be held in cash, short-term bonds or sukuk from now."
        )
        c1, c2 = st.columns([3, 1])
        needs = c1.data_editor(
            pd.DataFrame(p.liquidity_needs or [], columns=["description", "amount_usd", "date"]),
            num_rows="dynamic",
            key="cl_needs" + k,
            width="stretch",
            column_config={
                "description": st.column_config.TextColumn("Cash need"),
                "amount_usd": st.column_config.NumberColumn(
                    "Amount (USD)", format="%.0f", min_value=0
                ),
                "date": st.column_config.TextColumn("When (YYYY-MM)", help="e.g. 2028-10"),
            },
        )
        vals["reserve_pct"] = c2.number_input(
            "Extra cash reserve (% of portfolio)",
            0.0,
            100.0,
            float(p.reserve_pct),
            1.0,
            key="cl_reserve" + k,
        )
        vals["liquidity_needs"] = _need_rows(needs)
        st.caption("Goals: an amount (or a multiple of today's value) and when, in years.")
        goals = st.data_editor(
            pd.DataFrame(p.goals or [], columns=["description", "amount_usd", "multiple", "years"]),
            num_rows="dynamic",
            key="cl_goals" + k,
            width="stretch",
            column_config={
                "description": st.column_config.TextColumn("Goal"),
                "amount_usd": st.column_config.NumberColumn("Amount (USD)", format="%.0f"),
                "multiple": st.column_config.NumberColumn("Or multiple of today's value"),
                "years": st.column_config.NumberColumn("In years"),
            },
        )
        vals["goals"] = [
            {c: row[c] for c in goals.columns if pd.notna(row[c]) and row[c] != ""}
            for _, row in goals.iterrows()
        ]
        vals["goals"] = [g for g in vals["goals"] if g]
        need = goals_required_return(
            ws.portfolio_value(),
            vals["goals"],
            vals["liquidity_needs"],
            vals["annual_contribution_usd"],
        )
        st.markdown(f"**Return needed to meet the goals:** {need.reason}")

    # 5 Risk profile ---------------------------------------------------------------------
    with _card("risk", parts, p):
        st.markdown("**Risk questionnaire** (the client's own answers)")
        answers = {}
        q1, q2 = st.columns(2)
        liquidity_score, liquidity_share = liquidity_need_score(
            replace(
                p, liquidity_needs=vals["liquidity_needs"], portfolio_value_usd=ws.portfolio_value()
            ),
            date.today(),
        )
        for i, q in enumerate(QUESTIONS):
            if q["dimension"] not in SCORED:
                continue
            col = q1 if q["dimension"] == "capacity" else q2
            derived = q["id"] == "liquidity_need" and liquidity_score is not None
            stored = liquidity_score if derived else p.answers.get(q["id"])
            choice = col.radio(
                f"{q['text']} ({q['dimension']})",
                q["options"],
                index=stored - 1 if stored else None,
                key=f"cl_q{i}_{stored}" + k if derived else f"cl_q{i}" + k,
                disabled=derived,
                help="Set from the Cash needs table: the share of the portfolio due within 24 "
                "months (none 5, under 5% 4, under 10% 3, under 25% 2, otherwise 1)."
                if q["id"] == "liquidity_need"
                else None,
            )
            if derived:
                col.caption(
                    f"Set from the Cash needs table: {liquidity_share:.2%} of the portfolio is "
                    "due within 24 months."
                )
            if choice is not None:
                answers[q["id"]] = q["options"].index(choice) + 1
        c1, c2, c3 = st.columns([2, 1, 3])
        vals["max_acceptable_drawdown_pct"] = _field(c1, SECTION_BY_KEY["risk"].fields[1], p, k)
        scores = [None, 1, 2, 3, 4, 5]
        vals["override_score"] = c2.selectbox(
            "Adviser override (optional)",
            scores,
            index=scores.index(p.override_score) if p.override_score in scores else 0,
            format_func=lambda s: "none" if s is None else f"{s} / 5",
            key="cl_f_override_score" + k,
        )
        vals["override_reason"] = c3.text_input(
            "Reason for the override (required to apply it)",
            p.override_reason,
            key="cl_f_override_reason" + k,
            disabled=vals["override_score"] is None,
        )
        with st.expander("Adviser overrides of individual limits (optional)"):
            o1, o2, o3 = st.columns(3)
            ov = {
                "max_volatility_pct": o1.number_input(
                    "Max volatility (%)",
                    1.0,
                    60.0,
                    _opt(p.overrides.get("max_volatility_pct")),
                    0.5,
                    key="cl_ov_vol" + k,
                    placeholder="from band",
                ),
                "max_drawdown_pct": o2.number_input(
                    "Max drawdown (%)",
                    1.0,
                    90.0,
                    _opt(p.overrides.get("max_drawdown_pct")),
                    1.0,
                    key="cl_ov_dd" + k,
                    placeholder="from band",
                ),
                "max_weight_pct": o3.number_input(
                    "Max in one holding (%)",
                    1.0,
                    100.0,
                    _opt(p.overrides.get("max_weight_pct")),
                    1.0,
                    key="cl_ov_w" + k,
                    placeholder="from band",
                ),
            }
        vals["overrides"] = {key: val for key, val in ov.items() if val is not None}

    # 6 Knowledge & experience -----------------------------------------------------------
    with _card("knowledge", parts, p):
        vals |= _fields(p, k, "knowledge")
        for i, q in enumerate(QUESTIONS):
            if q["dimension"] in SCORED:
                continue
            choice = st.radio(
                q["text"] + " (client's own view)",
                q["options"],
                index=p.answers[q["id"]] - 1 if q["id"] in p.answers else None,
                key=f"cl_q{i}" + k,
                horizontal=True,
            )
            if choice is not None:
                answers[q["id"]] = q["options"].index(choice) + 1
        held = sorted(set((product_types or {}).values()))
        for flag in complex_product_flags(vals["products_used"], vals["years_investing"], held):
            st.warning(flag)
    vals["answers"] = answers

    # 7 Constraints & preferences --------------------------------------------------------
    with _card("constraints", parts, p):
        c1, c2, c3 = st.columns(3)
        vals["sharia_only"] = c1.checkbox(
            "Sharia-compliant holdings only", p.sharia_only, key="cl_sharia" + k
        )
        vals["home_bias_gcc_pct"] = c2.number_input(
            "Minimum in GCC holdings (%)",
            0.0,
            100.0,
            _opt(p.home_bias_gcc_pct),
            5.0,
            key="cl_home" + k,
            placeholder="none",
        )
        vals["max_non_pegged_currency_pct"] = c3.number_input(
            "Max in non-USD-pegged currencies (%)",
            0.0,
            100.0,
            float(p.max_non_pegged_currency_pct),
            5.0,
            key="cl_fx" + k,
        )
        st.caption("Minimum allocations to a country or region (e.g. Saudi Arabia at least 10%).")
        regions_table = st.data_editor(
            pd.DataFrame(
                [{"region": r, "min_pct": v} for r, v in p.region_minimums.items()],
                columns=["region", "min_pct"],
            ),
            num_rows="dynamic",
            key="cl_regions" + k,
            width="stretch",
            column_config={
                "region": st.column_config.SelectboxColumn("Region", options=REGIONS),
                "min_pct": st.column_config.NumberColumn(
                    "At least (%)", min_value=0.0, max_value=100.0, format="%.2f"
                ),
            },
        )
        vals["region_minimums"] = {
            str(row["region"]): float(row["min_pct"])
            for _, row in regions_table.iterrows()
            if pd.notna(row["region"]) and pd.notna(row["min_pct"]) and row["min_pct"] > 0
        }
        vals["excluded_sectors"] = list(
            st.multiselect(
                "Excluded sectors (ESG or values)",
                sorted(set(SECTOR_CHOICES) | set(p.excluded_sectors)),
                default=p.excluded_sectors,
                key="cl_sectors" + k,
                accept_new_options=True,
            )
        )
        vals |= _fields(p, k, "constraints")
        vals["unique_notes"] = st.text_area(
            "Other unique circumstances (optional)", p.unique_notes, key="cl_unique" + k
        )

    # 8 Estate & succession --------------------------------------------------------------
    with _card("estate", parts, p):
        vals |= _fields(p, k, "estate")
        st.info(SPECIALIST_REMINDER)

    # 9 Service & review -----------------------------------------------------------------
    with _card("service", parts, p):
        if not p.next_review and p.review_frequency:
            p = replace(p, next_review=next_review(p.review_frequency, date.today()))
        vals |= _fields(p, k, "service")
        if vals["review_frequency"] and not vals["next_review"]:
            vals["next_review"] = next_review(vals["review_frequency"], date.today())
            st.caption(
                f"Next review defaults to {vals['next_review']} "
                f"({vals['review_frequency'].lower()})."
            )
        vals["adviser_notes"] = st.text_area(
            "Adviser notes (optional)", p.adviser_notes, key="cl_adv" + k
        )

    out = replace(p, **vals, portfolio_value_usd=ws.portfolio_value())
    for warning in consistency_warnings(out):
        st.warning(warning)
    return out


def _need_rows(needs: pd.DataFrame) -> list[dict]:
    """Valid dated cash needs from the editor; incomplete or badly dated rows are warned about."""
    rows = []
    for _, row in needs.iterrows():
        if pd.isna(row["amount_usd"]) or not str(row["date"] or "").strip():
            if pd.notna(row["amount_usd"]) or str(row["description"] or "").strip():
                st.warning(f"Cash need '{row['description']}' needs an amount and a date.")
            continue
        try:
            parse_month(str(row["date"]))
        except ValueError:
            st.warning(f"Cash need date '{row['date']}' isn't a month like 2028-10; ignored.")
            continue
        rows.append(
            {
                "description": str(row["description"] or "Cash need"),
                "amount_usd": float(row["amount_usd"]),
                "date": str(row["date"]).strip()[:7],
            }
        )
    return rows


def _client_pickers(where: str) -> None:
    """Example client, saved profile (JSON) and a new blank client (welcome or Switch client)."""
    st.selectbox(
        "Load a worked example",
        list(EXAMPLE_CLIENTS),
        key="client_example",
        format_func=_example_label,
    )
    st.button("Load", key="client_load_example", on_click=_load_example)
    st.file_uploader(
        "Reopen a saved client (JSON)", type=["json"], key="client_upload", on_change=_load_json
    )
    if where == "switch":
        st.button("New blank client", key="client_new", on_click=_new_client)


def render_welcome() -> None:
    """Step 1 with no client: what the tool does, the demo, an example or a saved client."""
    _, mid, _ = st.columns([1, 3.2, 1])
    with mid:
        with st.container(border=True, key="welcome"):
            st.markdown(
                f'<div class="gpl-eyebrow">For GCC wealth managers</div><h1>{escape(WELCOME_TITLE)}'
                f'</h1><p class="gpl-lead">{escape(WELCOME)}</p>',
                unsafe_allow_html=True,
            )
            st.button(DEMO_BUTTON, key="cta_welcome_demo", on_click=fill_demo, type="primary")
            st.button(
                "or start a blank client",
                key="welcome_blank",
                type="tertiary",
                on_click=start_blank,
            )
        c1, c2 = st.columns(2)
        with c1.container(border=True, key="welcome_example"):
            ex, go = st.columns([3, 1], vertical_alignment="bottom")
            ex.selectbox(
                "Load a worked example",
                list(EXAMPLE_CLIENTS),
                key="client_example",
                format_func=_example_label,
            )
            go.button("Load", key="client_load_example", on_click=_load_example, width="stretch")
        with c2.container(border=True, key="welcome_upload"):
            st.file_uploader(
                "Reopen a saved client (JSON)",
                type=["json"],
                key="client_upload",
                on_change=_load_json,
            )
        st.markdown(f'<p class="gpl-meta" style="text-align:center">{escape(PUBLIC_NOTICE)}</p>',
                    unsafe_allow_html=True)  # fmt: skip
        if st.session_state.get("client_error"):
            st.error(st.session_state.client_error)


def _risk_card(ips) -> None:
    r = ips.risk
    calculated = r.override in ("", "ignored")

    def bar(n):
        return (
            '<span class="gpl-bar">'
            + "".join(f'<i class="{"on" if n and i < n else ""}"></i>' for i in range(5))
            + "</span>"
        )

    word = RISK_WORDS.get(r.score, "") if r.score else "Questionnaire incomplete"
    over = "" if calculated else f" · adviser override (calculated {r.calculated})"
    with st.container(border=True, key="risk_card"):
        st.markdown('<div class="gpl-section">Risk score</div>', unsafe_allow_html=True,
                    help=r.explanation, width="content")  # fmt: skip
        st.markdown(
            f'<div style="display:flex;align-items:baseline;gap:10px;margin:4px 0 10px">'
            f'<span style="font-size:2.4rem;font-weight:700">{r.score or "–"}</span>'
            f'<span style="color:var(--gpl-muted)">/ 5</span>'
            f'<span style="margin-left:6px">{escape(word + over)}</span></div>'
            f'<div class="gpl-kv" style="grid-template-columns:40% 45% 15%">'
            f"<div>Capacity</div><div>{bar(r.capacity)}</div><div>{r.capacity or '–'}</div>"
            f"<div>Willingness</div><div>{bar(r.willingness)}</div>"
            f"<div>{r.willingness or '–'}</div></div>",
            unsafe_allow_html=True,
        )
        if r.mismatch or r.score is None or r.override in ("higher", "ignored"):
            st.warning(r.explanation)


def _ips_card(profile: ClientProfile, ips, notes: list[str]) -> None:
    missing = ips_missing(profile)
    badge = layout.pill("Draft", "warn") if missing else layout.pill("Ready", "pass")
    with st.container(border=True, key="ips_card"):
        st.markdown(
            f'<div class="gpl-section">Investment Policy Statement &nbsp;{badge}</div>'
            '<div class="gpl-kv">'
            + "".join(f"<div>{escape(k)}</div><div>{escape(v)}</div>" for k, v in ips_rows(ips))
            + "</div>",
            unsafe_allow_html=True,
        )
        render_ips_downloads(profile, ips, "client")
        with st.expander("Full IPS text (RRTTLLU)"):
            st.markdown("\n\n".join(f"**{name}.** {text}" for name, text in ips.sections.items()))
            for note in notes:
                st.caption("• " + note)


def render_client_tab(
    cfg: Mapping,
    holdings: list[str],
    groups: Mapping[str, str],
    attributes: Mapping,
    cash_like: list[str] | None = None,
    regions: Mapping[str, str] | None = None,
    product_types: Mapping[str, str] | None = None,
    data_as_of: str = "",
    data_help: str | None = None,
) -> None:
    profile = current_profile()
    if profile is None:
        render_welcome()
        return
    first = (profile.name or "the client").split(" ")[0]
    right = layout.page_header(
        "Client fact-find & IPS",
        f"Check {first}'s facts; the risk score and IPS update as you go.",
        help="Decision support for a licensed adviser: record the client's facts and their own "
        "questionnaire answers; the risk score, IPS limits and checks come from tested rules. "
        + PUBLIC_NOTICE,
    )
    with right:
        m, sw = st.columns([1.6, 1], vertical_alignment="center")
        if data_as_of:
            layout.meta(data_as_of, help=data_help, container=m)
        with sw.popover("Switch client", width="stretch"):
            _client_pickers("switch")
    if st.session_state.get("client_error"):
        st.error(st.session_state.client_error)
    note = closest_mix_note(build_ips(profile, cfg))
    if note:
        st.warning(
            f"**{CLOSEST_MIX_LABEL}** {note} The measured figures are on step 5 (Suitability)."
        )
    left, side = st.columns([1.75, 1], gap="medium")
    with left.container(border=True, key="factfind"):
        bar = st.container()
        profile = _form(profile, st.session_state.get(VERSION, 0), product_types)
    st.session_state[CLIENT] = profile.to_dict()  # the form is the source of truth
    parts = completeness(profile)
    share = overall_completeness(parts)
    with bar:
        h1, h2 = st.columns([2, 1])
        h1.markdown('<div class="gpl-section">Fact-find</div>', unsafe_allow_html=True)
        h2.markdown(f'<p class="gpl-meta" style="color:var(--gpl-brand-btn);font-weight:600">'
                    f"{share:.0%} complete</p>", unsafe_allow_html=True)  # fmt: skip
        st.progress(
            share,
            text=None
            if share == 1
            else "To do: "
            + ", ".join(SECTION_BY_KEY[key].title for key, c in parts.items() if c.missing),
        )
    ips = build_ips(profile, cfg)
    notes = []
    if holdings:
        _, notes = ips_constraints(
            ips, holdings, dict(groups), dict(attributes), cash_like, dict(regions or {})
        )
    with side:
        _risk_card(ips)
        _ips_card(profile, ips, notes)


def render_ips_downloads(profile: ClientProfile, ips, where: str) -> None:
    """Final IPS (only when complete), a watermarked DRAFT, and the profile as JSON."""
    missing = ips_missing(profile)
    if missing:
        sections = list(dict.fromkeys(m.split(":", 1)[0] for m in missing))
        st.warning(
            f"The final IPS can't be issued yet: {len(missing)} required "
            f"item{'s are' if len(missing) != 1 else ' is'} missing in "
            + ", ".join(sections)
            + ". A watermarked DRAFT is available for discussion."
        )
        with st.expander("What's missing"):
            st.markdown("\n".join(f"- {m}" for m in missing))
    ref = (profile.reference or "client").replace(" ", "_")
    with st.container(key=f"cta_{where}_ips"):
        st.download_button(
            "Download IPS (PDF)",
            b"" if missing else ips_pdf(ips),
            file_name=f"IPS_{ref}_{ips.created}.pdf",
            mime="application/pdf",
            key=f"{where}_ips_pdf",
            type="primary",
            disabled=bool(missing),
            icon=":material/download:",
            width="stretch",
        )
    d2, d3 = st.columns(2)
    d2.download_button(
        "Draft IPS (PDF)",
        ips_pdf(ips, draft=True),
        file_name=f"IPS_DRAFT_{ref}_{ips.created}.pdf",
        mime="application/pdf",
        key=f"{where}_ips_draft",
        type="tertiary",
    )
    d3.download_button(
        "Save profile (JSON)",
        json.dumps(profile.to_dict(), indent=2),
        file_name=f"client_{ref}.json",
        mime="application/json",
        key=f"{where}_save_json",
        type="tertiary",
    )
