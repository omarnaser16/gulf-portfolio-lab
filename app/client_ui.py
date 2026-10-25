"""Client tab: profile form, risk questionnaire, IPS preview and downloads.

The profile lives in ``st.session_state["client"]`` (a dict, ``ClientProfile.to_dict()``).
Widgets are keyed with a version number so loading a profile (example, JSON or AI notes)
redraws the form. Every number shown comes from ``gpl.client`` (tested rules).
"""

import json
from collections.abc import Callable, Mapping

import pandas as pd
import streamlit as st

from gpl.agent.profile import apply_proposal, propose_profile
from gpl.client import (
    CLASSIFICATIONS,
    EXAMPLE_CLIENTS,
    QUESTIONS,
    REGULATORS,
    ClientProfile,
    build_ips,
    ips_constraints,
)
from gpl.report import ips_pdf

CLIENT, VERSION = "client", "client_v"
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
    data = st.session_state.get(CLIENT)
    return None if data is None else ClientProfile.from_dict(data)


def _set_profile(profile: ClientProfile) -> None:
    st.session_state[CLIENT] = profile.to_dict()
    st.session_state[VERSION] = st.session_state.get(VERSION, 0) + 1
    st.session_state.pop("ws_draft", None)  # the Optimise form re-reads the IPS
    st.session_state.pop("client_proposal", None)


def _load_example() -> None:
    name = st.session_state.get("client_example")
    if name in EXAMPLE_CLIENTS:
        _set_profile(ClientProfile.from_dict(EXAMPLE_CLIENTS[name]))


def _load_json() -> None:
    upload = st.session_state.get("client_upload")
    if upload is None:
        return
    try:
        _set_profile(ClientProfile.from_dict(json.loads(upload.getvalue())))
        st.session_state.pop("client_error", None)
    except (ValueError, TypeError) as e:  # bad JSON or bad values
        st.session_state.client_error = f"Couldn't read that profile: {e}"


def _new_client() -> None:
    _set_profile(ClientProfile())


def _interpret(make_client: Callable, model: str) -> None:
    notes = st.session_state.get("client_notes", "")
    st.session_state.pop("client_error", None)
    if not notes.strip():
        st.session_state.client_error = "Write some notes about the client first."
        return
    try:
        proposal = propose_profile(make_client(), model, notes)
    except Exception as e:  # API or model errors are shown, not raised
        st.session_state.client_error = str(e)
        return
    base = current_profile() or ClientProfile()
    _set_profile(apply_proposal(base, proposal.draft))
    st.session_state.client_proposal = proposal


def _opt(value):
    return None if value is None else float(value)


def _form(p: ClientProfile, v: int) -> ClientProfile:
    """Draw the form from ``p`` and return the profile the widgets now describe."""
    k = f"_{v}"
    c1, c2, c3 = st.columns(3)
    reference = c1.text_input("Client reference", p.reference, key="cl_ref" + k)
    name = c2.text_input("Name (optional)", p.name, key="cl_name" + k)
    classification = c3.selectbox(
        "Classification",
        CLASSIFICATIONS,
        index=CLASSIFICATIONS.index(p.classification) if p.classification in CLASSIFICATIONS else 0,
        key="cl_class" + k,
    )
    c1, c2, c3 = st.columns(3)
    regulator = c1.selectbox(
        "Regulator",
        REGULATORS,
        index=REGULATORS.index(p.regulator) if p.regulator in REGULATORS else len(REGULATORS) - 1,
        key="cl_reg" + k,
    )
    residency = c2.text_input("Residency (country)", p.residency, key="cl_res" + k)
    us_person = c3.checkbox(
        "US person (citizen, green card or US tax resident)", p.us_person, key="cl_us" + k
    )

    st.markdown("**Return, horizon and liquidity**")
    c1, c2, c3 = st.columns(3)
    horizon = c1.number_input(
        "Time horizon (years)", 0.5, 60.0, float(p.horizon_years), 0.5, key="cl_hor" + k
    )
    value = c2.number_input(
        "Portfolio value (USD)",
        0.0,
        1e10,
        _opt(p.portfolio_value_usd),
        1000.0,
        key="cl_val" + k,
        placeholder="not set",
    )
    target = c3.number_input(
        "Return target (% a year, optional)",
        -20.0,
        50.0,
        _opt(p.return_target_pct),
        0.5,
        key="cl_ret" + k,
        placeholder="none",
    )
    c1, c2, c3 = st.columns(3)
    need = c1.number_input(
        "Near-term cash need (USD)",
        0.0,
        1e10,
        float(p.liquidity_need_usd),
        1000.0,
        key="cl_need" + k,
    )
    months = c2.number_input(
        "Needed within (months)", 0, 120, int(p.liquidity_months), 1, key="cl_months" + k
    )
    reserve = c3.number_input(
        "Extra cash reserve (% of portfolio)",
        0.0,
        100.0,
        float(p.reserve_pct),
        1.0,
        key="cl_reserve" + k,
    )
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

    st.markdown("**Risk questionnaire** (the client's own answers)")
    answers = {}
    q1, q2 = st.columns(2)
    for i, q in enumerate(QUESTIONS):
        col = q1 if q["dimension"] == "capacity" else q2
        choice = col.radio(
            f"{q['text']} ({q['dimension']})",
            q["options"],
            index=p.answers[q["id"]] - 1 if q["id"] in p.answers else None,
            key=f"cl_q{i}" + k,
            horizontal=False,
        )
        if choice is not None:
            answers[q["id"]] = q["options"].index(choice) + 1

    st.markdown("**Constraints and preferences**")
    c1, c2, c3 = st.columns(3)
    sharia = c1.checkbox("Sharia-compliant holdings only", p.sharia_only, key="cl_sharia" + k)
    home_bias = c2.number_input(
        "Minimum in GCC holdings (%)",
        0.0,
        100.0,
        _opt(p.home_bias_gcc_pct),
        5.0,
        key="cl_home" + k,
        placeholder="none",
    )
    non_pegged = c3.number_input(
        "Max in non-USD-pegged currencies (%)",
        0.0,
        100.0,
        float(p.max_non_pegged_currency_pct),
        5.0,
        key="cl_fx" + k,
    )
    sectors = st.multiselect(
        "Excluded sectors",
        sorted(set(SECTOR_CHOICES) | set(p.excluded_sectors)),
        default=p.excluded_sectors,
        key="cl_sectors" + k,
        accept_new_options=True,
    )
    with st.expander("Adviser overrides of the risk band (optional)"):
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
    with st.expander("Notes (taxes, legal, unique circumstances, adviser)"):
        tax_notes = st.text_area("Tax notes", p.tax_notes, key="cl_tax" + k)
        legal_notes = st.text_area("Legal / regulatory notes", p.legal_notes, key="cl_legal" + k)
        unique_notes = st.text_area(
            "Other unique circumstances", p.unique_notes, key="cl_unique" + k
        )
        adviser_notes = st.text_area("Adviser notes", p.adviser_notes, key="cl_adv" + k)

    goal_rows = []
    for _, row in goals.iterrows():
        g = {c: row[c] for c in goals.columns if pd.notna(row[c]) and row[c] != ""}
        if g:
            goal_rows.append(g)
    return ClientProfile(
        reference=reference,
        name=name,
        classification=classification,
        regulator=regulator,
        residency=residency,
        us_person=us_person,
        horizon_years=horizon,
        goals=goal_rows,
        return_target_pct=target,
        portfolio_value_usd=value,
        liquidity_need_usd=need,
        liquidity_months=int(months),
        reserve_pct=reserve,
        tax_notes=tax_notes,
        legal_notes=legal_notes,
        sharia_only=sharia,
        excluded_sectors=list(sectors),
        home_bias_gcc_pct=home_bias,
        max_non_pegged_currency_pct=non_pegged,
        unique_notes=unique_notes,
        answers=answers,
        overrides={key: val for key, val in ov.items() if val is not None},
        adviser_notes=adviser_notes,
    )


def render_client_tab(
    cfg: Mapping,
    holdings: list[str],
    groups: Mapping[str, str],
    attributes: Mapping,
    ai_ok: bool,
    make_client: Callable,
    model: str,
) -> None:
    st.subheader("Client profile and Investment Policy Statement")
    st.caption(
        "Decision support for a licensed adviser: record the client's facts and their own "
        "questionnaire answers; the risk score, IPS limits and checks come from tested rules."
    )
    c1, c2, c3 = st.columns([3, 3, 2])
    with c1:
        st.selectbox("Example client", list(EXAMPLE_CLIENTS), key="client_example")
        st.button("Load example", key="client_load_example", on_click=_load_example)
    c2.file_uploader(
        "Load a saved profile (JSON)", type=["json"], key="client_upload", on_change=_load_json
    )
    c3.button("New blank client", key="client_new", on_click=_new_client)
    with st.expander("Describe the client in plain English (AI fills the form)"):
        st.text_area(
            "Adviser's notes",
            key="client_notes",
            placeholder="e.g. Dubai resident, $500k, needs 50k next year for a "
            "property deposit, Sharia-compliant only, at least 20% in the Gulf.",
        )
        st.button(
            "Fill the form with AI",
            key="client_interpret",
            on_click=_interpret,
            args=(make_client, model),
            disabled=not ai_ok,
            help=None if ai_ok else "Needs an API key (see the Analyst tab).",
        )
        proposal = st.session_state.get("client_proposal")
        if proposal:
            if proposal.summary:
                st.info(f"**The AI recorded:** {proposal.summary}")
            for item in proposal.unmapped:
                st.warning(f"Not recorded (no field for it): “{item}”")
            if proposal.numbers_not_in_notes:
                st.warning(
                    "These numbers are not in your notes, please check them: "
                    + ", ".join(proposal.numbers_not_in_notes)
                )
            st.caption(
                "The AI never answers the risk questionnaire: those are the client's own answers."
            )
    if st.session_state.get("client_error"):
        st.error(st.session_state.client_error)

    profile = current_profile()
    if profile is None:
        st.info("Load an example, a saved profile, or start a new blank client.")
        return
    profile = _form(profile, st.session_state.get(VERSION, 0))
    st.session_state[CLIENT] = profile.to_dict()  # the form is the source of truth

    ips = build_ips(profile, cfg)
    st.divider()
    r = ips.risk
    m1, m2, m3 = st.columns(3)
    m1.metric("Risk capacity", "–" if r.capacity is None else f"{r.capacity} / 5")
    m2.metric("Risk willingness", "–" if r.willingness is None else f"{r.willingness} / 5")
    m3.metric("Risk score (lower of the two)", "–" if r.score is None else f"{r.score} / 5")
    (st.warning if r.mismatch or r.score is None else st.caption)(r.explanation)
    st.markdown("**Investment Policy Statement (RRTTLLU)**")
    st.markdown("\n\n".join(f"**{name}.** {text}" for name, text in ips.sections.items()))
    if holdings:
        _, notes = ips_constraints(ips, holdings, dict(groups), dict(attributes))
        for note in notes:
            st.caption("• " + note)
    d1, d2 = st.columns(2)
    ref = (profile.reference or "client").replace(" ", "_")
    d1.download_button(
        "Download IPS (PDF)",
        ips_pdf(ips),
        file_name=f"IPS_{ref}_{ips.created}.pdf",
        mime="application/pdf",
        key="client_ips_pdf",
        type="primary",
    )
    d2.download_button(
        "Save profile (JSON)",
        json.dumps(profile.to_dict(), indent=2),
        file_name=f"client_{ref}.json",
        mime="application/json",
        key="client_save_json",
    )
