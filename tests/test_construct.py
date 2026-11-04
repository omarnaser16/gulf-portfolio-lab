"""Tests for portfolio construction and improvement (gpl.construct)."""

import pandas as pd
import pytest

from gpl.construct import limit_turnover, turnover


def test_turnover_counts_buys_and_sells() -> None:
    before = pd.Series({"A": 0.6, "B": 0.4})
    after = pd.Series({"A": 0.45, "C": 0.55})
    assert turnover(before, after) == pytest.approx(0.15 + 0.40 + 0.55)  # A -15, B -40, C +55


def test_limit_turnover_moves_part_way_by_hand() -> None:
    w, share = limit_turnover(pd.Series({"A": 1.0}), pd.Series({"B": 1.0}), 0.50)
    assert share == pytest.approx(0.25)  # full move trades 2.00; 0.50 / 2.00
    assert w["A"] == pytest.approx(0.75) and w["B"] == pytest.approx(0.25)
    assert turnover(pd.Series({"A": 1.0}), w) == pytest.approx(0.50)
    small, share = limit_turnover(
        pd.Series({"A": 0.5, "B": 0.5}), pd.Series({"A": 0.6, "B": 0.4}), 0.3
    )
    assert share == 1.0 and small["A"] == pytest.approx(0.6)  # within the limit: all the way
    same, share = limit_turnover(pd.Series({"A": 1.0}), pd.Series({"A": 1.0}), 0.1)
    assert share == 1.0 and same.to_dict() == {"A": 1.0}


# ------------------------------------------------------------------ universe, scores, build
from datetime import date  # noqa: E402

import numpy as np  # noqa: E402

from gpl.backtest import run_backtest  # noqa: E402
from gpl.client import ClientProfile, build_ips  # noqa: E402
from gpl.construct import (  # noqa: E402
    BuildSettings,
    benchmark_core,
    build_once,
    build_strategy,
    momentum_score,
    screen_universe,
    shortlist,
    turnover_limited,
)
from gpl.data import load_universe  # noqa: E402
from gpl.optimise import OptimisationSpec, active_moments  # noqa: E402

CFG = load_universe()
TODAY = date(2026, 11, 4)
ANSWERS = {
    "income_stability": 4,
    "wealth_share": 3,
    "dependence": 4,
    "liquidity_need": 4,
    "loss_reaction": 3,
    "tradeoff": 3,
}


def _tag(asset_class, region="United States", sharia="not_compliant", **kw):
    return {
        "name": kw.pop("name", "Fund"),
        "asset_class": asset_class,
        "region": region,
        "currency": "USD",
        "sharia": sharia,
        "domicile": kw.pop("domicile", "United States"),
        "history_from": kw.pop("history_from", "2012-01-03"),
        "verified": "2026-11-04",
        **kw,
    }


UNI = {
    "EQ1": _tag("Equity"),
    "EQ2": _tag("Equity", "Developed ex-US"),
    "EQ3": _tag("Equity", "Emerging markets"),
    "EQ4": _tag("Equity", factor="quality"),
    "EQ5": _tag("Equity", factor="low_vol"),
    "EQ6": _tag("Equity", "Japan"),
    "BD1": _tag("Bonds"),
    "BD2": _tag("Bonds", "Developed ex-US"),
    "CSH": _tag("Cash"),
    "GLD": _tag("Gold", "Global", sharia="unknown"),
    "SUK": _tag("Sukuk", "Global", sharia="compliant"),
    "SH1": _tag("Equity", sharia="compliant"),
    "IRL": _tag("Equity", domicile="Ireland", ucits_of="EQ1"),
    "NEW": _tag("Equity", history_from="2024-01-02"),
    "GCC": _tag(
        "Equity", "Saudi Arabia", sharia="compliant", gcc_stock=True, domicile="Saudi Arabia"
    ),
}


def _returns(n=1300, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    mkt = rng.normal(0.0004, 0.01, n)
    vol = {"Equity": 0.004, "Bonds": 0.003, "Cash": 0.0002, "Gold": 0.009, "Sukuk": 0.002}
    beta = {"Equity": 1.0, "Bonds": 0.1, "Cash": 0.0, "Gold": 0.1, "Sukuk": 0.05}
    return pd.DataFrame(
        {
            t: beta[r["asset_class"]] * mkt + rng.normal(0.0001, vol[r["asset_class"]], n)
            for t, r in UNI.items()
        },
        index=idx,
    )


def _ips(**kw):
    return build_ips(ClientProfile(horizon_years=10, answers=ANSWERS, **kw), CFG, TODAY)


def test_screen_universe_reasons() -> None:
    s = screen_universe(UNI, None, TODAY, min_history_years=7)
    assert "GCC" in s.excluded and s.excluded["GCC"].startswith("GCC single stock")
    assert s.excluded["NEW"] == "history only from 2024-01-02: under 7 years"
    assert "EQ1" in s.eligible and "IRL" in s.eligible
    us = screen_universe(UNI, _ips(us_person=True), TODAY)
    assert us.excluded["IRL"].startswith("non-US fund")
    sharia = screen_universe(UNI, _ips(sharia_only=True), TODAY)
    assert set(sharia.eligible) == {"SUK", "SH1", "GCC"}  # GCC joins: Sharia asks for it
    assert sharia.excluded["GLD"] == "Sharia status unknown: not picked automatically"
    assert sharia.excluded["EQ1"].startswith("excluded by the IPS")
    restricted = screen_universe(UNI, _ips(restricted_securities=["eq2"]), TODAY)
    assert restricted.excluded["EQ2"] == "restricted by the client"


def test_momentum_score_skips_the_last_month_by_hand() -> None:
    idx = pd.bdate_range("2024-01-01", periods=30)
    window = pd.DataFrame({"A": [0.01] * 10 + [0.0] * 20}, index=idx)
    # length 30, skip 21: only the first 9 days count -> 1.01^9 - 1
    assert momentum_score(window, length=30, skip=21)["A"] == pytest.approx(1.01**9 - 1)


def test_shortlist_keeps_core_and_tilt_names() -> None:
    r = _returns()
    eligible = screen_universe(UNI, None, TODAY, 3).eligible
    names, reasons, tilt = shortlist(r.iloc[-252:], eligible, UNI, "quality", core=("BD1",))
    assert "BD1" in names  # the benchmark's own fund (here also top of its sleeve)
    assert tilt == ["EQ4"]  # quality: the quality-factor ETF
    assert not ("EQ1" in names and "IRL" in names)  # a UCITS twin is a duplicate
    assert sum(1 for t in names if UNI[t]["asset_class"] == "Cash") == 1


def test_build_once_meets_every_limit() -> None:
    r = _returns()
    bench = 0.6 * r["EQ1"] + 0.4 * r["BD1"]
    eligible = screen_universe(UNI, None, TODAY, 3).eligible
    settings = BuildSettings(tracking_error_max=0.05, n_min=5, n_max=8)
    built = build_once(r.iloc[-252:], bench, eligible, UNI, None,
                       OptimisationSpec(max_weight=0.5), settings, core=("EQ1", "BD1"))  # fmt: skip
    assert built.feasible, built.notes
    w = built.weights
    assert 5 <= len(w) <= 8 and w.sum() == pytest.approx(1.0)
    assert (w >= 0.02 - 1e-6).all() and (w <= 0.5 + 1e-6).all()
    act = active_moments(r.iloc[-252:][list(w.index)], bench.iloc[-252:])
    te = float(np.sqrt(w @ act.cov.loc[w.index, w.index] @ w))
    assert te <= 0.05 + 1e-4
    assert set(built.reasons) == set(w.index)


def test_build_walk_forward_uses_no_future_data() -> None:
    r = _returns()
    bench = 0.6 * r["EQ1"] + 0.4 * r["BD1"]
    eligible = screen_universe(UNI, None, TODAY, 3).eligible
    settings = BuildSettings(tracking_error_max=0.05, n_min=5, n_max=8)
    prices = (1 + r).cumprod()

    def run(p, b):
        st = build_strategy(eligible, UNI, b, None, OptimisationSpec(max_weight=0.5),
                            settings, core=("EQ1", "BD1"))  # fmt: skip
        return run_backtest(p, st, lookback=252, cost_bps=10).weights

    before = run(prices, bench)
    cut = r.index[900]
    shocked = r.copy()
    shocked.loc[shocked.index > cut, "EQ3"] += 0.02  # EQ3 soars after the cut
    after = run((1 + shocked).cumprod(), 0.6 * shocked["EQ1"] + 0.4 * shocked["BD1"])
    pd.testing.assert_frame_equal(before.loc[:cut], after.loc[:cut])  # nothing before changes
    assert not before.loc[cut:].equals(after.loc[cut:])  # later choices react


def test_turnover_limited_strategy_caps_each_move() -> None:
    r = _returns()
    target = pd.Series({"EQ1": 0.5, "BD1": 0.5})
    st = turnover_limited(lambda window: target, {"CSH": 1.0}, 0.3)
    out = [st(r.iloc[:10]) for _ in range(5)]
    assert turnover(pd.Series({"CSH": 1.0}), out[0]) == pytest.approx(0.3)
    for a, b in zip(out, out[1:], strict=False):
        assert turnover(a, b) <= 0.3 + 1e-9
    assert out[-1].get("CSH", 0) < out[0]["CSH"]


def test_benchmark_core_maps_to_ucits_when_needed() -> None:
    assert benchmark_core({"EQ1": 0.6, "BD1": 0.4}, UNI, ["EQ1", "BD1"]) == ("EQ1", "BD1")
    assert benchmark_core({"EQ1": 0.6, "BD1": 0.4}, UNI, ["IRL", "BD1"]) == ("IRL", "BD1")


def test_swap_ideas_duplicates_and_missing_diversifier() -> None:
    from gpl.construct import apply_swaps, domicile_suggestions, swap_ideas

    r = _returns()
    r["DUP"] = r["EQ1"] * 1.001  # a near-copy of EQ1
    uni = {**UNI, "DUP": _tag("Equity")}
    current = pd.Series({"EQ1": 0.5, "DUP": 0.3, "EQ2": 0.2})
    classes = {"EQ1": "Equity", "DUP": "Equity", "EQ2": "Equity"}
    eligible = screen_universe(uni, None, TODAY, 3).eligible
    ideas = swap_ideas(r.iloc[-252:], current, eligible, uni, classes)
    kinds = [s.kind for s in ideas]
    assert kinds == ["duplicate", "diversifier"]
    assert {ideas[0].remove} <= {"EQ1", "DUP"} and "move almost together" in ideas[0].reason
    assert uni[ideas[1].add]["asset_class"] in {"Bonds", "Sukuk"}
    after = apply_swaps(current, ideas)
    assert ideas[0].remove not in after and ideas[1].add in after
    with_bonds = pd.Series({"EQ1": 0.6, "BD1": 0.4})
    assert swap_ideas(r.iloc[-252:], with_bonds, eligible, uni, {"EQ1": "Equity",
                      "BD1": "Bonds"}) == []  # fmt: skip
    dom = domicile_suggestions(pd.Series({"EQ1": 1.0}), UNI, us_person=False)
    assert dom[0].add == "IRL" and "not tax advice" in dom[0].reason
    assert domicile_suggestions(pd.Series({"EQ1": 1.0}), UNI, us_person=True) == []


def test_improve_strategy_swaps_and_caps_turnover() -> None:
    from gpl.construct import improve_strategy

    r = _returns()
    current = pd.Series({"EQ1": 0.7, "EQ2": 0.3})
    eligible = screen_universe(UNI, None, TODAY, 3).eligible
    st = improve_strategy(
        current, eligible, UNI, {"EQ1": "Equity", "EQ2": "Equity"},
        lambda names: OptimisationSpec(objective="max_sharpe", max_weight=0.6),
        max_turnover=0.3,
    )  # fmt: skip
    res = run_backtest((1 + r).cumprod(), st, lookback=252, cost_bps=10)
    assert [s.kind for s in st.latest_swaps] == ["diversifier"]
    w = res.weights
    added = st.latest_swaps[0].add
    assert w[added].iloc[-1] > 0  # the diversifier is held
    steps = w.diff().abs().sum(axis=1).iloc[1:]
    assert (steps <= 0.3 + 1e-6).all()  # every rebalance within the 30% limit
