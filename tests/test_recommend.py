"""Tests for goals, candidate specs and the recommendation rule (synthetic results tables)."""

import pandas as pd
import pytest

from gpl.optimise import OptimisationSpec
from gpl.recommend import (
    CANDIDATES,
    KEEP,
    candidate_specs,
    held_up,
    recommend,
    volatility_target,
)


def table(rows: dict[str, tuple]) -> pd.DataFrame:
    """rows: name -> (cagr, volatility, sharpe, in_sample_sharpe)."""
    frame = pd.DataFrame(rows, index=["cagr", "volatility", "sharpe", "in_sample_sharpe"]).T.astype(
        float
    )
    frame["feasible"] = [n in CANDIDATES for n in frame.index]
    return frame


BASE = {
    "Current": (0.08, 0.12, 0.50, 0.55),
    "Benchmark": (0.09, 0.11, 0.60, 0.60),
    "Max Sharpe": (0.10, 0.11, 0.70, 0.90),
    "Min variance": (0.05, 0.06, 0.45, 0.50),
    "Risk parity": (0.07, 0.08, 0.62, 0.70),
    "Max diversification": (0.08, 0.09, 0.64, 0.70),
    "Target volatility": (0.09, 0.12, 0.55, 0.80),
}


def test_held_up() -> None:
    assert held_up(1.0, 0.5) and not held_up(1.0, 0.49)
    assert held_up(-0.2, -0.1) and not held_up(-0.2, -0.3)  # nothing to lose if <= 0
    assert not held_up(float("nan"), 0.5)


def test_max_risk_adjusted_picks_highest_walk_forward_sharpe() -> None:
    rec = recommend("max_risk_adjusted", table(BASE))
    assert rec.choice == "Max Sharpe" and rec.ranking[0] == "Max Sharpe"
    assert "Highest walk-forward Sharpe" in rec.rule
    assert (
        rec.reasons[0]
        == "Max Sharpe ranks first under the rule: highest walk-forward Sharpe ratio after costs, "
        "and at least 0.05 above your current portfolio's."
    )


def test_a_method_that_did_not_hold_up_is_skipped() -> None:
    rows = dict(BASE)
    rows["Max Sharpe"] = (0.10, 0.11, 0.70, 1.60)  # kept only 44% of its in-sample Sharpe
    rec = recommend("max_risk_adjusted", table(rows))
    assert rec.choice == "Max diversification" and not rec.held_up["Max Sharpe"]
    assert any("did not hold up" in r for r in rec.reasons)


def test_reduce_risk_tie_break_on_sharpe() -> None:
    rows = dict(BASE)
    rows["Risk parity"] = (0.07, 0.064, 0.62, 0.70)  # within 0.50 pp of min variance's 6%
    rec = recommend("reduce_risk", table(rows))
    assert rec.choice == "Risk parity" and rec.ranking[:2] == ["Risk parity", "Min variance"]


def test_reduce_risk_keeps_current_when_nothing_is_safer() -> None:
    rows = {k: v for k, v in BASE.items() if k in ("Current", "Max Sharpe")}
    rows["Max Sharpe"] = (0.10, 0.15, 0.60, 0.70)
    rec = recommend("reduce_risk", table(rows))
    assert rec.choice == KEEP and "No method had lower volatility" in rec.reasons[0]


def test_target_return_reached_and_not_reached() -> None:
    rec = recommend("target_return", table(BASE), target=0.08)
    # reached 8%: Max Sharpe (11% vol), Max diversification (9%), Target volatility (12%)
    assert rec.choice == "Max diversification"
    rec = recommend("target_return", table(BASE), target=0.20)
    assert rec.choice == "Max Sharpe"  # highest CAGR when none reached
    assert any("No method reached the 20.00% target" in r for r in rec.reasons)


def test_target_volatility() -> None:
    rec = recommend("target_volatility", table(BASE), target=0.09)
    # within 9.5%: Min variance (5%), Risk parity (7%), Max diversification (8%) -> highest CAGR
    assert rec.choice == "Max diversification"
    rec = recommend("target_volatility", table(BASE), target=0.03)
    assert rec.choice == "Min variance" and any("stayed within" in r for r in rec.reasons)


def test_beat_benchmark_with_less_risk() -> None:
    rows = dict(BASE)
    rows["Max diversification"] = (0.095, 0.10, 0.75, 0.80)  # beats 9% with 10% < 11% vol
    rec = recommend("beat_benchmark_less_risk", table(rows))
    assert rec.choice == "Max diversification"
    rec = recommend("beat_benchmark_less_risk", table(BASE))  # nobody beats 9% with < 11% vol
    assert rec.choice == "Max diversification"  # best Sharpe among the safer ones
    assert any("also beat its return" in r for r in rec.reasons)


def test_keep_current_when_the_best_does_not_improve() -> None:
    rows = dict(BASE)
    rows["Current"] = (0.12, 0.10, 0.90, 0.95)
    rec = recommend("max_risk_adjusted", table(rows))
    assert rec.choice == KEEP and any("did not improve" in r for r in rec.reasons)


def test_infeasible_candidates_are_ignored() -> None:
    frame = table(BASE)
    frame.loc["Max Sharpe", "feasible"] = False
    assert recommend("max_risk_adjusted", frame).choice == "Max diversification"
    frame["feasible"] = False
    rec = recommend("max_risk_adjusted", frame)
    assert rec.choice == KEEP and rec.reasons == ["No method could meet your limits."]


def test_targets_are_required() -> None:
    with pytest.raises(ValueError, match="needs a target"):
        recommend("target_return", table(BASE))


def test_volatility_target_and_candidate_specs() -> None:
    assert volatility_target("target_volatility", 0.10, 0.12, 0.11) == 0.10
    assert volatility_target("beat_benchmark_less_risk", None, 0.12, 0.11) == 0.11
    assert volatility_target("max_risk_adjusted", None, 0.12, 0.11) == 0.12
    assert volatility_target("max_risk_adjusted", None, 0.12, 0.11, cap=0.10) == 0.10
    limits = OptimisationSpec(
        max_weight=0.4, groups={"A": "GCC"}, group_bounds={"GCC": (0.2, None)}
    )
    specs = candidate_specs(limits, 0.10)
    assert list(specs) == list(CANDIDATES)
    assert specs["Target volatility"].max_volatility == 0.10
    assert specs["Max Sharpe"].max_volatility is None  # only the target-vol candidate is capped
    assert all(
        s.max_weight == 0.4 and s.group_bounds == {"GCC": (0.2, None)} for s in specs.values()
    )
    assert specs["Risk parity"].objective == "risk_parity"


# ------------------------------------------------------------- minimum Sharpe improvement


def _one_candidate(current_sharpe: float, pick_sharpe: float) -> pd.DataFrame:
    return table(
        {
            "Current": (0.08, 0.10, current_sharpe, current_sharpe),
            "Benchmark": (0.07, 0.12, 0.40, 0.40),
            "Max Sharpe": (0.09, 0.10, pick_sharpe, pick_sharpe),
        }
    )


@pytest.mark.parametrize(
    ("pick", "expected"),
    [(0.80, "Max Sharpe"), (0.799, KEEP), (0.76, KEEP), (0.74, KEEP)],
)
def test_minimum_sharpe_gain_boundary(pick, expected) -> None:
    # current 0.75: a gain of exactly 0.05 counts; 0.049 and 0.01 are within noise
    assert recommend("max_risk_adjusted", _one_candidate(0.75, pick)).choice == expected


def test_small_sharpe_gain_is_explained() -> None:
    rec = recommend("max_risk_adjusted", _one_candidate(0.7512, 0.7575))  # the README case
    assert rec.choice == KEEP and rec.ranking == ["Max Sharpe"]
    assert any("by less than the 0.05 minimum" in r for r in rec.reasons)
    assert "at least 0.05 above" in rec.rule


def test_minimum_gain_in_the_benchmark_goal() -> None:
    # current already beats the benchmark (8% > 7% with 10% < 12% volatility): change needs +0.05
    assert recommend("beat_benchmark_less_risk", _one_candidate(0.75, 0.77)).choice == KEEP
    assert recommend("beat_benchmark_less_risk", _one_candidate(0.75, 0.81)).choice == "Max Sharpe"
    # a current portfolio that does NOT qualify is replaced by one that does, whatever the Sharpe
    frame = _one_candidate(0.75, 0.70)
    frame.loc["Current", "volatility"] = 0.15  # riskier than the benchmark
    assert recommend("beat_benchmark_less_risk", frame).choice == "Max Sharpe"


def test_threshold_is_configurable_and_off_at_zero() -> None:
    frame = _one_candidate(0.75, 0.76)
    assert recommend("max_risk_adjusted", frame, min_sharpe_gain=0.0).choice == "Max Sharpe"
    assert recommend("max_risk_adjusted", frame, min_sharpe_gain=0.2).choice == KEEP


def test_volatility_goals_are_not_affected() -> None:
    # reduce risk is judged on volatility: a lower Sharpe is fine if risk falls
    frame = _one_candidate(0.75, 0.60)
    frame.loc["Max Sharpe", "volatility"] = 0.07
    assert recommend("reduce_risk", frame).choice == "Max Sharpe"


def test_boundary_is_not_broken_by_rounding() -> None:
    # 0.1 + 0.2 != 0.3 in floating point; a gain of exactly 0.05 must still count
    assert recommend("max_risk_adjusted", _one_candidate(0.1 + 0.2, 0.35)).choice == "Max Sharpe"


# ------------------------------------------------------------------- suitability gate


def gated(passes: dict[str, tuple[int, float]]) -> pd.DataFrame:
    """BASE with IPS columns; passes: name -> (rules failed, summed relative breach)."""
    t = table(BASE)
    t["ips_fails"] = [passes.get(n, (0, 0.0))[0] for n in t.index]
    t["ips_breach"] = [passes.get(n, (0, 0.0))[1] for n in t.index]
    t["passes_ips"] = t["ips_fails"] == 0
    return t


def test_gate_skips_a_candidate_that_fails_the_ips() -> None:
    # Max Sharpe ranks first but fails (e.g. volatility); Max diversification is next by Sharpe
    rec = recommend("max_risk_adjusted", gated({"Max Sharpe": (1, 0.1)}))
    assert rec.choice == "Max diversification" and rec.status == "recommended"
    assert "Max Sharpe" not in rec.ranking
    assert any("passes every IPS rule" in r for r in rec.reasons)
    assert rec.current_passes_ips is True


def test_nothing_passes_shows_the_closest() -> None:
    fails = {n: (2, 0.5) for n in CANDIDATES} | {"Risk parity": (1, 0.3), "Current": (2, 0.1)}
    rec = recommend("max_risk_adjusted", gated(fails))
    assert rec.status == "closest_fails" and rec.choice == "Risk parity"
    assert rec.current_passes_ips is False
    assert any("current portfolio fails the IPS too" in r for r in rec.reasons)
    assert any("not as a suitable portfolio" in r for r in rec.reasons)
    # ties on rules failed are broken by the smaller breach; here the current portfolio
    fails["Current"] = (1, 0.05)
    rec = recommend("max_risk_adjusted", gated(fails))
    assert rec.choice == KEEP and rec.status == "closest_fails"
    assert any("current portfolio is the closest" in r for r in rec.reasons)


def test_current_fails_so_a_passing_candidate_is_recommended_without_a_gain() -> None:
    # Sharpe gains below the 0.05 minimum would normally mean "keep"; the current portfolio
    # fails the IPS, so the best passing method is recommended anyway
    t = gated({"Current": (1, 0.2), "Max Sharpe": (1, 0.1)})
    t.loc["Current", "sharpe"] = 0.62
    rec = recommend("max_risk_adjusted", t)
    assert rec.choice == "Max diversification" and rec.status == "recommended"
    assert "fails the client's IPS" in rec.reasons[0]


def test_keep_needs_a_passing_current_portfolio_and_always_has_reasons() -> None:
    t = gated({})
    t.loc["Current", "sharpe"] = 0.69  # nothing beats it by 0.05
    rec = recommend("max_risk_adjusted", t)
    assert rec.choice == KEEP and rec.status == "keep" and rec.reasons
    assert rec.reasons[-1] == "Your current portfolio passes every IPS rule."
    ungated = recommend("max_risk_adjusted", table(BASE).assign(sharpe=lambda f: f.sharpe))
    assert ungated.current_passes_ips is None  # no IPS: no claim either way


# ------------------------------------------------------------------------ Build mode


def _build_table(**overrides) -> pd.DataFrame:
    from gpl.recommend import DIVERSIFIED, TRACKER

    rows = {
        "Max IR": dict(volatility=0.10, sharpe=0.80, in_sample_sharpe=1.0, excess_return=0.01,
                       information_ratio=0.30, tracking_error=0.03, passes_ips=True),
        "Max IR · momentum tilt": dict(volatility=0.10, sharpe=0.85, in_sample_sharpe=1.0,
                                       excess_return=0.015, information_ratio=0.45,
                                       tracking_error=0.035, passes_ips=True),
        TRACKER: dict(volatility=0.11, sharpe=0.70, in_sample_sharpe=0.7, excess_return=-0.001,
                      information_ratio=-0.5, tracking_error=0.002, passes_ips=True),
        DIVERSIFIED: dict(volatility=0.08, sharpe=0.75, in_sample_sharpe=0.9, excess_return=-0.01,
                          information_ratio=-0.2, tracking_error=0.05, passes_ips=True),
        "Benchmark": dict(volatility=0.11, sharpe=0.72, in_sample_sharpe=None, excess_return=0.0,
                          information_ratio=None, tracking_error=0.0, passes_ips=None),
    }  # fmt: skip
    for name, change in overrides.items():
        rows[name] = {**rows[name], **change}
    table = pd.DataFrame(rows).T
    table["ips_fails"] = [0 if p else 1 for p in table["passes_ips"]]
    table["ips_breach"] = 0.0
    return table


def test_build_picks_the_highest_ir_that_beats_the_benchmark() -> None:
    from gpl.recommend import recommend_build

    rec = recommend_build(_build_table(), tracking_error_max=0.04)
    assert rec.status == "recommended" and rec.choice == "Max IR · momentum tilt"
    assert rec.ranking == ["Max IR · momentum tilt", "Max IR"]
    assert "not a promise" in rec.reasons[-1]


def test_build_excludes_too_much_tracking_error_volatility_or_ips_fails() -> None:
    from gpl.recommend import recommend_build

    rec = recommend_build(
        _build_table(**{"Max IR · momentum tilt": {"tracking_error": 0.06}}), 0.04
    )
    assert rec.choice == "Max IR"  # 6.00% tracking error is over the 4.00% limit
    rec = recommend_build(_build_table(**{"Max IR · momentum tilt": {"volatility": 0.13}}), 0.04)
    assert rec.choice == "Max IR"  # riskier than the benchmark
    rec = recommend_build(_build_table(**{"Max IR · momentum tilt": {"passes_ips": False}}), 0.04)
    assert rec.choice == "Max IR"
    rec = recommend_build(_build_table(**{"Max IR · momentum tilt": {"sharpe": 0.3}}), 0.04)
    assert rec.choice == "Max IR"  # 0.3 is under half its in-sample 1.0: didn't hold up


def test_build_falls_back_to_tracker_then_diversified_then_closest() -> None:
    from gpl.recommend import DIVERSIFIED, TRACKER, recommend_build

    none_beat = {
        "Max IR": {"excess_return": -0.01},
        "Max IR · momentum tilt": {"information_ratio": -0.1},
    }
    rec = recommend_build(_build_table(**none_beat), 0.04)
    # every test passed, none beat the benchmark: the highest Sharpe (0.85) is next best
    assert rec.status == "next_best" and rec.choice == "Max IR · momentum tilt"
    assert "does not beat the benchmark reliably" in rec.reasons[1]
    assert "others: Max IR 0.80, Benchmark tracker 0.70" in rec.reasons[1]
    weak = {"Max IR": {"excess_return": -0.01, "sharpe": 0.45},
            "Max IR · momentum tilt": {"information_ratio": -0.1, "sharpe": 0.45}}  # fmt: skip
    rec = recommend_build(_build_table(**weak), 0.04)
    assert rec.choice == TRACKER  # 0.70 beats the builds; Diversified's 5% TE is over the limit
    rec = recommend_build(
        _build_table(**weak, **{TRACKER: {"passes_ips": False}, DIVERSIFIED:
                                {"tracking_error": 0.03}}), 0.04
    )  # fmt: skip
    assert rec.status == "next_best" and rec.choice == DIVERSIFIED
    all_fail = {n: {"passes_ips": False} for n in ("Max IR", "Max IR · momentum tilt", TRACKER,
                                                  DIVERSIFIED)}  # fmt: skip
    all_fail["Max IR"]["ips_fails"] = 1
    rec = recommend_build(_build_table(**all_fail), 0.04)
    assert rec.status == "closest_fails" and "Not suitable as it stands" in rec.reasons[1]


def test_build_needs_a_material_information_ratio() -> None:
    from gpl.recommend import recommend_build

    tiny = {"Max IR": {"information_ratio": 0.02}, "Max IR · momentum tilt":
            {"information_ratio": 0.08}}  # fmt: skip
    rec = recommend_build(_build_table(**tiny), 0.04)
    assert rec.status == "next_best"  # 0.08 < 0.10: within noise, so not "recommended"
    assert rec.choice == "Max IR · momentum tilt"  # still the best Sharpe of the rest
    assert "at least 0.10" in rec.reasons[0]
    assert recommend_build(_build_table(**tiny), 0.04, min_information_ratio=0.05).choice == (
        "Max IR · momentum tilt"
    )


def test_build_keeps_a_compliant_current_portfolio_over_a_worse_fallback() -> None:
    from gpl.recommend import KEEP, TRACKER, recommend_build

    none_beat = {
        "Max IR": {"excess_return": -0.01},
        "Max IR · momentum tilt": {"information_ratio": -0.1},
    }
    weak = {n: {**c, "sharpe": 0.45} for n, c in none_beat.items()}  # below the tracker's 0.70
    table = _build_table(**weak)
    table.loc["Current"] = table.loc[TRACKER]
    table.loc["Current", ["sharpe", "passes_ips"]] = [0.75, True]  # better than the 0.70
    rec = recommend_build(table, 0.04)
    assert rec.choice == KEEP and rec.status == "keep" and "so keep it" in rec.reasons[1]
    table.loc["Current", "sharpe"] = 0.60  # worse than the tracker: move to it
    assert recommend_build(table, 0.04).choice == TRACKER
    table.loc["Current", ["sharpe", "passes_ips"]] = [0.90, False]  # better, but fails the IPS
    assert recommend_build(table, 0.04).choice == TRACKER


def _seed_44593_table() -> pd.DataFrame:
    """Demo seed 44593 against 60/40, real prices, 2022-01-04 to 2026-11-05 (live run,
    8 Nov 2026). No candidate passes every test: the builds didn't hold up and are over the
    4.00% tracking-error limit; the tracker breaks the IPS (home bias, liquidity)."""
    from gpl.recommend import DIVERSIFIED, TRACKER

    cols = [
        "volatility",
        "sharpe",
        "max_drawdown",
        "in_sample_sharpe",
        "excess_return",
        "information_ratio",
        "tracking_error",
        "turnover_per_year",
        "passes_ips",
    ]
    rows = {
        "Max IR": [0.1194, 0.30, -0.1941, 1.01, 0.0027, 0.06, 0.0518, 2.81, True],
        "Max IR · low-vol tilt": [0.1140, 0.49, -0.1892, 1.07, 0.0255, 0.47, 0.0495, 2.07, True],
        "Max IR · momentum tilt": [0.1199, 0.34, -0.1832, 0.97, 0.0075, 0.15, 0.0505, 2.60, True],
        "Max IR · quality tilt": [0.1203, 0.31, -0.1974, 0.94, 0.0038, 0.08, 0.0523, 2.78, True],
        DIVERSIFIED: [0.1376, 0.39, -0.2429, 1.44, 0.0187, 0.14, 0.1477, 3.59, True],
        TRACKER: [0.1052, 0.29, -0.2053, 0.29, -0.0012, -0.05, 0.0420, 0.11, False],
        "Current": [0.1142, 0.32, -0.1769, None, 0.0039, 0.07, 0.0487, 0.19, True],
        "Benchmark": [0.1161, 0.28, -0.2027, None, 0.0, None, 0.0, None, None],
    }  # fmt: skip
    table = pd.DataFrame(rows, index=cols).T
    table["ips_fails"] = [0 if p is True else 1 for p in table["passes_ips"]]
    table["ips_breach"] = 0.0
    return table


def test_build_never_falls_back_to_a_portfolio_worse_than_a_rejected_candidate() -> None:
    # Live finding: the old rule recommended Diversified (Sharpe 0.39, 14.77% tracking error)
    # because only the fallback skipped the tests; the low-vol tilt (0.49) was better on all
    # five measures and also passes the IPS.
    from gpl.recommend import recommend_build

    rec = recommend_build(_seed_44593_table(), tracking_error_max=0.04)
    assert rec.status == "next_best" and rec.choice == "Max IR · low-vol tilt"
    assert "no portfolio passed every test" in rec.reasons[1]
    assert (
        "fails the fewest tests (didn't hold up out of sample; too much tracking error)"
        in (rec.reasons[1])
    )
    assert "Diversified (IPS) 0.39" in rec.reasons[1]


def test_build_fallback_must_pass_the_same_tests_as_the_builds() -> None:
    from gpl.recommend import DIVERSIFIED, recommend_build

    table = _seed_44593_table()
    # Diversified within every test, the builds still failing: it is the only full pass
    table.loc[DIVERSIFIED, ["tracking_error", "in_sample_sharpe"]] = [0.03, 0.6]
    table.loc[DIVERSIFIED, "volatility"] = 0.11
    rec = recommend_build(table, tracking_error_max=0.04)
    assert rec.choice == DIVERSIFIED and "pass the IPS, held up out of sample" in rec.reasons[1]


def test_dominance_guard_switches_to_the_better_portfolio() -> None:
    from gpl.recommend import DIVERSIFIED, dominated_by

    table = _seed_44593_table()
    pool = [n for n in table.index if table.loc[n, "passes_ips"] is True and n != "Current"]
    assert dominated_by(table, DIVERSIFIED, pool) == "Max IR · low-vol tilt"
    assert dominated_by(table, "Max IR · low-vol tilt", pool) is None
    # worse on one measure (more turnover) is enough to not be dominated
    table.loc["Max IR · low-vol tilt", "turnover_per_year"] = 4.0
    table.loc[["Max IR · momentum tilt", "Max IR", "Max IR · quality tilt"], "sharpe"] = 0.1
    assert dominated_by(table, DIVERSIFIED, pool) is None


def test_build_keep_still_applies_to_the_new_fallback() -> None:
    from gpl.recommend import KEEP, recommend_build

    table = _seed_44593_table()
    table.loc["Current", "sharpe"] = 0.55  # better than the low-vol tilt's 0.49
    rec = recommend_build(table, tracking_error_max=0.04)
    assert rec.choice == KEEP and "so keep it" in rec.reasons[-1]


def test_build_test_failures_names_each_test() -> None:
    from gpl.recommend import TRACKER, build_test_failures

    fails = build_test_failures(_seed_44593_table(), tracking_error_max=0.04)
    assert fails["Max IR · low-vol tilt"] == ["held_up", "tracking_error"]
    assert fails[TRACKER] == ["ips"]  # its 4.20% is within 4.00% + 0.50 pp tolerance
    assert "Current" not in fails and "Benchmark" not in fails


def test_build_fallback_prefers_the_fewest_failed_tests_then_sharpe() -> None:
    # Same client after the 25%-a-quarter turnover limit (live, 8 Nov 2026): Diversified's
    # Sharpe rose to 0.54 but it fails three tests (held up, volatility, 12.77% tracking
    # error); the low-vol tilt fails two (held up; 4.87% tracking error, just over the limit).
    from gpl.recommend import DIVERSIFIED, recommend_build

    table = _seed_44593_table()
    table.loc[DIVERSIFIED, ["volatility", "sharpe", "max_drawdown", "tracking_error"]] = [
        0.1220, 0.54, -0.2086, 0.1277,
    ]  # fmt: skip
    table.loc["Max IR · low-vol tilt", ["volatility", "tracking_error"]] = [0.1152, 0.0487]
    rec = recommend_build(table, tracking_error_max=0.04)
    assert rec.choice == "Max IR · low-vol tilt"
    assert "Diversified (IPS) 0.54" in rec.reasons[1]  # the higher Sharpe is named, not hidden


def test_candidate_statuses_build_in_plain_english() -> None:
    from gpl.recommend import TRACKER, candidate_statuses, recommend_build

    table = _seed_44593_table()
    rec = recommend_build(table, tracking_error_max=0.04)
    st = candidate_statuses(table, rec, "build", {TRACKER: ["Home-market bias", "Liquidity"]}, 0.04)
    assert st["Max IR · low-vol tilt"][0] == "✓ Next best"
    assert st[TRACKER] == ("✗ Breaks IPS", "Fails the IPS: Home-market bias; Liquidity.")
    label, why = st["Max IR"]
    assert label == "✗ Didn't hold up out of sample"
    assert why == (
        "Sharpe 0.30 in the test against 1.01 when fitted: it kept less than 50% of it; "
        "Tracking error 5.18%, over the 4.00% limit."
    )
    assert st["Current"][0] == "— Today's portfolio" and st["Benchmark"][0] == "— Reference"


def test_candidate_statuses_build_names_a_weak_information_ratio() -> None:
    from gpl.recommend import candidate_statuses, recommend_build

    tiny = {"Max IR": {"information_ratio": 0.02}}
    table = _build_table(**tiny)
    rec = recommend_build(table, 0.04)
    st = candidate_statuses(table, rec, "build", None, 0.04)
    assert rec.choice == "Max IR · momentum tilt" and st[rec.choice][0] == "✓ Recommended"
    assert st["Max IR"][0] == "✗ Didn't beat the benchmark enough"
    assert "Information ratio 0.02 (at least 0.10 is needed)" in st["Max IR"][1]


def test_candidate_statuses_improve() -> None:
    from gpl.recommend import KEEP, Recommendation, candidate_statuses

    table = pd.DataFrame(
        {
            "sharpe": [0.5, 0.9, 0.3, 0.7, 0.6],
            "in_sample_sharpe": [None, 1.0, 1.2, 0.8, 0.7],
            "passes_ips": [True, True, True, False, True],
            "feasible": [True, True, True, True, True],
        },
        index=["Current", "Max Sharpe", "Min variance", "Risk parity", "Equal weight"],
    )
    rec = Recommendation("Max Sharpe", "rule", ["r"], ["Max Sharpe", "Equal weight"],
                         {"Max Sharpe": True, "Min variance": False, "Risk parity": True,
                          "Equal weight": True})  # fmt: skip
    st = candidate_statuses(table, rec, "improve", {"Risk parity": ["Concentration"]})
    assert st["Max Sharpe"][0] == "✓ Recommended"
    assert st["Min variance"][0] == "✗ Didn't hold up out of sample"
    assert st["Risk parity"] == ("✗ Breaks IPS", "Fails the IPS: Concentration.")
    assert st["Equal weight"] == ("Ranked lower", "Qualified under the rule; ranked 2 of 2.")
    keep = Recommendation(KEEP, "rule", ["not enough"], ["Max Sharpe"], rec.held_up, "keep")
    st = candidate_statuses(table, keep, "improve")
    assert st["Current"] == ("✓ Keep (recommended)", "not enough")
    assert st["Max Sharpe"][0] == "✗ Not enough better than today"


# --------------------------------------- lower risk, keep today's return; too cautious


def test_lower_risk_keep_return_allows_at_most_a_quarter_point_less_return() -> None:
    # current: CAGR 8.00%, volatility 12%
    rows = {
        "Current": (0.0800, 0.12, 0.50, 0.55),
        "Min variance": (0.0770, 0.07, 0.62, 0.70),  # 0.30 pp less return: rejected
        "Risk parity": (0.0780, 0.09, 0.60, 0.70),  # 0.20 pp less return: allowed
        "Max Sharpe": (0.0850, 0.125, 0.60, 0.70),  # more volatile: not lower risk
    }
    rec = recommend("reduce_risk_keep_return", table(rows))
    assert rec.choice == "Risk parity" and rec.ranking == ["Risk parity"]
    assert "at most 0.25 percentage points a year below" in rec.rule
    # plain "lower risk" would take the lowest volatility, whatever the return
    assert recommend("reduce_risk", table(rows)).choice == "Min variance"
    # a looser tolerance lets Min variance in
    assert recommend("reduce_risk_keep_return", table(rows), return_tolerance=0.004).choice == (
        "Min variance"
    )
    none = {**rows, "Risk parity": (0.0760, 0.09, 0.60, 0.70)}
    kept = recommend("reduce_risk_keep_return", table(none))
    assert kept.choice == KEEP and "gave up more than 0.25 percentage points" in kept.reasons[0]


GROWTH = {  # a 5/5 client: IPS volatility cap 20%, so under 10% is too cautious
    "Current": (0.090, 0.150, 0.45, 0.50),
    "Min variance": (0.055, 0.080, 0.60, 0.70),  # best Sharpe, but half the volatility
    "Max Sharpe": (0.105, 0.140, 0.58, 0.70),
    "Risk parity": (0.070, 0.095, 0.52, 0.70),
}


def test_too_cautious_candidates_are_ruled_out_for_a_growth_client() -> None:
    rec = recommend("max_risk_adjusted", table(GROWTH), min_volatility=0.10)
    assert rec.choice == "Max Sharpe"
    assert set(rec.too_cautious) == {"Min variance", "Risk parity"}
    assert "under 10.00%" in rec.too_cautious["Min variance"]
    # without the floor the rule would steer the 5/5 client into Min variance
    assert recommend("max_risk_adjusted", table(GROWTH)).choice == "Min variance"
    # a lower-risk goal allows it: the client chose to lower risk
    lower = recommend("reduce_risk", table(GROWTH), min_volatility=0.10)
    assert lower.choice == "Min variance" and lower.too_cautious == {}


def test_too_cautious_below_the_required_return_and_when_all_are() -> None:
    rec = recommend("max_risk_adjusted", table(GROWTH), min_return=0.06)
    assert set(rec.too_cautious) == {"Min variance"}  # 5.50% < 6.00% needed
    assert "below the 6.00% the client's goals need" in rec.too_cautious["Min variance"]
    assert rec.choice == "Max Sharpe"
    every = recommend("max_risk_adjusted", table(GROWTH), min_return=0.20)
    assert every.choice == KEEP and "too cautious for this client" in every.reasons[0]


def test_too_cautious_status_label() -> None:
    from gpl.recommend import candidate_statuses

    rec = recommend("max_risk_adjusted", table(GROWTH), min_volatility=0.10)
    status = candidate_statuses(table(GROWTH), rec, "improve")
    assert status["Min variance"][0] == "✗ Too cautious for this client"
    assert status["Max Sharpe"][0] == "✓ Recommended"
