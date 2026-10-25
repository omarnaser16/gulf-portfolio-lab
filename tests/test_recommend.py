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
