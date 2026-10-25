"""Tests for goals -> spec (the model transcribes limits, never weights) and the
optimisation report."""

import numpy as np
import pandas as pd
import pytest

from gpl import backtest as bt
from gpl.agent.spec import (
    describe_goal,
    describe_limits,
    numbers_not_in_goals,
    propose_spec,
    spec_from_draft,
    spec_tool,
)
from gpl.agent.tools import (
    AnalystContext,
    ToolError,
    diagnose_portfolio,
    execute_tool,
    run_candidates,
)
from gpl.data import BenchmarkData
from gpl.metrics import beta
from gpl.optimise import OptimisationSpec
from gpl.recommend import CANDIDATES, KEEP
from gpl.recommend import GOALS as RECOMMEND_GOALS
from test_agent import CFG, FakeClient, response, text, tool_use

LIMIT_FIELDS = ("max_weight", "asset_bounds", "groups", "group_bounds", "max_volatility")

GROUPS = {"SPY": "Global equity", "AGG": "Bonds", "KSA": "GCC", "QAT": "GCC", "GLD": "Gold"}
GOALS = "max 12% vol, at least 20% GCC, no single asset over 25%, maximise Sharpe"
DRAFT = {
    "goal": "max_risk_adjusted",
    "max_weight_pct": 25,
    "asset_limits": [],
    "group_limits": [{"group": "GCC", "min_pct": 20, "max_pct": None}],
    "max_volatility_pct": 12,
    "min_return_pct": None,
    "unmapped_requests": [],
    "summary": "Maximise Sharpe with volatility at most 12%, at least 20% GCC, max 25% per asset.",
}


# ------------------------------------------------------------------------------ the spec tool


def test_spec_tool_has_no_place_for_weights_or_methods() -> None:
    schema = spec_tool(list(GROUPS), ["GCC", "Bonds"])["input_schema"]
    assert "weights" not in str(schema).lower().replace("max_weight_pct", "")
    assert schema["properties"]["goal"]["enum"] == list(RECOMMEND_GOALS)
    assert not {"objective", "method"} & set(schema["properties"])
    assert schema["additionalProperties"] is False and set(schema["required"]) == set(
        schema["properties"]
    )
    assert schema["properties"]["asset_limits"]["items"]["properties"]["ticker"]["enum"] == list(
        GROUPS
    )


def test_spec_from_draft_converts_percent_to_fractions() -> None:
    goal, limits, target = spec_from_draft(DRAFT, GROUPS)
    assert goal == "max_risk_adjusted" and target is None and limits.max_weight == 0.25
    assert limits.group_bounds == {"GCC": (0.20, None)} and limits.max_volatility == 0.12
    assert limits.min_return is None and limits.groups == GROUPS


def test_target_goals_take_their_number_out_of_the_limits() -> None:
    goal, limits, target = spec_from_draft({**DRAFT, "goal": "target_volatility"}, GROUPS)
    assert goal == "target_volatility" and target == 0.12 and limits.max_volatility is None
    draft = {**DRAFT, "goal": "target_return", "min_return_pct": 8}
    goal, limits, target = spec_from_draft(draft, GROUPS)
    assert target == 0.08 and limits.min_return is None and limits.max_volatility == 0.12
    assert describe_goal(goal, target) == (
        "Reach a target return with the least risk: at least 8.00% a year"
    )


def test_spec_from_draft_rejects_unknown_assets_and_missing_targets() -> None:
    with pytest.raises(ValueError, match="TSLA"):
        spec_from_draft(
            {**DRAFT, "asset_limits": [{"ticker": "TSLA", "min_pct": None, "max_pct": 5}]}, GROUPS
        )
    with pytest.raises(ValueError, match="Enter the target return"):
        spec_from_draft({**DRAFT, "goal": "target_return"}, GROUPS)


def test_numbers_not_in_goals_flags_invented_limits() -> None:
    assert numbers_not_in_goals(DRAFT, GOALS) == []
    invented = {**DRAFT, "min_return_pct": 7}
    assert numbers_not_in_goals(invented, GOALS) == ["7%"]


def test_describe_limits() -> None:
    lines = describe_limits(spec_from_draft(DRAFT, GROUPS)[1])
    assert "No single asset above 25.00%" in lines and "Group GCC: at least 20.00%" in lines
    assert "Volatility at most 12.00% a year" in lines


# --------------------------------------------------------------------------- propose_spec


def test_propose_spec_returns_the_tool_input_and_checks_numbers() -> None:
    client = FakeClient(
        [response([tool_use("s1", "propose_spec", {**DRAFT, "min_return_pct": 7})], "tool_use")]
    )
    proposal = propose_spec(client, "claude-haiku-5-5", GOALS, GROUPS)
    assert proposal.draft["max_volatility_pct"] == 12
    assert proposal.numbers_not_in_goals == ["7%"]  # the model added a limit nobody asked for
    request = client.requests[0]
    assert [t["name"] for t in request["tools"]] == ["propose_spec"]
    assert (
        "never calculate" in request["system"]
        and "SPY (Global equity)" in request["messages"][0]["content"]
    )


def test_propose_spec_without_a_tool_call_is_an_error() -> None:
    client = FakeClient([response([text("Here are some weights: SPY 40%...")], "end_turn")])
    with pytest.raises(ValueError, match="did not return a specification"):
        propose_spec(client, "m", GOALS, GROUPS)


# ------------------------------------------------------------------------ optimisation report


@pytest.fixture(scope="module")
def ctx_opt(ctx) -> AnalystContext:
    return AnalystContext(
        prices=ctx.prices,
        rf=ctx.rf,
        groups=ctx.groups,
        benchmarks=ctx.benchmarks,
        regime_features=ctx.regime_features,
        regime_labels=ctx.regime_labels,
        settings=ctx.settings,
        universe=["SPY", "AGG", "GLD", "KSA", "QAT"],
        user_portfolio={"SPY": 0.5, "AGG": 0.3, "GLD": 0.1, "KSA": 0.1},
    )


CURRENT = {"SPY": 0.5, "AGG": 0.3, "GLD": 0.1, "KSA": 0.1}
HELD = {t: GROUPS[t] for t in CURRENT}


def test_optimisation_tool(ctx_opt) -> None:
    with pytest.raises(ToolError, match="no optimisation"):
        execute_tool(ctx_opt, "get_optimisation_result", {})
    ctx_opt.optimisation = {"feasible": True}
    assert execute_tool(ctx_opt, "get_optimisation_result", {}) == {"feasible": True}
    ctx_opt.optimisation = None


def test_cfg_groups_include_gcc() -> None:
    assert "KSA" in CFG["groups"]["gcc_equity"]


# ------------------------------------------------------- diagnosis and candidate methods


def _bench(ctx_opt, cash: bool = False) -> BenchmarkData:
    rets = ctx_opt.all_returns
    if cash:
        # T-bill rates drift a little day to day, as real ^IRX data does
        drift = 1e-6 * np.sin(np.arange(len(rets.index) - 1))
        r = pd.Series(1.02 ** (1 / 252) - 1 + drift, index=rets.index[1:])
    else:
        r = 0.6 * rets["SPY"] + 0.4 * rets["AGG"]
    return BenchmarkData("60/40" if not cash else "Cash", "test", r, cash, [])


def test_diagnosis_wraps_the_metrics(ctx_opt) -> None:
    ctx_opt.benchmark = _bench(ctx_opt)
    report = diagnose_portfolio(ctx_opt, CURRENT)
    rets = ctx_opt.all_returns[list(CURRENT)]
    assert report["holdings"]["SPY"]["weight"] == "50.00%"
    shares = [float(h["share_of_risk"].rstrip("%")) for h in report["holdings"].values()]
    assert sum(shares) == pytest.approx(100, abs=0.05)
    expected_beta = beta(rets["SPY"], ctx_opt.benchmark.returns.reindex(rets.index))
    assert report["holdings"]["SPY"]["beta_to_benchmark"] == f"{expected_beta:.2f}"
    rel = report["vs_benchmark"]["relative"]
    assert set(rel) >= {"beta", "alpha_annual_jensen", "tracking_error", "up_capture"}
    assert set(report["vs_benchmark"]["absolute"]) == {"Your portfolio", "Benchmark"}
    ctx_opt.benchmark = _bench(ctx_opt, cash=True)
    cash = diagnose_portfolio(ctx_opt, CURRENT)
    assert cash["vs_benchmark"]["relative"]["beta"].startswith("not defined")
    ctx_opt.benchmark = None


def test_candidates_are_walk_forward_within_limits_and_recommended_by_the_rule(ctx_opt) -> None:
    ctx_opt.benchmark = _bench(ctx_opt)
    limits = OptimisationSpec(max_weight=0.45, groups=HELD, group_bounds={"GCC": (0.10, None)})
    report, run = run_candidates(ctx_opt, "max_risk_adjusted", limits, CURRENT)
    wf = report["walk_forward_after_costs"]
    assert {"Current", "Benchmark", *CANDIDATES} <= set(wf)
    # every candidate's latest weights respect the limits
    latest = run.latest_weights.drop("Current")
    assert (latest.max(axis=1) <= 0.45 + 1e-6).all() and (latest["KSA"] >= 0.10 - 1e-6).all()
    # the recommendation is the rule's, applied to the reported walk-forward Sharpe ratios
    rec = report["recommendation"]
    assert rec["choice"] == run.recommendation.choice and rec["rule"].startswith("Highest")
    if rec["choice"] != KEEP:
        sharpe = {n: float(wf[n]["sharpe"]) for n in rec["ranking_under_the_rule"]}
        assert rec["ranking_under_the_rule"] == sorted(sharpe, key=sharpe.get, reverse=True)
        assert set(report["recommended_latest_weights"]) == set(CURRENT)
    assert "not a forecast or a promise" in report["caveats"]
    # a candidate's walk-forward equals the backtester run directly with Ledoit-Wolf
    direct = bt.run_backtest(
        ctx_opt.prices[list(CURRENT)],
        bt.spec_strategy(
            OptimisationSpec("min_variance", **{k: getattr(limits, k) for k in LIMIT_FIELDS}),
            rf=ctx_opt.rf,
            fallback=CURRENT,
            cov_method="ledoit_wolf",
        ),
        rebalance="ME",
        lookback=126,
        cost_bps=10.0,
    )
    np.testing.assert_allclose(
        # from the first full day held: day one (buying, which pays costs) is excluded
        run.equity["Min variance"].iloc[-1],
        direct.equity.iloc[-1] / direct.equity.iloc[0],
        rtol=1e-9,
    )
    ctx_opt.benchmark = None


def test_candidates_with_impossible_limits(ctx_opt) -> None:
    impossible = OptimisationSpec(max_weight=0.2, groups=HELD, group_bounds={"GCC": (0.9, None)})
    report, run = run_candidates(ctx_opt, "reduce_risk", impossible, CURRENT)
    assert set(report["infeasible_candidates"]) == set(CANDIDATES)
    assert report["recommendation"]["choice"] == KEEP
    assert "recommended_latest_weights" not in report and list(run.latest_weights.index) == [
        "Current"
    ]


def test_diagnosis_tool_needs_a_diagnosis(ctx_opt) -> None:
    with pytest.raises(ToolError, match="no diagnosis"):
        execute_tool(ctx_opt, "get_portfolio_diagnosis", {})


def test_candidates_use_the_configured_minimum_sharpe_gain(ctx_opt) -> None:
    limits = OptimisationSpec(max_weight=0.45, groups=HELD)
    ctx_opt.settings = {**ctx_opt.settings, "optimise_lab": {"min_sharpe_gain": 5.0}}
    report, _ = run_candidates(ctx_opt, "max_risk_adjusted", limits, CURRENT)
    assert report["recommendation"]["choice"] == KEEP  # no method gains 5.0 of Sharpe
    assert "at least 5.00 above" in report["recommendation"]["rule"]
    ctx_opt.settings = {k: v for k, v in ctx_opt.settings.items() if k != "optimise_lab"}
