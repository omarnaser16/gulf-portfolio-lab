"""Tests for goals -> spec (the model transcribes limits, never weights) and the
optimisation report."""

import pandas as pd
import pytest

from gpl.agent.spec import (
    describe_spec,
    numbers_not_in_goals,
    propose_spec,
    spec_from_draft,
    spec_tool,
)
from gpl.agent.tools import AnalystContext, ToolError, execute_tool, run_optimisation
from gpl.optimise import OptimisationSpec, optimise
from test_agent import CFG, FakeClient, response, text, tool_use

GROUPS = {"SPY": "Global equity", "AGG": "Bonds", "KSA": "GCC", "QAT": "GCC", "GLD": "Gold"}
GOALS = "max 12% vol, at least 20% GCC, no single asset over 25%, maximise Sharpe"
DRAFT = {
    "objective": "max_sharpe",
    "max_weight_pct": 25,
    "asset_limits": [],
    "group_limits": [{"group": "GCC", "min_pct": 20, "max_pct": None}],
    "max_volatility_pct": 12,
    "min_return_pct": None,
    "unmapped_requests": [],
    "summary": "Maximise Sharpe with volatility at most 12%, at least 20% GCC, max 25% per asset.",
}


# ------------------------------------------------------------------------------ the spec tool


def test_spec_tool_has_no_place_for_weights() -> None:
    schema = spec_tool(list(GROUPS), ["GCC", "Bonds"])["input_schema"]
    assert "weights" not in str(schema).lower().replace("max_weight_pct", "")
    assert schema["additionalProperties"] is False and set(schema["required"]) == set(
        schema["properties"]
    )
    assert schema["properties"]["asset_limits"]["items"]["properties"]["ticker"]["enum"] == list(
        GROUPS
    )


def test_spec_from_draft_converts_percent_to_fractions() -> None:
    spec = spec_from_draft(DRAFT, GROUPS)
    assert spec.objective == "max_sharpe" and spec.max_weight == 0.25
    assert spec.group_bounds == {"GCC": (0.20, None)} and spec.max_volatility == 0.12
    assert spec.min_return is None and spec.groups == GROUPS


def test_spec_from_draft_rejects_unknown_assets_and_missing_targets() -> None:
    with pytest.raises(ValueError, match="TSLA"):
        spec_from_draft(
            {**DRAFT, "asset_limits": [{"ticker": "TSLA", "min_pct": None, "max_pct": 5}]}, GROUPS
        )
    with pytest.raises(ValueError, match="target_return needs min_return"):
        spec_from_draft({**DRAFT, "objective": "target_return"}, GROUPS)


def test_numbers_not_in_goals_flags_invented_limits() -> None:
    assert numbers_not_in_goals(DRAFT, GOALS) == []
    invented = {**DRAFT, "min_return_pct": 7}
    assert numbers_not_in_goals(invented, GOALS) == ["7%"]


def test_describe_spec() -> None:
    lines = describe_spec(spec_from_draft(DRAFT, GROUPS))
    assert lines[0].startswith("Maximise the Sharpe ratio")
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


def test_report_wraps_the_optimiser_and_backtests(ctx_opt) -> None:
    spec = OptimisationSpec(
        "min_variance", max_weight=0.6, groups=HELD, group_bounds={"GCC": (0.15, None)}
    )
    report, weights = run_optimisation(ctx_opt, spec, CURRENT)
    rets = ctx_opt.all_returns[list(CURRENT)]
    direct = optimise(rets.mean() * 252, rets.cov() * 252, spec, 0.02)
    pd.testing.assert_series_equal(weights, direct.weights, atol=1e-6)
    assert report["feasible"] and report["weights"]["KSA"]["current"] == "10.00%"
    assert weights["KSA"] >= 0.15 - 1e-6
    assert set(report["backtests_after_costs"]) == {
        "Current",
        "Optimised (hindsight)",
        "Optimised (walk-forward)",
        "60/40 (SPY/AGG)",
    }
    assert "not a guarantee" in report["caveats"]
    assert report["in_sample_historical"]["Current"]["max_drawdown"].startswith("-")


def test_report_explains_infeasible_specs_without_weights(ctx_opt) -> None:
    impossible = OptimisationSpec(
        "min_variance", groups=HELD, group_bounds={"GCC": (0.9, None)}, max_weight=0.2
    )
    report, weights = run_optimisation(ctx_opt, impossible, CURRENT)
    assert weights is None and not report["feasible"] and "weights" not in report
    assert report["reasons_if_not_feasible"]
    assert set(report["backtests_after_costs"]) == {"Current", "60/40 (SPY/AGG)"}


def test_optimisation_tool(ctx_opt) -> None:
    with pytest.raises(ToolError, match="no optimisation"):
        execute_tool(ctx_opt, "get_optimisation_result", {})
    ctx_opt.optimisation = {"feasible": True}
    assert execute_tool(ctx_opt, "get_optimisation_result", {}) == {"feasible": True}
    ctx_opt.optimisation = None


def test_cfg_groups_include_gcc() -> None:
    assert "KSA" in CFG["groups"]["gcc_equity"]
