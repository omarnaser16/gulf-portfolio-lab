"""Charts requested by the analyst are built only from data the engine computes.

The model picks a chart type and a section; it cannot pass numbers, and what it writes in its
answer cannot change a chart. Each chart is checked against an independent engine calculation.
"""

import json

import numpy as np
import pandas as pd
import pytest

from gpl import backtest as bt
from gpl.agent.analyst import Analyst
from gpl.agent.charts import CHART_TYPES, build_chart, chart_data
from gpl.agent.tools import AnalystContext, ToolError, execute_tool, tool_definitions
from gpl.data import BenchmarkData
from gpl.optimise import risk_contributions
from test_agent import CFG, FakeClient, response, text, tool_use

CURRENT = {"SPY": 0.5, "AGG": 0.3, "GLD": 0.1, "KSA": 0.1}


@pytest.fixture
def cctx(ctx) -> AnalystContext:
    rets = ctx.all_returns
    settings = {
        **ctx.settings,
        "stress_episodes": CFG["stress_episodes"],
        "exposure_meta": CFG["exposure_meta"],
    }
    return AnalystContext(
        prices=ctx.prices,
        rf=ctx.rf,
        groups=ctx.groups,
        benchmarks=ctx.benchmarks,
        regime_features=ctx.regime_features,
        regime_labels=ctx.regime_labels,
        settings=settings,
        universe=list(CURRENT),
        user_portfolio=CURRENT,
        holding_groups={"SPY": "Global equity", "AGG": "Bonds", "GLD": "Gold", "KSA": "GCC"},
        benchmark=BenchmarkData("60/40", "test", 0.6 * rets["SPY"] + 0.4 * rets["AGG"], False, []),
    )


def _numeric_fields(schema) -> list:
    """Every property in a JSON schema whose type is (or includes) number/integer."""
    found = []
    if isinstance(schema, dict):
        t = schema.get("type")
        types = t if isinstance(t, list) else [t]
        if {"number", "integer"} & set(types):
            found.append(schema)
        for v in schema.values():
            found += _numeric_fields(v)
    elif isinstance(schema, list):
        for v in schema:
            found += _numeric_fields(v)
    return found


def test_show_chart_schema_has_no_numbers(cctx) -> None:
    schema = next(t for t in tool_definitions(cctx) if t["name"] == "show_chart")["input_schema"]
    assert set(schema["properties"]) == {"chart", "section"}
    assert schema["properties"]["chart"]["enum"] == list(CHART_TYPES)
    assert schema["properties"]["section"] == {"type": "string"}
    assert _numeric_fields(schema) == []


@pytest.mark.parametrize("chart", list(CHART_TYPES))
def test_every_chart_type_builds(cctx, chart) -> None:
    out = execute_tool(cctx, "show_chart", {"chart": chart, "section": "Risk"})
    assert out["chart_id"] == len(cctx.charts) - 1 and out["key_figures"]
    built = build_chart(cctx.charts[out["chart_id"]])
    assert isinstance(built, pd.DataFrame) or built.data


def test_money_vs_risk_bars_equal_the_engine(cctx) -> None:
    fig = build_chart(chart_data(cctx, "money_vs_risk"))
    w = pd.Series(CURRENT)
    expected = risk_contributions(w, cctx.all_returns[list(w.index)].cov())
    risk = next(t for t in fig.data if t.name == "Share of risk")
    np.testing.assert_allclose(list(risk.x), expected.reindex(list(risk.y)).to_numpy())


def test_growth_lines_equal_the_backtester(cctx) -> None:
    fig = build_chart(chart_data(cctx, "growth_of_1"))
    direct = bt.run_backtest(
        cctx.prices[list(CURRENT)],
        bt.fixed_weights(CURRENT),
        rebalance="ME",
        lookback=126,
        cost_bps=10.0,
    ).returns.iloc[1:]
    current = next(t for t in fig.data if t.name == "Current")
    np.testing.assert_allclose(current.y[-1], (1 + direct).prod(), rtol=1e-12)
    bench = next(t for t in fig.data if t.name.startswith("Benchmark"))
    assert bench.line.dash == "dash"


def test_stress_bars_equal_a_direct_calculation(cctx) -> None:
    fig = build_chart(chart_data(cctx, "stress_test"))
    held = cctx.all_returns[list(CURRENT)] @ pd.Series(CURRENT)
    covid = held[(held.index > "2020-02-19") & (held.index <= "2020-03-23")]
    current = next(t for t in fig.data if t.name == "Current")
    assert current.y[0] == pytest.approx((1 + covid).prod() - 1)
    assert "COVID crash<br>19 Feb 2020 – 23 Mar 2020" == current.x[0]
    # the synthetic data ends in 2021, so the 2022 episode is reported, not invented
    assert np.isnan(current.y[2])
    assert any("not covered" in a.text for a in fig.layout.annotations)


def test_model_text_cannot_change_a_chart(cctx) -> None:
    client = FakeClient(
        [
            response(
                [tool_use("c1", "show_chart", {"chart": "money_vs_risk", "section": "Risk"})],
                "tool_use",
            ),
            response([text("### Risk\nSPY is 99.99% of the risk and AGG is 123.45%.")], "end_turn"),
        ]
    )
    reply = Analyst(cctx, client, model="m").ask("Review my portfolio")
    ((section, data),) = reply.charts
    assert section == "Risk"
    fresh = build_chart(chart_data(cctx, "money_vs_risk"))
    assert json.loads(build_chart(data).to_json()) == json.loads(fresh.to_json())
    assert "99.99%" in reply.unverified  # and the made-up figures are flagged in the text


def test_unknown_chart_is_rejected(cctx) -> None:
    with pytest.raises(ToolError, match="unknown chart"):
        execute_tool(cctx, "show_chart", {"chart": "pie_of_dreams", "section": "x"})


# ------------------------------------------------------------------------------ what-if


def test_what_if_uses_the_users_numbers_and_sets_the_proposed_portfolio(cctx) -> None:
    cctx.last_user_message = "What if I cut SPY to 20%?"
    out = execute_tool(
        cctx, "what_if_portfolio", {"changes": [{"ticker": "SPY", "weight_pct": 20}]}
    )
    # AGG, GLD, KSA share the other 80% in their 30:10:10 ratio -> 48%, 16%, 16%
    assert (
        out["weights"]["AGG"]["what_if"] == "48.00%"
        and out["weights"]["SPY"]["change"] == "-30.00 pp"
    )
    assert cctx.proposed["weights"]["GLD"] == pytest.approx(0.16)
    fig = build_chart(chart_data(cctx, "weights_current_vs_proposed"))
    what_if = next(t for t in fig.data if t.name == "What-if")
    assert dict(zip(what_if.y, what_if.x, strict=True))["SPY"] == pytest.approx(0.20)
    assert "Current vs benchmark" in out["key_figures_after_costs"]  # the table has both
    assert "What-if vs benchmark" in out["key_figures_after_costs"]


def test_what_if_refuses_weights_the_user_did_not_write(cctx) -> None:
    cctx.last_user_message = "What if I halve SPY?"
    with pytest.raises(ToolError, match="not in the user's message: 25%"):
        execute_tool(cctx, "what_if_portfolio", {"changes": [{"ticker": "SPY", "weight_pct": 25}]})
    assert cctx.proposed is None
