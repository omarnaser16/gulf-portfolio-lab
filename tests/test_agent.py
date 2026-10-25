"""Tests for the agent: tools wrap gpl functions; the loop runs tools and audits numbers.

No network: tools run on synthetic data and the model is a scripted fake client.
"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from gpl.agent import analyst as an
from gpl.agent.analyst import Analyst, estimate_cost, unverified_numbers
from gpl.agent.tools import (
    EXECUTORS,
    AnalystContext,
    ToolError,
    execute_tool,
    num,
    pct,
    tool_definitions,
)
from gpl.data import load_universe, to_returns
from gpl.metrics import summary_table
from gpl.optimise import min_variance, risk_contributions

CFG = load_universe()


# ----------------------------------------------------------------------------------- tools


def test_every_tool_has_a_strict_schema(ctx) -> None:
    defs = {d["name"]: d for d in tool_definitions(ctx)}
    assert set(defs) == set(EXECUTORS)
    for d in defs.values():
        schema = d["input_schema"]
        assert d["strict"] is True and schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        assert len(d["description"]) > 40


def test_asset_metrics_wrap_summary_table(ctx) -> None:
    out = execute_tool(ctx, "get_asset_metrics", {"tickers": ["SPY", "AGG"]})
    table = summary_table(to_returns(ctx.prices)[["SPY", "AGG"]], rf=ctx.rf)
    assert out["metrics"]["SPY"]["sharpe"] == num(table.loc["SPY", "sharpe"])
    assert out["metrics"]["AGG"]["max_drawdown"] == pct(table.loc["AGG", "max_drawdown"])
    assert len(execute_tool(ctx, "get_asset_metrics", {"tickers": []})["metrics"]) == 9


def test_optimise_wraps_min_variance_and_respects_cap(ctx) -> None:
    out = execute_tool(ctx, "optimise_portfolio", {"method": "min_variance"})
    expected = min_variance(to_returns(ctx.prices).cov() * 252, 0.40)
    top = expected.idxmax()
    assert out["weights"][top] == pct(expected[top])
    assert all(float(v[:-1]) <= 40.0 for v in out["weights"].values())
    assert "hindsight" in out["caveat"]


def test_backtest_tool_matches_the_backtester(ctx) -> None:
    out = execute_tool(
        ctx, "backtest_strategies", {"strategies": ["Equal weight", "SPY"], "regime_aware": True}
    )
    assert set(out["results"]) == {"Equal weight", "Equal weight + regime", "SPY"}
    res = ctx.backtest("Equal weight")
    sharpe = summary_table(res.returns.to_frame("x"), rf=ctx.rf).loc["x", "sharpe"]
    assert out["results"]["Equal weight"]["sharpe"] == num(sharpe)
    assert out["results"]["SPY"]["latest_weights"] == {"SPY": "100.00%"}


def test_simulation_tool_is_reproducible_and_deflates(ctx) -> None:
    args = {"strategy": "Risk parity", "horizon_years": 5, "target_multiple": 1.5}
    nominal = execute_tool(ctx, "simulate_future", {**args, "real_terms": False})
    again = execute_tool(ctx, "simulate_future", {**args, "real_terms": False})
    real = execute_tool(ctx, "simulate_future", {**args, "real_terms": True})
    assert nominal == again
    assert "probability_of_reaching_1.50×" in nominal
    assert float(real["median_outcome"][:-1]) < float(nominal["median_outcome"][:-1])
    assert "real USD" in real["units"]


def test_regime_tool_reports_track_record(ctx) -> None:
    out = execute_tool(ctx, "get_regime_view", {})
    assert out["model_signal"] in {"stressed", "calm"}
    assert out["threshold"] == "50.00%"
    assert {"roc_auc", "precision", "base_rate", "vol_rule_precision"} <= set(
        out["out_of_sample_track_record"]
    )


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("no_such_tool", {}),
        ("get_asset_metrics", {"tickers": ["TSLA"]}),
        ("optimise_portfolio", {"method": "magic"}),
        (
            "simulate_future",
            {"strategy": "SPY", "horizon_years": 99, "target_multiple": 2.0, "real_terms": False},
        ),
        ("optimise_portfolio", {}),  # missing argument
    ],
)
def test_bad_tool_calls_raise_tool_error(ctx, name, args) -> None:
    with pytest.raises(ToolError):
        execute_tool(ctx, name, args)


# ----------------------------------------------------------------------- number auditing


def test_unverified_numbers_compares_values_not_strings() -> None:
    sources = ['{"sharpe": "0.49", "cagr": "15.27%", "goal": "1.50×"}', "I am 35, horizon 10 years"]
    assert unverified_numbers("Sharpe 0.49 and CAGR 15.27%, target 1.5×.", sources) == []
    assert unverified_numbers("CAGR of about 15.3%", sources) == ["15.3%"]  # re-rounded
    assert unverified_numbers("Over 10 years from 2026, aged 35", sources) == []  # integers ok
    assert unverified_numbers("Expected 7.25% a year", sources) == ["7.25%"]  # invented


def test_dates_and_counts_do_not_whitelist_percentages() -> None:
    # Found in a live run: "a 20-30% fall" passed because a date "2026-09-30" contains 30.
    sources = ['{"backtest_period": "2016-09-30 to 2026-09-30", "trading_days": 2774}']
    assert unverified_numbers("how you would react to a 20–30% fall", sources) == ["30%"]


def test_estimate_cost_hand_value() -> None:
    reply = an.AnalystReply("", [], "end_turn", input_tokens=200_000, output_tokens=10_000)
    assert estimate_cost(reply, {"input": 0.10, "output": 0.50}) == pytest.approx(0.025)


def test_has_credentials_reads_dotenv(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(an, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not an.has_credentials()
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=test-key\n")
    assert an.has_credentials()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


# ----------------------------------------------------------------------------- tool loop


def text(t: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=t)


def tool_use(id_: str, name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=args)


def response(blocks: list, stop: str, tokens: tuple[int, int] = (100, 20)) -> SimpleNamespace:
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]),
    )


class FakeClient:
    """Replays scripted responses and records every request."""

    def __init__(self, script: list) -> None:
        self.script, self.requests = list(script), []
        self.messages = self

    def create(self, **kwargs) -> SimpleNamespace:
        self.requests.append(json.loads(json.dumps(kwargs, default=lambda o: o.__dict__)))
        return self.script.pop(0)


def test_loop_runs_tools_and_audits_the_memo(ctx) -> None:
    sharpe = execute_tool(
        ctx, "backtest_strategies", {"strategies": ["Equal weight"], "regime_aware": False}
    )["results"]["Equal weight"]["sharpe"]
    client = FakeClient(
        [
            response(
                [
                    text("Let me look."),
                    tool_use("t1", "get_universe", {}),
                    tool_use(
                        "t2",
                        "backtest_strategies",
                        {"strategies": ["Equal weight"], "regime_aware": False},
                    ),
                ],
                "tool_use",
                (500, 50),
            ),
            response(
                [
                    text(
                        f"Equal weight has Sharpe {sharpe} [backtest_strategies] and returns "
                        "12.34% a year."
                    )
                ],
                "end_turn",
                (900, 300),
            ),
        ]
    )
    analyst = Analyst(ctx, client, model="claude-haiku-5-5", effort="low")
    reply = analyst.ask("Moderate risk, 10 years.")

    assert [c.name for c in reply.tool_calls] == ["get_universe", "backtest_strategies"]
    second = client.requests[1]
    results = second["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]  # all results in ONE message
    assert second["messages"][1]["role"] == "assistant"  # assistant blocks kept as returned
    assert reply.unverified == ["12.34%"]  # invented number caught; quoted Sharpe is fine
    assert "not investment advice" in reply.text.lower()
    assert (reply.input_tokens, reply.output_tokens) == (1400, 350)
    assert client.requests[0]["model"] == "claude-haiku-5-5"
    assert client.requests[0]["output_config"] == {"effort": "low"}
    assert "Never calculate" in client.requests[0]["system"]


def test_tool_errors_go_back_to_the_model(ctx) -> None:
    client = FakeClient(
        [
            response([tool_use("t1", "optimise_portfolio", {"method": "magic"})], "tool_use"),
            response([text("Sorry, that method is not available.")], "end_turn"),
        ]
    )
    reply = Analyst(ctx, client, model="m").ask("Use magic optimisation")
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "magic" in result["content"]
    assert reply.tool_calls[0].is_error


def test_loop_stops_after_too_many_rounds(ctx) -> None:
    loop = [response([tool_use(f"t{i}", "get_universe", {})], "tool_use") for i in range(3)]
    reply = Analyst(ctx, FakeClient(loop), model="m", max_tool_rounds=2).ask("?")
    assert reply.stop_reason == "max_tool_rounds"
    assert len(reply.tool_calls) == 3


def test_follow_up_keeps_the_conversation(ctx) -> None:
    client = FakeClient(
        [response([text("First.")], "end_turn"), response([text("Second.")], "end_turn")]
    )
    analyst = Analyst(ctx, client, model="m")
    analyst.ask("Profile one")
    analyst.ask("And a follow-up?")
    roles = [m["role"] for m in client.requests[1]["messages"]]
    assert roles == ["user", "assistant", "user"]


def test_refusal_is_reported(ctx) -> None:
    reply = Analyst(ctx, FakeClient([response([], "refusal")]), model="m").ask("?")
    assert "declined" in reply.text


# ------------------------------------------------------------------------- My portfolio


@pytest.fixture(scope="module")
def ctx_mine(ctx) -> AnalystContext:
    """Same data; a smaller strategy universe; a user portfolio with an asset outside it."""
    return AnalystContext(
        prices=ctx.prices,
        rf=ctx.rf,
        groups=ctx.groups,
        benchmarks=ctx.benchmarks,
        regime_features=ctx.regime_features,
        regime_labels=ctx.regime_labels,
        settings={**ctx.settings, "periods_per_year": 200},
        universe=["SPY", "EFA", "AGG", "GLD"],
        user_portfolio={"SPY": 0.5, "AGG": 0.3, "KSA": 0.2},
    )


def test_strategies_stay_in_the_universe_portfolio_does_not(ctx_mine) -> None:
    assert ctx_mine.assets == ["SPY", "EFA", "AGG", "GLD"]
    ew = ctx_mine.backtest("Equal weight").weights.iloc[-1]
    assert set(ew[ew > 0].index) == {"SPY", "EFA", "AGG", "GLD"}
    mine = ctx_mine.backtest("My portfolio").weights.iloc[-1]
    assert mine[mine > 0].to_dict() == pytest.approx({"SPY": 0.5, "AGG": 0.3, "KSA": 0.2})
    assert "My portfolio" in ctx_mine.strategy_names()
    enum = next(d for d in tool_definitions(ctx_mine) if d["name"] == "simulate_future")
    assert "My portfolio" in enum["input_schema"]["properties"]["strategy"]["enum"]


def test_get_my_portfolio_wraps_risk_contributions(ctx_mine) -> None:
    out = execute_tool(ctx_mine, "get_my_portfolio", {})
    w = pd.Series({"SPY": 0.5, "AGG": 0.3, "KSA": 0.2})
    rc = risk_contributions(w, to_returns(ctx_mine.prices)[w.index].cov() * 200)
    assert out["holdings"]["KSA"] == {"weight": "20.00%", "share_of_risk": pct(rc["KSA"])}
    assert out["largest_risk_share"].startswith(rc.idxmax())
    assert (
        out["backtest"]["sharpe"]
        == execute_tool(
            ctx_mine, "backtest_strategies", {"strategies": ["My portfolio"], "regime_aware": False}
        )["results"]["My portfolio"]["sharpe"]
    )


def test_tools_annualise_with_observed_periods(ctx_mine) -> None:
    out = execute_tool(ctx_mine, "get_asset_metrics", {"tickers": ["SPY"]})
    table = summary_table(
        to_returns(ctx_mine.prices)[["SPY"]], rf=ctx_mine.rf, periods_per_year=200
    )
    assert out["metrics"]["SPY"]["annual_return"] == pct(table.loc["SPY", "ann_return"])
    assert "200 observations a year" in out["note"]
    assert "about 200 a year" in execute_tool(ctx_mine, "get_universe", {})["calendar_note"]


def test_no_portfolio_is_a_clear_tool_error(ctx) -> None:
    with pytest.raises(ToolError, match="My portfolio tab"):
        execute_tool(ctx, "get_my_portfolio", {})


def test_backtest_output_explains_dates_and_confidence(ctx_mine) -> None:
    """Live run: the model mistook the later backtest start for a data error, and '95%' in
    '95% VaR' was flagged because no tool output stated the confidence level."""
    out = execute_tool(ctx_mine, "get_my_portfolio", {})
    assert "not a data problem" in out["why_the_backtest_starts_later"]
    assert out["data_start"] < out["backtest_period"][:10]
    assert unverified_numbers("Your 1-day 95% VaR is 0.93%", [json.dumps(out), '"0.93%"']) == []


# --------------------------------------------------------------- client profile and calculator


def _unions(schema) -> int:
    if isinstance(schema, dict):
        own = int(isinstance(schema.get("type"), list) or "anyOf" in schema)
        return own + sum(_unions(v) for v in schema.values())
    if isinstance(schema, list):
        return sum(_unions(v) for v in schema)
    return 0


def test_tool_schemas_stay_within_the_api_union_limit(ctx) -> None:
    for d in tool_definitions(ctx):
        assert _unions(d["input_schema"]) <= 16, d["name"]


def _client_ctx(ctx) -> AnalystContext:
    from datetime import date

    from gpl.client import EXAMPLE_CLIENTS, ClientProfile, build_ips

    profile = ClientProfile.from_dict(EXAMPLE_CLIENTS["Dubai, moderate, 10 years"])
    profile.annual_income_need_usd = 20_000
    ctx.ips = build_ips(profile, CFG, today=date(2026, 10, 25))
    return ctx


def test_client_profile_tool_reports_the_ips(ctx) -> None:
    with pytest.raises(ToolError, match="no client profile"):
        execute_tool(ctx, "get_client_profile", {})
    out = execute_tool(_client_ctx(ctx), "get_client_profile", {})
    assert (
        out["limits"]["max_volatility"] == "12.00%"
        and out["limits"]["max_weight_per_holding"] == "30.00%"
    )
    assert out["limits"]["cash_to_hold_usd"] == "50,000"
    # 20,000 / (500,000 - 50,000) = 4.44% real, with 2.5% inflation -> 7.06%
    assert out["required_return"] == pct((1 + 20_000 / 450_000) * 1.025 - 1)
    assert "Required return" in out["ips"]["Return"]


def test_calculate_only_uses_numbers_from_tools(ctx) -> None:
    ctx.number_sources = ['{"cash_to_hold_usd": "600,000", "value": "3,000,000", "vol": "9.36%"}']
    out = execute_tool(
        ctx,
        "calculate",
        {"operation": "share_of", "values": [600000, 3000000], "label": "cash share"},
    )
    assert out["result"] == "20.00%" and out["inputs"] == ["600,000", "3e+06"]
    out = execute_tool(
        ctx,
        "calculate",
        {"operation": "subtract", "values": [3000000, 600000], "label": "investable"},
    )
    assert out["result"] == "2,400,000.00"
    out = execute_tool(
        ctx, "calculate", {"operation": "percent_change", "values": [600000, 3000000], "label": "x"}
    )
    assert out["result"] == "400.00%"
    with pytest.raises(ToolError, match="did not come from any tool result"):
        execute_tool(
            ctx, "calculate", {"operation": "add", "values": [600000, 123456], "label": "x"}
        )
    with pytest.raises(ToolError, match="exactly two"):
        execute_tool(
            ctx,
            "calculate",
            {"operation": "divide", "values": [600000, 3000000, 9.36], "label": "x"},
        )


def test_calculate_result_can_be_quoted_in_the_memo(ctx) -> None:
    ctx = _client_ctx(ctx)
    client = FakeClient(
        [
            response([tool_use("t1", "get_client_profile", {})], "tool_use"),
            response(
                [
                    tool_use(
                        "t2",
                        "calculate",
                        {"operation": "share_of", "values": [50000, 500000], "label": "cash"},
                    )
                ],
                "tool_use",
            ),
            response([text("Cash to hold is 10.00% of the portfolio [calculate].")], "end_turn"),
        ]
    )
    ctx.portfolio_value = 500_000
    reply = Analyst(ctx, client, model="m").ask("How much cash must I keep?")
    assert [c.name for c in reply.tool_calls] == ["get_client_profile", "calculate"]
    assert not reply.tool_calls[1].is_error and reply.unverified == []
