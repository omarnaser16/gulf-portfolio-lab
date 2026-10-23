"""Tests for the agent: tools wrap gpl functions; the loop runs tools and audits numbers.

No network: tools run on synthetic data and the model is a scripted fake client.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gpl import regime as rg
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
from gpl.optimise import min_variance

CFG = load_universe()


@pytest.fixture(scope="module")
def ctx() -> AnalystContext:
    """Synthetic 9-ETF universe over ~6 years, plus synthetic regime data."""
    idx = pd.bdate_range("2015-01-01", periods=1600)
    rng = np.random.default_rng(11)
    vols = np.linspace(0.004, 0.015, len(CFG["assets"]))
    stress = 1 + 2 * (np.sin(np.arange(len(idx)) / 40) > 0.85)  # a stress spell every ~250 days
    rets = rng.normal(0.0003, vols, size=(len(idx), len(vols))) * stress[:, None]
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=CFG["assets"])
    vix = pd.Series(
        15 + 300 * pd.Series(rets[:, 3]).rolling(10, min_periods=1).std().to_numpy(), index=idx
    )
    settings = {
        "max_weight": 0.40,
        "cost_bps": 10.0,
        "lookback": 126,
        "rebalance": "ME",
        "var_level": 0.95,
        "regime": {**CFG["regime"], "min_train": 300},
        "montecarlo": CFG["montecarlo"],
    }
    return AnalystContext(
        prices=prices,
        rf=pd.Series((1.02) ** (1 / 252) - 1, index=idx),
        groups=CFG["groups"],
        benchmarks=CFG["benchmarks"],
        regime_features=rg.make_features(prices, vix),
        regime_labels=rg.make_labels(prices["SPY"].pct_change().iloc[1:]),
        settings=settings,
    )


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
