"""The plain-English summary card is written by code from the report's facts."""

from gpl.recommend import KEEP
from gpl.summary import GLOSSARY, change_text, kpi_comparison, result_summary, window_note


def _build_report(status="next_best", choice="Max IR · low-vol tilt", ranking=()):
    rows = {
        "Current": {"cagr": "7.10%", "annual_volatility": "11.42%", "sharpe": "0.32",
                    "max_drawdown": "-17.69%"},
        "Max IR": {"cagr": "8.09%", "annual_volatility": "11.71%", "sharpe": "0.39",
                   "max_drawdown": "-19.30%"},
        "Max IR · low-vol tilt": {"cagr": "9.21%", "annual_volatility": "11.52%",
                                  "sharpe": "0.49", "max_drawdown": "-19.12%"},
        "Benchmark tracker": {"cagr": "6.59%", "annual_volatility": "10.52%", "sharpe": "0.29",
                              "max_drawdown": "-20.53%"},
        "Diversified (IPS)": {"cagr": "10.27%", "annual_volatility": "12.20%", "sharpe": "0.54",
                              "max_drawdown": "-20.86%"},
        "Benchmark": {"cagr": "6.71%", "annual_volatility": "11.61%", "sharpe": "0.28",
                      "max_drawdown": "-20.27%"},
    }  # fmt: skip
    return {
        "mode": "build",
        "universe": {"eligible": 54, "excluded": {f"X{i}": "why" for i in range(8)}},
        "turnover_limit_per_rebalance": "25.00% a quarter after the first build",
        "walk_forward_after_costs": rows,
        "backtest_period": "2022-01-04 to 2026-11-05",
        "benchmark": {"name": "60/40 (SPY/AGG)"},
        "recommendation": {
            "choice": choice,
            "status": status,
            "reasons": ["No built portfolio beat the benchmark…", "Next best option: …"],
            "ranking_under_the_rule": list(ranking),
        },
    }


def test_change_text_largest_moves_first() -> None:
    now = {"SPY": 0.45, "EEM": 0.10, "AGG": 0.45}
    new = {"SPY": 0.30, "EEM": 0.05, "AGG": 0.55, "GLD": 0.10}
    assert change_text(now, new) == (
        "Sell 15.00% SPY and 5.00% EEM; buy 10.00% AGG and 10.00% GLD (turnover 40.00%)."
    )
    assert change_text(now, now) == "No trades needed."
    # a change too small to list still counts in the turnover, which matches the report's
    tiny = change_text({"A": 0.5, "B": 0.5}, {"A": 0.4, "B": 0.4998, "C": 0.1002})
    assert tiny == "Sell 10.00% A; buy 10.02% C (turnover 20.04%)."
    assert change_text(now, None) == "No trades needed."
    many = {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}
    text = change_text(many, {"F": 1.0}, max_moves=2)
    assert "buy 100.00% F; plus 4 smaller changes (turnover 200.00%)." in text


def test_build_summary_card() -> None:
    s = result_summary(
        _build_report(), {"SPY": 0.6, "AGG": 0.4}, {"SPY": 0.3, "AGG": 0.4, "USMV": 0.3}
    )
    assert s.did[0] == "Screened 62 securities against the client's IPS: 54 eligible."
    assert s.did[1].startswith("Built 3 candidate portfolios (plus a benchmark tracker)")
    assert "25.00% a quarter" in s.did[1]
    assert s.did[2] == (
        "Tested each one month by month from 2022-01-04 to 2026-11-05, after trading costs, "
        "against 60/40 (SPY/AGG)."
    )
    assert s.found == (
        "None of the 4 portfolios beat 60/40 (SPY/AGG) reliably after costs within every test."
    )
    assert s.recommend == "Next best option: …"  # the rule's own reason, not a re-wording
    assert s.changes == "Sell 30.00% SPY; buy 30.00% USMV (turnover 60.00%)."


def test_summary_for_keep_and_recommended() -> None:
    keep = result_summary(_build_report("keep", KEEP), {"SPY": 1.0}, None)
    assert keep.changes == "No trades needed." and keep.recommend.startswith("Keep your current")
    rec = result_summary(
        _build_report("recommended", "Max IR", ["Max IR"]), {"SPY": 1.0}, {"SPY": 1.0}
    )
    assert rec.found.startswith("1 of the 4 portfolios beat 60/40 (SPY/AGG) reliably")
    assert rec.recommend.startswith("Move to Max IR.")


def test_kpi_comparison_arrows() -> None:
    kpis = {k["label"]: k for k in kpi_comparison(_build_report(), "Max IR · low-vol tilt")}
    assert kpis["Return (CAGR)"]["change"] == "+2.11 pp" and kpis["Return (CAGR)"]["better"]
    assert kpis["Volatility"]["change"] == "+0.10 pp" and kpis["Volatility"]["better"] is False
    assert kpis["Sharpe ratio"]["change"] == "+0.17" and kpis["Sharpe ratio"]["better"]
    assert kpis["Max drawdown"]["change"] == "-1.43 pp"
    assert kpis["Max drawdown"]["better"] is False  # a deeper fall is worse
    same = kpi_comparison(_build_report(), "Current")
    assert all(k["change"] in ("+0.00 pp", "+0.00") and k["better"] is None for k in same)


def test_window_note_and_glossary() -> None:
    note = window_note("build", "4 Jan 2022 – 5 Nov 2026")
    assert note.startswith("Figures here cover 4 Jan 2022 – 5 Nov 2026.")
    assert "Analyse" in note and "Improve" in note and "Build" in note
    assert set(GLOSSARY) == {
        "Sharpe ratio", "Information ratio", "Tracking error", "Held up out of sample", "Turnover",
    }  # fmt: skip
    assert all(text.count(".") <= 2 for text in GLOSSARY.values())  # one plain sentence each


def test_keep_reason_skips_the_ips_note() -> None:
    from gpl.summary import main_reason

    reasons = ["None of the qualifying methods held up out of sample.",
               "Your current portfolio passes every IPS rule."]  # fmt: skip
    assert main_reason(reasons) == "None of the qualifying methods held up out of sample."
    assert main_reason(["so keep it."]) == "so keep it."


def test_improve_found_says_when_the_current_portfolio_breaks_the_ips() -> None:
    report = {
        "mode": "improve",
        "goal": "Maximise risk-adjusted return (Sharpe)",
        "max_turnover": "30.00%",
        "walk_forward_after_costs": {"Current": {}, "Target volatility": {}, "Benchmark": {}},
        "backtest_period": "2021-11-02 to 2026-11-05",
        "recommendation": {"choice": "Target volatility", "status": "recommended",
                           "reasons": ["Your current portfolio fails the client's IPS…"]},
        "suitability_gate": {"current_passes_ips": False},
    }  # fmt: skip
    s = result_summary(report, {"SPY": 0.45, "AGG": 0.55}, {"SPY": 0.30, "AGG": 0.70})
    assert s.found.startswith("Your current portfolio breaks the client's IPS")
    report["suitability_gate"]["current_passes_ips"] = True
    s = result_summary(report, {"SPY": 0.45, "AGG": 0.55}, {"SPY": 0.30, "AGG": 0.70})
    assert s.found.startswith("Target volatility did better")


def test_compliant_move_says_the_full_move_is_not_recommended() -> None:
    why = "the full portfolio fails the IPS over the full history (Volatility)"
    report = {
        "mode": "improve",
        "goal": "Maximise risk-adjusted return (Sharpe)",
        "max_turnover": "30.00%",
        "walk_forward_after_costs": {"Current": {}, "Max Sharpe": {}, "Benchmark": {}},
        "backtest_period": "2017-05-01 to 2026-11-05",
        "recommendation": {"choice": "Max Sharpe", "status": "compliant_move",
                           "reasons": ["Max Sharpe ranks first…"],
                           "full_move_not_recommended_because": [why]},
        "suitability_gate": {"current_passes_ips": True},
    }  # fmt: skip
    s = result_summary(report, {"SPY": 0.45, "AGG": 0.55}, {"SPY": 0.50, "AGG": 0.50})
    assert s.found == (
        "Max Sharpe did better under the rule for your goal, but the full move fails the IPS: "
        f"{why}."
    )
    assert s.recommend.startswith("Move part-way towards Max Sharpe: the closest compliant move")
