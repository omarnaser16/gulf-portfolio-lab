"""Tests for the dashboard's input helpers (app/inputs.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from inputs import asset_groups, combine_tickers, parse_tickers  # noqa: E402


def test_parse_tickers_splits_cleans_and_dedupes() -> None:
    assert parse_tickers(" aapl, 2222.sr;VOD.L  aapl\n emaar.ae ") == [
        "AAPL", "2222.SR", "VOD.L", "EMAAR.AE",
    ]  # fmt: skip
    assert parse_tickers("") == [] and parse_tickers(None) == []


def test_combine_keeps_order_without_duplicates() -> None:
    assert combine_tickers([["SPY", "AGG"], ["AGG", "GLD"]], ["SPY", "2222.SR"]) == [
        "SPY", "AGG", "GLD", "2222.SR",
    ]  # fmt: skip


def test_asset_groups() -> None:
    config = {"bonds": ["AGG", "TLT"], "gcc_equity": ["KSA"]}
    labels = {"bonds": "Bonds", "gcc_equity": "GCC equity"}
    groups = asset_groups(["AGG", "KSA", "2222.SR", "NBK.KW", "AAPL"], config, labels)
    assert groups == {
        "Bonds": ["AGG"],
        "GCC equity": ["KSA"],
        "GCC stocks": ["2222.SR", "NBK.KW"],
        "Other": ["AAPL"],
    }


from inputs import parse_holdings  # noqa: E402


def test_parse_holdings_weights() -> None:
    weights, problems, note = parse_holdings("SPY 40\n2222.sr: 30%\nGLD,30")
    assert weights == {"SPY": 0.4, "2222.SR": 0.3, "GLD": 0.3}
    assert problems == [] and note == ""


def test_parse_holdings_rescales_and_reports() -> None:
    weights, problems, note = parse_holdings("SPY 50\nAGG 45\nnonsense line\nGLD -5\n\nSPY 0")
    assert weights == {"SPY": 50 / 95, "AGG": 45 / 95}
    assert len(problems) == 3 and "Line 3" in problems[0]
    assert note == "Weights add up to 95.00%, so they were scaled to 100.00%."


def test_parse_holdings_amounts_and_duplicates() -> None:
    weights, problems, note = parse_holdings("SPY 10,000\nAGG 5000\nSPY 5,000", mode="amounts")
    assert weights == {"SPY": 0.75, "AGG": 0.25}
    assert problems == [] and note == ""  # amounts are never "rescaled"


def test_parse_holdings_empty() -> None:
    assert parse_holdings("") == ({}, [], "")
