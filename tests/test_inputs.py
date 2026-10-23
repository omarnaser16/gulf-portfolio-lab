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
