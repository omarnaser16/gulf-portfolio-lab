"""Tests for the dashboard's input helpers (app/inputs.py)."""

import sys
from pathlib import Path

import pytest

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


# ------------------------------------------------------------------- workstation holdings

from inputs import guess_group, normalise_holdings  # noqa: E402

CONFIG = {"gcc_equity": ["KSA"], "bonds": ["AGG"], "real_assets": ["GLD"], "global_equity": ["SPY"]}


def test_guess_group() -> None:
    assert guess_group("KSA", "NYSEArca", "ETF", "iShares MSCI Saudi Arabia", CONFIG) == "GCC"
    assert guess_group("2222.SR", "Saudi", "Stock", "Saudi Arabian Oil Co.", CONFIG) == "GCC"
    assert guess_group("XYZ", "DFM", "Stock", "Some Dubai Co", CONFIG) == "GCC"
    assert (
        guess_group("IEF", "NasdaqGM", "ETF", "iShares 7-10 Year Treasury Bond ETF", CONFIG)
        == "Bonds"
    )
    assert guess_group("IAU", "NYSEArca", "ETF", "iShares Gold Trust", CONFIG) == "Gold"
    assert guess_group("AAPL", "NasdaqGS", "Stock", "Apple Inc.", CONFIG) == "Global equity"
    assert guess_group("^GSPC", "SNP", "Index", "S&P 500", CONFIG) == "Other"


def test_normalise_holdings_usd_and_pct() -> None:
    assert normalise_holdings({"SPY": 6000, "AGG": 4000}, "usd") == (
        {"SPY": 0.6, "AGG": 0.4},
        [],
        "",
    )
    w, problems, note = normalise_holdings({"SPY": 50, "AGG": 45, "GLD": 0}, "pct")
    assert w == {"SPY": 50 / 95, "AGG": 45 / 95}
    assert problems == ["GLD: enter a positive weight."]
    assert note == "Weights add up to 95.00%, so they were scaled to 100.00%."
    assert normalise_holdings({}, "usd") == ({}, [], "")


from inputs import equal_values, normalise_values  # noqa: E402


def test_equal_values() -> None:
    assert equal_values(["A", "B", "C"], usd=False, total=0) == pytest.approx(
        {"A": 33.333333, "B": 33.333333, "C": 33.333333}
    )
    assert equal_values(["A", "B"], usd=True, total=10_000) == {"A": 5_000.0, "B": 5_000.0}
    assert equal_values([], usd=False, total=0) == {}


def test_normalise_values() -> None:
    out = normalise_values({"A": 50, "B": 30, "C": 30})  # 110 in total
    assert out == pytest.approx({"A": 45.4545, "B": 27.2727, "C": 27.2727}, abs=1e-4)
    assert sum(out.values()) == pytest.approx(100)
    assert normalise_holdings(out, "pct")[2] == ""  # no "scaled to 100%" note afterwards
    assert normalise_values({"A": 0, "B": 0}) == {"A": 0, "B": 0}  # nothing to scale
    assert normalise_values({"A": 20, "B": 0}) == {"A": 100.0, "B": 0}
