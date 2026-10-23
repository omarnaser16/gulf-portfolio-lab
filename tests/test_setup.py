"""Smoke tests: the package imports and the fixtures have the agreed shape."""

import importlib

import pandas as pd
import pytest

MODULES = [
    "gpl",
    "gpl.data",
    "gpl.metrics",
    "gpl.optimise",
    "gpl.backtest",
    "gpl.montecarlo",
    "gpl.regime",
    "gpl.plots",
    "gpl.agent",
    "gpl.agent.tools",
    "gpl.agent.analyst",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_prices_small_shape(prices_small: pd.DataFrame) -> None:
    assert prices_small.shape == (10, 2)
    assert list(prices_small.columns) == ["A", "B"]
    assert isinstance(prices_small.index, pd.DatetimeIndex)
    assert prices_small.index.tz is None
    assert prices_small.index.is_monotonic_increasing
    assert not prices_small.isna().any().any()


def test_prices_with_gaps_shape(prices_with_gaps: pd.DataFrame) -> None:
    assert prices_with_gaps.shape == (12, 2)
    assert prices_with_gaps["A"].isna().sum() == 4
    assert prices_with_gaps["B"].isna().sum() == 7


def test_fake_yf_download_mimics_yfinance(fake_yf_download) -> None:
    out = fake_yf_download(["A", "B"], start="2024-01-01", end="2024-01-05")
    assert out.columns.names == ["Price", "Ticker"]
    assert len(out) == 4  # end is exclusive
    assert out[("Close", "A")].iloc[0] == 100.0
    assert len(fake_yf_download.calls) == 1
