"""Tests for gpl.data. yfinance is always mocked: these tests never touch the network."""

import logging
import math

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

from gpl.data import clean_prices, get_prices, get_risk_free, load_universe, to_returns

# --------------------------------------------------------------------------- to_returns


def test_to_returns_simple_hand_values(prices_small: pd.DataFrame) -> None:
    r = to_returns(prices_small)
    assert r.shape == (9, 2)  # one row shorter: day 1 has no previous price
    assert r.index[0] == pd.Timestamp("2024-01-02")
    assert r.loc["2024-01-02", "A"] == pytest.approx(0.02)  # 102/100 - 1
    assert r.loc["2024-01-02", "B"] == pytest.approx(0.01)  # 50.5/50 - 1
    assert r.loc["2024-01-03", "A"] == pytest.approx(99 / 102 - 1)  # -0.029412
    assert r.loc["2024-01-04", "B"] == pytest.approx(49 / 51 - 1)  # -0.039216


def test_to_returns_simple_compound_to_total_return(prices_small: pd.DataFrame) -> None:
    r = to_returns(prices_small, kind="simple")
    total = (1 + r).prod() - 1
    assert total["A"] == pytest.approx(0.06)  # 106/100 - 1
    assert total["B"] == pytest.approx(0.05)  # 52.5/50 - 1


def test_to_returns_log_hand_values(prices_small: pd.DataFrame) -> None:
    r = to_returns(prices_small, kind="log")
    assert r.loc["2024-01-02", "A"] == pytest.approx(math.log(1.02))  # 0.019803
    # log returns add up over time
    assert r["A"].sum() == pytest.approx(math.log(106 / 100))


def test_to_returns_rejects_unknown_kind(prices_small: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        to_returns(prices_small, kind="arithmetic")


def test_to_returns_does_not_mutate_input(prices_small: pd.DataFrame) -> None:
    before = prices_small.copy()
    to_returns(prices_small)
    pd.testing.assert_frame_equal(prices_small, before)


# ------------------------------------------------------------------------- clean_prices


def test_clean_prices_drops_leading_nans(prices_with_gaps: pd.DataFrame) -> None:
    out = clean_prices(prices_with_gaps)
    # A has no price until 3 Jan, so the first two rows go
    assert out.index[0] == pd.Timestamp("2024-01-03")
    assert len(out) == 10


def test_clean_prices_fills_short_gap(prices_with_gaps: pd.DataFrame) -> None:
    out = clean_prices(prices_with_gaps)
    # A is missing on 5 and 8 Jan (2 days <= 5): carried forward from 101
    assert out.loc["2024-01-05", "A"] == 101.0
    assert out.loc["2024-01-08", "A"] == 101.0
    assert out["A"].notna().all()


def test_clean_prices_respects_ffill_limit(prices_with_gaps: pd.DataFrame) -> None:
    out = clean_prices(prices_with_gaps, max_ffill=5)
    # B is missing for 7 days (4-12 Jan): only the first 5 are filled with 52
    filled = out.loc["2024-01-04":"2024-01-10", "B"]
    assert (filled == 52.0).all()
    assert out.loc["2024-01-11":"2024-01-12", "B"].isna().all()
    assert out.loc["2024-01-15", "B"] == 55.0


def test_clean_prices_bigger_limit_fills_everything(prices_with_gaps: pd.DataFrame) -> None:
    out = clean_prices(prices_with_gaps, max_ffill=10)
    assert not out.isna().any().any()


def test_clean_prices_logs_what_changed(
    prices_with_gaps: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="gpl.data"):
        clean_prices(prices_with_gaps)
    assert any(rec.name == "gpl.data" for rec in caplog.records), "nothing was logged"


def test_clean_prices_leaves_clean_data_alone(prices_small: pd.DataFrame) -> None:
    out = clean_prices(prices_small)
    pd.testing.assert_frame_equal(out, prices_small, check_freq=False)


def test_clean_prices_does_not_mutate_input(prices_with_gaps: pd.DataFrame) -> None:
    before = prices_with_gaps.copy()
    clean_prices(prices_with_gaps)
    pd.testing.assert_frame_equal(prices_with_gaps, before)


# --------------------------------------------------------------------------- get_prices


@pytest.fixture
def fake_yf(monkeypatch: pytest.MonkeyPatch, fake_yf_download):
    """Replace yfinance.download with the fake. Use ``import yfinance as yf`` in data.py."""
    monkeypatch.setattr(yf, "download", fake_yf_download)
    return fake_yf_download


def _same_values(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(actual, expected, check_freq=False, check_names=False)


def test_get_prices_returns_adjusted_close(fake_yf, prices_small, tmp_path) -> None:
    out = get_prices(["A", "B"], start="2024-01-01", cache_dir=tmp_path)
    _same_values(out, prices_small)  # Close, not Open/High/Low
    assert fake_yf.calls[0]["auto_adjust"] is True


def test_get_prices_shape_and_index(fake_yf, tmp_path) -> None:
    out = get_prices(["B", "A"], start="2024-01-01", cache_dir=tmp_path)
    assert list(out.columns) == ["B", "A"]  # flat columns, in the order asked
    assert not isinstance(out.columns, pd.MultiIndex)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is None
    assert out.index.is_monotonic_increasing


def test_get_prices_single_ticker_is_dataframe(fake_yf, prices_small, tmp_path) -> None:
    out = get_prices(["A"], start="2024-01-01", cache_dir=tmp_path)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["A"]
    np.testing.assert_allclose(out["A"].to_numpy(), prices_small["A"].to_numpy())


def test_get_prices_respects_dates(fake_yf, tmp_path) -> None:
    out = get_prices(["A", "B"], start="2024-01-03", end="2024-01-06", cache_dir=tmp_path)
    assert out.index.min() >= pd.Timestamp("2024-01-03")
    assert out.index.max() <= pd.Timestamp("2024-01-06")


def test_get_prices_empty_tickers_raises(fake_yf, tmp_path) -> None:
    with pytest.raises(ValueError):
        get_prices([], start="2024-01-01", cache_dir=tmp_path)


def test_get_prices_unknown_ticker_raises(fake_yf, tmp_path) -> None:
    with pytest.raises(ValueError, match="NOPE"):
        get_prices(["A", "NOPE"], start="2024-01-01", cache_dir=tmp_path)


def test_get_prices_writes_and_reads_cache(fake_yf, tmp_path) -> None:
    first = get_prices(["A", "B"], start="2024-01-01", cache_dir=tmp_path)
    assert list(tmp_path.glob("*.parquet")), "no parquet file written"
    second = get_prices(["A", "B"], start="2024-01-01", cache_dir=tmp_path)
    assert len(fake_yf.calls) == 1, "second call should be served from the cache"
    _same_values(second, first)


def test_get_prices_without_cache_always_downloads(fake_yf, tmp_path) -> None:
    get_prices(["A", "B"], start="2024-01-01", use_cache=False, cache_dir=tmp_path)
    get_prices(["A", "B"], start="2024-01-01", use_cache=False, cache_dir=tmp_path)
    assert len(fake_yf.calls) == 2
    assert not list(tmp_path.glob("*.parquet"))


# ------------------------------------------------------------------------ get_risk_free


@pytest.fixture
def fake_irx(monkeypatch: pytest.MonkeyPatch, fake_irx_download):
    monkeypatch.setattr(yf, "download", fake_irx_download)
    return fake_irx_download


def test_get_risk_free_converts_percent_to_daily(fake_irx, tmp_path) -> None:
    rf = get_risk_free(start="2024-01-01", cache_dir=tmp_path)
    assert isinstance(rf, pd.Series)
    assert rf.name == "rf"
    assert fake_irx.calls[0]["tickers"] == ["^IRX"]
    # 5.0 means 5% a year: daily = 1.05^(1/252) - 1 = 0.00019363
    assert rf.loc["2024-01-01"] == pytest.approx(1.05 ** (1 / 252) - 1)
    assert rf.loc["2024-01-04"] == pytest.approx(1.04 ** (1 / 252) - 1)


def test_get_risk_free_compounds_back_to_annual(fake_irx, tmp_path) -> None:
    rf = get_risk_free(start="2024-01-01", cache_dir=tmp_path)
    assert (1 + rf.loc["2024-01-01"]) ** 252 - 1 == pytest.approx(0.05)


def test_get_risk_free_fills_missing_day(fake_irx, tmp_path) -> None:
    rf = get_risk_free(start="2024-01-01", cache_dir=tmp_path)
    assert len(rf) == 5
    assert rf.notna().all()
    assert rf.loc["2024-01-03"] == rf.loc["2024-01-02"]


# ------------------------------------------------------------------------ load_universe


def test_load_universe_default_config() -> None:
    config = load_universe()
    assert config["assets"] == ["KSA", "UAE", "QAT", "SPY", "EFA", "EEM", "AGG", "TLT", "GLD"]
    assert config["defaults"]["max_weight"] == 0.40
    assert config["non_assets"]["risk_free"] == "^IRX"
