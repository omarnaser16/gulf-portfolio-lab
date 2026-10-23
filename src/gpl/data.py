"""Download, cache, clean and transform price data.

This is the only module in ``gpl`` allowed to do network or disk I/O.

Conventions
-----------
Prices are adjusted closes in USD: a ``pd.DataFrame`` with a tz-naive ``DatetimeIndex``
and one column per ticker. Returns are simple daily returns unless stated otherwise.
"""

import logging
import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def get_prices(
    tickers: Sequence[str],
    start: str | date,
    end: str | date | None = None,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance, with a parquet cache.

    Parameters
    ----------
    tickers : Sequence[str]
        Yahoo Finance tickers, e.g. ``["SPY", "AGG", "KSA"]``.
    start : str or datetime.date
        First date to include (inclusive), e.g. ``"2015-01-01"``.
    end : str or datetime.date, optional
        Last date to include. ``None`` means up to the latest available date.
    use_cache : bool, default True
        If True, read from and write to parquet files in ``cache_dir`` so that repeated
        calls do not hit the network.
    cache_dir : pathlib.Path, optional
        Directory for parquet files. Defaults to ``DEFAULT_CACHE_DIR`` (``data/cache/``).

    Returns
    -------
    pd.DataFrame
        Adjusted close prices. Tz-naive ``DatetimeIndex`` sorted ascending, one column
        per ticker in the order given. NaNs are left in place for ``clean_prices``.

    Raises
    ------
    ValueError
        If ``tickers`` is empty, or if any ticker comes back with no data
        (yfinance returns an all-NaN column instead of raising).
    """
    tickers = list(tickers)
    if not tickers:
        raise ValueError("tickers is empty")

    cache_path = (cache_dir or DEFAULT_CACHE_DIR) / _cache_filename(tickers, start, end)
    if use_cache and cache_path.exists():
        logger.info("Loaded %s from cache", cache_path.name)
        return pd.read_parquet(cache_path)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no data for {tickers} from {start} to {end}")

    # yfinance columns are (Price field, Ticker); with auto_adjust=True "Close" is adjusted.
    prices = raw["Close"]
    if isinstance(prices, pd.Series):  # defensive: older yfinance flattened one ticker
        prices = prices.to_frame(name=tickers[0])
    prices = prices.reindex(columns=tickers)

    missing = [t for t in tickers if prices[t].isna().all()]
    if missing:
        raise ValueError(f"No data returned for: {missing}")

    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.sort_index()
    prices.index.name = "date"
    prices.columns.name = None

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)
    return prices


def _cache_filename(tickers: list[str], start: str | date, end: str | date | None) -> str:
    """Parquet filename for one request; an open ``end`` is keyed by today (daily refresh)."""
    safe = "-".join(re.sub(r"[^A-Za-z0-9.]", "_", t) for t in tickers)
    end_key = end if end is not None else date.today()
    return f"{safe}_{start}_{end_key}.parquet"


def clean_prices(prices: pd.DataFrame, max_ffill: int = 5) -> pd.DataFrame:
    """Fill short gaps, drop leading NaNs and report what changed.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw prices as returned by ``get_prices``.
    max_ffill : int, default 5
        Maximum number of consecutive missing days to forward-fill per column.
        Longer gaps are only filled up to this limit.

    Returns
    -------
    pd.DataFrame
        Cleaned prices, starting on the first date where every column has a price.
        Gaps longer than ``max_ffill`` keep their remaining NaNs.

    Raises
    ------
    ValueError
        If a column has no prices at all.

    Notes
    -----
    Never drops data silently: the number of leading rows dropped, the number of cells
    forward-filled and any NaNs remaining are logged at WARNING level through the
    module logger.
    """
    if prices.empty:
        return prices.copy()

    first_valid = prices.apply(lambda col: col.first_valid_index())
    empty_cols = first_valid[first_valid.isna()].index.tolist()
    if empty_cols:
        raise ValueError(f"Columns with no prices at all: {empty_cols}")

    filled = prices.ffill(limit=max_ffill)
    n_filled = int(filled.notna().sum().sum() - prices.notna().sum().sum())

    # Start once every asset has traded; earlier rows can't be part of a common sample.
    start = first_valid.max()
    n_dropped = int((prices.index < start).sum())
    cleaned = filled.loc[start:]

    if n_dropped:
        logger.warning(
            "Dropped %d leading rows before %s (not all assets trading)", n_dropped, start.date()
        )
    if n_filled:
        logger.warning(
            "Forward-filled %d missing prices (max %d days in a row)", n_filled, max_ffill
        )
    remaining = cleaned.isna().sum()
    remaining = remaining[remaining > 0]
    if not remaining.empty:
        logger.warning(
            "NaNs left after a gap longer than %d days: %s", max_ffill, remaining.to_dict()
        )
    return cleaned


def to_returns(
    prices: pd.DataFrame,
    kind: Literal["simple", "log"] = "simple",
) -> pd.DataFrame:
    """Convert prices to daily returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Cleaned prices.
    kind : {"simple", "log"}, default "simple"
        ``"simple"``: :math:`r_t = P_t / P_{t-1} - 1`.
        ``"log"``: :math:`r_t = \\ln(P_t / P_{t-1})`.

    Returns
    -------
    pd.DataFrame
        Returns with the same columns as ``prices``, one row shorter (the first date
        has no previous price).

    Raises
    ------
    ValueError
        If ``kind`` is not ``"simple"`` or ``"log"``.
    """
    if kind not in ("simple", "log"):
        raise ValueError(f"kind must be 'simple' or 'log', got {kind!r}")

    ratio = prices / prices.shift(1)
    returns = ratio - 1 if kind == "simple" else np.log(ratio)
    return returns.iloc[1:]


def get_risk_free(
    start: str | date,
    end: str | date | None = None,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> pd.Series:
    """Daily risk-free rate derived from the 13-week T-bill yield (``^IRX``).

    Parameters
    ----------
    start, end, use_cache, cache_dir
        As in ``get_prices``.

    Returns
    -------
    pd.Series
        Daily risk-free rate as a decimal, named ``"rf"``. ``^IRX`` is quoted as an
        annual percentage (``5.2`` means 5.2%), so it is divided by 100 and converted
        with ``(1 + rf) ** (1 / 252) - 1``.
    """
    raise NotImplementedError
