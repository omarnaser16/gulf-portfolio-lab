"""Download, cache, clean and transform price data.

This is the only module in ``gpl`` allowed to do network or disk I/O.

Conventions
-----------
Prices are adjusted closes in USD: a ``pd.DataFrame`` with a tz-naive ``DatetimeIndex``
and one column per ticker. Returns are simple daily returns unless stated otherwise.
"""

import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd

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
        If ``tickers`` is empty, or if no data is returned for any ticker.
    """
    raise NotImplementedError


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

    Notes
    -----
    Never drops data silently: the number of leading rows dropped, the number of cells
    forward-filled and any NaNs remaining are logged at WARNING level through the
    module logger.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
