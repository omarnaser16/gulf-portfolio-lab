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
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
DEFAULT_UNIVERSE = PROJECT_ROOT / "config" / "universe.yaml"
RISK_FREE_TICKER = "^IRX"
PERIODS_PER_YEAR = 252


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
        Last date to include (inclusive, unlike yfinance). ``None`` means up to the
        latest available date.
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

    # yfinance's end is exclusive; ours is inclusive, so ask for one extra day.
    yf_end = None if end is None else pd.Timestamp(end) + pd.Timedelta(days=1)
    raw = yf.download(tickers, start=start, end=yf_end, auto_adjust=True, progress=False)
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
        with ``(1 + rf) ** (1 / 252) - 1``. Missing days are forward-filled as in
        ``clean_prices``.

    Notes
    -----
    ``^IRX`` is a discount yield rather than a true compounding yield; the difference is a
    few basis points and is ignored here.
    """
    irx = clean_prices(get_prices([RISK_FREE_TICKER], start, end, use_cache, cache_dir))
    annual = irx[RISK_FREE_TICKER] / 100
    daily = (1 + annual) ** (1 / PERIODS_PER_YEAR) - 1
    return daily.rename("rf")


def load_universe(path: Path | None = None) -> dict:
    """Read the universe and defaults from ``config/universe.yaml``.

    Parameters
    ----------
    path : pathlib.Path, optional
        YAML file to read. Defaults to ``DEFAULT_UNIVERSE``.

    Returns
    -------
    dict
        The parsed config, plus ``"assets"``: every ticker in ``groups``, flattened in
        file order.
    """
    with open(path or DEFAULT_UNIVERSE) as f:
        config = yaml.safe_load(f)
    config["assets"] = [t for tickers in config["groups"].values() for t in tickers]
    return config


# ------------------------------------------------------------------ any asset, any currency

# Local currency units per 1 USD. These Gulf currencies are pegged to the dollar; Yahoo's FX
# series for them are noisy (e.g. SAR=X is >1% off the peg on ~100 days since 2015), so the
# official peg is used instead of market data.
PEGS = {"SAR": 3.75, "AED": 3.6725, "QAR": 3.64, "BHD": 0.376, "OMR": 0.3845}
# Yahoo quotes some markets in a sub-unit: Kuwaiti fils, UK pence, SA cents, Israeli agorot.
SUBUNITS = {
    "KWF": ("KWD", 1000),
    "GBp": ("GBP", 100),
    "GBX": ("GBP", 100),
    "ZAc": ("ZAR", 100),
    "ILA": ("ILS", 100),
}
# Fallback when Yahoo's currency lookup fails: exchange suffix -> quote currency.
SUFFIX_CURRENCY = {
    ".SR": "SAR",
    ".AE": "AED",
    ".AD": "AED",
    ".QA": "QAR",
    ".KW": "KWF",
    ".BH": "BHD",
    ".OM": "OMR",
    ".L": "GBp",
}
RETRY_DELAYS = (2.0, 5.0)  # seconds between download attempts when Yahoo blocks or errors
TICKER_PATTERN = re.compile(r"^[A-Za-z0-9^][A-Za-z0-9.\-=^]{0,19}$")


class DataUnavailable(RuntimeError):
    """Yahoo Finance could not be reached or refused the request (e.g. rate limiting)."""


def _download_error(ticker: str) -> str:
    """The error yfinance recorded for ``ticker`` in its last download ("" if none)."""
    try:
        from yfinance import shared

        return str(shared._ERRORS.get(ticker, ""))
    except Exception:  # private API; absence just means "unknown"
        return ""


def _looks_blocked(ticker: str) -> bool:
    message = _download_error(ticker).lower()
    return any(s in message for s in ("ratelimit", "too many requests", "connection", "timed out"))


def download_series(
    ticker: str, start: str | date, end: str | date | None, cache_dir: Path | None = None
) -> pd.Series | None:
    """One ticker's adjusted close (cached per ticker); ``None`` if Yahoo has no data for it.

    Retries with backoff when Yahoo appears to block the request, then raises
    ``DataUnavailable``, so "this ticker doesn't exist" and "Yahoo is refusing us" stay distinct.
    """
    import time

    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            return get_prices([ticker], start, end, cache_dir=cache_dir)[ticker]
        except ValueError:
            if not _looks_blocked(ticker):
                return None
            if attempt < len(RETRY_DELAYS):
                logger.warning("Yahoo blocked %s; retrying in %.0fs", ticker, RETRY_DELAYS[attempt])
                time.sleep(RETRY_DELAYS[attempt])
    raise DataUnavailable(f"Yahoo Finance did not respond for {ticker}: {_download_error(ticker)}")


def get_currency(ticker: str, cache_dir: Path | None = None) -> tuple[str, str]:
    """Quote currency of ``ticker`` and where it came from ("us listing", "yahoo", "suffix").

    US listings (no exchange suffix) are USD without a network call. Others are looked up on
    Yahoo once and cached in ``currencies.json``; if that fails, the exchange suffix decides.
    """
    import json

    if "." not in ticker and "=" not in ticker:
        return "USD", "us listing"
    path = (cache_dir or DEFAULT_CACHE_DIR) / "currencies.json"
    known = json.loads(path.read_text()) if path.exists() else {}
    if ticker in known:
        return known[ticker], "yahoo"
    try:
        currency = yf.Ticker(ticker).fast_info["currency"]
    except Exception:
        currency = None
    if currency:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**known, ticker: currency}, indent=1))
        return currency, "yahoo"
    suffix = "." + ticker.rsplit(".", 1)[-1].upper()
    return SUFFIX_CURRENCY.get(suffix, "USD"), "suffix"


def to_usd(
    local: pd.DataFrame, currencies: dict[str, str], fx: dict[str, pd.Series]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Convert local-currency prices to USD.

    Parameters
    ----------
    local : pd.DataFrame
        Prices in each asset's quote currency (NaN where an asset didn't trade).
    currencies : dict
        Ticker -> Yahoo quote currency (may be a sub-unit such as ``"KWF"``).
    fx : dict
        Currency -> Series of local units per 1 USD (Yahoo ``C=X``), for floating currencies.

    Returns
    -------
    (pd.DataFrame, dict)
        USD prices, and ticker -> description of the conversion used.

    Notes
    -----
    Sub-units are scaled first (851 fils = 0.851 KWD). Pegged currencies use ``PEGS``. Market FX
    is carried forward up to 5 days onto the asset's dates, so a Sunday price in Kuwait uses
    Friday's rate.
    """
    usd, notes = local.copy(), {}
    for ticker in local.columns:
        code = currencies.get(ticker, "USD")
        base, divisor = SUBUNITS.get(code, (code, 1))
        prices = local[ticker] / divisor
        unit = f" ({code} = {base}/{divisor})" if divisor != 1 else ""
        if base == "USD":
            notes[ticker] = "USD" + unit
        elif base in PEGS:
            prices = prices / PEGS[base]
            notes[ticker] = f"{base} peg {PEGS[base]:g} per USD{unit}"
        else:
            if base not in fx:
                raise ValueError(f"no FX rate for {base} (needed for {ticker})")
            rate = fx[base]
            rate = rate.reindex(rate.index.union(prices.index)).ffill(limit=5).reindex(prices.index)
            prices = prices / rate
            notes[ticker] = f"{base} market rate ({base}=X){unit}"
        usd[ticker] = prices
    return usd, notes


def trading_weekdays(series: pd.Series) -> frozenset[int]:
    """Weekdays (Mon=0 ... Sun=6) on which an asset trades on at least 10% of weeks."""
    days = series.dropna().index
    if len(days) == 0:
        return frozenset()
    counts = pd.Series(days.day_of_week).value_counts()
    weeks = max(1, len(days) / 5)
    return frozenset(int(d) for d, c in counts.items() if c >= 0.1 * weeks)


def align_common_dates(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep only dates on which every asset traded (no forward-filling across calendars).

    Returns the aligned prices and a report: rows kept and dropped, and for each asset how many
    of the other assets' trading dates it was missing.
    """
    start = prices.apply(lambda c: c.first_valid_index()).max()
    window = prices.loc[start:]
    aligned = window.dropna()
    report = {
        "kept": len(aligned),
        "dropped": len(window) - len(aligned),
        "missing_by_asset": {t: int(window[t].isna().sum()) for t in window.columns},
    }
    if report["dropped"]:
        logger.warning("Kept %d common dates, dropped %d", report["kept"], report["dropped"])
    return aligned, report


def observations_per_year(index: pd.DatetimeIndex) -> int:
    """Observed price dates per year (used to annualise when calendars were aligned)."""
    years = (index[-1] - index[0]).days / 365.25
    return int(round((len(index) - 1) / years)) if years > 0 else PERIODS_PER_YEAR


@dataclass
class AssetData:
    """USD prices ready for analysis, plus everything the user should be told about them."""

    prices: pd.DataFrame
    history: pd.DataFrame
    invalid: dict[str, str]
    mixed_calendars: bool
    periods_per_year: int
    alignment: dict
    price_as_of: pd.Timestamp | None
    loaded_at: str


def load_assets(
    tickers: Sequence[str],
    start: str | date,
    end: str | date | None,
    lookback: int = 252,
    max_ffill: int = 5,
    cache_dir: Path | None = None,
) -> AssetData:
    """Download any Yahoo tickers, convert to USD and align calendars.

    Tickers with no data are reported in ``invalid`` rather than raising. Raises
    ``DataUnavailable`` if Yahoo blocks the requests. If every asset trades on the same
    weekdays, prices are cleaned exactly as ``clean_prices`` does (so results for the default
    US ETFs are unchanged); if calendars differ (e.g. Saudi Sun-Thu vs US Mon-Fri), only the
    dates on which every asset traded are kept and annualisation uses the observed number of
    dates per year.
    """
    from datetime import UTC, datetime

    invalid: dict[str, str] = {}
    raw: dict[str, pd.Series] = {}
    currencies: dict[str, str] = {}
    sources: dict[str, str] = {}
    for t in dict.fromkeys(tickers):  # de-duplicate, keep order
        if not TICKER_PATTERN.match(t):
            invalid[t] = "Not a valid ticker format"
            continue
        series = download_series(t, start, end, cache_dir)
        if series is None or series.dropna().empty:
            invalid[t] = "No data on Yahoo Finance"
            continue
        raw[t] = series.dropna()
        currencies[t], sources[t] = get_currency(t, cache_dir)

    fx: dict[str, pd.Series] = {}
    for code in sorted({SUBUNITS.get(c, (c, 1))[0] for c in currencies.values()}):
        if code in ("USD", *PEGS):
            continue
        rate = download_series(f"{code}=X", start, end, cache_dir)
        if rate is None:
            for t in [t for t, c in currencies.items() if SUBUNITS.get(c, (c, 1))[0] == code]:
                invalid[t] = f"No {code}=X exchange rate on Yahoo Finance"
                raw.pop(t)
        else:
            fx[code] = rate.dropna()

    loaded_at = datetime.now(UTC).strftime("%d %b %Y %H:%M UTC")
    if not raw:
        empty = pd.DataFrame()
        return AssetData(empty, empty, invalid, False, PERIODS_PER_YEAR, {}, None, loaded_at)

    local = pd.DataFrame(raw)
    usd, conversions = to_usd(local, {t: currencies[t] for t in raw}, fx)
    calendars = {t: trading_weekdays(usd[t]) for t in usd.columns}
    mixed = len(set(calendars.values())) > 1
    if mixed:
        prices, alignment = align_common_dates(usd)
        periods = observations_per_year(prices.index)
    else:
        prices, alignment, periods = clean_prices(usd, max_ffill=max_ffill), {}, PERIODS_PER_YEAR

    day_names = "Mon Tue Wed Thu Fri Sat Sun".split()
    history = pd.DataFrame(
        {
            "first_date": [raw[t].index[0] for t in raw],
            "last_date": [raw[t].index[-1] for t in raw],
            "observations": [len(raw[t]) for t in raw],  # the asset's own trading days
            "currency": [currencies[t] for t in raw],
            "currency_source": [sources[t] for t in raw],
            "conversion": [conversions[t] for t in raw],
            "trading_days": [" ".join(day_names[d] for d in sorted(calendars[t])) for t in raw],
        },
        index=list(raw),
    )
    # Too short for one full estimation window plus a month of results on its own history.
    history["too_short"] = history["observations"] < lookback + 21
    history["sets_start"] = history["first_date"] == history["first_date"].max()
    return AssetData(
        prices,
        history,
        invalid,
        mixed,
        periods,
        alignment,
        prices.index[-1] if len(prices) else None,
        loaded_at,
    )
