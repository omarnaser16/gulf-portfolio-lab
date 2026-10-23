"""Synthetic price fixtures small enough to check by hand in Excel.

``prices_small`` (10 business days, 2-12 Jan 2024)::

    date        A      B
    2024-01-01  100.0  50.0
    2024-01-02  102.0  50.5
    2024-01-03   99.0  51.0
    2024-01-04  101.0  49.0
    2024-01-05  103.0  49.5
    2024-01-08  100.0  50.0
    2024-01-09  104.0  51.0
    2024-01-10  105.0  50.0
    2024-01-11  103.0  52.0
    2024-01-12  106.0  52.5

First simple returns: A = 102/100 - 1 = 0.02, B = 50.5/50 - 1 = 0.01.
Total return over the period: A = 106/100 - 1 = 0.06, B = 52.5/50 - 1 = 0.05.
"""

import numpy as np
import pandas as pd
import pytest

DATES_10 = pd.bdate_range("2024-01-01", periods=10)
DATES_12 = pd.bdate_range("2024-01-01", periods=12)

PRICES_A = [100.0, 102.0, 99.0, 101.0, 103.0, 100.0, 104.0, 105.0, 103.0, 106.0]
PRICES_B = [50.0, 50.5, 51.0, 49.0, 49.5, 50.0, 51.0, 50.0, 52.0, 52.5]


@pytest.fixture
def prices_small() -> pd.DataFrame:
    """Clean prices: 2 assets x 10 business days, no NaNs."""
    return pd.DataFrame({"A": PRICES_A, "B": PRICES_B}, index=DATES_10)


@pytest.fixture
def prices_with_gaps() -> pd.DataFrame:
    """Prices with the gaps ``clean_prices`` must handle (12 business days).

    - ``A``: 2 leading NaNs (asset not yet listed), then a 2-day interior gap.
    - ``B``: a 7-day interior gap, longer than the default ``max_ffill=5``.

    ::

        date        A      B
        2024-01-01  NaN    50.0
        2024-01-02  NaN    51.0
        2024-01-03  100.0  52.0
        2024-01-04  101.0  NaN
        2024-01-05  NaN    NaN
        2024-01-08  NaN    NaN
        2024-01-09  103.0  NaN
        2024-01-10  104.0  NaN
        2024-01-11  105.0  NaN
        2024-01-12  106.0  NaN
        2024-01-15  107.0  55.0
        2024-01-16  108.0  56.0
    """
    nan = np.nan
    a = [nan, nan, 100.0, 101.0, nan, nan, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]
    b = [50.0, 51.0, 52.0, nan, nan, nan, nan, nan, nan, nan, 55.0, 56.0]
    return pd.DataFrame({"A": a, "B": b}, index=DATES_12)


class FakeYfDownload:
    """Stand-in for ``yfinance.download`` that never touches the network.

    Mimics ``yf.download(..., auto_adjust=True)``: columns are a ``MultiIndex`` of
    ``(Price, Ticker)`` with Price in Close/High/Low/Open/Volume, even for one ticker.
    ``Close`` holds the adjusted prices from ``prices_small``; Open/High/Low are
    deliberately different so a test can tell which field was used. Unknown tickers
    come back as all-NaN columns, like a failed yfinance download. ``end`` is exclusive.

    Every call is recorded in ``self.calls`` so tests can assert on cache hits.
    """

    def __init__(self, prices: pd.DataFrame) -> None:
        self.prices = prices
        self.calls: list[dict] = []

    def __call__(
        self,
        tickers: str | list[str],
        start=None,
        end=None,
        auto_adjust: bool = True,
        progress: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers = tickers.split()
        self.calls.append(
            {"tickers": list(tickers), "start": start, "end": end, "auto_adjust": auto_adjust}
        )

        close = self.prices.reindex(columns=list(tickers))
        if start is not None:
            close = close[close.index >= pd.Timestamp(start)]
        if end is not None:
            close = close[close.index < pd.Timestamp(end)]

        fields = {
            "Close": close,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Open": close * 0.99,
            "Volume": close * 0 + 1_000,
        }
        out = pd.concat(fields, axis=1, names=["Price", "Ticker"])
        out.index.name = "Date"
        return out


@pytest.fixture
def fake_yf_download(prices_small: pd.DataFrame) -> FakeYfDownload:
    """A ``FakeYfDownload`` serving ``prices_small``. Use with ``monkeypatch``."""
    return FakeYfDownload(prices_small)


@pytest.fixture
def fake_irx_download() -> FakeYfDownload:
    """A ``FakeYfDownload`` serving ``^IRX`` (annual T-bill yield in percent), 5 days.

    ::

        date        ^IRX
        2024-01-01  5.0
        2024-01-02  5.0
        2024-01-03  NaN    <- missing day, should be forward-filled to 5.0
        2024-01-04  4.0
        2024-01-05  4.0
    """
    irx = pd.DataFrame(
        {"^IRX": [5.0, 5.0, np.nan, 4.0, 4.0]}, index=pd.bdate_range("2024-01-01", periods=5)
    )
    return FakeYfDownload(irx)


@pytest.fixture
def returns_small() -> pd.Series:
    """Four daily returns, small enough for metrics by hand.

    ::

        r = [0.01, -0.02, 0.03, 0.00]
        mean        = 0.005
        sample std  = sqrt(((0.005)^2 + (-0.025)^2 + (0.025)^2 + (-0.005)^2) / 3)
                    = sqrt(0.0013 / 3) = 0.0208167
        growth      = 1.01 * 0.98 * 1.03 * 1.00 = 1.019494
        downside    = sqrt((0 + 0.02^2 + 0 + 0) / 4) = 0.01   (target 0)
    """
    return pd.Series(
        [0.01, -0.02, 0.03, 0.00], index=pd.bdate_range("2024-01-02", periods=4), name="X"
    )
