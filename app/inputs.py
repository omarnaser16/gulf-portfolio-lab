"""Parsing and grouping helpers for the dashboard's inputs (pure functions, unit-tested)."""

import re
from collections.abc import Iterable, Mapping, Sequence

GCC_SUFFIXES = (".SR", ".AE", ".AD", ".QA", ".KW", ".BH", ".OM")


def parse_tickers(text: str) -> list[str]:
    """Tickers typed by the user, split on commas, semicolons or spaces; upper-cased, unique."""
    tokens = [t.strip().upper() for t in re.split(r"[,;\s]+", text or "")]
    return list(dict.fromkeys(t for t in tokens if t))


def combine_tickers(presets: Iterable[Sequence[str]], typed: Sequence[str]) -> list[str]:
    """Tickers from the chosen presets then typed ones, without duplicates, order kept."""
    out: list[str] = []
    for group in [*presets, typed]:
        out += [t for t in group if t not in out]
    return out


def asset_groups(
    tickers: Sequence[str], config_groups: Mapping[str, Sequence[str]], labels: Mapping[str, str]
) -> dict[str, list[str]]:
    """Group tickers for charts: configured groups first, then GCC stocks, then Other."""
    groups: dict[str, list[str]] = {}
    placed: set[str] = set()
    for key, members in config_groups.items():
        chosen = [t for t in tickers if t in members]
        if chosen:
            groups[labels.get(key, key.replace("_", " ").capitalize())] = chosen
            placed |= set(chosen)
    gcc = [t for t in tickers if t not in placed and t.upper().endswith(GCC_SUFFIXES)]
    other = [t for t in tickers if t not in placed and t not in gcc]
    if gcc:
        groups["GCC stocks"] = gcc
    if other:
        groups["Other"] = other
    return groups


DEFAULT_HOLDINGS = "SPY 50\nAGG 30\nGLD 10\nKSA 10"


def parse_holdings(text: str, mode: str = "weights") -> tuple[dict[str, float], list[str], str]:
    """Parse "TICKER value" lines into portfolio weights.

    Parameters
    ----------
    text : str
        One holding per line, e.g. ``"SPY 40"``, ``"2222.SR: 30%"`` or ``"GLD, 10,000"``.
    mode : {"weights", "amounts"}
        ``"weights"``: values are percentages; ``"amounts"``: values are money amounts.

    Returns
    -------
    (dict, list, str)
        Ticker -> weight (sums to 1), a list of problems with individual lines, and a note
        when weights had to be rescaled to 100%.
    """
    holdings: dict[str, float] = {}
    problems: list[str] = []
    for n, raw in enumerate((text or "").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        match = re.match(r"^([A-Za-z0-9^][A-Za-z0-9.\-=^]*)[\s,:;]+([\d,]*\.?\d+)\s*%?$", line)
        if not match:
            problems.append(f"Line {n} (“{line}”): expected a ticker and a number, e.g. “SPY 40”.")
            continue
        ticker, value = match.group(1).upper(), float(match.group(2).replace(",", ""))
        if value <= 0:
            problems.append(f"Line {n} (“{line}”): the {mode[:-1]} must be positive.")
            continue
        holdings[ticker] = holdings.get(ticker, 0.0) + value
    total = sum(holdings.values())
    note = ""
    if holdings and mode == "weights" and abs(total - 100) > 0.01:
        note = f"Weights add up to {total:.2f}%, so they were scaled to 100.00%."
    weights = {t: v / total for t, v in holdings.items()} if total else {}
    return weights, problems, note


# ------------------------------------------------------------------- workstation holdings

GROUPS = ["GCC", "Global equity", "Bonds", "Gold", "Other"]
CONFIG_GROUP = {
    "gcc_equity": "GCC",
    "global_equity": "Global equity",
    "bonds": "Bonds",
    "real_assets": "Gold",
}
GCC_EXCHANGES = {
    "saudi",
    "sau",
    "dfm",
    "dubai",
    "adx",
    "abu dhabi",
    "doh",
    "qatar",
    "kuw",
    "kuwait",
    "bah",
    "bahrain",
    "msm",
    "muscat",
}
BOND_WORDS = ("bond", "treasury", "aggregate", "fixed income", "sukuk", "t-bill", "gilt")


def guess_group(
    symbol: str,
    exchange: str,
    quote_type: str,
    name: str,
    config_groups: Mapping[str, Sequence[str]],
) -> str:
    """Best guess of a holding's group; the user can change it in the holdings table."""
    for key, members in config_groups.items():
        if symbol in members and key in CONFIG_GROUP:
            return CONFIG_GROUP[key]
    if symbol.upper().endswith(GCC_SUFFIXES) or (exchange or "").lower() in GCC_EXCHANGES:
        return "GCC"
    lowered = (name or "").lower()
    if any(w in lowered for w in BOND_WORDS):
        return "Bonds"
    if "gold" in lowered:
        return "Gold"
    if quote_type in ("Stock", "ETF", "Fund", "EQUITY", "MUTUALFUND"):
        return "Global equity"
    return "Other"


def normalise_holdings(
    values: Mapping[str, float], mode: str
) -> tuple[dict[str, float], list[str], str]:
    """Holding values (USD amounts or % weights) -> weights summing to 1.

    Returns weights, problems (non-positive or missing values; those holdings are left out)
    and a note when percentages had to be rescaled to 100%.
    """
    problems = [
        f"{t}: enter a positive {'amount' if mode == 'usd' else 'weight'}."
        for t, v in values.items()
        if v is None or not v > 0
    ]
    valid = {t: float(v) for t, v in values.items() if v is not None and v > 0}
    total = sum(valid.values())
    note = ""
    if valid and mode == "pct" and abs(total - 100) > 0.01:
        note = f"Weights add up to {total:.2f}%, so they were scaled to 100.00%."
    return ({t: v / total for t, v in valid.items()} if total else {}), problems, note


def equal_values(tickers: Sequence[str], usd: bool, total: float) -> dict[str, float]:
    """Inputs giving every holding the same weight: 100/n %, or ``total``/n USD."""
    if not tickers:
        return {}
    return {t: (total if usd else 100.0) / len(tickers) for t in tickers}


def normalise_values(values: Mapping[str, float]) -> dict[str, float]:
    """Rescale % inputs so they add up to 100 (non-positive inputs stay as they are).

    Full precision is kept (the table shows two decimals), so the result adds up to exactly
    100 and no rescaling note appears afterwards.
    """
    total = sum(v for v in values.values() if v is not None and v > 0)
    if total <= 0:
        return dict(values)
    return {t: (v * 100 / total if v is not None and v > 0 else v) for t, v in values.items()}
