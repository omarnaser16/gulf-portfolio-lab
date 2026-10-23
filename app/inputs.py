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
