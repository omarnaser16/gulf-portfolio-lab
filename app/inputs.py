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
