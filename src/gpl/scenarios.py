"""Scenario analysis: what-if weights, historical stress episodes and exposures.

Pure functions: no I/O, no printing, no plotting.
"""

from collections.abc import Mapping, Sequence

import pandas as pd

from gpl.metrics import drawdown_series

UNKNOWN = "Other / unknown"
_TOL = 1e-9

# Exchange suffix -> country of listing, for holdings without config metadata.
SUFFIX_REGION = {
    ".SR": "Saudi Arabia",
    ".AE": "UAE",
    ".AD": "UAE",
    ".QA": "Qatar",
    ".KW": "Kuwait",
    ".BH": "Bahrain",
    ".OM": "Oman",
    ".L": "United Kingdom",
}
GROUP_ASSET_TYPE = {
    "GCC": "Equity",
    "Global equity": "Equity",
    "Bonds": "Bonds",
    "Gold": "Gold / commodities",
}


def apply_what_if(weights: Mapping[str, float], targets: Mapping[str, float]) -> pd.Series:
    """Set some holdings to target weights; scale the others pro-rata so the total stays 1.

    Parameters
    ----------
    weights : mapping
        Current weights (sum to 1).
    targets : mapping
        Ticker -> new weight (fraction). A ticker not currently held is added.

    Returns
    -------
    pd.Series
        New weights: current holdings in their order, then any added tickers.

    Raises
    ------
    ValueError
        If a target is outside 0-100%, the targets add up to more than 100%, or the targets
        leave weight to spread but no other holdings to absorb it.
    """
    for t, v in targets.items():
        if not 0 <= v <= 1:
            raise ValueError(f"{t}: a weight must be between 0% and 100%, not {v:.2%}.")
    fixed = sum(targets.values())
    if fixed > 1 + _TOL:
        raise ValueError(f"The weights you named add up to {fixed:.2%}, more than 100%.")
    current = pd.Series(weights, dtype=float)
    others = current.drop([t for t in targets if t in current.index])
    remaining = 1 - fixed
    if remaining > _TOL and others.sum() <= _TOL:
        raise ValueError(
            f"The weights you named add up to {fixed:.2%}, and there are no other holdings to "
            "hold the rest. Name weights that add up to 100%."
        )
    scaled = others * (remaining / others.sum()) if others.sum() > _TOL else others * 0
    order = list(current.index) + [t for t in targets if t not in current.index]
    out = pd.Series({**scaled.to_dict(), **dict(targets)}, dtype=float)
    return out.reindex(order)


def episode_performance(
    returns: pd.DataFrame, episodes: Sequence[Mapping[str, str]]
) -> pd.DataFrame:
    """How each series did in each historical episode.

    The episode return runs from the close on ``start`` to the close on ``end`` (returns dated
    after ``start`` up to and including ``end``). An episode is covered only if the data
    starts on or before ``start`` and ends on or after ``end``; otherwise its figures are NaN.

    Returns
    -------
    pd.DataFrame
        One row per (episode, series): ``episode``, ``series``, ``start``, ``end``,
        ``covered``, ``total_return``, ``max_drawdown`` (within the episode) and
        ``worst_day``.
    """
    rows = []
    for ep in episodes:
        start, end = pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"])
        covered = len(returns) > 0 and returns.index[0] <= start and returns.index[-1] >= end
        window = returns.loc[(returns.index > start) & (returns.index <= end)]
        for name in returns.columns:
            r = window[name]
            rows.append(
                {
                    "episode": ep["name"],
                    "series": name,
                    "start": start,
                    "end": end,
                    "covered": bool(covered),
                    "total_return": float((1 + r).prod() - 1) if covered else float("nan"),
                    "max_drawdown": float(drawdown_series(r).min()) if covered else float("nan"),
                    "worst_day": float(r.min()) if covered else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def classify_holdings(
    tickers: Sequence[str],
    config_meta: Mapping[str, Mapping[str, str]],
    groups: Mapping[str, str],
    currencies: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    """Region, asset type and trading currency for each holding.

    Config metadata wins (e.g. KSA, a US-listed ETF, is Saudi exposure); otherwise the region
    comes from the exchange suffix (no suffix = United States listing) and the asset type
    from the holding's group. Anything unknown is labelled ``Other / unknown``.
    """
    out = {}
    for t in tickers:
        known = config_meta.get(t, {})
        suffix = "." + t.rsplit(".", 1)[-1] if "." in t else ""
        group = groups.get(t, "")
        if "region" in known:
            region = known["region"]
        elif suffix in SUFFIX_REGION:
            region = SUFFIX_REGION[suffix]
        elif not suffix and group in GROUP_ASSET_TYPE:
            region = "United States (listing)"
        else:
            region = UNKNOWN
        out[t] = {
            "region": region,
            "asset_type": known.get("asset_type", GROUP_ASSET_TYPE.get(group, UNKNOWN)),
            "currency": currencies.get(t) or UNKNOWN,
        }
    return out


def exposures(weights: pd.Series, meta: Mapping[str, Mapping[str, str]]) -> dict[str, pd.Series]:
    """Portfolio weight by region, asset type and currency (largest first; sums to 1 each)."""
    out = {}
    for key in ("region", "asset_type", "currency"):
        labels = weights.index.map(lambda t, k=key: meta.get(t, {}).get(k, UNKNOWN))
        out[key] = weights.groupby(labels).sum().sort_values(ascending=False)
    return out
