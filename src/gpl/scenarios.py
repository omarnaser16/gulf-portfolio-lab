"""Scenario analysis: what-if weights, historical stress episodes and exposures.

Pure functions: no I/O, no printing, no plotting.
"""

from collections.abc import Mapping, Sequence

import pandas as pd

from gpl.backtest import cost_rates
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


def drift(current: pd.Series, target: pd.Series, band: float = 0.05) -> pd.DataFrame:
    """Current vs target weights, drift (current minus target) and whether it's outside the band.

    Holdings in either list are included (missing weights count as 0). Exactly at the band is
    inside it.
    """
    tickers = list(dict.fromkeys([*target.index, *current.index]))
    out = pd.DataFrame(
        {
            "current": current.reindex(tickers).fillna(0.0),
            "target": target.reindex(tickers).fillna(0.0),
        }
    )
    out["drift"] = out["current"] - out["target"]
    out["outside_band"] = out["drift"].abs() > band + 1e-9
    return out


def needs_rebalance(drift_table: pd.DataFrame) -> bool:
    """True if any holding has drifted outside the band."""
    return bool(drift_table["outside_band"].any())


def trade_list(
    current_values: pd.Series,
    target: pd.Series,
    prices_usd: pd.Series,
    cost_bps: float,
    total_value: float | None = None,
    cost_by_suffix: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Trades that bring every holding back to its target weight, cash-neutral after costs.

    Parameters
    ----------
    current_values : pd.Series
        Current holdings in USD (or proportions, if ``total_value`` is given).
    target : pd.Series
        Target weights; must add up to 100%.
    prices_usd : pd.Series
        Latest price per unit in USD, for the estimated number of units.
    cost_bps : float
        Trading cost in basis points of the amount traded.
    total_value : float, optional
        Portfolio value in USD; defaults to the sum of ``current_values``.
    cost_by_suffix : mapping, optional
        Market-specific rates by exchange suffix (``gpl.backtest.cost_rates``); others pay
        ``cost_bps``.

    Returns
    -------
    pd.DataFrame
        Per holding: ``current_usd``, ``target_usd``, ``trade_usd`` (positive = buy),
        ``action``, ``units`` (estimated, fractional) and ``cost_usd``. Costs are paid from
        the sales: targets are applied to the value after costs, so buys + costs = sells.
    """
    if abs(float(target.sum()) - 1) > 1e-6:
        raise ValueError(f"Target weights add up to {target.sum():.2%}; they must add up to 100%.")
    tickers = list(dict.fromkeys([*target.index, *current_values.index]))
    held = current_values.reindex(tickers).fillna(0.0).astype(float)
    value = float(held.sum()) if total_value is None else float(total_value)
    if total_value is not None and held.sum() > 0:
        held = held / held.sum() * value
    w = target.reindex(tickers).fillna(0.0)
    rate = cost_rates(tickers, cost_bps, cost_by_suffix) / 1e4
    costs = 0.0
    for _ in range(20):  # costs depend on the trades, which depend on costs: iterate
        trades = w * (value - costs) - held
        new_costs = float((trades.abs() * rate).sum())
        if abs(new_costs - costs) < 1e-9:
            break
        costs = new_costs
    out = pd.DataFrame(
        {
            "current_usd": held,
            "target_usd": w * (value - costs),
            "trade_usd": trades,
        }
    )
    out["action"] = ["Buy" if x > 0.005 else "Sell" if x < -0.005 else "Hold" for x in trades]
    out["units"] = trades / prices_usd.reindex(tickers)
    out["cost_usd"] = trades.abs() * rate
    return out


def cash_need_check(after_usd: pd.Series, cash_like: Sequence[str], need_usd: float) -> dict:
    """Do the cash-like holdings after the trades cover the cash need?

    ``after_usd``: USD value of each holding after rebalancing (``trade_list``'s
    ``target_usd``). Returns ``held``, ``need``, ``shortfall`` (0 if covered) and ``ok``.
    """
    held = float(after_usd.reindex(list(cash_like)).fillna(0.0).sum()) if cash_like else 0.0
    shortfall = max(0.0, need_usd - held)
    return {"held": held, "need": float(need_usd), "shortfall": shortfall, "ok": shortfall < 0.5}


def cash_floor(
    target: pd.Series, cash_like: Sequence[str], need_usd: float, investable_usd: float
) -> pd.Series:
    """Raise the cash-like holdings' target weights to cover ``need_usd`` of ``investable_usd``.

    The cash-like holdings are scaled up together (keeping their mix; an equal split if they
    are all at zero) and the others scaled down pro-rata, so the weights still add up to 1.
    Unchanged when the need is already covered or there is no cash-like holding.
    """
    cash = [t for t in cash_like if t in target.index]
    if not cash or need_usd <= 0 or investable_usd <= 0:
        return target
    floor = min(1.0, need_usd / investable_usd)
    held = float(target[cash].sum())
    if held >= floor - 1e-12:
        return target
    out = target.astype(float).copy()
    mix = target[cash] / held if held > 0 else pd.Series(1 / len(cash), index=cash)
    rest = [t for t in target.index if t not in cash]
    rest_total = float(target[rest].sum())
    out[cash] = mix * floor
    if rest_total > 0:
        out[rest] = target[rest] / rest_total * (1 - floor)
    return out


def trades_covering_cash_need(
    current_values: pd.Series,
    target: pd.Series,
    prices_usd: pd.Series,
    cost_bps: float,
    cash_like: Sequence[str],
    need_usd: float,
    total_value: float | None = None,
    cost_by_suffix: Mapping[str, float] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """``trade_list`` whose cash-like holdings end at or above ``need_usd`` after costs.

    Costs are paid from the sales, so a target that holds exactly the need *before* costs
    ends slightly short; the cash-like targets are topped up (``cash_floor``) until the
    after-cost amount covers the need. Returns the adjusted target and the trades.
    """
    adjusted = target
    trades = trade_list(current_values, adjusted, prices_usd, cost_bps, total_value, cost_by_suffix)
    for _ in range(10):
        if cash_need_check(trades["target_usd"], cash_like, need_usd)["ok"]:
            break
        adjusted = cash_floor(
            adjusted, cash_like, need_usd + 1.0, float(trades["target_usd"].sum())
        )
        trades = trade_list(
            current_values, adjusted, prices_usd, cost_bps, total_value, cost_by_suffix
        )
    return adjusted, trades
