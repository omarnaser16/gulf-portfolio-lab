# Gulf Portfolio Lab — Project Brief

**An AI-agent portfolio analyst for GCC and global investors.**
A Python finance engine (data, risk, optimisation, backtesting, simulation) with an ML regime classifier,
an Anthropic-API agent that calls the engine's functions as tools and writes an investment memo, and a
Streamlit dashboard with a chat box.

Core design principle: **the maths is deterministic, tested Python; the AI only orchestrates and explains.**

---

## 1. Why this project
- Shows the finance a master's in Investment & Wealth Management builds on: return/risk measurement,
  modern portfolio theory, risk parity, backtesting discipline, Monte Carlo.
- Shows engineering habits: tests, clean modules, reproducibility, no look-ahead bias.
- Shows sensible AI use: an agent that is auditable because it cannot invent numbers.
- GCC angle: USD-pegged currencies, concentrated local markets (energy and banks), and how Gulf
  allocations interact with global equities, bonds and gold.

## 2. Architecture
```
gulf-portfolio-lab/
├── CLAUDE.md               # how Claude Code works with me
├── BRIEF.md                # this file
├── PROGRESS.md             # learning log, appended each session
├── README.md               # public-facing write-up (Week 3)
├── pyproject.toml
├── .env.example            # ANTHROPIC_API_KEY=
├── config/
│   └── universe.yaml       # tickers, groups, defaults
├── data/cache/             # parquet price cache (gitignored)
├── src/gpl/
│   ├── data.py             # download, cache, clean, returns
│   ├── metrics.py          # risk and performance metrics
│   ├── optimise.py         # mean-variance, risk parity
│   ├── backtest.py         # monthly-rebalanced backtester
│   ├── montecarlo.py       # path simulation, goal probability
│   ├── regime.py           # ML regime classifier (Week 2)
│   └── agent/
│       ├── tools.py        # tool schemas wrapping the finance functions (Week 3)
│       └── analyst.py      # tool-use loop + memo writer (Week 3)
├── app/
│   └── streamlit_app.py    # dashboard + chat (Weeks 2–3)
├── notebooks/              # exploration only, never imported
└── tests/
    ├── conftest.py         # synthetic price fixtures
    └── test_*.py
```

## 3. Default universe (`config/universe.yaml`)
US-listed ETFs are the default because they share one trading calendar, are priced in USD and have
clean history on Yahoo Finance.

| Group        | Tickers                         | Note |
|--------------|---------------------------------|------|
| GCC equity   | `KSA`, `UAE`, `QAT`             | iShares MSCI country ETFs; history starts ~2014–15 |
| Global equity| `SPY`, `EFA`, `EEM`             | US, developed ex-US, emerging |
| Bonds        | `AGG`, `TLT`                    | aggregate, long Treasuries |
| Real assets  | `GLD`                           | gold |
| Rates / risk | `^IRX`, `^VIX`                  | risk-free proxy; regime feature (not portfolio assets) |

Optional extension: local Tadawul listings (e.g. `2222.SR`, `1120.SR`). Caveats: priced in SAR,
different trading days (Sun–Thu) from US markets, patchier data. Align on common dates and convert to USD.

## 4. Module specs

### `data.py`
- `get_prices(tickers, start, end, use_cache=True) -> pd.DataFrame` — adjusted closes, cached to parquet.
- `clean_prices(prices, max_ffill=5) -> pd.DataFrame` — limited forward-fill, drop leading NaNs, report gaps.
- `to_returns(prices, kind="simple") -> pd.DataFrame`
- `get_risk_free(start, end) -> pd.Series` — daily rate from `^IRX`.

### `metrics.py`
- `annualised_return`, `cagr`, `annualised_volatility`, `sharpe_ratio`, `sortino_ratio`
- `drawdown_series`, `max_drawdown`
- `historical_var`, `parametric_var`, `expected_shortfall`
- `summary_table(returns, rf) -> pd.DataFrame` — one row per asset or strategy

### `optimise.py`
- `portfolio_performance(weights, mu, cov) -> (ret, vol)`
- `min_variance(cov, max_weight)`, `max_sharpe(mu, cov, rf, max_weight)` via `scipy.optimize.minimize` (SLSQP)
- `efficient_frontier(mu, cov, n_points)`
- `risk_contributions(weights, cov)`, `risk_parity(cov)`

### `backtest.py`
- `run_backtest(prices, strategy, rebalance="ME", lookback=252, cost_bps=10) -> BacktestResult`
- `BacktestResult` dataclass: `equity`, `returns`, `weights`, `turnover`, `costs`
- Strategies: equal weight, min variance, max Sharpe, risk parity; benchmarks: 60/40 (`SPY`/`AGG`), `SPY`.
- Weights estimated on the last trading day of each month from the trailing window, applied from the next day.

### `montecarlo.py`
- `simulate_paths(returns, weights, horizon_days, n_paths, method="bootstrap" | "gbm", seed)`
- `terminal_stats(paths)` — percentiles of terminal wealth
- `goal_probability(paths, target)` — e.g. P(portfolio ≥ 1.5× in 10 years)

### `regime.py` (Week 2)
- Target: **next month's** regime (calm vs stressed), so the model is predictive rather than circular.
- Label: stressed if forward 21-day realised volatility of `SPY` is above its rolling 80th percentile (tunable).
- Features (known at *t*): trailing vol, drawdown, `^VIX` level and change, momentum, bond–equity correlation.
- Models: logistic regression baseline, then random forest. `TimeSeriesSplit` only, no shuffling.
- Report: precision/recall on the stressed class, confusion matrix, comparison with "always calm" baseline.
- Use: a regime-aware strategy in the backtester (e.g. de-risk when P(stressed) is high).

### Agent (Week 3)
- Anthropic API with tool use. Tools wrap: `summary_table`, `optimise`, `run_backtest`,
  `simulate_paths`, `predict_regime`.
- Flow: user describes profile ("Dubai-based, USD, 10-year horizon, moderate risk") → agent calls tools →
  writes a structured memo (allocation, expected risk/return, drawdown history, regime view, risks, disclaimer).
- Every number in the memo traces back to a tool call; log tool calls and show them in the UI.

### Dashboard (Weeks 2–3)
Streamlit + plotly: asset picker, metrics table, efficient frontier, backtest equity curves and drawdowns,
weight history, Monte Carlo fan chart, regime probability chart, and a chat panel for the agent.

## 5. Plan

### Week 1 — finance core (Claude writes and explains it; I learn to explain it)
| Day | Work |
|-----|------|
| 1 | Repo, venv, `pyproject`, skeleton, synthetic fixtures; `data.py` |
| 2 | Returns and core metrics (return, CAGR, vol, Sharpe, Sortino) + tests |
| 3 | Drawdowns, VaR, Expected Shortfall, `summary_table` + tests |
| 4 | Mean-variance: min variance, max Sharpe, efficient frontier |
| 5 | Risk contributions and risk parity |
| 6 | Backtester with monthly rebalancing and costs |
| 7 | Monte Carlo; tidy, tests green, first results notebook |

**Done when:** `pytest` passes; one script/notebook produces a summary table and backtest chart comparing
all strategies and benchmarks on the default universe; `PROGRESS.md` has an entry per day.

### Week 2 — ML + dashboard
Regime classifier with honest evaluation; regime-aware strategy; Streamlit dashboard with plotly charts.

### Week 3 — agent + GitHub + live demo
Tool-use agent and memo; chat panel; README with screenshots, methodology and limitations;
deploy to Streamlit Community Cloud with the API key in secrets.

## 6. Known pitfalls
- **yfinance** is unofficial and can break or rate-limit: cache everything, pin the version, fail clearly.
- **Short GCC ETF history** (~2015 onwards) limits lookbacks; state the sample period in every result.
- **Mean-variance is sensitive to expected-return estimates**: compare with min variance and risk parity,
  and say so in the README.
- **Survivorship and look-ahead bias**: fixed ETF universe, strict lag in the backtester.
- **Python 3.14**: if a dependency won't install, fall back to a 3.13 venv rather than fight it.

## 7. Stretch ideas
Black-Litterman with a GCC view; Sharia-screened universe option; rolling-window metrics; PDF export of
the memo; CVaR optimisation.
