# CLAUDE.md — Gulf Portfolio Lab

Instructions for Claude Code on this project. Scope, architecture and plan live in @BRIEF.md.

## Context
Omar Naser, a finance and business graduate, is building this as a portfolio project for master's
applications in finance. Claude Code writes the code; Omar directs the work, reviews it and tests
the results. Interviewers may ask Omar to explain any part of the finance code, so every piece of
code comes with an explanation he can learn from.

## How we work: build + explain mode
The project began in a tutor mode, in which Omar would write the finance core with hints. On Day 1
it switched to build + explain mode, which has been used since:

- **Claude Code writes all the code:** the finance core (`src/gpl/data.py`, `metrics.py`,
  `optimise.py`, `backtest.py`, `montecarlo.py`, `regime.py`), the agent, the dashboard, tests,
  notebooks and documentation.
- **Omar directs, reviews and tests:** he sets priorities and requirements, approves dependencies,
  inspects the dashboard and notebooks, reports problems, and decides between options.
- **Tests first where practical**, with expected values small enough to check by hand or in Excel.
- **After each function:** explain in 2–4 plain-English sentences what it does and the finance idea
  behind it. No jargon without a definition.
- **After each module:** ask Omar 2–3 interview-style questions so he can explain it himself
  (e.g. "Why scale volatility by √252?", "Why can historical VaR understate tail risk?").
- **Verify before reporting:** run the code on real data as well as tests, check rendered pages
  (screenshots) for layout problems, and check that every number in prose matches the computed
  output.
- **Report results honestly**, including negative or surprising ones, and say what was not checked.
- **Commit only when the full test suite passes**, with small, clear commits.
- Agent tools must **wrap the finance functions**, never reimplement them.

## Finance conventions (single source of truth)
- **Prices:** adjusted close, in USD. `pd.DataFrame`, tz-naive `DatetimeIndex`, one column per ticker.
- **Returns:** simple daily returns (`pct_change`). Log returns only where stated (GBM in Monte Carlo).
- **Annualisation:** 252 trading days. Volatility × √252; arithmetic mean × 252. Also report CAGR.
- **Risk-free rate:** annual rate, configurable (default derived from `^IRX`, 13-week T-bill).
  Daily rate = `(1 + rf) ** (1/252) - 1`.
- **Sharpe:** annualised mean excess return ÷ annualised volatility.
- **Max drawdown:** reported as a negative number (e.g. `-0.34`), from the running peak of cumulative wealth.
- **VaR / CVaR:** 1-day horizon, 95% default, **reported as a positive loss**. Historical and parametric
  (Gaussian). Expected Shortfall alongside VaR.
- **Weights:** sum to 1, long-only, default max weight 0.40 per asset.
- **No look-ahead bias:** any estimate used to set weights at date *t* uses data up to *t* only, and the
  new weights apply from the **next** trading day.
- **Transaction costs:** bps on turnover, default 10 bps.
- **Randomness:** every stochastic function takes a `seed` argument.
- **FX:** every asset is analysed in USD. SAR, AED, QAR, BHD and OMR use their official USD pegs
  (Yahoo's quotes for them are noisy); other currencies, including KWD (pegged to a basket), use
  Yahoo market rates (`C=X`, local units per USD). Sub-unit quotes (KWF fils, GBp pence) are scaled
  first. Conversion lives in `data.py` (`to_usd`, `load_assets`).
- **Universe:** the user's holdings (Portfolio tab) are the universe for every tab; the regime
  model alone uses SPY/AGG/VIX. The per-asset cap is set on Portfolio › Optimise.
- **Benchmarks:** total-return ETF proxies in USD (config `benchmark_choices`), put on the
  portfolio's dates without shortening its sample; cash has no market risk, so beta, alpha, capture
  and correlation against it are undefined.
- **Covariance for the Optimise tab:** Ledoit-Wolf shrinkage, fitted per window (no look-ahead);
  other tabs keep the sample covariance.
- **Calendars:** assets on different weekly calendars (Sun–Thu vs Mon–Fri) are aligned on the dates
  all of them traded, and annualised with the observed periods per year instead of 252.
- **Display:** percentages shown with exactly two decimals (e.g. `9.00%`).
- **UI:** dark theme (`.streamlit/config.toml`) and one plotly template (`gpl_dark`); show figures
  with `formatting.plot` (theme=None). Benchmarks are grey dashed reference series. No sidebar:
  settings live in the top-bar popover.

## Code style
- Python 3.14, type hints everywhere, numpy-style docstrings.
- Core modules are **pure functions**: no I/O, no printing, no plotting. Network and disk I/O live in `data.py` only.
- Vectorised pandas/numpy. Loops only where the logic is sequential (e.g. the backtester's rebalance loop).
- Never drop NaNs silently: log or return what was removed.
- `ruff` for lint and format; `pytest` for tests. Every public function has at least one test.
- Small commits with clear messages (`feat: add historical VaR`, `test: drawdown edge cases`).

## Commands
```bash
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ruff check . && ruff format .
streamlit run app/streamlit_app.py
```

## AI agent rules
- **The LLM never calculates numbers.** Every figure in a memo comes from a tool result, and the memo
  says which tool produced it. The system prompt requires this and code flags any figure no tool
  returned.
- **The LLM never chooses weights or methods.** For optimisation it only transcribes the user's
  goals into a spec (one of five goals, a target, limits; percentages as written) through a strict
  tool with no weights or method field; the user confirms the spec before anything runs.
- **Recommendations are chosen by code.** `gpl/recommend.py` applies a fixed, tested rule per goal
  to walk-forward, after-cost results (where it compares Sharpe ratios, a change must gain at
  least 0.05, `min_sharpe_gain`); the AI only explains the rule's choice and trade-offs, and
  never presents results as guaranteed or higher future returns.
- **Charts come only from tool data.** The AI chooses chart type, order and section via
  `show_chart` (no numeric fields); `gpl/agent/charts.py` computes and draws every chart from the
  engine. What-if weights must appear in the user's own message; the other holdings are scaled
  pro-rata.
- API key in `.env` as `ANTHROPIC_API_KEY` (or Streamlit secrets when deployed); never committed.
  The model name lives in config, not in code.
- Memos carry a "not investment advice" disclaimer, appended by code.

## Don'ts
- Don't add a dependency without asking Omar first.
- Don't commit `.env`, `data/cache/`, or notebook outputs with large data.
- Don't change finance behaviour without explaining why and showing the effect on results.

## Learning log
At the end of every session, append to `PROGRESS.md`: date, what was built (by Claude Code, at
Omar's direction), what Omar reviewed or decided, concepts learned, open questions and next step.
Written in Omar's voice; short and honest.
