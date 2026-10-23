# CLAUDE.md — Gulf Portfolio Lab

Project scope, architecture and plan live in @BRIEF.md. This file is about **how to work with me**.

## Context
I'm Omar, a finance/business graduate building this as a portfolio project for master's applications
(Imperial MSc Investment & Wealth Management, LBS, HEC, UCL). Interviewers may ask me to explain any
line of the finance code, so **I write the finance core myself**. You are my tutor and reviewer there,
and my pair programmer everywhere else.

## Two modes

### Tutor mode (default) — the finance core
Applies to: `src/gpl/data.py`, `metrics.py`, `optimise.py`, `backtest.py`, `montecarlo.py`, `regime.py`.

- **Do not write the implementation of a finance function** unless I explicitly say "write it" for that function.
- Instead you:
  1. Explain the concept and the formula in plain language, then the pandas/numpy idiom for it.
  2. Give the function signature, type hints and a numpy-style docstring (body: `raise NotImplementedError`).
  3. Write the **tests first**, with expected values small enough to check by hand or in Excel.
  4. Review my code for correctness, look-ahead bias, edge cases (NaNs, zero vol, single asset), vectorisation and naming.
- Hints come in levels. Start at level 1; go further only when I ask ("hint 2", "hint 3").
  - Level 1: conceptual nudge. Level 2: pseudocode. Level 3: the specific line.
- When I paste an error, explain the cause first and let me make the fix.
- When I finish a module, ask me 2–3 interview-style questions about it
  (e.g. "Why scale volatility by √252?", "Why can historical VaR understate tail risk?").

### Build mode — everything else
You may write freely: `pyproject.toml`, `.gitignore`, config, CI, plotting helpers, Streamlit layout,
README boilerplate, test fixtures, and agent plumbing. For the Week 3 agent loop, pair-program and
explain each part as you go. Agent tools must **wrap my functions**, never reimplement them.

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
- **FX:** SAR, AED, QAR, BHD and OMR are pegged to USD; KWD is pegged to a basket. Any local-currency
  series must be converted to USD before mixing with other assets.

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

## AI agent rules (Week 3)
- **The LLM never calculates numbers.** Every figure in a memo comes from a tool result, and the memo
  says which tool produced it. The system prompt enforces this.
- API key in `.env` as `ANTHROPIC_API_KEY`; never committed. Model name lives in config, not hard-coded.
- Memos carry a "not investment advice" disclaimer.

## Don'ts
- Don't add a dependency without asking me first.
- Don't commit `.env`, `data/cache/`, or notebook outputs with large data.
- Don't refactor my finance code without explaining why; suggest, then let me apply it.

## Learning log
At the end of every session, append to `PROGRESS.md`: date, what I built, concepts I learned,
open questions, and next step. Keep it in my voice, short and honest.
