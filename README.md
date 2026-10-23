# Gulf Portfolio Lab

**An AI-assisted portfolio analyst for GCC and global investors, where the AI is not allowed to do the maths.**

Gulf Portfolio Lab is a tested Python finance engine (data, risk metrics, portfolio optimisation,
backtesting, Monte Carlo simulation and a machine-learning market-regime model) with a Streamlit
dashboard on top. Its AI analyst, built on the Anthropic API, writes an investment memo for an
investor profile ("I'm 35, in Dubai, investing in USD for 10 years…") by calling the engine's
functions as tools. **Every number in the memo is copied from a tool result and cites the tool, and
code flags any figure that no tool produced.**

> 🔗 **Live demo:** *add the Streamlit Community Cloud link here after deploying*
>
> **Not investment advice.** This is an educational portfolio project. Past performance does not
> predict future results.

> 📸 *Screenshot placeholder: `docs/screenshots/backtest.png` (Backtest tab: strategies vs benchmarks)*

---

## Why it exists

- **Finance that holds up to scrutiny.** Return and risk measurement, mean-variance and risk parity,
  look-ahead-free backtesting with costs, and Monte Carlo, each with tests whose expected values can
  be checked by hand.
- **A Gulf investor's angle.** GCC equity ETFs (Saudi Arabia, UAE, Qatar) alongside global equities,
  bonds and gold. The Gulf currencies SAR, AED, QAR, BHD and OMR are pegged to the US dollar, so
  USD-priced assets carry little currency risk for these investors.
- **AI that can be audited.** The language model orchestrates and explains; deterministic, tested
  Python produces every number.

## What's inside

| Part | What it does |
|---|---|
| **Data** | Any Yahoo Finance ticker, adjusted prices cached per ticker as parquet; converted to USD (official pegs for SAR, AED, QAR, BHD and OMR; market rates for other currencies; sub-units such as Kuwaiti fils scaled); different trading calendars aligned on common dates; history and data notes shown for every asset. |
| **Metrics** | Annual return, CAGR, volatility, Sharpe, Sortino, Calmar, drawdowns, historical and normal VaR, expected shortfall, trailing 1- and 3-year returns; against a benchmark: beta, Jensen's alpha, tracking error, information ratio, up/down capture, relative drawdown; correlation clusters. |
| **Optimisation** | Min variance, max Sharpe (solved in convex form), efficient frontier, risk parity with a 40% weight cap; a constrained optimiser (max Sharpe, min variance, max return, target volatility, target return, risk parity and max diversification) with per-asset and group limits, exact feasibility checks and plain-language reasons when limits can't all be met; Ledoit-Wolf covariance shrinkage. |
| **Backtester** | Monthly rebalancing on a trailing 252-day window, new weights from the **next** day, weight drift, 10 bps costs on turnover. |
| **Monte Carlo** | 21-day block bootstrap (default), daily bootstrap or GBM; goal probabilities, bad-case outcomes, drawdowns along the way, nominal or real (2.50% inflation). |
| **Regime model** | Predicts whether next month will be stressed; logistic regression, time-series cross-validation with a 21-day gap, monthly walk-forward refits, compared with a simple volatility rule. |
| **AI analyst** | Claude (Haiku 5.5 by default) with nine tools wrapping the engine; memos for an investor profile, about the user's own portfolio, or explaining a recommendation, with audit trail and number check. It also turns plain-English goals into a goal and limits the user confirms; it never chooses weights or methods. |
| **Dashboard** | A *Portfolio* workstation (holdings, screen, analyse & backtest, benchmark comparison, optimise) whose holdings drive every other tab: Assets, Frontier, Backtest, Monte Carlo, Regime and Analyst. Latest data by default, or the fixed sample that reproduces this README. |

> 📸 *Screenshot placeholder: `docs/screenshots/analyst.png` (Analyst tab: memo, audit trail, cost line)*

## Using it as an investor

- **Any asset, one portfolio.** Your holdings are the universe for every tab. Search by name or
  ticker ("Aramco", "Emaar", "SPY"), paste a list, or start from a preset (the README universe of
  nine ETFs, GCC stocks such as Saudi Aramco `2222.SR`, Al Rajhi `1120.SR`, Emaar `EMAAR.AE`,
  Emirates NBD `EMIRATESNBD.AE`, QNB `QNBK.QA` and NBK `NBK.KW`, or global ETFs). Tickers with no
  data get a clear message. Abu Dhabi (ADX) listings such as FAB are not on Yahoo Finance.
- **One currency.** Everything is analysed in USD. The Gulf pegs are used for SAR, AED, QAR, BHD and
  OMR because Yahoo's quotes for these currencies are noisy (the riyal series is more than 1% off its
  3.75 peg on about 100 days since 2015). Other currencies use market rates (e.g. `KWD=X`).
- **Different trading weeks.** Saudi, Qatari and Kuwaiti markets trade Sunday to Thursday; US markets
  Monday to Friday. When both are selected, only the dates on which every market traded are used
  (about 183 a year for the GCC preset with SPY), and figures are annualised with that number instead
  of 252. The dashboard says so, and shows each asset's start date, currency, conversion and trading
  days, warning about assets with too little history.
- **Your own portfolio (the *Portfolio* tab).** Holdings show each security's latest price and
  quote time; edit them in USD or percent (white columns are editable), use *Equal weight* or
  *Normalise to 100%*, and assign groups. *Screen* ranks holdings by any of 14 measures (return,
  CAGR, 1Y and 3Y return, volatility, Sharpe, Sortino, Calmar, max drawdown, VaR, expected
  shortfall, beta to the benchmark, share of portfolio risk, correlation to the rest). *Analyse &
  backtest* shows share of money vs share of risk and a backtest after costs vs the strategies.
- **Benchmarks.** Compare against 60/40 (default), the S&P 500 (SPY), FTSE 100 (ISF.L, converted
  from pence to USD), MSCI World (URTH), the Saudi (KSA) or UAE (UAE) market, cash (13-week
  T-bills) or any loaded asset. These are total-return ETF proxies, so dividends count as they do
  in your portfolio. The *Benchmark* tab shows return, volatility, Sharpe, max drawdown, beta,
  alpha, tracking error, information ratio, up/down capture and correlation, with growth-of-$1 and
  relative-drawdown charts. A benchmark's holidays never shorten your sample.
- **Optimise like an analyst.** (1) *Diagnose*: each holding's share of money vs share of risk,
  volatility, Sharpe and beta, clusters of holdings that move together, and the portfolio vs the
  benchmark. (2) *Goal*: maximise risk-adjusted return, reduce risk, reach a target return, hit a
  target volatility, or beat the benchmark with less risk, picked from a list or described in plain
  English (the AI fills in the goal and limits for you to check; numbers not in your text are
  flagged). (3) *Candidates*: max Sharpe, min variance, risk parity, max diversification and target
  volatility, each within your limits, re-optimised monthly on past data only (Ledoit-Wolf
  covariance) and evaluated after costs against your current portfolio and the benchmark. (4)
  *Recommendation*: a fixed rule per goal picks the method that fits the goal **and** held up out
  of sample, or says to keep your current portfolio, and shows its reasons. (5) An optional AI memo
  explains it. Results are historical and risk-adjusted, never presented as higher returns to come.
- **Latest or reproducible.** The dashboard opens on the latest data ("Data as of …" at the top). The
  *Reproduce README results* option uses the fixed sample below and reproduces this README's tables
  exactly.

## Key results

Data from 17 Sep 2015 (when the youngest ETF, KSA, starts) to **30 Sep 2026**; the end date is
fixed in `config/universe.yaml` (the dashboard's *Reproduce README results* option), so results are
reproducible.

**Backtests (Sep 2016 – Sep 2026, after 10 bps costs, max 40% per asset):**

| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Turnover / yr |
|---|---|---|---|---|---|
| SPY | 15.27% | 17.95% | 0.75 | −33.72% | 0.00× |
| 60/40 (SPY/AGG) | 9.64% | 11.18% | 0.67 | −21.63% | 0.19× |
| Max Sharpe | 8.72% | 11.25% | 0.59 | −26.71% | 3.90× |
| Equal weight | 7.14% | 10.31% | 0.49 | −21.64% | 0.30× |
| Risk parity | 5.14% | 7.81% | 0.38 | −16.29% | 0.51× |
| Min variance | 4.28% | 6.95% | 0.30 | −15.21% | 1.05× |

A decade of strong US equities and weak bonds favoured SPY and 60/40. The diversified strategies
had smaller drawdowns but lower returns, and the GCC ETFs added risk without matching return over
this sample.

**Regime model (out of sample, Jul 2009 – Aug 2026):**

| | ROC AUC | Precision | Recall | Accuracy |
|---|---|---|---|---|
| Logistic model | 0.72 | 47.89% | 40.45% | 77.73% |
| "Volatility already high" rule | 0.71 | 58.33% | 51.35% | 81.65% |
| Always predict calm | 0.50 | – | 0.00% | 78.49% |

The model ranks stressed months modestly well, and its probabilities are calibrated (28.86%
predicted on average vs 21.51% actual). However, a one-line volatility rule does at least as well at
deciding. De-risking on the forecast cut the COVID drawdown of the equal-weight portfolio from
−20.32% to −16.06%, but lowered its Sharpe from 0.49 to 0.39.

**Monte Carlo (equal weight, 10 years, dashboard defaults):** 81.55% chance of ending at 1.50× or
more, median 2.00×, bad case (5th percentile) 1.16×, 1.90% chance of losing money. Along the way, the
bad-case drawdown is −35.15% with block bootstrap vs −31.53% with daily draws: keeping crashes
together makes the path rougher.

**AI analyst:** a typical memo uses 5 tool calls and about 15,500 input + 2,700 output tokens, roughly
**$0.003** with Claude Haiku 5.5.

## Methodology

**Conventions** (see `CLAUDE.md`): simple daily returns; 252 trading days a year; Sharpe = annualised
mean excess return over the 13-week T-bill ÷ annualised volatility; VaR and expected shortfall are
1-day, 95%, reported as positive losses; weights are long-only and sum to 1.

**No look-ahead bias.** At each month-end rebalance a strategy sees only returns up to that day, and
its new weights earn from the next day. A test runs a "cheating" strategy that buys after a jump and
checks that it never earns the jump. The regime model is refit monthly on labels already known: a
label looks 21 days ahead, so a model refit on day *t* trains only on labels up to *t − 21*.
Cross-validation leaves the same 21-day gap.

**Optimisation details.** Max Sharpe is solved as an equivalent convex problem (minimise risk for a
fixed excess return, then rescale), which removed solver failures seen with the direct ratio. Risk
parity uses cyclical coordinate descent; when an asset would exceed the 40% cap it is held at the
cap and the rest share risk equally. Solver output that breaks the cap is rejected, never rescaled.

**Regime labels are relative.** "Stressed" means SPY's volatility over the next 21 trading days is
above the 80th percentile of its own past year, so the bar moves with recent conditions. The model
uses SPY, AGG and VIX history from 2005, longer than the GCC ETF sample. Class weighting was removed
because it inflated the average predicted probability to 47.41% against a 21.51% actual rate.

**AI analyst.** A manual tool-use loop with the Anthropic Python SDK; the model ID is set in config.
The nine tools (`get_universe`, `get_asset_metrics`, `optimise_portfolio`,
`backtest_strategies`, `simulate_future`, `get_my_portfolio`, `get_portfolio_diagnosis`,
`get_optimisation_result`, `get_regime_view`) wrap the engine and return pre-formatted numbers.
The system prompt forbids arithmetic and requires a bracketed tool citation for every figure. After
each answer,
code compares every decimal, percentage and multiple in it with the tool outputs and flags any that
don't match (a re-rounded "15.3%" for "15.27%" is flagged). Every tool call is shown in an audit
trail, and the disclaimer is appended by code.

**Goals to a recommendation.** A separate, single model call fills one strict tool,
`propose_spec`, whose schema is the spec: one of five goals, a target if the goal needs one, a
per-asset cap, asset and group limits, a volatility cap and a return floor, in percent as the user
wrote them. There is no field for weights or methods. Code converts percent to fractions and checks
every number against the user's text. Each candidate method is the constrained optimiser with a
different objective; it first checks feasibility exactly (a linear program for the weight and group
limits, then the lowest achievable volatility and highest achievable return), so impossible limits
get a specific reason instead of a solver failure, and solver output is re-checked against every
limit. Covariances use Ledoit-Wolf shrinkage, fitted each month on the trailing window only.

The recommendation comes from `gpl/recommend.py`, not from the model. For each goal there is one
written rule applied to the walk-forward results after costs (e.g. *reduce risk*: lowest volatility
below the current portfolio's, with the higher Sharpe ratio winning when two are within 0.50
percentage points). A method "held up" if it kept at least half of its in-sample Sharpe ratio out of
sample; one that ranked first but didn't hold up is skipped, and if nothing improves on the current
portfolio for the goal, the rule says to keep it. Every branch is unit-tested.

Example on the README sample (holdings SPY 50%, AGG 30%, GLD 10%, KSA 10%; at least 10% GCC;
walk-forward 3 Oct 2016 – 30 Sep 2026, after costs, against 60/40):
- *Reduce risk* (at most 45% per asset): the rule chose risk parity, with volatility 7.69% vs
  10.73% and max drawdown −14.47% vs −21.18%, but CAGR 7.34% vs 10.29% and Sharpe 0.65 vs 0.75. It is
  a lower-risk choice, not a better risk-adjusted one.
- *Beat the benchmark with less risk* (at most 40% per asset): the rule chose max Sharpe (CAGR
  9.95% vs the benchmark's 9.83%, volatility 10.16% vs 11.28%). Its Sharpe ratio beat the current
  portfolio's by under 0.01 (0.7575 vs 0.7512) while trading 1.70× a year vs 0.24×; with data to 23
  Oct 2026 the same request recommended keeping the current portfolio. The rule has no minimum
  improvement, so a near-tie can flip with a few weeks of data (see Limitations).

**Testing.** 479 tests (`pytest`), most with expected values small enough to check by hand. They
include property tests (no optimiser output exceeds the weight cap across random problems; features
and labels don't change when the future changes) and headless dashboard tests with synthetic data and
a scripted fake model, so the suite never touches the network.

## Limitations

- **One sample period.** About ten years dominated by US equities, COVID and the 2022 bond sell-off.
  Results may not generalise; every figure is tied to the sample shown.
- **Short GCC history** (from 2015) and a universe of ETFs chosen today (survivorship).
- **Simple costs:** 10 bps per unit traded, with no bid-ask spread on thinner GCC ETFs and no taxes.
- **The efficient frontier is hindsight:** it uses full-sample means and covariances. The backtests
  are the honest test.
- **Regime whipsaw.** 2022 was clearly stressed (55.38% of days), but the month-end probability
  crossed 50% back and forth (January above, February and March below, April above, May below, June
  above). So the regime strategy switched in and out of min variance, paid 0.44% in costs that year
  vs 0.04% static, and lost 13.18% vs 12.57%. Min variance itself gave only modest shelter because
  bonds fell too.
- **Regime model scope:** stress is defined by SPY volatility, so GCC-specific shocks (oil, regional
  politics) are invisible to it. There are few independent stress episodes, and fold ROC AUCs range
  from 0.63 to 0.88.
- **Monte Carlo** assumes the future resembles a reshuffle of this sample, with daily rebalancing and
  no costs.
- **AI analyst:** the number check covers decimals, percentages and multiples, not bare integers or
  numbers written in words, and the model can still mis-describe a correct number. Treat the audit
  trail as the source of truth.
- **Optimisation uses history.** Expected returns are historical means, which are noisy (Ledoit-Wolf
  only shrinks the covariance); max Sharpe is especially sensitive to them. Picking the best of five
  methods on the same period is itself a mild form of hindsight, and the target-volatility level is
  set from the whole period. Tight limits can leave little room: with four holdings and a 25% cap,
  25% each is the only allowed portfolio. The spec check catches invented numbers, not a misread
  goal, so read the confirmation before running.
- **The recommendation rule has no materiality threshold:** any improvement on the goal metric
  counts, so a method can be recommended for a tiny gain (e.g. 0.006 of Sharpe) despite much higher
  turnover, and near-ties can flip as data is added.
- **Benchmarks** are ETF proxies with fund fees, compared without trading costs (as indices are).
  On dates when the benchmark's market was closed but yours traded, it shows 0% and catches up next
  day. Against cash, beta, alpha, capture and correlation are not defined.
- **Search and quotes** come from Yahoo Finance too: suggestions can be incomplete and quotes delayed;
  groups are guessed from the exchange and name and can be changed in the table.
- **Data source:** `yfinance` is unofficial and can break or rate-limit. Downloads are cached per
  ticker (latest data refreshes at most every 6 hours) and retried; if Yahoo still refuses, the
  dashboard says so and offers a Retry button. No price data is stored in this repository. "Latest
  data" can include today's price before the market closes.
- **Coverage and conversions:** Yahoo does not cover every exchange (e.g. Abu Dhabi), and market FX
  for non-pegged currencies comes from Yahoo too. Pegs are assumed to hold.
- **Common dates:** aligning Sunday–Thursday and Monday–Friday markets drops each market's other
  days, so some returns span more than one day and a market's Sunday or Friday moves are only seen
  in the next common return. With mixed markets, the sample starts when the newest asset starts.
- **Known issue:** once, with default settings, the Frontier tab showed weights matching a 75% cap
  instead of 40%. It could not be reproduced; safeguards and an on-screen cap caption were added.

## Future work

- **Smoothing rules for the regime switch, tested out of sample:** require two consecutive months
  above the threshold, add a hysteresis band (enter at 60%, exit at 40%), or scale risk gradually with
  P(stressed). Choose among them on 2008–2018 only, then test once on 2019–2026, to avoid tuning to
  2022.
- GCC-specific stress features (oil, local market volatility) and local Tadawul listings converted
  from SAR to USD.
- Black-Litterman with a GCC view; CVaR optimisation; a Sharia-screened universe option.
- PDF export of the memo, and an evaluation set for the analyst (does every memo cite every number?).

## Run it locally

```bash
git clone <your-repo-url> && cd gulf-portfolio-lab
python3.14 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # then add your ANTHROPIC_API_KEY (only needed for the AI features)
pytest -q                       # 479 tests, no network needed
streamlit run app/streamlit_app.py
```

Notebooks with the full results: `notebooks/01_week1_results.ipynb` (risk, optimisation, backtests,
Monte Carlo) and `notebooks/02_week2_regime.ipynb` (regime model and regime-aware strategies).

## Project layout

```
config/universe.yaml     tickers, groups, defaults, regime, Monte Carlo and agent settings
src/gpl/data.py          download, cache, clean, returns, risk-free rate (all I/O lives here)
src/gpl/metrics.py       risk and performance metrics
src/gpl/optimise.py      min variance, max Sharpe, frontier, risk parity, constrained optimiser,
                         Ledoit-Wolf covariance
src/gpl/recommend.py     goals, candidate methods and the recommendation rule
src/gpl/backtest.py      monthly-rebalanced backtester and strategies
src/gpl/montecarlo.py    simulations, goal probability, path drawdowns
src/gpl/regime.py        labels, features, models, evaluation, walk-forward
src/gpl/plots.py         plotly charts shared by notebooks and the app
src/gpl/agent/           tools.py (tool schemas wrapping gpl), analyst.py (tool-use loop),
                         spec.py (plain-English goals -> goal and limits)
app/                     Streamlit dashboard (workstation.py: Portfolio tab)
tests/                   pytest suite
```

## How this was built

**The code in this repository was written by [Claude Code](https://claude.com/claude-code),
Anthropic's coding agent**, in a series of sessions following the three-week plan in `BRIEF.md`.

My role (Omar Naser) was to **direct, review and test**:
- **Directed:** I defined the project idea, scope, tech stack and plan; the brief and Claude Code
  instructions (`BRIEF.md`, `CLAUDE.md`) were drafted with Claude from those requirements. I set the
  priorities for each session, approved every new dependency, and chose among the options
  presented, for example capping risk parity at 40%, fixing the end date for reproducibility, and
  making the block bootstrap the default. I specified the Portfolio workstation, the benchmark
  set, the analyst-style Optimise workflow, and the rules that the AI only turns goals into limits I
  confirm (never weights) and that a coded rule, not the AI, makes the recommendation.
- **Reviewed:** I read the results, notebooks and explanations, and asked for changes where something
  looked wrong or unclear, for example checking whether the regime model's probabilities were
  calibrated, leading with ROC AUC rather than accuracy, and stating limitations such as the 2022
  whipsaw.
- **Tested:** I used the dashboard and reported problems, including a weight-cap violation on the
  Frontier tab, labels overlapping the frontier, a truncated table and inconsistent percentage
  formats.

Claude Code worked in a "build and explain" mode (described in `CLAUDE.md`): after each function it
explained the finance idea in plain language, and after each module it asked me interview-style
questions, so that I can explain every part of the code. `PROGRESS.md` is the session-by-session log
of what was built, what I directed and what I learned, including mistakes found and fixed.

## Licence and disclaimer

Educational project. Nothing here is investment advice or a recommendation to buy or sell any
security. Data © its providers, via Yahoo Finance.
