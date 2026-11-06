# Methodology, results and limitations

The full write-up behind the [README](../README.md): every module, the methods and their tests,
the detailed results tables and every known limitation. For how the parts fit together, and what
the AI may and may not do, see [ARCHITECTURE.md](../ARCHITECTURE.md).

**Not investment advice.** Educational project; past performance does not predict future results.
Do not enter real client data in the public demo.

## Contents

- [What's inside](#whats-inside)
- [For a wealth manager: plan, execute, review](#for-a-wealth-manager-plan-execute-review)
- [Build a portfolio for me](#build-a-portfolio-for-me)
- [Improve my current portfolio](#improve-my-current-portfolio)
- [Benchmarks, including the policy benchmark](#benchmarks-including-the-policy-benchmark)
- [Using it as an investor](#using-it-as-an-investor)
- [Key results](#key-results)
- [Methodology](#methodology)
- [Limitations](#limitations)

## What's inside

| Part | What it does |
|---|---|
| **Data** | Any Yahoo Finance ticker, adjusted prices cached per ticker as parquet; converted to USD (official pegs for SAR, AED, QAR, BHD and OMR; market rates for other currencies; sub-units such as Kuwaiti fils scaled); different trading calendars aligned on common dates, date-aware (Dubai and Abu Dhabi moved from a Sunday to a Monday week in 2022); history and data notes shown for every asset. |
| **Metrics** | Annual return, CAGR, volatility, Sharpe, Sortino, Calmar, drawdowns, historical and normal VaR, expected shortfall, trailing 1- and 3-year returns; against a benchmark: beta, Jensen's alpha, tracking error, information ratio, up/down capture, relative drawdown; correlation clusters. |
| **Optimisation** | Min variance, max Sharpe (solved in convex form), efficient frontier, risk parity with a 40% weight cap; a constrained optimiser (max Sharpe, min variance, max return, target volatility, target return, risk parity and max diversification) with per-asset and group limits, exact feasibility checks and plain-language reasons when limits can't all be met; Ledoit-Wolf covariance shrinkage. |
| **Backtester** | Monthly rebalancing on a trailing 252-day window, new weights from the **next** day, weight drift, costs on turnover by market (10 bps for US listings; judgement estimates of 30–40 bps for GCC exchanges, 15 bps for London). |
| **Monte Carlo** | 21-day block bootstrap (default), daily bootstrap or GBM; goal probabilities, bad-case outcomes, drawdowns along the way, nominal or real (2.50% inflation). |
| **Regime model** | Predicts whether next month will be stressed; logistic regression, time-series cross-validation with a 21-day gap, monthly walk-forward refits, compared with a simple volatility rule. |
| **AI analyst** | Claude (Haiku 5.5 by default) with seventeen tools wrapping the engine, including the client's IPS and a `calculate` tool so arithmetic is done in code; portfolio reviews, stress tests, what-ifs and explanations of a recommendation, each memo section paired with charts the AI chooses but code builds, with audit trail and number check. It also turns plain-English goals into a goal and limits the user confirms; it never chooses weights or methods. |
| **Adviser workflow** | Full fact-find (nine sections, completeness bar), risk questionnaire, IPS (PDF, DRAFT until the fact-find is complete) feeding the optimiser, ten suitability rules including knowledge & experience, fund-domicile check, drift monitor and trade list, client report (PDF). |
| **Dashboard** | Light, private-bank-style Streamlit app in six steps (1 Client, 2 Portfolio, 3 Analyse, 4 Optimise, 5 Suitability, 6 Report & trades) with a stepper showing each step's status, a sticky header (client, risk score, portfolio value, IPS status, benchmark, *Fill demo client*, *Ask the analyst*, ⚙ Settings) and a *Next* button per step. The holdings drive every step; Analyse holds the benchmark comparison, assets, Monte Carlo and regime views; Optimise holds the frontier and strategy backtests; step 6 holds trades, reports and the AI analyst. Latest data by default, or the fixed sample that reproduces the results below. |


## For a wealth manager: plan, execute, review

The tool follows the CFA portfolio-management loop as **decision support for a licensed
adviser**. Every number comes from tested code; the AI only fills forms from the adviser's notes
and explains results.

| Step | In the tool |
|---|---|
| Discovery and risk profiling | *1 Client*: a full fact-find in nine sections (client & household, KYC/AML, financial situation, objectives & goals, risk profile, knowledge & experience, constraints & preferences, estate & succession, service & review) with a completeness bar, consistency warnings and a next-review date; profile, classification and regulator (DFSA, FSRA, CMA…), a seven-question risk questionnaire scored by a tested rule (risk score = the lower of capacity and willingness; a gap of two or more is flagged), "describe the client in plain English" for the AI to fill the facts (never the questionnaire): name, age, dated cash needs (the AI copies "Oct 2028" or "in 18 months"; code dates it, so Oct 2028 is 24 months away), tax notes, region minimums, and the regulator from the booking jurisdiction by a fixed rule (Abu Dhabi/ADGM → FSRA, DIFC → DFSA, onshore UAE → SCA, Saudi → CMA, QFC → QFCRA). Profiles save and load as JSON. One portfolio value is used everywhere: the total of USD holdings, or entered once (no default). |
| Investment Policy Statement | RRTTLLU (Return, Risk, Time horizon, Taxes, Liquidity, Legal, Unique) generated from the profile, with limits from the risk band (e.g. band 3: volatility 12%, drawdown 20%) or adviser overrides; exported as a signable PDF. |
| Strategic allocation | *Optimise* uses the IPS limits automatically (volatility cap, max weight, GCC minimum, region minimums such as Saudi ≥ 10%, excluded holdings at 0%), stricter of IPS and form. Cash needs due within 36 months are a hard minimum on cash-like holdings (cash, short-term bonds, sukuk); with none held the app offers to add BIL. |
| Suitability | Ten rules for the current and proposed portfolios: volatility, max drawdown, worst stress episode, concentration, home-market bias, region minimums, liquidity, exclusions (Sharia, sectors), currency, and knowledge & experience (complex products the client has not used are flagged for review). Each is pass, fail or review, with a reason written from the numbers; in Optimise, Analyst, the memo and the report. **Optimise only recommends portfolios that pass every rule**; if none does, it shows the closest with each breach (e.g. "Volatility: 13.10%, 1.10 pp beyond the 12.00% limit") and says so if the current portfolio fails too. Identical methods are shown once. |
| Implementation | *Fund domicile check* for non-US clients: US-situs holdings (US-listed funds and stocks, including KSA), the illustrative US estate tax (18%–40% schedule minus the 13,000 USD non-resident credit, i.e. a 60,000 exemption), 30% vs 15% dividend withholding, and verified Irish-domiciled alternatives (SPY → CSPX.L, AGG → IUAA.L, GLD → IGLN.L…). Information, not tax advice. |
| Monitoring and rebalancing | *6 Report & trades › Trades*: current vs editable target weights (default: the Optimise recommendation), a ±5 pp drift band, and a cash-neutral trade list (buy/sell in USD, estimated units and costs; buys + costs = sells), downloadable as CSV. The trades are checked against the cash need, and default targets are topped up so the need is still covered after costs. |
| Reporting | One-click, date-stamped *client report* PDF in plain language: in brief, goals and limits, holdings in USD, past performance, **the optimiser's recommendation** (never a chat what-if), proposed trades, the policy checks, the adviser's commentary (AI memo without tool tags), an adviser sign-off block (name, licence, regulator, date, client acknowledgement), disclaimers and the IPS as an appendix. |

## Build a portfolio for me

*4 Optimise › Build a portfolio for me* proposes a portfolio from a vetted universe that aims to
beat the benchmark on a risk-adjusted basis, judged walk-forward after costs.

- **Curated universe** (`config/investable_universe.yaml`): 62 liquid securities, each tagged with
  asset class, region, currency, Sharia status, domicile, factor and its UCITS twin. Global and
  regional equity ETFs (US-listed and Irish UCITS), GCC country ETFs, GCC blue chips (Aramco, Al
  Rajhi, SNB, Emaar, Emirates NBD, QNB, NBK and others), bonds, sukuk, gold, cash, real estate,
  factor ETFs and Sharia funds. `scripts/verify_universe.py` checks every ticker on Yahoo Finance
  and records its first price date (`config/investable_history.yaml`).
- **Screen**: each security is kept or excluded by the client's IPS (Sharia, excluded sectors,
  restricted securities, US-person status) and history (10 years; 6 for Sharia funds, which start
  in 2019), with the reason shown. GCC blue chips join only when the IPS asks for GCC exposure
  (a GCC or GCC-region minimum, or Sharia only); otherwise GCC exposure comes through the
  US-listed country ETFs.
- **Candidates**:
  - *Max information ratio*: maximise w'(μ − μ_b) / √(w'Σₐw), where Σₐ is the covariance of each
    security's return minus the benchmark's, with tracking error √(w'Σₐw) at most 4% a year
    (editable), volatility at most the benchmark's, and the IPS and weight limits. Holdings under
    2% are pruned and the problem re-solved, to 8-15 holdings.
  - *Max IR with a tilt*: the shortlist favours low volatility or momentum (12-1 month return),
    both scored from trailing prices, or quality through quality-factor ETFs (QUAL, IQLT).
  - *Diversified (IPS)*: max Sharpe on the screened sleeves, the fallback.
  - *Benchmark tracker*: the benchmark's own funds (UCITS twins for a non-US person), split
    across twins when one fund would break the weight cap.
- **No look-ahead**: securities as well as weights are re-chosen every quarter using only data
  available then (risk from all history so far, returns from the last year), held to the next
  quarter, after costs per market. Quarterly re-selection keeps a run under a minute.
- **The rule** (`recommend.recommend_build`, every branch tested): among builds that pass the IPS,
  held up out of sample, kept volatility at most the benchmark's (+0.50 pp) and tracking error
  within the limit (+0.50 pp), and beat the benchmark out of sample after costs (excess return
  above zero and information ratio at least 0.10), pick the highest information ratio. Below 0.10
  the difference is treated as noise. If none qualifies, the *next best option*, clearly labelled
  "does not beat the benchmark": the tracker if it passes the IPS, else the diversified portfolio.
  If the current portfolio passes the IPS and its walk-forward Sharpe ratio is at least the
  fallback's, the rule says keep it. If nothing passes the IPS, the closest is shown with its
  breaches (`closest_fails`), never as suitable.
- **Shown**: securities chosen and why (tags and the reason each was picked), latest weights,
  walk-forward table (CAGR, volatility, Sharpe, max drawdown, turnover, excess return,
  information ratio, tracking error, hit rate, worst relative drawdown), growth and
  relative-drawdown charts against the benchmark (grey dashed), and the suitability checks.

## Improve my current portfolio

*4 Optimise › Improve my current portfolio* keeps the client's holdings and asks whether a better
mix, or a few explained swaps, would have done better.

- **Re-weighting**: max Sharpe, min variance, risk parity, max diversification and target
  volatility, each within the IPS and the form's limits, re-optimised monthly on the trailing
  window (Ledoit-Wolf covariance), after costs.
- **Turnover limit** (30% of the portfolio, editable): each month's move, and today's suggested
  change, go part-way from the current weights to the target until turnover is within the limit
  (`construct.limit_turnover`). Because every limit is linear or convex, every point on that line
  is feasible when both ends are. Limits come first: if today's portfolio breaks an IPS limit, the
  move needed to meet it is made even when it is larger, and the app says so.
- **Swaps** (up to 3, from the curated universe, trailing data only): rule-based, each with a
  written reason. *Duplicates*: two holdings whose weekly returns correlate 0.95 or more (or a
  UCITS fund and the US fund it mirrors): sell the one with the lower risk-adjusted return and
  give its weight to the other. *Missing diversifier*: no bond, sukuk or cash holding: add the
  eligible bond or sukuk fund least correlated with the portfolio. "Re-weight + swaps" is one more
  candidate under the same rule.
- **Domicile ideas**: for a non-US person, US-domiciled funds are paired with their Irish UCITS
  equivalents, labelled information, not tax advice, and kept out of the recommendation rule.
- **The rule**: recommend a change only if it passes the IPS and beats the current portfolio's
  walk-forward Sharpe ratio by at least 0.05 (`min_sharpe_gain`); otherwise keep the current
  portfolio. If the current portfolio fails the IPS, the best passing method is recommended and
  labelled as needed for suitability, even without a gain.

## Benchmarks, including the policy benchmark

60/40 (SPY/AGG) is the default; the S&P 500, MSCI World, FTSE 100, Saudi and UAE market ETFs,
cash or any loaded asset can be chosen. A 60/40 benchmark can itself break a conservative client's
limits, so Build also suggests a **policy benchmark that fits the IPS**: a mix of global equity,
GCC equity, bonds and cash funds (Sharia funds for a Sharia client) with the most equity whose
full-history volatility and largest fall stay within the IPS, that meets its GCC minimum and cash
need, and that holds no fund over the weight cap. For the Dubai example it is 45% equity, 10% GCC,
35% bonds and 10% cash; when no mix fits (the conservative Sharia example), the app says so and
keeps 60/40. 60/40 stays selectable.

**Mixed calendars.** A benchmark on US trading days compared with a universe that includes Gulf
stocks (fewer common dates) must be compounded onto the same dates first
(`metrics.compound_to`, `align_compounded`). Before this was fixed, one day of benchmark was
compared with several days of portfolio, which inflated tracking error (SPY against 60/40:
10.98% instead of 7.04%) and made Max IR fail the limits in 36 of 39 quarters for the Dubai
example; afterwards, 0 of 39. A regression test pins it.

## Using it as an investor

- **Any asset, one portfolio.** Your holdings are the universe for every step. Search by name or
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
- **Your own portfolio (step *2 Portfolio*).** Holdings show each security's latest price and
  quote time; edit them in USD or percent (white columns are editable), use *Equal weight* or
  *Normalise to 100%*, and assign groups. *Screen* ranks holdings by any of 14 measures (return,
  CAGR, 1Y and 3Y return, volatility, Sharpe, Sortino, Calmar, max drawdown, VaR, expected
  shortfall, beta to the benchmark, share of portfolio risk, correlation to the rest). *Analyse &
  backtest* shows share of money vs share of risk and a backtest after costs vs the strategies.
- **Benchmarks.** Compare against 60/40 (default), the S&P 500 (SPY), FTSE 100 (ISF.L, converted
  from pence to USD), MSCI World (URTH), the Saudi (KSA) or UAE (UAE) market, cash (13-week
  T-bills) or any loaded asset. These are total-return ETF proxies, so dividends count as they do
  in your portfolio. The *Benchmark* view (step 3) shows return, volatility, Sharpe, max drawdown, beta,
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
- **Ask the analyst.** The AI analyst (step 6, or *Ask the analyst* in the header) opens on your portfolio (KPI cards vs the benchmark,
  holdings with share of risk, allocation). *Review my portfolio*, *Stress test my portfolio* or
  *Explain the Optimise recommendation*, then ask follow-ups such as "what if I cut NVDA to 10%?".
  Each memo section is followed by the charts the AI picked from a fixed list: current vs proposed
  weights, growth of $1, drawdowns, a metrics table with deltas, money vs risk, a correlation
  heatmap, rolling one-year volatility, the efficient frontier, a Monte Carlo fan, exposure by
  region, asset type and currency, and stress tests (COVID crash 19 Feb – 23 Mar 2020, oil crash
  6 Mar – 21 Apr 2020, rate shock 3 Jan – 12 Oct 2022). A what-if sets the holdings you name to
  your exact weights and scales the others pro-rata; later charts compare it with today.
- **Latest or reproducible.** The dashboard opens on the latest data ("Data as of …" at the top). The
  *Reproduce README results* option uses the fixed sample below and reproduces the tables in
  *Key results* exactly.

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
The seventeen tools (`get_universe`, `get_asset_metrics`, `optimise_portfolio`,
`backtest_strategies`, `simulate_future`, `get_my_portfolio`, `get_portfolio_diagnosis`,
`get_optimisation_result`, `show_chart`, `what_if_portfolio`, `stress_test`, `get_regime_view`,
`check_suitability`, `check_domicile`, `get_drift_and_trades`, `get_client_profile`, `calculate`)
wrap the engine and return pre-formatted numbers. `calculate` does simple arithmetic (sums,
differences, shares) in code, and only on numbers already returned by a tool or written by the
user.
The system prompt forbids arithmetic and requires a bracketed tool citation for every figure. After
each answer,
code compares every decimal, percentage and multiple in it with the tool outputs and flags any that
don't match (a re-rounded "15.3%" for "15.27%" is flagged). Every tool call is shown in an audit
trail, and the disclaimer is appended by code.

**Charts the AI can't fake.** `show_chart` takes only a chart type from a whitelist and the
section it belongs to; its schema has no numeric fields. Code computes the chart's data with the
engine (`gpl/agent/charts.py`), keeps it, draws it from that data alone, and returns formatted
key figures the AI may quote. Tests check every chart type against an independent calculation
and that the AI's text cannot change a chart. `what_if_portfolio` refuses any weight that isn't
in the user's own message ("halve NVDA" gets a question back, not a computed 10%).

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
portfolio for the goal, the rule says to keep it. Where the rule compares Sharpe ratios
(*maximise risk-adjusted return*, and the Sharpe test in *beat the benchmark with less risk*), a
change must beat the current portfolio's Sharpe ratio by at least 0.05 (`min_sharpe_gain` in
config): smaller gains are within estimation noise and not worth the trading. Every branch is
unit-tested.

Example on the fixed sample (holdings SPY 50%, AGG 30%, GLD 10%, KSA 10%; at least 10% GCC;
walk-forward 3 Oct 2016 – 30 Sep 2026, after costs, against 60/40):
- *Reduce risk* (at most 45% per asset): the rule chose risk parity, with volatility 7.69% vs
  10.73% and max drawdown −14.47% vs −21.18%, but CAGR 7.34% vs 10.29% and Sharpe 0.65 vs 0.75. It is
  a lower-risk choice, not a better risk-adjusted one.
- *Beat the benchmark with less risk* (at most 40% per asset): max Sharpe ranked first (CAGR
  9.95% vs the benchmark's 9.83%, volatility 10.16% vs 11.28%), but its Sharpe ratio beat the
  current portfolio's by under 0.01 (0.7575 vs 0.7512) while trading 1.70× a year vs 0.24×, so the
  rule says to keep the current portfolio. Before the 0.05 minimum was added it recommended max
  Sharpe here, while three more weeks of data said "keep": the minimum removed that flip. Across
  20 goal-and-portfolio checks it changed only these near-ties; larger gains (e.g. +0.20 of
  Sharpe with NVDA in the portfolio) are still recommended.

**Testing.** 936 tests (`pytest`), run on every push by GitHub Actions, most with expected values small enough to check by hand. They
include property tests (no optimiser output exceeds the weight cap across random problems; features
and labels don't change when the future changes) and headless dashboard tests with synthetic data and
a scripted fake model, so the suite never touches the network.

## Limitations

- **Build and Improve.** The curated universe is today's funds, so a backtest over it can't
  include funds that closed (survivorship). Choosing the best of several candidates on the same
  period is itself a mild form of hindsight; the 0.10 information-ratio and 0.05 Sharpe
  thresholds are judgements, not estimated from data. Sharia and UCITS funds have shorter
  histories (Sharia from 2019), and Gulf stocks reduce the common dates (about 180-200 a year
  with US funds), so samples are shorter and noisier. Tilts are simple price-based scores, not
  a factor model. The "held up" test needs an in-sample Sharpe from a full-sample build; when
  that build can't meet the limits, the candidate counts as not held up.
- **Turnover limit when limits bind.** If today's portfolio breaks an IPS limit, the turnover limit
  gives way, and because the move runs along the line to the target, it can trade the whole way
  (e.g. 107% of the portfolio in the demo screenshot) rather than the least needed to meet the
  limit. The app states the turnover in each case.
- **Adviser tools are decision support, not advice.** The risk bands (e.g. band 3: 12%
  volatility, 20% drawdown, 30% in one holding) are judgement defaults, not calibrated to client
  outcomes. The required return (income ÷ investable assets after cash set aside, plus inflation)
  assumes a flat inflation rate. Sharia
  status and sectors come from a partial config list plus Yahoo sectors; anything unknown is
  "review", never a pass. Cash-like means the "Cash & short-term" group, a short config list
  (T-bill, short Treasury and sukuk funds) or volatility at most 3%; it is a proxy, not market
  depth, and a broad bond fund such as AGG does not count. The estate-tax figure is an illustration under 2026 statutory rates with no
  treaty, deductions or other US assets; the UCITS list is small, curated and only verified to
  exist on Yahoo (costs and tracking not compared). Suitability risk figures use the full common
  history, while performance uses the backtest period, so their drawdowns can differ.
- **One sample period.** About ten years dominated by US equities, COVID and the 2022 bond sell-off.
  Results may not generalise; every figure is tied to the sample shown.
- **Short GCC history** (from 2015) and a universe of ETFs chosen today (survivorship).
- **Simple costs:** a flat rate per market (10 bps for US listings; 30–40 bps for GCC exchanges are
  judgement estimates, not quotes), with no market impact by trade size and no taxes.
- **UAE prices on Yahoo** have almost no Fridays since Dubai and Abu Dhabi moved to a Monday week
  in 2022, so UAE stocks still trade on fewer common dates than US assets.
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
- **The minimum improvement covers Sharpe comparisons only:** goals judged on volatility or
  return (reduce risk, target return, target volatility) accept any improvement on their own
  metric, so a near-tie there can still flip as data is added. The 0.05 level is a judgement,
  not estimated from the data.
- **Analyst charts:** "proposed" growth, drawdown and metrics apply today's proposed weights to the
  past (hindsight). Stress tests hold today's weights through each episode, rebalanced daily
  without costs, and need the holdings' common history to cover it; with Sunday–Thursday
  holdings an episode starts at the nearest earlier common date (e.g. the oil crash from 5 Mar
  2020). Region tags are approximations (known fund exposure, otherwise country of listing).
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
- **The once-seen 75% cap.** Early on, the Frontier once showed weights for a 75% cap with the 40%
  default in place. The cause was never found. Tests now check the Frontier against the cap in use
  for 30%, 50% and 75%, in a fresh session, and after the cap is changed to 75% and back to 40%;
  all pass, and the cap in use is shown on screen.
