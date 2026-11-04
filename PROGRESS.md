# Progress log

How to read this log: the code in this repository was written by Claude Code (Anthropic's coding
agent) at my direction. Each entry records what was built, what I directed, reviewed or decided, and
what I learned from the explanations and quizzes. The plan in `BRIEF.md` is organised in weeks and
days; the work itself was done in a series of sessions.

## Day 1 — 2026-10-23 · project setup and data.py
- **Built by Claude Code:** the repository skeleton (src-layout `gpl` package, `pyproject.toml`, config,
  test fixtures, a documented stub for every module), then `get_prices` (Yahoo Finance download with a
  parquet cache), `clean_prices` and `to_returns`. All dependencies installed on Python 3.14.
- **Directed:** I approved the Week 1 plan. I started in a tutor mode, intending to write the finance
  code myself with hints, and then changed the approach to "build + explain": Claude Code writes the
  code, explains each function, and quizzes me after each module. CLAUDE.md and BRIEF.md were updated.
- **Learned:** adjusted vs raw close prices (dividends and splits); why only short gaps are
  forward-filled (long fills understate volatility); simple returns add across assets, log returns
  add across time.
- **Open:** the data.py quiz is still to be answered.
- **Next:** risk-free rate and core metrics.

## Day 2 — 2026-10-23 · risk-free rate and core metrics
- **Built by Claude Code:** `get_risk_free` (^IRX to a daily rate), annualised return, CAGR,
  volatility, Sharpe and Sortino, with tests checked against hand-worked values.
- **Learned:** volatility scales with √time because variances add; CAGR is below the arithmetic mean
  when returns are volatile ("volatility drag"); Sortino only penalises downside.
- **Open:** ^IRX is a discount yield, not a compounding yield; the difference (a few basis points) is
  ignored for now.
- **Next:** drawdowns and tail risk.

## Day 3 — 2026-10-23 · drawdowns, VaR, expected shortfall, summary table
- **Built by Claude Code:** drawdown series, max drawdown, historical and parametric VaR, expected
  shortfall, `summary_table`.
- **Learned:** VaR describes a normal bad day; expected shortfall describes the days beyond it.
  Historical VaR only knows the crashes in its sample, so it can understate tail risk.
- **Next:** the optimiser.

## Day 4 — 2026-10-23 · mean-variance optimiser
- **Built by Claude Code:** `portfolio_performance`, min variance, max Sharpe and the efficient
  frontier (SLSQP, long-only, 40% cap).
- **Learned:** with uncorrelated assets, min variance weights are proportional to 1/variance, a useful
  sanity check. Max Sharpe is very sensitive to expected-return estimates.
- **Next:** risk parity.

## Day 5 — 2026-10-23 · risk contributions and risk parity
- **Built by Claude Code:** `risk_contributions`, `risk_parity`.
- **Learned:** equal *weights* are not equal *risk*; in a 1/n portfolio the volatile assets dominate.
- **Open:** risk parity had no 40% cap and reached 43% in AGG in the backtest.
- **Next:** backtester.

## Day 6 — 2026-10-23 · backtester
- **Built by Claude Code:** a monthly-rebalanced backtester with weight drift, turnover and 10 bps costs,
  and strategy builders (equal weight, fixed weights, min variance, max Sharpe, risk parity).
- **Learned:** no look-ahead means weights chosen at the close of day *t* only earn from *t + 1*; a
  test with a deliberately "cheating" strategy checks this.
- **Fix:** the first risk parity solver (L-BFGS-B) stalled on real data and was replaced by cyclical
  coordinate descent.
- **Next:** Monte Carlo and results.

## Day 7 — 2026-10-23 · Monte Carlo and results notebook
- **Built by Claude Code:** `simulate_paths` (bootstrap and GBM), `terminal_stats`,
  `goal_probability`, plotly chart helpers, and `notebooks/01_week1_results.ipynb`.
- **Results (Oct 2016 – Oct 2026, net of costs):** SPY had the best Sharpe (0.74), then 60/40 (0.67)
  and max Sharpe (0.55). Min variance was lowest (0.31) but had the smallest drawdown (−15% vs −34% for
  SPY). Max Sharpe traded about 3.9× its value per year.
- **Open:** whether to add Jupyter, and reviewing the charts myself.
- **Status:** 125 tests passing.
- **Next:** answer the quizzes, decide on capping risk parity, then Week 2.

## Week 1 follow-up — 2026-10-23 · weight cap, fixed dates, Jupyter
- **Decided:** cap risk parity at 40% to match the max-weight rule; fix the end date at 2026-09-30 so
  results are reproducible; add Jupyter as a development dependency.
- **Built by Claude Code:** capped risk parity (capped assets sit at the cap, the rest share risk
  equally).
- **Bug found:** `get_prices` treated `end` as exclusive (yfinance's rule) while its documentation said
  inclusive; fixed and tested.
- **Learned:** the cap only applied in 9 of 120 months (AGG, 2018–19) and moved risk parity's Sharpe by
  0.0003. Most of the small changes in results came from moving the end date, not from the cap.
  Rankings and max drawdowns did not change.
- **Next:** review the charts in Jupyter, answer the quizzes, then Week 2.

## Week 2 — 2026-10-23 · regime classifier, regime-aware strategy, dashboard
- **Built by Claude Code:** `regime.py` (forward-looking stress labels, eight backward-looking
  features, logistic regression and random forest, time-series cross-validation with a 21-day gap,
  monthly walk-forward probabilities, a simple volatility-rule baseline); `regime_switch_strategy` in
  the backtester; and a Streamlit dashboard. I approved adding scikit-learn and Streamlit.
- **Results:** logistic regression caught 67% of stressed days, but only 34% of its "stressed" calls
  were right (base rate 22%); its accuracy (65%) was *below* always predicting calm (79%). Switching to
  min variance on its signal cut drawdowns (equal weight: −21.6% → −17.1%; COVID −20% → −14%) but
  lowered Sharpe for every strategy (equal weight 0.49 → 0.42). The simple volatility rule beat the
  model on Sharpe.
- **Learned:** a label that looks 21 days ahead leaks into the test fold unless a gap is left, and a
  model refit on day *t* can only train on labels up to *t − 21*. Accuracy is the wrong metric when 80%
  of days are calm. Reducing risk usually costs return.
- **Open:** whether a higher threshold would reduce false alarms (choosing it on this sample would
  overfit). GCC-specific stress is not modelled.
- **Next:** review the dashboard, then Week 3.

## Week 2 fixes — 2026-10-23 · weight cap, max Sharpe robustness, frontier labels
- **Reviewed and reported:** while testing the dashboard I found the Frontier tab showing weights
  above the 40% cap, and asked for the cap to be enforced everywhere, for a test that no optimiser
  exceeds it, and for the frontier labels not to overlap the curve.
- **Built by Claude Code:** the optimiser now rejects infeasible solver output instead of rescaling
  it; max Sharpe is solved in convex form (minimise risk for a fixed excess return, then rescale) with
  a multi-start fallback, giving 0 failures in 6,000 randomised tests; frontier labels are placed away
  from the curve; the tabs show the cap in use.
- **Unresolved:** once, with default settings (no slider, asset or date changes, no reload), the
  Frontier tab showed min variance at 75% AGG and max Sharpe at 52.6% SPY / 47.4% GLD, exactly the
  results for a 75% cap. It could not be reproduced and now shows 40%.
- **Learned:** maximising a ratio directly is fragile; rewriting it as a convex problem is the standard
  remedy. "Cannot reproduce" is not the same as "not a bug", so it is recorded here.

## Regime fixes — 2026-10-23 · calibration, honest metrics, monthly chart
- **Directed:** I asked for a check on whether the model's probabilities were calibrated, for ROC AUC
  and precision against the base rate as headline metrics, for the volatility rule as a baseline, and
  for a monthly chart.
- **Found:** class weighting made the average predicted P(stressed) 47.41% against a 21.51% actual
  rate, so it was a score rather than a probability. The weighting was removed instead of adding
  time-series calibration (`CalibratedClassifierCV`), because the calibrated version ranked worse (AUC
  0.67) and had a worse Brier score than always predicting the base rate. Out of sample, the mean
  prediction is now 28.86% vs 21.51% actual; most of the gap is the first fold, which was trained on
  the 2008 crisis, and later folds are within 1–4 points.
- **Comparison:** the simple volatility rule (precision 58.33%, recall 51.35%, accuracy 81.65%) beats
  the model at the 50% threshold (47.89% / 40.45% / 77.73%). The model ranks slightly better (AUC 0.72
  vs 0.71). Neither beats always-calm accuracy (78.49%) by much.
- **Knock-on:** with calibrated probabilities, equal weight + regime de-risks in 11% of months (was
  36%); Sharpe 0.39 (was 0.42; static equal weight 0.49). The Week 2 strategy figures above are out of
  date.
- **Learned:** class weights trade calibration for recall; ROC AUC ignores calibration, the Brier score
  does not. A threshold only means something if the probabilities are calibrated.
- **Notebook 02 updated:** new figures and a key-results summary, with every figure in the text printed
  by the notebook. New finding: 2022 *was* clearly stressed (55.38% of days), but the month-end
  probability crossed 50% back and forth (January above, February–March below, April above, May below,
  June above), so equal weight + regime switched in and out: 0.44% in costs vs 0.04%, and −13.18% vs
  −12.57% for the static portfolio. A gradual or two-month rule might help (untested).

## Week 3 — 2026-10-23 · AI analyst, chat panel, README
- **Directed:** I asked for a tool-use agent with a cheap default model, a chat panel and a README, and
  approved adding `anthropic` and `python-dotenv`.
- **Built by Claude Code:** an agent (Claude Haiku 5.5 by default, set in config) with six tools that
  wrap the finance engine; a memo writer whose numbers must be copied from tool results and cite the
  tool; a checker that flags any figure no tool returned; an Analyst tab with example profiles,
  follow-up chat, an audit trail and a cost line; and the README.
- **Live runs:** a moderate-risk, 10-year Dubai profile took 5 tool calls, about 15,500 input and
  2,700 output tokens, about $0.003 and 23 seconds. The memo recommended 60/40 with every figure cited.
- **Caught:** the first live memo mentioned "a 20–30% fall", which no tool produced, and the checker
  missed it because a date (2026-09-30) contains "30". Fixed: only figures with units or decimals
  count as quotable.
- **Learned:** "the LLM never calculates" has to be enforced in code, not just in the prompt; tools
  should return formatted numbers so there is nothing to round; the agent loop can be tested with a
  scripted fake model.
- **Next:** publish to GitHub and deploy on Streamlit Community Cloud; answer the outstanding quizzes.

## Real-investor features — 2026-10-23 · any asset, currencies, calendars, My portfolio
- **Directed:** I asked for any Yahoo ticker with GCC quick picks, USD conversion (pegs for the Gulf
  currencies, market rates otherwise), aligned trading calendars, history checks, a My portfolio mode
  with an analyst memo, and latest data by default with the README sample as an option. I chose
  caching only (no price data in the repo), a "Data as of" line and a Retry button.
- **Checked before building:** FAB has no Yahoo data (Abu Dhabi isn't covered), Kuwaiti prices come
  in fils, and Yahoo's pegged-currency FX series are noisy, which supports using the official pegs.
- **Built by Claude Code:** `load_assets` (per-ticker cached downloads with retry, currency lookup,
  USD conversion, common-date alignment, history report); annualisation with observed periods per
  year; the new sidebar, data notes and My portfolio tab; and analyst support for my portfolio.
- **Caught:** a too-short common sample crashed the backtest (it now explains which asset limits it).
  A first idea for charging risk-free interest across date gaps changed existing results, so it was
  replaced by a simpler, consistent rule. A live memo misread the backtest dates as an error, so the
  tools now explain them.
- **Learned:** mixing Sunday–Thursday and Monday–Friday markets leaves about 183 common dates a year,
  so annualising with 252 would overstate returns and volatility; currency pegs are better than noisy
  market quotes; and a portfolio's risk can be very different from its weights (SPY at 50% of the
  money was 79% of the risk).
- **Next:** deploy on Streamlit Community Cloud; answer the outstanding quizzes.

## Portfolio workstation — 2026-10-23 · search, holdings, screen, AI-assisted optimisation
- **Directed:** I asked for a mini portfolio workstation (like Bloomberg PORT) as the first tab, built
  around my own portfolio, with an optimiser I can drive in plain English, where the AI turns my goals
  into limits I confirm and never invents weights. I approved `streamlit-searchbox` and merging the
  old My portfolio tab into it.
- **Built by Claude Code:** a constrained optimiser (five objectives, per-asset and group limits,
  volatility cap, return floor, exact feasibility checks with plain-language reasons); security
  search and live quotes; the workstation (holdings table in USD or %, allocation chart, screen with
  correlation to the rest, analyse & backtest); goals → spec → confirm → optimise, with current vs
  optimised weights, hindsight and walk-forward backtests, and a memo.
- **Live test:** "max 12% vol, at least 20% GCC, no single asset over 25%, maximise Sharpe" on SPY,
  AGG, GLD, Aramco and Emaar was read correctly. Walk-forward after costs: 14.45% CAGR, Sharpe 1.00 vs
  11.90% and 0.87 for my current mix, and below the hindsight 1.08. An impossible request (60% GCC with
  8% volatility) was refused with the reason: the lowest volatility allowed was 12.31%.
- **Caught:** a quote lookup for a currency looped forever; the walk-forward failed outright when the
  first month couldn't meet the limits (it now keeps my current weights and counts those months); the
  Interpret button stayed disabled until the text box lost focus; a live memo questioned the sample
  dates, so the report now explains them.
- **Learned:** an optimiser should say *why* limits can't be met, not just fail; hindsight backtests
  flatter, walk-forward is the fair test; with four holdings and a 25% cap there is only one possible
  portfolio, so "optimised" can have a lower Sharpe than what I hold.
- **Status:** 417 tests passing.
- **Next:** deploy on Streamlit Community Cloud; answer the outstanding quizzes.

## Quant-style Optimise, benchmarks, holdings everywhere — 2026-10-24
- **Directed:** I asked for the Optimise tab to work like a quant analyst (diagnose, agree the goal,
  try several methods with Ledoit-Wolf, judge them walk-forward after costs, recommend the one that
  fits and held up), plus Portfolio fixes: obviously editable weights with Equal weight and
  Normalise buttons, holdings driving every tab, a richer Screen, and a benchmark comparison. I chose
  the benchmark set (60/40 default, S&P 500, FTSE 100, MSCI World, Saudi, UAE, cash, any asset) and
  decided that a coded rule, not the AI, makes the recommendation.
- **Built by Claude Code:** beta, Jensen's alpha, tracking error, information ratio, up/down
  capture, Calmar, trailing returns, relative drawdown and correlation clusters (hand-value tests);
  benchmark loading on my portfolio's dates; risk parity and max diversification within any limits;
  Ledoit-Wolf covariance; `recommend.py` with one tested rule per goal; diagnosis and candidate tools;
  the new Optimise and Benchmark tabs. The sidebar asset picker is gone; README mode plus the README
  preset still reproduces the README table exactly.
- **Caught:** against cash, the drifting T-bill rate gave a beta of 0.22 and a 355% "up capture",
  so those are now undefined for cash. A live memo read "months the limits couldn't be met" as
  "breached the limits", so the label now says the previous weights were kept. The diagnosis and
  the candidates table disagreed by 0.02% of CAGR (one included the buying day); now aligned.
  Streamlit can't colour editable columns, so the read-only ones are greyed instead. I could not
  reproduce the empty pie chart.
- **Learned:** a method that wins in sample can fall apart out of sample (synthetic test: max Sharpe
  1.13 in sample, 0.17 walk-forward); "reduce risk" usually costs return (risk parity: volatility
  7.69% vs 10.73%, Sharpe 0.65 vs 0.75); against cash the information ratio is the Sharpe ratio.
  The rule recommended max Sharpe for a 0.006 Sharpe gain at 1.70× turnover, and the answer flipped
  with three more weeks of data, so it may need a minimum improvement.
- **Status:** 479 tests passing.
- **Next:** decide on a minimum-improvement threshold; deploy; answer the outstanding quizzes.

## Terminal-style UI and a charting analyst — 2026-10-25
- **Directed:** I asked for a Bloomberg-style dark UI without a sidebar (settings in a popover,
  benchmark always visible), KPI cards, an Analyst page centred on my portfolio whose memo
  sections each come with a chart the AI picks but code builds, historical stress tests, and
  what-if follow-ups. I chose pro-rata rescaling for what-ifs and kept the profile memos.
- **Built by Claude Code:** the theme and one plotly template (palette checked for colour-blind
  separation on the dark surface); KPI cards with deltas vs the benchmark; `scenarios.py`
  (what-if weights, stress episodes, exposures); `show_chart` with eleven whitelisted chart types,
  `what_if_portfolio` and `stress_test`; the new Analyst page with sections paired with charts.
- **Live test:** with SPY 35%, NVDA 20%, AGG 25%, GLD 10%, Aramco 10%, the review showed NVDA at
  20% of the money but 56.06% of the risk; in the 2022 rate shock the portfolio lost 25.83% vs
  20.19% for 60/40. "What if I cut NVDA to 10%?" gave volatility 12.13% vs 15.92% and max drawdown
  −21.18% vs −26.98%, but CAGR 14.33% vs 19.99% (hindsight). Stress numbers matched a separate
  calculation.
- **Caught:** charts whose section the AI misnamed ended up after the disclaimer (now before it);
  a "2x" the tool had returned was flagged because it was only inside a field name; exposure labels
  were clipped; KPI deltas were truncated; importing Streamlit replaces plotly's default template,
  so the template is set on every figure.
- **Learned:** a chart is only trustworthy if the model can't pass it numbers; with
  Sunday–Thursday holdings an episode starts at the nearest earlier common date.
- **Status:** 516 tests passing.
- **Next:** decide on a minimum-improvement threshold for recommendations; deploy; quizzes.

## Minimum Sharpe improvement — 2026-10-25
- **Decided:** a recommended change must beat my current Sharpe ratio by at least 0.05 wherever the
  rule compares Sharpe ratios (configurable as `min_sharpe_gain`).
- **Built by Claude Code:** the threshold in `recommend.py`, a reason that says when a gain was too
  small, and tests at the boundary (0.05 counts, 0.049 doesn't, rounding can't break it).
- **Effect:** of 20 goal-and-portfolio checks, only the two near-ties changed (0.7575 vs 0.7512 on
  the README sample) and now say "keep", which matches the latest data; a +0.20 gain is still
  recommended.
- **Learned:** a rule with no materiality threshold turns estimation noise into trades. Goals
  judged on volatility or return still have no minimum.
- **Status:** 526 tests passing.

## Wealth-manager workflow — 2026-10-25
- **Directed:** following my roadmap document, I asked for a client profile and IPS builder
  feeding the optimiser, a suitability checker, a client report PDF, a drift monitor with a trade
  list, and a GCC fund-domicile check, all as decision support for a licensed adviser. I approved
  reportlab and matplotlib for the PDFs, chose an editable target column for drift, and agreed to
  make data.py writable for a cached Yahoo sector lookup.
- **Built by Claude Code:** `client.py` (questionnaire: risk score = lower of capacity and
  willingness; RRTTLLU IPS; IPS limits -> optimiser limits), AI form filling that never touches
  the questionnaire, `suitability.py` (eight rules, reasons written from the numbers),
  `report.py` (IPS and client report PDFs), drift and cash-neutral trade list, `domicile.py`
  (US-situs check, estate-tax schedule, Irish alternatives verified on Yahoo).
- **Checked live:** the AI filled a Dubai client from notes and its own conversion of "12 months"
  to "1 year" was flagged; a memo's suitability section listed three fails without softening
  them; a USD 500,000 all-US-situs portfolio shows an illustrative USD 142,800 estate tax.
- **Caught:** the strict tool schema rejected an enum with null (fixed with anyOf); "2x" wasn't
  read as a number; the PDF had near-empty pages, a clipped label and a legend over the data; the
  report's two drawdowns (-13.81% backtest vs -21.67% full history) needed an explanation; AGGU
  is a global index, so AGG maps to IUAA instead.
- **Learned:** the lower of risk capacity and willingness governs; US-listed ETFs, including the
  KSA fund, are US-situs for a Gulf investor; a trade list must pay its own costs.
- **Status:** 582 tests passing.
- **Next:** deploy; answer the outstanding quizzes; review the risk bands with a practitioner.

## QA fixes from a full client walkthrough — 2026-10-25
- **Directed:** after walking a client through the whole workflow, I asked for six fixes, most
  severe first: a suitability gate on Optimise, cash needs as hard limits, one portfolio value,
  a stable holdings grid, a better AI intake, and a client-ready report.
- **Built by Claude Code:**
  - *Gate:* candidates and the current portfolio are checked against the IPS; only passing ones
    can be recommended. If none passes, the closest is shown with each breach; identical
    methods are merged.
  - *Cash needs:* dated needs; those due within 36 months bind as an optimiser set over
    cash-like holdings. The trade list is checked, and topped up so costs don't leave it short.
  - *One value:* from USD holdings or entered once; the $10,000 default is gone; importing USD
    amounts keeps them.
  - *Grid:* edits applied in a callback, blank cells stored as None, the searchbox pick handled
    once, tabs tracked.
  - *Intake:* name, age, jurisdiction → regulator by rule, tax notes, dated needs (code does the
    date maths), region minimums.
  - *Report:* plain language, the optimiser's recommendation, trades, sign-off, clean memo.
- **Finance behaviour change:** AGG (5.26% volatility) no longer counts as cash. On the default
  portfolio, the Dubai and Riyadh examples' liquidity checks change from pass to fail; with 15%
  in BIL they pass.
- **Caught:** a blank cell looped because NaN never equals itself; a removed stock came back
  from the searchbox's stale pick; the live API rejected the intake schema (20 union fields,
  limit 16); a real trade list was USD 360 short of the cash need after costs; suitability
  crashed with no stress episodes configured.
- **Checked:** the real grid in Chrome (two quick edits kept, blank cell, remove, tab kept); a
  live intake of an Abu Dhabi client (FSRA, Oct 2028 = 24 months, Saudi ≥ 10%); a real 7-page
  report rendered page by page (bullets, no tool tags).
- **Learned:** constraints must hold after costs, not just in the optimiser; a client report needs
  the optimiser's result, not the last chat answer.
- **Status:** 625 tests passing.
- **Next:** a practitioner review of the cash-like list and region names; deploy.

## QA batch 2 — 2026-10-25
- **Directed:** a second QA pass with nine fixes: periods on headline figures, date-aware
  calendars, Sharia funds and benchmark, optimiser limits, market costs, data quality, the IPS
  required return and band caps, a client-aware analyst with a calculator, and small UI fixes.
- **Built by Claude Code:**
  - *Periods:* every headline block names its dates; suitability shows full history and the
    backtest period.
  - *Calendars:* split at the UAE change.
  - *Sharia:* a curated list (SPSK sukuk, SPRE, UMMA, iShares Islamic UCITS) and a Sharia 60/40
    benchmark; domicile wording that tells funds from stocks.
  - *Optimise:* goal changes reset limits; boxes clear; an active-limits table with sources.
  - *Costs:* per-market rates; tickers without prices rejected; late starters named, with an
    Exclude button.
  - *IPS:* required return 6.60% in the QA case; caps 20–40% by band.
  - *Analyst:* `get_client_profile` and `calculate` tools.
  - *Small fixes:* compact KPI value; no 100.01% notice; 1-2-5 percent ticks; one sample
    count; usage stats off.
- **Findings:**
  - Yahoo has almost no UAE Friday prices after 2022 (Mon–Thu in the data), so UAE stocks
    still align on fewer dates.
  - GCC costs cut the example Max Sharpe CAGR from 15.41% to 15.17%.
  - The new 25% cap makes the Riyadh Sharia example infeasible: only four eligible holdings.
  - A live memo listed both IPS failures and used `calculate` for a what-if.
- **Mistake:** one commit (limits) went in with a failing test because my check read the wrong
  exit code; fixed in the next commit, and every later commit checked pytest's own exit code.
- **Learned:** a rule that is right in general (lower cap for cautious clients) can leave no
  feasible portfolio for a narrow Sharia universe; the adviser must widen the holdings or the
  IPS.
- **Status:** 655 tests passing.
- **Next:** practitioner review of the cost estimates, caps and Sharia list; deploy.

## Professional upgrade (5 batches) — 2026-11-04
- **Directed:** make the tool something a wealth manager could use day to day, in five
  batches (visible bugs, full fact-find, step-by-step UX, light private-bank theme, a
  "Fill demo client" button), tested and committed after each. I chose an azure accent and
  the AI analyst as a step-6 sub-tab with an "Ask the analyst" button in the header.
- **Built by Claude Code:**
  - *Bugs:* the regulator defaults from booking location or residency (incl. CBB, CMA
    Kuwait, Oman's FSA); the final IPS is blocked until complete, with a watermarked DRAFT;
    domicile asks for the value in place; "cash needs in the next two years" is set from the
    dated Cash needs table; experience moved out of willingness.
  - *Fact-find:* `gpl/factfind.py`, one registry of nine sections that drives the form
    cards, the completeness bar, the IPS gate and the PDFs. It calculates net worth,
    investable assets, savings rate, emergency-fund months and the return needed to meet
    the goals. It adds an adviser override that needs a reason, capacity for loss, a
    knowledge & experience check, and an editable suitability record built from the rules.
  - *UX:* six steps with a stepper, a sticky client header, and a Next button that lists
    what is still to do. Advanced settings sit in an expander.
  - *Theme:* off-white, white cards, navy and azure across the app, charts and PDFs.
  - *Demo:* `gpl/demo.py` generates a seeded, consistent GCC client and 5–8 real tickers
    without AI. Both are marked DEMO DATA, with the seed shown in the header.
- **Finance changes (and their effect):**
  - Willingness is now the mean of two answers. Doha example: 5/4/4 → 5/5/5 (band 5).
  - The cash-need answer is set by rule from the Cash needs table. Dubai and Riyadh examples:
    the answer falls from 4 and 3 to 2, but their scores don't move.
  - "Can't lose more than X%" now tightens the drawdown limit (capacity for loss). Before,
    it was an override that could loosen it.
- **Checked:** pytest after each batch; live-data screenshots of the Client, Analyse,
  Suitability and Report steps; a full demo run through to a prepared client report; the
  demo IPS PDF rendered.
- **Findings:**
  - Navy and azure are too close to use as two chart series (ΔE 12.8), so "proposed" is
    orange.
  - Strict tools allow at most 16 nullable fields, so some new AI fields use "" or 0 for
    "not stated".
  - UMMA starts in 2022, so it was left out of the demo pool (it cut the sample to 4 years).
  - Oman's CMA became the FSA in March 2024, not 2023 as I first said.
- **Learned:** KYC (who the client is) and suitability (what suits them) are separate
  records. Knowledge & experience decides which products are appropriate, not how much
  risk to take.
- **Open questions:**
  - Should willingness round half down now that it has two answers? Half up tilts
    towards more risk.
  - The analyst's suitability tool doesn't run the knowledge check yet.
  - The three example clients have incomplete fact-finds, so they only produce a DRAFT IPS.
- **Status:** 884 tests passing (655 at the start of the session).
- **Next:** a practitioner review of the fact-find fields and rules; deploy the public demo
  with the "do not enter real client data" notice.
