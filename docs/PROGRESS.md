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

## Open items closed — 2026-11-04
- **Directed:** finish the open items from the upgrade, then check the Optimise step, the AI
  chat and every page of the client report.
- **Built by Claude Code:**
  - The three example clients have complete fact-finds and now produce a final IPS. Their
    risk scores and limits are unchanged (a test locks them).
  - Benchmarks are grey in every backtest chart: 60/40 dashed, SPY dotted, and neither takes
    a colour slot.
  - The analyst's suitability tool runs the knowledge & experience check too.
- **Checked:**
  - Optimise on live data: with BIL added, Max Sharpe is recommended and passes every rule.
  - A live AI review: every figure cites a tool and the charts are drawn by code.
  - The real 10-page client report, downloaded from the app and read page by page. The
    trade totals add up.
- **Found in the PDF and fixed:**
  - Chart markers ("[show_chart: …]") and a citation with a note were left in the
    commentary.
  - "Important information" sat alone at the foot of a page.
  - The goals' required return printed "-0.00%" and gave the rate twice.
  - Capacity for loss had no % sign.
- **Status:** 889 tests passing.

## Example clients work out of the box — 2026-11-04
- **Directed:** each example should load its own portfolio with a cash-like holding, and a
  goal with a meaningful required return.
- **Built by Claude Code:**
  - `EXAMPLE_HOLDINGS`, loaded with the example as USD amounts of its portfolio value. Each
    includes a cash-like fund: BIL for Dubai and Doha, SPSK sukuk for Riyadh.
  - New goals:
    - Dubai: USD 1.1m for retirement in 10 years (4.84% needed).
    - Riyadh: USD 1.1m capital in 5 years (4.06%).
    - Doha: USD 2.5m financial independence in 20 years (6.70%).
- **Checked on live data:**
  - Dubai and Doha pass every IPS rule out of the box. For Doha the rule advises keeping the
    portfolio: no method beats it by 0.05 Sharpe.
  - Riyadh meets its cash need but fails volatility, 9.35% against 9.00%. The optimiser
    confirms the lowest volatility its limits allow is 9.32%.
- **Finding:** in this Sharia universe, no portfolio meets band 2 under a 25% cap with at
  least 30% in GCC (a 5% grid search: 0 of all mixes). SPSK is the only low-risk Sharia fund
  in the list, and BIL can't be used because T-bills pay interest. An early search looked
  feasible only because SPRE shortened the sample and left out the 2020 crash.
- **Open question:** add a second verified Sharia cash or sukuk fund, or let the adviser
  widen the Riyadh IPS?
- **Status:** 891 tests passing.

## Riyadh example labelled — 2026-11-04
- **Decided:** keep the Riyadh IPS unchanged, and label the example clearly instead.
- **Built by Claude Code:**
  - The label "No portfolio meets every rule: closest mix shown, with the reasons." appears
    in four places:
    - the example picker ("· closest mix");
    - the Client step;
    - the Suitability step, with each failed rule's measured breach;
    - the suitability record, so it carries into the client report.
  - The reason is written from the IPS's own limits (25% cap, 30% GCC, 9% volatility, 15%
    drawdown). The label shows only while the example's own portfolio is loaded, so it
    disappears once the holdings change or another client is loaded.
- **Checked on live data:** the Suitability step shows "Volatility: 9.35%, 0.35 pp beyond
  the 9.00% limit" under the label.
- **Status:** 893 tests passing.

## Two Optimise modes: Build and Improve — 2026-11-04
- **Directed:** two modes on one engine, run walk-forward after costs with a tested rule,
  and the AI only explaining:
  - "Build a portfolio for me": 8–15 holdings from a curated universe, aiming to beat the
    benchmark on a risk-adjusted basis.
  - "Improve my current portfolio": re-weight within a turnover limit, plus explained
    swaps.
- **I decided:**
  - momentum and low-vol tilts are scored from prices; the quality tilt uses
    quality-factor ETFs;
  - UCITS swaps are shown separately, as information outside the 0.05 Sharpe rule;
  - defaults are a 4% tracking-error limit and a 30% turnover limit;
  - GCC blue chips join only when the IPS asks for GCC exposure.
- **Built by Claude Code:**
  - A curated universe of 62 securities. Every one was verified on Yahoo (2026-11-04).
  - Max information ratio and a tracking-error limit in the optimiser, plus hit rate and a
    relative summary.
  - `construct.py`: screen, shortlist, de-duplication, builds of 8–15 holdings, swaps and
    turnover limits.
  - `recommend_build`, the tools, the app modes and the report sections.
- **Found on live data and fixed:**
  - Yahoo quotes NBK.KW and KFH.KW about 1000x too low on 31 Aug and 3 Sep 2017, which gave
    a +100,000% daily return. Short unit glitches are now repaired and logged. This affected
    any portfolio holding Kuwaiti stocks.
  - Builds used one year of risk while the IPS judges the full history. Risk now uses all
    history up to each date, and a build whose past fall breaks the IPS is rebuilt with
    less risk.
  - "Beats the benchmark" now needs an information ratio of at least 0.10, not just above 0.
    Doha's low-vol tilt had 0.02 (+0.09% a year), which is noise.
  - The fallback could recommend a portfolio worse than the client's own. The current
    portfolio is now kept when it passes the IPS with at least the fallback's Sharpe.
    Dubai: keep, 0.67 against 0.51.
  - The tracker splits SPY with its UCITS twin to respect the cap.
  - Builds re-choose quarterly: 38 s instead of 122 s.
- **Results (Nov 2026 data, walk-forward after costs):**
  - Dubai: no build beat 60/40 within the IPS. The 4% tracking-error limit can't be met
    alongside the GCC minimum and cash need (the lowest possible was 5.12–5.71% for two
    tilts). Keep the current portfolio.
  - Doha (script run): next best was the benchmark tracker, and the current portfolio's IR
    (0.12) was higher than any build's.
  - My script and the app gave different feasibility for Dubai's Max IR; I didn't find why.
- **Learned:** with a 60/40 benchmark whose own worst fall (−21.72%) breaks a 20% IPS
  limit, "beat the benchmark within the IPS" is often impossible. The honest answer is
  "keep", or a labelled next best.
- **Open questions:**
  - Should the 0.10 information-ratio threshold stay?
  - Should Build use a benchmark that fits the IPS?
  - Why did the script and the app differ on Dubai's Max IR feasibility?
- **Status:** 928 tests passing.
- **Next:** have a practitioner review the universe and thresholds.

## Calendar bug, policy benchmark, re-runs — 2026-11-04
- **Decided:**
  - keep the 0.10 information-ratio threshold;
  - Build suggests a benchmark that itself passes the IPS, with 60/40 still selectable;
  - find the cause of the script/app gap first.
- **Root cause (a bug), fixed and tested:**
  - The app's benchmark is daily on the holdings' US calendar. Dubai's Build universe (with
    Gulf stocks) keeps only the dates every market traded, 181 a year.
  - The daily benchmark was dropped onto those dates, so one day of benchmark was compared
    with several days of portfolio. SPY vs 60/40 showed 10.98% tracking error instead of
    7.04%, and Max IR missed the limits in 39 of 39 quarters (now 0 of 39).
  - The fix compounds returns onto common dates (`metrics.compound_to`), in Build, in
    Improve with swaps and in the charts.
- **Found by reading the reports, fixed and tested:**
  - Improve tables on the universe's dates were annualised as 252 a year: the current
    portfolio showed 10.81% CAGR instead of 7.77%.
  - My first fix compounded the risk-free rate as well, which the metrics already scale,
    so interest was charged twice: Sharpe 0.57 instead of 0.67. I removed it; the test
    fails with the double charge and passes without.
  - Domicile ideas claimed dividend withholding savings for gold and T-bill funds.
  - Headings could sit alone at the foot of a page.
- **Policy benchmark:**
  - Dubai: 45% equity, 10% GCC, 35% bonds, 10% cash (volatility 9.19%, largest fall
    −19.57%).
  - Doha: 80% equity, 20% bonds.
  - Riyadh: none, because SPSK, the only Sharia bond fund, can't exceed the 25% cap.
- **Re-runs:**
  - Dubai Build: keep the current portfolio against both 60/40 and the policy benchmark
    (Sharpe 0.67 vs the fallback's 0.52 and 0.46).
  - Dubai Improve (app): Target volatility, Sharpe 0.74 vs 0.66, about 30% traded.
  - Riyadh: closest only; it fails volatility in both modes.
  - Doha: Build next best is the tracker (vs 60/40) or keep (vs policy); Improve is keep.
- **Read:** every page of the Improve report and the Build "keep" report. The Build report
  shows "keep", no trades, and the reasons.
- **Status:** 935 tests passing.
- **Follow-up (same day):** I regenerated the Build "keep" report on the corrected code
  and read all 9 pages. It recommends keeping the current portfolio (Sharpe 0.67 vs the
  fallback's 0.52; the earlier version understated both, 0.54 vs 0.41) with no trades. The
  domicile heading, note and table now stay on one page. 935 tests passing.

## Public-ready repo — 2026-11-06 · README, methodology, CI, screenshots
- **Directed:** I asked for the repo to be ready for employers and admissions readers and to
  match the current app exactly: a short README, the detail moved to `docs/METHODOLOGY.md`,
  three real screenshots, CI with a badge, and the docs moved into `docs/`. I chose to keep
  Claude Code's instructions at `.claude/CLAUDE.md` so they still load automatically, and to
  describe my QA walkthrough as what it was: an AI agent (Claude in Chrome) I set up to play a
  GCC wealth manager through the browser, whose bug report I reviewed and acted on.
- **Built by Claude Code:**
  - A README of about 200 lines (pitch, demo link placeholder, screenshots, three features,
    five findings, six limitations, how it was built) and `docs/METHODOLOGY.md` with the
    full method, results and limitations, updated for Build and Improve, the policy
    benchmark, the fact-find, ten suitability rules and the six-step workflow.
  - Screenshots from the real app with *Fill demo client* (seed 2): the Build result, the
    Improve recommendation, and page 3 of the client report.
  - `.github/workflows/tests.yml`: ruff and the full suite on every push.
  - A regression test for the old "75% cap" sighting: the Frontier obeys 40% in a fresh
    session and after the cap goes to 75% and back. It passes, so the known-issue bullet is
    gone; the cause was never found, and METHODOLOGY says so.
- **Checked:**
  - The fixed-sample backtest and regime numbers reproduce exactly when the notebooks re-run.
  - The live Dubai numbers moved a little with new data: the Build fallback's Sharpe is now
    0.50 (was 0.52), and Improve's target volatility is 0.73 vs 0.66 (was 0.74).
  - The suite passes in a fresh copy with an empty price cache and the network blocked, the
    same conditions as CI.
  - Git has only ever tracked `data/cache/.gitkeep`.
- **Found while taking screenshots (not changed yet, my decision):**
  - Build counts a candidate as "not held up" when its full-sample build can't meet the
    limits (no in-sample Sharpe). The headline reason then says no build "beat the
    benchmark", even when one had an information ratio of 0.30 but failed the other tests.
    It didn't change today's outcome.
  - When today's portfolio breaks the IPS, the turnover limit gives way and the move can be
    the whole way to the target (107% traded in the demo).
  - The report's trade summary says "10 basis points" even when GCC trades cost 30-35.
- **Status:** 936 tests passing.
- **Next:** decide on the three findings above; push, so the CI badge runs; deploy and add the
  live URL.

## Turnover choice and real cost rates — 2026-11-06
- **Directed:** I asked for two of the three issues found while taking screenshots to be fixed,
  and left the "held up" rule as it is.
  - The report's trade summary must show the cost rate each market actually charges.
  - When the current portfolio breaks the IPS, the turnover limit stays by default, as the
    "closest compliant move within 30% turnover". A "full restructure to target" option shows
    its turnover and cost, and the adviser chooses.
- **Built by Claude Code:**
  - Each trade now carries its market's rate: on screen, in the CSV and in the report (a "Cost
    rate" column). The summary reads e.g. "10 bps on SPY, AGG…; 30 bps on 2222.SR, QNBK.QA;
    35 bps on EMAAR.AE" instead of a blanket "10 basis points".
  - `construct.compliant_move`: a linear program finds the least-turnover portfolio that meets
    every weight, group and cash limit; ties go to the one closest to the method's target. If
    it is within the limit, the move goes there and spends the rest heading for the target. If
    not, it goes as far as the limit allows. The walk-forward history follows the same rule
    month by month. The old "limits come first" move is gone.
  - The app adds a choice under the recommendation (closest compliant move, or full
    restructure with turnover, estimated cost at each market's rate, and IPS result). The
    choice feeds the proposal, suitability, trades and the client report; the report also
    says when a full restructure was available.
- **Effect on results:** in the seed-2 demo, Target volatility now moves 30% (SPY 45% → 30%,
  Aramco +15 pp), where before it traded 107%. Its walk-forward Sharpe is 0.92 (was 0.94).
  The full restructure is offered at 107.17% turnover and about USD 696. Dubai is unchanged:
  target volatility, 0.73 vs 0.66, because that portfolio already meets its limits.
- **Learned:** with several portfolios at the same least turnover, a solver picks a corner at
  random (the first version moved the sold 15% into KSA). A tiny second objective, distance to
  the target, makes the move sensible and deterministic.
- **Status:** 942 tests passing.

## 2026-11-07 — Streamlit Cloud start-up crash
- **Decided:** the deployed app crashed on start-up (`FileNotFoundError` in `load_universe`).
  Fix it properly, check every file the app reads, and test a non-editable install.
- **Cause:** Streamlit Cloud runs `pip install .`, which copies `src/gpl` into
  `site-packages`. The code found `config/` by going two folders up from its own file. That
  works in my editable install, but on the server it points into the Python installation.
- **Built by Claude Code:**
  - `data.find_project_root` tries, in order: the `GPL_PROJECT_ROOT` environment variable, two
    levels above the package (editable install), then the working directory and its parents
    (Streamlit starts from the repository root). It takes the first folder that has
    `config/universe.yaml`, and otherwise fails with a message listing where it looked.
  - The config files, the price cache and `.env` (the analyst) all use that one root. The
    PDFs use reportlab's built-in fonts, so they read no font files. The app itself opens no
    other files.
  - `tests/test_install.py`:
    - unit tests of the resolver;
    - a copy of the package outside `src/` run as a separate process: it finds the repo's
      config from the repo and from `app/`, follows the environment variable, and fails clearly
      outside the repo;
    - a real `pip install --target` test, which runs in CI (setuptools added there).
  - I reproduced the crash with the old code installed outside `src/` before fixing it.
- **Learned:** an editable install hides packaging bugs: paths relative to `__file__` only
  point at the repository when the code is run from inside it.
- **Status:** 951 tests (950 run locally; the real-install test needs setuptools and runs in CI,
  and passed locally with setuptools added temporarily).

## 2026-11-08 — Fixes from a live end-to-end test: Build rule, turnover, clearer results
- **Decided:** a live test of the deployed app (demo seed 44593) recommended Diversified
  (Sharpe 0.39, tracking error 14.77%) over the rejected low-vol tilt (0.49, 4.95%), which was
  better on every measure. I asked for the fallback to face the same tests as the builds, for a
  Build turnover limit, a note on why CAGR/Sharpe differ between steps, one global equity fund
  in the policy benchmark, some polish, and results a non-expert can follow. I chose ACWI alone,
  exempt from the per-holding cap (a benchmark isn't a holding), and 25% turnover a quarter.
- **Built by Claude Code:**
  - **Build rule:** every candidate faces the same four tests (IPS, held up, volatility,
    tracking error). The fallback is the best Sharpe among those passing all four; if none do,
    the IPS-passing one failing the fewest tests, with the failures named. A guard means the
    pick is never a portfolio another IPS-passing candidate beats on Sharpe, volatility,
    drawdown, tracking error and turnover. The seed-44593 numbers are a test.
  - **Turnover:** "Turnover / yr" was annualised ×12 although Build rebalances quarterly, so the
    6–10× figures were really 2–3.5×; `backtest.annual_turnover` fixes it for every mode. Build
    now moves at most 25% a quarter after the first build, and leftovers under 0.5% the new build
    doesn't want are sold within that limit, so no tail of tiny holdings.
  - **Policy benchmark:** ACWI alone (no URTH); the Dubai mix is unchanged (45/10/35/10,
    volatility 9.21%, largest fall −19.50%).
  - **Polish:** Data notes stay collapsed after Fill demo client; the Portfolio value box follows
    every USD holdings edit; nothing shows as −0.00% (one helper, `metrics.no_negative_zero`).
  - **Easier to follow:** live progress (4 steps and a bar, "about a minute"); a summary card
    first (what we did, found, recommend and why, what changes), written by code
    (`gpl/summary.py`); current vs recommended (weights and four cards with arrows); a status
    and one-line reason per candidate, with the full table in "Full results (for analysts)";
    "How to read this"; the page scrolls to the summary and highlights it; a note on each step's
    dates.
- **Effect on results:** seed 44593 now gets the low-vol tilt (fewest failed tests) instead of
  Diversified. Dubai against 60/40: still keep (current 0.67 vs next best 0.55). Dubai against
  the policy benchmark: the quality tilt is now recommended (information ratio 0.21, +0.67% a
  year, tracking error 3.02%), where on 6 Nov nothing reached 0.10. Both changes contributed:
  ACWI alone without the limit gives the low-vol tilt 0.13. A better result after a method
  change is a mild form of hindsight, and the margins are small, so I say so in the README.
- **Learned:** a fallback must be held to the same standard as the main choice, or it can pick
  something worse than what it rejected. A turnover number is only as good as its annualisation.
  A turnover limit on its own leaves dust; selling it has to fit inside the same limit.
- **Status:** 972 tests (971 pass locally; the real-install test runs in CI).

## 2026-11-08 — "Fill the form with AI" failed on the live app
- **Problem:** the Client tab's AI intake returned a raw 400: "The compiled grammar is too
  large… Simplify your tool schemas or reduce the number of strict tools." The fact-find tool
  had grown to about 40 fields, too many for the API's strict mode.
- **Decided:** fix it without loosening the rule that the AI only copies facts and never fills
  the risk questionnaire; check every other AI tool; show plain messages; add live tests.
- **Built by Claude Code:**
  - A live probe with my key first: only the intake tool failed. `propose_spec` and the
    analyst's 17 tools compile even with all 62 securities in their lists.
  - The intake tool is no longer strict. `profile.clean_draft` checks every field in code before
    the form changes (numbers in range, choices from the form's options, yes/no as yes/no) and
    drops anything else, including any field the form doesn't have (a risk score, say), and
    tells the adviser what was dropped. Field descriptions in the schema now point "at least
    15% in the Gulf" to the GCC minimum: before, the model filed it under unmapped requests in
    4 of 6 tries; after, 8 of 8.
  - `agent/api.py`: strict tools retry once without strict mode if the grammar is ever too
    large (their inputs are also checked in code); `friendly_error` turns API failures into
    plain messages (key rejected, busy, unavailable, couldn't process) and logs the details.
    Used by the client form, Optimise's goals and the analyst.
  - Live smoke tests (`pytest -m live`, skipped without a key, so CI skips them): a short note
    ("Risky investor aiming for high returns in Europe"), a long detailed note, the goals tool,
    and both strict tool sets with the full universe. All pass with my key.
- **Learned:** strict mode guarantees the schema but has a size budget; past it, validate in
  code instead. Model output varies run to run, so a live test needs repeating before trusting
  it (the first run passed by luck).
- **Status:** 989 tests: 988 pass locally with my key (including 6 live API tests), the real-install test runs in CI; CI skips the live ones.

## 2026-11-08 — AI intake removed; demo client front and centre and realistic
- **Decided:** remove the AI client intake entirely (the box, its button, `agent/profile.py` and
  its tests): it was the riskiest AI piece for the least gain. Optimise's plain-English goals and
  the analyst stay. Make the demo impossible to miss, and make every demo client realistic.
- **Built by Claude Code:**
  - *▶ Try a demo client* is now the primary button in the header (brand blue, larger). On a
    first visit a welcome banner says what the tool does, with a big demo button and an "or start
    a blank client" link; the empty Client tab shows the same button first. After it runs, a toast
    says "Demo client loaded. Walk through steps 1–6, or open Optimise to see a recommendation."
  - The demo portfolio now follows the client's own computed risk score and IPS: 5-10 tickers
    from the verified investable universe (SGOV, which wasn't in it, is gone), an equity share
    that fits the score, nothing above the per-holding cap, the client's own market first among
    GCC holdings, a region minimum only when a holding covers it, cash needs due within 36 months
    in cash-like funds, KWD holdings kept small (the IPS caps non-pegged currencies at 20%), and
    screened compliant funds only for Sharia clients (no GLD: its Sharia status is unknown). The
    loss tolerance now matches the score. Sharia demo clients are growth investors only: with one
    sukuk fund and no Sharia cash fund, a cautious Sharia portfolio can't meet the cap.
  - A test checks 50 random seeds against every weight-based rule. The same checks failed 49 of
    50 old demo clients (SGOV, holdings above the cap, loss tolerance not matching the score).
- **Checked on real prices** (31 Dec 2019 – 3 Nov 2026, seeds 0-199): all 200 within their
  volatility limit; 186 of 200 within their drawdown limit, the other 14 over by 0.6 pp on
  average (2.1 pp at most), mostly score-1 clients whose 10% limit is tight against COVID. To
  get there I made the cautious mixes hold more cash and kept TLT for growth clients only.
- **Not done:** the README screenshots are from seed 2 before this change; seed 2 is now a
  different client, so they need re-shooting when the screens next change.
- **Learned:** a demo is a claim about the product; if it breaks its own rules, the product
  looks broken. Test the generator against the rules, not against itself.
- **Status:** 1025 tests: 1024 pass locally with my key (including 4 live API tests); the real-install test runs in CI.

## 2026-11-08 — Step cards are the navigation; README screenshots re-shot
- **Checked first:** the AI intake was already gone and *▶ Try a demo client* was the primary
  button with the welcome banner. The step cards were still display-only above a duplicate row
  of six tabs.
- **Built by Claude Code:** each step card now opens its step (a full-size button over the card
  sets the step, as the Next buttons already did); the top-level tab row is hidden, so the cards
  are the only step navigation. Optimise and Report & trades got page titles, since the tab
  label used to name them. Checked in headless Chrome with real mouse clicks: no tab row
  visible, the active card and the page content follow each click.
- **Screenshots** re-shot from the new demo client, seed 2: Build (no build beats 60/40 reliably;
  Diversified (IPS) is the next best fallback), Improve (Max Sharpe, Sharpe 0.52 against 0.33,
  within the 30% turnover limit) and page 3 of the client report PDF.
- **Noticed, not changed:** the report's swap note says "sell IEF" while the turnover-limited
  move only trims IEF (7.31% to 4.97%); the wording should say the full sale happens in the full
  restructure.
- **Status:** 1025 tests: 1024 pass locally; the real-install test runs in CI.
