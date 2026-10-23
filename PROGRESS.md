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
