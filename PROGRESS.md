# Progress log

## Day 1 — 2026-10-23 · setup + data.py
- **Built:** repo skeleton (src-layout `gpl`, pyproject, config, fixtures, stubs for every module);
  `get_prices` (yfinance + parquet cache), `clean_prices`, `to_returns`. Everything installed fine on 3.14.
- **Change of plan:** I switched from writing the finance code myself to "build + explain". Claude writes
  it, explains each function, and quizzes me after each module. CLAUDE.md and BRIEF.md updated.
- **Learned:** adjusted vs raw close (dividends and splits); why forward-fill only short gaps (long fills
  fake low volatility); simple returns add across assets, log returns add across time.
- **Open:** I haven't answered the Day 1 quiz yet.
- **Next:** risk-free rate and core metrics.

## Day 2 — 2026-10-23 · risk-free rate + core metrics
- **Built:** `get_risk_free` (^IRX → daily rate), annualised return, CAGR, volatility, Sharpe, Sortino.
- **Learned:** volatility scales with √time because variances add; CAGR < arithmetic mean when returns
  are volatile ("volatility drag"); Sortino only penalises downside.
- **Open:** ^IRX is a discount yield, not a compounding yield. Ignored for now (a few bps).
- **Next:** drawdowns and tail risk.

## Day 3 — 2026-10-23 · drawdowns, VaR, ES, summary table
- **Built:** drawdown series, max drawdown, historical and parametric VaR, Expected Shortfall, `summary_table`.
- **Learned:** VaR says "how bad is a normal bad day"; ES says "how bad are the days beyond that".
  Historical VaR only knows the crashes in its sample, so it can understate tail risk.
- **Next:** the optimiser.

## Day 4 — 2026-10-23 · mean-variance optimiser
- **Built:** `portfolio_performance`, min variance, max Sharpe, efficient frontier (SLSQP, long-only, 40% cap).
- **Learned:** with uncorrelated assets min variance is just 1/variance weights, a good sanity check.
  Max Sharpe is very sensitive to expected-return estimates.
- **Next:** risk parity.

## Day 5 — 2026-10-23 · risk contributions + risk parity
- **Built:** `risk_contributions`, `risk_parity`.
- **Learned:** equal *weights* are not equal *risk*; in a 1/n portfolio the volatile assets dominate the risk.
- **Open:** risk parity has no 40% cap. In the backtest it reached 43% in AGG. Should I cap it?
- **Next:** backtester.

## Day 6 — 2026-10-23 · backtester
- **Built:** monthly-rebalanced backtester with drift, turnover and 10 bps costs; strategy builders
  (equal weight, fixed weights, min variance, max Sharpe, risk parity).
- **Learned:** no look-ahead means the weights you choose at the close of day *t* only earn from *t+1*.
  There's a test with a "cheating" strategy that proves it.
- **Fix:** the first risk parity solver (L-BFGS-B) stalled on real data and was replaced by coordinate descent.
- **Next:** Monte Carlo and results.

## Day 7 — 2026-10-23 · Monte Carlo + results notebook
- **Built:** `simulate_paths` (bootstrap and GBM), `terminal_stats`, `goal_probability`; plotly chart helpers;
  `notebooks/01_week1_results.ipynb`.
- **Results (Oct 2016 – Oct 2026, net of costs):** SPY had the best Sharpe (0.74), then 60/40 (0.67) and
  max Sharpe (0.55). Min variance was lowest (0.31) but had the smallest drawdown (−15% vs −34% for SPY).
  Max Sharpe traded about 3.9× its value per year.
- **Open:** Jupyter isn't installed, so I need to decide whether to add it. I also need to open the
  notebook and actually look at the charts.
- **Week 1 status:** 125 tests passing. "Done when" is met apart from me viewing the charts.
- **Next:** answer the quizzes, decide on capped risk parity, then Week 2 (regime model + dashboard).

## Week 1 follow-up — 2026-10-23 · cap, fixed dates, Jupyter
- **Changed:** risk parity is now capped at 40% (capped assets sit at the cap, the rest share risk equally).
  The config end date is fixed at 2026-09-30 so results don't move daily. Jupyter is a dev dependency and
  the notebook runs with nbconvert.
- **Bug found:** `get_prices` treated `end` as exclusive (yfinance's rule), while the docs said inclusive.
  Fixed and tested. Without the fix the data would have stopped on 29 Sep.
- **Learned:** the cap only bit in 9 of 120 months (AGG, 2018–19) and moved risk parity's Sharpe by 0.0003.
  Most of the small changes in the results came from moving the end date (dropping 3 weeks of data), not
  from the cap. Rankings and max drawdowns didn't change.
- **Next:** look at the charts in Jupyter, answer the quizzes, then Week 2.

## Week 2 — 2026-10-23 · regime classifier, regime-aware strategy, dashboard
- **Built:** `regime.py` (forward-looking stress labels, 8 backward-looking features, logistic regression
  and random forest, time-series CV with a 21-day gap, monthly walk-forward probabilities, a simple
  volatility-rule baseline); `regime_switch_strategy` in the backtester; a Streamlit dashboard with
  assets, frontier, backtest, Monte Carlo and regime tabs. Added scikit-learn and streamlit (approved).
- **Results:** logistic catches 67% of stressed days, but only 34% of its "stressed" calls are right
  (base rate 22%). Its accuracy (65%) is *below* always-calm (79%). Switching to min variance on its
  signal cut drawdowns (equal weight: −21.6% → −17.1%; COVID −20% → −14%) but lowered Sharpe for every
  strategy (equal weight 0.49 → 0.42). The plain "vol is already high" rule beat the ML model on Sharpe.
- **Learned:** a label that looks 21 days ahead leaks into the test fold unless you leave a gap, and a
  model refit on day *t* can only train on labels up to *t − 21*. Accuracy is the wrong metric when 80%
  of days are calm. Reducing risk usually costs return.
- **Open:** should the threshold be higher than 0.5 (fewer false alarms)? Tuning it on this sample would
  overfit, so it would need its own hold-out period. GCC-specific stress isn't modelled at all.
- **Next:** open the dashboard and check the charts by eye, then Week 3 (agent + README + deploy).

## Week 2 fixes — 2026-10-23 · weight cap, max Sharpe robustness, frontier labels
- **Fixed:** the optimiser now refuses infeasible solver output instead of rescaling it (rescaling could
  push weights past the cap). Max Sharpe is solved in convex form (minimise risk for a fixed excess
  return, then rescale), with a multi-start fallback. It no longer fails on short or awkward samples:
  0 failures in 6,000 random tests. Frontier labels no longer sit on the curve. Tabs show the cap in use.
- **Unresolved:** once, with untouched defaults (no slider, asset or date changes, no reload, server
  started right after Week 2), the Frontier tab showed min variance 75% AGG and max Sharpe 52.6% SPY /
  47.4% GLD. Those are exactly the results at a 75% cap. It couldn't be reproduced and now shows 40%.
  If it happens again: screenshot the tab with the sidebar and the "at most X% per asset" caption.
- **Learned:** maximising a ratio directly is fragile; rewriting it as a convex problem is the standard
  fix. "Can't reproduce" isn't the same as "not a bug", so I wrote it down.
