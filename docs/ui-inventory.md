# UI inventory (for design)

Everything on screen in Gulf Portfolio Lab, step by step: what each element is for, and whether
it is **essential** (the workflow or a decision depends on it) or *secondary* (detail, help or a
power-user option that could move to an expander, a tooltip or a second level).

- Captured on 8 Nov 2026 from the running app (`streamlit run app/streamlit_app.py`) with
  *▶ Try a demo client*, seed 2 (a fictional client, Layla Haddad, USD 360,000, risk score 3).
- Element lists were read from the rendered page, in on-screen order, then checked against the
  code; widgets inside collapsed expanders come from the code.
- Full-page screenshots (1440 px wide) are in [`docs/ui-screens/`](ui-screens/). The app is
  Streamlit with a light private-bank theme (`.streamlit/config.toml`, brand blue `#2563EB`);
  layout CSS is in `app/layout.py`.

Priority key: **E** = essential, S = secondary.

| Screenshot | Screen |
|---|---|
| [01-welcome.png](ui-screens/01-welcome.png) | First visit, nothing loaded |
| [02-client.png](ui-screens/02-client.png) | 1 Client, demo loaded (sections collapsed) |
| [02b-client-expanded.png](ui-screens/02b-client-expanded.png) | 1 Client, all nine fact-find sections open |
| [03-portfolio-holdings.png](ui-screens/03-portfolio-holdings.png) | 2 Portfolio › Holdings |
| [03b-portfolio-screen.png](ui-screens/03b-portfolio-screen.png) | 2 Portfolio › Screen |
| [04-analyse-performance.png](ui-screens/04-analyse-performance.png) | 3 Analyse › Performance & backtest |
| [04b-analyse-benchmark.png](ui-screens/04b-analyse-benchmark.png) | 3 Analyse › Benchmark |
| [04c-analyse-assets.png](ui-screens/04c-analyse-assets.png) | 3 Analyse › Assets |
| [04d-analyse-monte-carlo.png](ui-screens/04d-analyse-monte-carlo.png) | 3 Analyse › Monte Carlo |
| [04e-analyse-regime.png](ui-screens/04e-analyse-regime.png) | 3 Analyse › Regime |
| [05-optimise-improve-form.png](ui-screens/05-optimise-improve-form.png) | 4 Optimise › Improve, before running |
| [05b-optimise-improve-result.png](ui-screens/05b-optimise-improve-result.png) | 4 Optimise › Improve, with the recommendation |
| [05c-optimise-build-form.png](ui-screens/05c-optimise-build-form.png) | 4 Optimise › Build, before running |
| [05d-optimise-frontier.png](ui-screens/05d-optimise-frontier.png) | 4 Optimise › Frontier |
| [05e-optimise-strategy-backtests.png](ui-screens/05e-optimise-strategy-backtests.png) | 4 Optimise › Strategy backtests |
| [06-suitability.png](ui-screens/06-suitability.png) | 5 Suitability |
| [07-report-trades.png](ui-screens/07-report-trades.png) | 6 Report & trades › Trades |
| [07b-report-reports.png](ui-screens/07b-report-reports.png) | 6 Report & trades › Reports |
| [08-ai-analyst.png](ui-screens/08-ai-analyst.png) | 6 Report & trades › AI analyst |

---

## Global frame (on every screen)

### Header (sticky at the top)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| "Gulf Portfolio Lab" | Title | Product name | **E** |
| Client chip | Chip | Client name, or "none loaded" | **E** |
| Risk score chip | Chip | The client's risk score, e.g. "3 / 5" (from the questionnaire) | **E** |
| Portfolio chip | Chip | Portfolio value, e.g. "USD 360,000", or "not set" | **E** |
| IPS chip | Chip (green / amber / grey) | IPS status: Ready, Draft or – | **E** |
| DEMO DATA · seed N | Chip (solid accent) | Marks fictional data; the seed reproduces it | **E** when a demo is loaded |
| Benchmark | Select | Reference portfolio for every comparison (60/40, S&P 500, MSCI World, Saudi, UAE, Sharia 60/40, cash, policy benchmark…) | **E** |
| ▶ Try a demo client | Primary button (brand blue, larger) | Fills a realistic client and portfolio and opens step 1 | **E** (the main call to action for visitors) |
| 💬 Ask the analyst | Button | Opens Report & trades › AI analyst | S |
| ⚙ | Popover | Settings (below) | S |

**⚙ Settings popover (all secondary):**
- Data period (radio): latest data, or the fixed sample that reproduces the README.
- Start date (date input).
- Costs for US listings, in bps on turnover (number input).
- **Demo seed** (expander): a seed number input and a "Load the demo client with this seed" button.
- **Advanced settings** (expander): estimation window (select slider 126/252/504), strategies to
  show (multiselect), "Add regime-aware versions" (toggle), the de-risk threshold (slider).

### Step cards (navigation)

Six cards in a row under the header: 1 Client, 2 Portfolio, 3 Analyse, 4 Optimise,
5 Suitability, 6 Report & trades. They are the only step navigation (there is no tab row).

| Element | Type | Purpose | Priority |
|---|---|---|---|
| Card (× 6) | Clickable card | Opens the step. Shows a number or ✓ (done), the title and one status line (e.g. "fact-find 100% complete", "7 holdings · USD 360,000", "analysis ready", "not run yet", "Current: passes every rule", "report not prepared"). The active step has a blue outline | **E** |

On a first visit, steps 2-6 read "not started".

### Data strip (under the cards)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| "Data as of …" | Text | Latest price date and load time; latest data or the fixed README sample | S |
| Sample | Caption | Date range and number of daily returns behind every result | S |
| Data notes | Expander (table inside) | Start date, trading days, currency, USD conversion and trading calendar per holding; warnings for short histories or mixed calendars | S |
| Benchmark description | Caption | What the chosen benchmark holds | S |

### Step footer (bottom of every step)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| "Still to do: …" or the step's status | Caption | What is missing in this step | S |
| Next: <step> → | Primary button | Opens the next step (not on step 6) | **E** |
| Disclaimer | Caption | "Educational project, not investment advice…" | **E** (compliance) |

### Feedback messages

- **Toast** after *Try a demo client*: "Demo client loaded. Review the client and IPS, then use
  Next to walk through each step." (S)
- **Data errors:** "Yahoo Finance didn't respond just now…" with a Retry button; "Couldn't load
  <ticker>…" warning. (**E** when shown)

---

## Welcome screen (first visit, no client)

Screenshot: [01-welcome.png](ui-screens/01-welcome.png)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| Welcome banner | Bordered card | One line on what the tool does | **E** |
| ▶ Try a demo client | Large primary button | The quickest way in | **E** |
| "or start a blank client" | Text link (tertiary button) | Starts an empty client on step 1 | S |
| Client step (below) | Empty state | "No client yet" heading; a caption pointing to the demo button; example client select + Load example; JSON upload; New blank client | **E** |

The banner disappears as soon as a client is loaded. The header and step cards show neutral
states ("none loaded", "not started").

---

## Step 1 · Client

Screenshots: [02-client.png](ui-screens/02-client.png) (collapsed),
[02b-client-expanded.png](ui-screens/02b-client-expanded.png) (every section open).

**Top of the page**

| Element | Type | Purpose | Priority |
|---|---|---|---|
| "Client fact-find and Investment Policy Statement" | Heading | Step title | **E** |
| Adviser notice | Caption | Decision support for a licensed adviser; "do not enter real client data in the public version" | **E** (compliance) |
| Example client | Select + "Load example" button | Three worked examples (Dubai, Riyadh, Doha), each with its own portfolio | S |
| Load a saved profile (JSON) | File upload | Reopens a saved client | S |
| New blank client | Button | Empty fact-find | S |
| DEMO DATA notice | Info box | Marks the demo client | **E** when a demo is loaded |
| Closest-mix warning | Warning | Only for the Riyadh example: no portfolio meets every rule | **E** when shown |
| Fact-find progress | Progress bar | "Fact-find 100% complete", or the sections still to do | **E** |

**Nine fact-find sections** (expanders, each titled with ✅ when complete, or "optional"):

| Section | Inputs | Priority |
|---|---|---|
| Client & household | Name, client reference, age, date of birth, residency, nationality, tax residencies, booking location, regulator, US person (checkbox), marital status, dependants and their ages, occupation, employer, employment type, retirement age, reporting currency | **E** |
| KYC / AML | Source of wealth, source of funds (text areas), PEP (radio), client classification + reason, expected activity | **E** (regulatory) |
| Financial situation | Income (salary, bonus, business, rental, other), annual expenses, assets (cash, investments, property, business, end-of-service gratuity, pensions), liabilities (mortgage, loans, other). Four metric cards: net worth, investable assets, savings rate, emergency fund (months), with a formula caption | **E** inputs; metrics S |
| Objectives & goals | Primary objective, time horizon, portfolio value, return target, yearly income drawn, yearly addition, inflation; **Cash needs** table (description, amount, YYYY-MM date); extra cash reserve; **Goals** table (amount or multiple, years) | **E** |
| Risk profile | Six questionnaire radios (4 capacity, 2 willingness), with "set from the Cash needs table" for the cash question; capacity for loss (%); adviser override (select) + reason; nested expander "Adviser overrides of individual limits" (max volatility, max drawdown, max in one holding) | **E** questionnaire; overrides S |
| Knowledge & experience | Years investing, trading frequency, products used (multiselect), background, client's own experience (radio) | S (feeds a review flag only) |
| Constraints & preferences | Sharia only (checkbox), minimum in GCC (%), max in non-USD-pegged currencies (%), region minimums table, excluded sectors (multiselect), concentrated holdings, home-currency preference, legal notes, restricted tickers, tax notes, other circumstances | **E** for Sharia, GCC, regions and exclusions; S for the free-text notes |
| Estate & succession | Will (select), structures (multiselect), beneficiaries noted (checkbox), notes; a warning to involve a lawyer | S |
| Service & review | Service type, review frequency, next review date, adviser name, licence number, fee basis, notes | S |

**Result area (below the sections)**

| Element | Type | Purpose | Priority |
|---|---|---|---|
| Risk capacity / Risk willingness / Risk score | 3 metric cards | The score is the lower of the two; it sets the IPS limits | **E** |
| Score explanation | Caption (warning if capacity and willingness differ by 2+, or an override) | How the score was reached | S |
| Investment Policy Statement (RRTTLLU) | Text block | Return, Risk, Time horizon, Taxes, Liquidity, Legal, Unique circumstances, generated by rules | **E** |
| Constraint notes | Bullet captions | How the IPS becomes optimiser limits (e.g. "the 20.00% drawdown limit is checked by the suitability check") | S |
| Missing items | Warning + "What's missing" expander | Shown when the final IPS can't be issued | **E** when shown |
| Download IPS (PDF) | Primary download button (disabled until complete) | The final IPS | **E** |
| Download DRAFT IPS (PDF) | Download button | Watermarked draft for discussion | S |
| Save profile (JSON) | Download button | Saves the client | S |

---

## Step 2 · Portfolio

### Holdings (default sub-tab)

Screenshot: [03-portfolio-holdings.png](ui-screens/03-portfolio-holdings.png)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| "Current portfolio" | Heading | Step title | **E** |
| Holdings / Screen | Sub-tabs | Two views | **E** |
| Enter holdings as | Radio (USD amount / Weight %) | How values are typed | **E** |
| Portfolio value (USD) | Number input | Only in weight mode (in USD mode the total is the value) | **E** |
| Holdings table | Editable grid | Ticker, name, group, ✏️ amount or weight, price, units, weight; white columns are editable | **E** |
| Equal weight / Normalise to 100% | Buttons | Quick fixes | S |
| "Prices as of …" | Caption | Quote time and tickers | S |
| Import a list or start from a preset | Expander | Preset list (select) + "Replace holdings with this preset"; paste box ("SPY 40" per line) + Weights/Amounts radio + "Replace holdings"; import errors | S |
| Show allocation by | Radio (Holding / Group) | Chooses the chart grouping | S |
| Allocation chart | Bar chart | "Allocation by holding" | **E** |
| Quote source | Caption | Yahoo Finance, refreshed every 15 minutes | S |

### Screen

Screenshot: [03b-portfolio-screen.png](ui-screens/03b-portfolio-screen.png)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| Rank by | Select | Metric to rank holdings by | S |
| Ranking table | Grid | Each holding's risk and return measures, ranked | S |
| Ranking and period notes | Captions | Which direction ranks higher; the sample dates | S |

---

## Step 3 · Analyse

Shared top of the step (on every sub-tab):

| Element | Type | Purpose | Priority |
|---|---|---|---|
| "Analyse the current portfolio" | Heading | Step title | **E** |
| KPI row | 5 metric cards | Portfolio value, Return (CAGR), Volatility, Sharpe, Max drawdown, each with the difference from the benchmark (green/red arrows) | **E** |
| Period note | Caption | Backtest dates, and why figures differ slightly between Analyse, Improve and Build | S |
| Performance & backtest / Benchmark / Assets / Monte Carlo / Regime | Sub-tabs | Five views | **E** |

### Performance & backtest

Screenshot: [04-analyse-performance.png](ui-screens/04-analyse-performance.png)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| Share of money vs share of risk | Table + paired bar chart | Which holdings drive the risk (e.g. EEM: 26% of the money, 58% of the risk) | **E** |
| Backtest after costs | Table | Your portfolio vs equal weight, min variance, max Sharpe, risk parity and 60/40: return, CAGR, volatility, Sharpe, Sortino, drawdown, turnover, VaR, expected shortfall | S |
| Growth of $1 | Line chart | Wealth over time vs strategies and the benchmark (grey dashed) | **E** |
| Drawdown from peak | Line chart | Falls from the running peak | S |
| Notes | Captions | Rebalancing and costs; where to find Monte Carlo | S |

### Benchmark

Screenshot: [04b-analyse-benchmark.png](ui-screens/04b-analyse-benchmark.png)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| Your portfolio vs benchmark | Table (11 rows) | Return, risk, beta, alpha, tracking error, information ratio, capture ratios… | **E** |
| Growth of $1 | Line chart | Portfolio vs benchmark | S |
| Relative drawdown | Line chart | How far the portfolio fell behind the benchmark | S |
| Explanations | Captions | Period and method | S |

### Assets

Screenshot: [04c-analyse-assets.png](ui-screens/04c-analyse-assets.png)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| "Risk and return by asset" | Heading + table | Per-holding return, CAGR, volatility, Sharpe, Sortino, drawdown, VaR, expected shortfall | S |
| Method note | Caption | Annualisation, VaR definition | S |

### Monte Carlo

Screenshot: [04d-analyse-monte-carlo.png](ui-screens/04d-analyse-monte-carlo.png)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| "Monte Carlo: where could today's weights go?" | Heading | Sub-tab title | S |
| Use the latest weights of | Select | Your portfolio or a strategy | **E** within this view |
| Horizon (years), Goal (× start), Method, Paths, Seed | Slider, number, radio, slider, number | Simulation settings | S (Horizon and Goal: **E**) |
| Show in real terms | Checkbox | After inflation | S |
| Outcome metrics | 5 metric cards | P(≥ goal), median outcome, bad case (5th percentile), P(lose money), drawdown (5th percentile) | **E** |
| Simulated growth | Fan chart | Range of paths | **E** |
| Method note | Caption | Fixed weights, no costs, nominal USD | S |

### Regime

Screenshot: [04e-analyse-regime.png](ui-screens/04e-analyse-regime.png)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| "Will next month be stressed?" | Heading | Sub-tab title | S |
| Model metrics | 5 metric cards | ROC AUC, precision, recall, accuracy (vs simple baselines), latest P(stressed) | S |
| Probability chart | Line chart | Month-end probability of a stressed month | S |
| Walk-forward table + notes | Table, captions | Results by test period vs a volatility rule and "always calm" | S |

The whole Regime view is secondary: it informs, but does not feed the recommendation.

---

## Step 4 · Optimise

Shared top of the step:

| Element | Type | Purpose | Priority |
|---|---|---|---|
| "Optimise" | Heading | Step title | **E** |
| Optimise / Frontier / Strategy backtests | Sub-tabs | Three views | **E** |
| Max weight per asset (optimiser limit) | Slider | Cap on any one holding | S (the client's IPS cap applies anyway) |
| What should Optimise do? | Radio: Improve my current portfolio / Build a portfolio for me | The two modes | **E** |

### Improve, before running

Screenshot: [05-optimise-improve-form.png](ui-screens/05-optimise-improve-form.png)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| 1. Diagnosis | Table + "Money vs risk" chart + current-vs-benchmark table + caption | Where today's risk comes from, before any change | **E** |
| Describe it in plain English (optional) | Expander: "Goals in plain English" text area + "Interpret with AI" button | The AI turns text into a goal and limits for the adviser to check | S |
| Goal | Radio: Maximise risk-adjusted return (Sharpe) / Reduce risk / Reach a target return with the least risk / Highest return at a target volatility / Beat the benchmark with less risk | Chooses the recommendation rule | **E** |
| Target box | Number input (only for target goals) | Target return or volatility | **E** when shown |
| More limits (optional) | Expander: volatility cap, return floor, group limits table, asset limits table, "Clear all limits" | Extra constraints | S |
| Use the client's IPS limits | Checkbox + limits table (Limit / Value / From) + notes | IPS limits applied to every method | **E** |
| Method note | Caption | The five methods, judged walk-forward after costs | S |
| Max turnover (% of the portfolio) | Number input | How much may be traded | **E** |
| Also try swaps from the curated universe / Max swaps | Checkbox, number | Rule-based swap ideas | S |
| Run the candidate methods | Primary button | Runs everything (about a minute) | **E** |

### Improve, after running

Screenshot: [05b-optimise-improve-result.png](ui-screens/05b-optimise-improve-result.png). The
page scrolls to the summary card and highlights it briefly.

| Element | Type | Purpose | Priority |
|---|---|---|---|
| Progress | Status box with steps and a bar ("1/4 … 4/4"), then "Done: the summary is below." | Shows the run is working | **E** while running |
| **Summary** card | Bordered card | What we did (3 bullets), What we found, What we recommend and why, What changes (the trades in words) | **E** |
| Weights: current vs recommended | Paired bar chart | The move at a glance | **E** |
| Current vs recommended | 4 metric cards | Return, Sharpe, volatility, max drawdown, with the change and a green/red arrow | **E** |
| Recommended method box | Success box | The rule's choice and its reasons | **E** |
| Rule caption | Caption | The fixed rule for this goal | S |
| How to read this | Popover | One sentence each on Sharpe, information ratio, tracking error, held up out of sample, turnover | S |
| Candidate status table | Table | Each candidate: ✓ Recommended / ✗ reason, Why, Sharpe, volatility, drawdown, turnover/yr | **E** |
| Window note | Caption | Why the current portfolio's figures differ slightly between steps | S |
| Full results (for analysts) | Expander with two wide tables | Every walk-forward measure, and today's weights per candidate | S |
| 🔁 Swap ideas | Markdown lines | What the swap rule found, and what the recommendation does with it | S |
| 🏛 Domicile notes | Captions | US-domiciled funds with an Irish UCITS alternative (information, not tax advice) | S |
| Turnover note | Caption | How the turnover limit shapes today's move | S |
| Today's move for <method> | Radio: within the turnover limit / full restructure (with its turnover and estimated cost) | Adviser chooses partial or full move | **E** |
| Growth of $1, walk-forward after costs | Line chart | Candidates vs current and benchmark | S |
| Drawdown relative to the benchmark | Line chart | For the recommended method | S |
| Caveats | Info box | Walk-forward results are historical, not forecasts | **E** (compliance) |
| Suitability table | Table (Current vs recommended) + "Reasons" expanders + period caption | Every IPS rule: pass / fail / review | **E** |
| Explain the recommendation (AI memo) | Button (only with an API key) | AI memo explaining the rule's choice | S |

### Build, before running

Screenshot: [05c-optimise-build-form.png](ui-screens/05c-optimise-build-form.png)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| Build description | Caption | Curated universe of 62 securities, how candidates are built and judged | S |
| Suggested benchmark | Info box + "Use the policy benchmark" button | A simple mix that passes the client's IPS | **E** |
| Tracking-error limit (% a year) | Number input | How far a build may stray from the benchmark | S |
| Turnover limit per quarter (%) | Number input | Trading limit after the first build | S |
| Limits note | Caption | Cap, IPS limits, how tilts are scored | S |
| Build the candidate portfolios | Primary button | Runs the build (about a minute) | **E** |

Build results (not in the screenshots; see the README's `docs/screenshots/build-result.png`)
use the same layout as Improve: progress, summary card, weights chart, metric cards, status
table, plus "Universe after the IPS screen" (expander) and "Securities chosen and why" with a reason
each.

### Frontier

Screenshot: [05d-optimise-frontier.png](ui-screens/05d-optimise-frontier.png)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| "Efficient frontier (full sample, in hindsight)" | Heading + 2 captions | The hindsight caveat and the cap | S |
| Efficient frontier | Scatter/line chart | Risk vs return of possible mixes, with the current portfolio | S |
| Frontier portfolios | Grid | Weights along the frontier | S |

### Strategy backtests

Screenshot: [05e-optimise-strategy-backtests.png](ui-screens/05e-optimise-strategy-backtests.png)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| "Strategies after costs" | Heading + method caption | Monthly rebalancing on a trailing window | S |
| Strategy table | Table | Each strategy's return, risk, turnover, VaR… | S |
| Growth of $1, Drawdown from peak | Line charts | Strategy paths | S |
| Weights over time for | Select + area chart | Target weights by group through time | S |

Frontier and Strategy backtests are both secondary: they explain the methods; the
recommendation comes from the Optimise sub-tab.

---

## Step 5 · Suitability

Screenshot: [06-suitability.png](ui-screens/06-suitability.png)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| "Suitability" | Heading | Step title | **E** |
| Suitability table | Table: rule rows (volatility, max drawdown, worst stress episode, concentration, home-market bias, region minimums, liquidity, exclusions, currency, knowledge & experience) × Current and recommended columns, each ✅ Pass / ❌ Fail / ⚠️ Review with value and limit | The compliance check | **E** |
| Reasons: Current / Reasons: <recommended> | Expanders | One sentence per rule | S |
| Periods | Caption | Full history for risk rules; backtest period for headline cards | S |
| Rationale | Text area + "Reset to the generated text" button + caption | Suitability rationale for the IPS and client report, generated by rules and editable | **E** |
| Domicile (information, not tax advice) | Info box + table (ticker, weight, value, US-situs, Irish alternative) + bullet captions | Illustrative US estate tax and dividend withholding for a non-US person | S |

---

## Step 6 · Report & trades

Shared: heading "Report & trades"; sub-tabs **Trades / Reports / AI analyst**.

### Trades

Screenshot: [07-report-trades.png](ui-screens/07-report-trades.png)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| Targets note | Caption | Targets start from the recommendation; drift definition | S |
| Targets | Editable grid | Target weight per holding | **E** |
| Drift band (± pp) | Number input | When a rebalance is suggested | S |
| Portfolio value (USD) | Number input | Same single value as elsewhere | S |
| Drift table | Table: current, target, drift, outside band | Which holdings are off target | **E** |
| Rebalance alert | Warning | Holdings outside the band | **E** when shown |
| Trade list | Table: action, amount, units, cost rate, cost, after | The trades | **E** |
| Totals | Caption | Sells, buys, estimated costs | **E** |
| Download trade list (CSV) | Download button | For execution | **E** |

### Reports

Screenshot: [07b-report-reports.png](ui-screens/07b-report-reports.png)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| "IPS and client report" | Heading | Section title | **E** |
| Download IPS (PDF) / DRAFT IPS / Save profile (JSON) | Download buttons | Same as on step 1 | S here (duplicate) |
| Adviser name / Licence number | Text inputs | Printed on the report | S |
| Prepare client report (PDF) | Button | Builds the client report PDF (10 pages for the demo) | **E** |
| After preparing: Download client report, a caption (uses the Optimise recommendation, never a chat what-if) and warnings for any problems | Download button, caption, warnings | The deliverable | **E** |

### AI analyst

Screenshot: [08-ai-analyst.png](ui-screens/08-ai-analyst.png)

| Element | Type | Purpose | Priority |
|---|---|---|---|
| "AI analyst" | Heading | Section title | **E** within this view |
| KPI row | 5 metric cards | Same as Analyse | S (duplicate) |
| Holdings table + allocation chart | Table, bar chart | Context for the conversation | S |
| How it works | Caption | The model calls the app's own functions as tools; numbers come from tools | S |
| About me (optional) | Expander: "Fill from an example" select + "About me" text area | Client context for the memo | S |
| Review my portfolio / Explain the Optimise recommendation / Stress test my portfolio | 3 buttons | Start a memo | **E** within this view |
| New conversation | Button | Clears the chat | S |
| Conversation | Chat messages (user / assistant), memo sections with charts, errors | The memo and follow-ups | **E** within this view |
| Follow-up box | Chat input ("Ask a follow-up, e.g. 'What if I cut SPY to 30%?'…") | Questions about the results | S |

Without an API key the buttons are disabled with a note. The whole analyst is optional: every
figure and the recommendation exist without it.

---

## Observations for the redesign

Facts from the inventory, not decisions:

- **Repeated elements:** the IPS downloads appear on step 1 and in Reports; the portfolio value
  input appears in Portfolio › Holdings, Client › Objectives and Trades; the KPI row appears on
  Analyse and AI analyst; the suitability table appears in the Optimise result and on step 5;
  domicile notes appear in the Optimise result and on step 5.
- **Long pages:** Client with every section open is about 7,900 px tall; the Improve result is
  about 7,100 px. Analyse › Performance is about 3,500 px.
- **Captions:** most views carry two to five explanatory captions (methods, periods, caveats).
  Some are compliance text that must stay visible (the disclaimer, "not tax advice", walk-forward
  caveats); most could move into tooltips or "How to read this" popovers.
- **Secondary views:** Analyse › Assets and Regime, Optimise › Frontier and Strategy backtests,
  and Portfolio › Screen are analyst detail; the core path is Client → Holdings → Analyse
  (Performance) → Optimise (Improve or Build) → Suitability → Report.
- **States to design for:** empty (first visit), demo loaded, a run in progress, a result
  (recommended / next best / keep / closest that fails), a missing API key, Yahoo unavailable,
  an incomplete fact-find (draft IPS only).
