# QA evidence: final polish (9 Nov 2026, updated the same day)

How it was checked:
- **Setup:** headless Chrome, driven over the DevTools protocol, against the app started with
  `--client.toolbarMode=viewer` (so Streamlit's toolbar is drawn as it is on Streamlit Cloud).
- **Clicks:** every click is a real mouse event (`Input.dispatchMouseEvent`) at the control's
  on-screen centre. Before each click, `document.elementFromPoint` at that point must be the
  control itself. After each click, the expected effect must appear: a dropdown opens, the step
  card *and* the page content change, the demo loads.
- **Logs:** `qa-log-clicks-goals-seed2-widths.json` (96 checks, 96 passed);
  `qa-log-seed2-final.json` (seed 2 and all six goals, re-run after the two-check rule);
  `qa-log-lower-risk-seed2.json` (both lower-risk goals on seed 2); `qa-log-live-site.json`
  (header and step cards on the deployed site).
- **Live site:** the same header and step-card clicks run on
  https://gulf-portfolio-lab.streamlit.app inside Streamlit's wrapper page: 47 of 47 passed.
  At each click point the wrapper's top element was the app's iframe, and inside the app it
  was the control itself.
- **Not checked:** the owner-only "Manage app" badge (not visible to a logged-out browser).

| Item | Fixed? | Evidence | Notes |
|---|---|---|---|
| A1 Header controls clickable | Yes, locally and live | `A1-*`, `LIVE-*` (both sizes, seeds 92713 and 2) | Header sits below Streamlit's toolbar; decorative layers ignore clicks; toasts moved bottom right (one covered ⚙). |
| A2 Clear of toolbar and badge | Yes (public view) | `A1-steps-*`, `LIVE-steps-*` | On the live site the Back/Next bar ends left of Streamlit's bottom-right badges. The owner-only "Manage app" badge isn't shown to a logged-out browser, so that one is covered by padding only. |
| A3 "Ask the analyst" removed | Yes | `A1-demo-loaded-1440x900.png` | |
| B4 Chat and free-text AI removed | Yes | `C7-optimise-form.png`, `D13-*` | Covered by `test_no_chat_or_free_text_ai_anywhere`. |
| B5 One AI feature | Yes | `B5-ai-memo.png`, `B5-memo-in-client-report.txt` | Real API call: 3 tool calls. The memo is in the downloaded client report. The no-key note is covered by an app test. |
| B6 Dead code removed, docs updated | Yes | git history | `scenarios.apply_what_if` kept: it's a tested finance function. |
| C7 "Optimise portfolio" with explanation | Yes | `C7-optimise-form.png` | Turnover and swaps sit under "Advanced settings". |
| C8 Goal dropdown and the new goal | Yes | `C8-goal-*.png` | Default goal follows the risk score. |
| C9 Too cautious for this client | Yes | `C11-results-seed2.png` | Seed 2: Min variance, Risk parity, Max diversification and Re-weight + swaps are marked too cautious (return below the 6.49% the client needs). |
| C10 Swaps vs Min variance (seed 92713) | Explained, tested | `C8-goal-lower-risk.png` | No swap fired, so it matched Min variance; now shown once, with a note. |
| C11 Results order | Yes | `C11-results-seed2.png`, `C11-results-seed2-1280x800.png` | |
| C12 Keep is not an error | Yes | `C8-goal-beat-the-benchmark.png` | |
| Both IPS checks for "recommended" | Yes | `C11-results-seed2.png`, `B5-memo-in-client-report.txt` | Seed 2 now reads "Move part-way towards Max Sharpe": the full target fails volatility, max drawdown and worst stress episode over the full history. The memo and PDF headline say the same. |
| Lower-risk return warning | Yes | `C9-seed2-goal-lower-risk.png` | Seed 2, Lower risk: Min variance returned 2.60% a year, below the 6.49% needed. Seed 92713 has no required return, so no warning. |
| D13 Report & trades states | Yes | `D13-report-before-optimise.png`, `D13-report-after-optimise-*.png` | |
| D14 One formatter, no "USD -0", typo | Yes | `D14-report-card-seed2.png` | Unit tests for `money()`. |
| E15 No truncated button labels | Yes | `E16-holdings-*.png` | |
| E16 Tables | Yes | `E16-holdings-*.png`, `E20-report-after-optimise-1280x800.png` | QA found clipped trade-table headers and a cut-off fund name at 1280 px; both widened and re-checked. |
| E17 Donut centre text | Yes | `E16-holdings-*.png` | |
| E18 "Enter as" radio, (?) by labels | Yes | `E16-holdings-*.png` | |
| E19 Sticky bar | Yes | `E20-*` | |
| E20 Three widths | Yes | `E20-*` (1280, 1440, 1920) | Optimise results seen at 1280 and 1440 only. |

## Final fixes (same day): cards, data notes, benchmark, footer

Checked at 1280×800, 1440×900 and 1920×1080 on seed 92713. A check flags any metric card,
table cell, header or footer text whose content is wider than its box or beyond the screen
(`qa-log-final-fixes.json`, 51 of 51 passed). The ⚙ icon's font ligature always trips the
width test; it isn't text and is ignored.

| Item | Fixed? | Evidence | Notes |
|---|---|---|---|
| F1 Client metric cards cut ("17,…", "Net worth (US…") | Yes | `F1-client-metrics-before-1280x800.png`, `F1-client-metrics-*` | 2 × 2 cards, compact values (`$17.95m`, `35.41%`, `4.0 months`), the exact amount in the (?) tooltip. Labels and values in *every* metric card now wrap instead of ending in "…"; KPI, Monte Carlo and Regime cards checked too. Percentages keep two decimals (project rule), so `35.41%`, not `35.4%`. |
| F1 Thousands separators in money inputs | Caption under each | `F1-client-metrics-*` | Streamlit's number input only takes a numeric printf format (no `%,d`), so each money input shows its amount below it (`$7,393,000`). |
| F2 Data notes names, date column | Yes | `F2-data-notes-*` | Name from the same source as the holdings table (the quote's name, else config, else the ticker); dates and day counts never wrap. |
| F3 Benchmark selector | Yes | `F3-header-benchmark-open-*`, `F3-analyse-benchmark-policy-*`, `F3-analyse-*` | Labelled "Benchmark" in the header, and again on Analyse beside the comparison; one setting (an app test changes each and checks the other). The policy benchmark is listed whenever the client's IPS allows one; otherwise the Analyse copy says why it is missing. |
| F4 Footer text clipped on the left | Fixed (not reproduced) | footers in `F2-*`, `F3-*` | Not reproduced in headless Chrome at any width or zoom (900 to 1920 px, zoom 1.1 to 1.5) locally, nor in the earlier live screenshots at 1280 and 1440. Likely cause: the text sits in a centring flex wrapper, so a column narrower than its text overflows on *both* sides. The text column now shrinks and wraps, left-aligned, so it can only grow downwards. |

**Live site** (https://gulf-portfolio-lab.streamlit.app, after the redeploy; every live page
had the new "Benchmark" label): at 1280, 1440 and 1920 px, real clicks open the header and
Analyse benchmark selectors, pick the policy benchmark on Analyse, then pick 60/40 in the
header. No card, table or footer text is cut (`LIVE-F*`, `qa-log-live-final-fixes.json`:
51 of 57). The 6 failures were my check, not the app: it read the select's text, but the
chosen value sits in the select's `<input>`. The screenshots (`LIVE-F3-policy-chosen-*`) show
the header, guidance line and KPI deltas following the Analyse choice. With the check fixed,
1280 px passed 19 of 19, both directions included (`qa-log-live-sync-recheck.json`).
The first live run also found the long "Policy benchmark (fits the IPS)" cut off in the
header select at 1280 px; the header's benchmark column is now wider (355 px at 1280) and the
re-run found nothing cut.

Found by this QA and fixed:
- **Step navigation.** Right after a demo client loaded, a step-card click could light the card
  but leave the old page showing. The step is now app state, and the hidden tabs are rebuilt
  open at that step.
- **Risk-vs-return labels.** On a crowded chart, labels could sit on neighbouring dots.
- **Tables at 1280 px.** Trade-table headers were clipped, and so was a long fund name in the
  holdings table.
