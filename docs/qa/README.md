# QA evidence: final polish (9 Nov 2026)

How it was checked:
- **Setup:** headless Chrome, driven over the DevTools protocol, against the app started with
  `--client.toolbarMode=viewer` (so Streamlit's toolbar is drawn as it is on Streamlit Cloud).
- **Clicks:** every click is a real mouse event (`Input.dispatchMouseEvent`) at the control's
  on-screen centre. Before each click, `document.elementFromPoint` at that point must be the
  control itself. After each click, the expected effect must appear: a dropdown opens, the step
  card *and* the page content change, the demo loads.
- **Logs:** `qa-log-clicks-goals-seed2-widths.json` (96 checks, 96 passed) and
  `qa-log-seed2-final.json` (the seed-2 run repeated on the final code).
- **Not checked:** the "Manage app" badge only exists on Streamlit Cloud, so its clearance is
  checked by padding, not seen.

| Item | Fixed? | Evidence | Notes |
|---|---|---|---|
| A1 Header controls clickable | Yes | `A1-*` (both sizes, seeds 92713 and 2) | Header sits below Streamlit's toolbar; decorative layers ignore clicks; toasts moved bottom right (one covered ⚙). |
| A2 Clear of toolbar and badge | Partly seen | `A1-steps-*`, `E20-*` | Toolbar: seen. Badge: padding only (see above). |
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
| D13 Report & trades states | Yes | `D13-report-before-optimise.png`, `D13-report-after-optimise-*.png` | |
| D14 One formatter, no "USD -0", typo | Yes | `D14-report-card-seed2.png` | Unit tests for `money()`. |
| E15 No truncated button labels | Yes | `E16-holdings-*.png` | |
| E16 Tables | Yes | `E16-holdings-*.png`, `E20-report-after-optimise-1280x800.png` | QA found clipped trade-table headers and a cut-off fund name at 1280 px; both widened and re-checked. |
| E17 Donut centre text | Yes | `E16-holdings-*.png` | |
| E18 "Enter as" radio, (?) by labels | Yes | `E16-holdings-*.png` | |
| E19 Sticky bar | Yes | `E20-*` | |
| E20 Three widths | Yes | `E20-*` (1280, 1440, 1920) | Optimise results seen at 1280 and 1440 only. |

Found by this QA and fixed:
- **Step navigation.** Right after a demo client loaded, a step-card click could light the card
  but leave the old page showing. The step is now app state, and the hidden tabs are rebuilt
  open at that step.
- **Risk-vs-return labels.** On a crowded chart, labels could sit on neighbouring dots.
- **Tables at 1280 px.** Trade-table headers were clipped, and so was a long fund name in the
  holdings table.
