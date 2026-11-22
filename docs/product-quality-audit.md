# PayLens product quality review

This is an implementation review of the existing Streamlit project, **not a certification of WCAG compliance or an end-to-end browser audit**. The project uses reproducible synthetic transactions and an educational fraud model.

## Accessibility review

| Area | Code-level status | Live verification required |
|---|---|---|
| Navigation | Horizontal radio-group workspace navigation with labeled choices, active state and 44px targets | Tab/arrow-key behaviour and mobile horizontal scrolling |
| Focus | Visible focus outlines on navigation, buttons, inputs, expander summaries, sliders and table elements | Keyboard-only walkthrough across every workspace |
| Text contrast | Readable dark-theme text, lighter muted colors and dark primary-button labels on warm accent background | Browser-rendered WCAG contrast measurements and browser zoom at 200% |
| Semantics | Page header uses an `h1`; reusable panel headings use `h2`; decorative branding symbols are hidden from assistive technology | VoiceOver/NVDA announcements and heading hierarchy inspection |
| Colour/risk | Risk classification always has text (Low/Medium/High), not just colour | High-contrast and colour-vision checks |
| Responsiveness | Breakpoints for KPIs, cards, status and rows; horizontally scrollable workspace navigation | Desktop/tablet/mobile widths and portrait orientation |
| Motion | `prefers-reduced-motion` disables transitions | Browser/device preference |
| Charts and tables | Plotly axis labels, tooltips and data tables accompany key metrics | Plotly keyboard access and screen-reader exploration |

**Known limitations:** Plotly canvas/SVG visualizations are not automatically equivalent to accessible underlying data tables. Some charts need descriptive data-table equivalents. Streamlit controls and custom CSS must be checked against the deployed Streamlit version. A browser audit is still mandatory before claiming accessibility conformance.

## UI consistency review

The application continues to use the same graphite background, orange accent, Inter/system font stack, KPI and panel components, Plotly styling and horizontal navigation across seven workspaces. Deliberate interaction changes include cross-workspace routing to a selected transaction or merchant and text-based feedback in the case workflow.

**Follow-up:** Verify narrow-screen overflow, chart toolbars, long IDs, resize behaviour and semantic hierarchy in a browser. Avoid decorative redesign unless it improves navigation or comprehension.

## Linked analyst workflow

- Overview high-risk queue → open its highest-ranked transaction in Investigations
- Transaction explorer → select a row in the current page → Investigations
- Merchant Risk → highest-risk transaction of the selected merchant → Investigations
- Investigations → selected merchant profile → Merchant Risk
- Network Intelligence → selected merchant → Merchant Risk
- Merchant Risk → Network Intelligence

Navigation uses Streamlit session-state callbacks and respects global filters. If a linked ID falls outside filters, the destination presents a message instead of silently opening the wrong record.

## Automated and manual testing boundary

GitHub Actions runs Python compilation, deterministic data integrity tests, network analytics tests, SHAP additive-model tests, and a Streamlit `AppTest` widget-level workflow. The latter exercises page changes, SHAP calculation, session-only case saving, threshold control and Merchant Risk rendering. Consult the latest Actions run for actual status.

These tests do **not** download a CSV in a real browser, test assistive technology, prove visual cross-browser rendering, or ensure Streamlit Cloud deployment is awake. Those steps remain manual until a browser-level automation environment is configured.

## Required live smoke test

1. Load the app on Streamlit Cloud and verify all seven workspaces render without exceptions
2. Investigate the top-risk payment; calculate SHAP; confirm baseline + contributions ≈ model score
3. Save a decision and note, change pages, return, then confirm note is still present
4. Download session case notes and inspect transaction ID, status and notes in CSV
5. Compare thresholds in Fraud Intelligence and verify summary counts change
6. Open a merchant from Investigations and follow it into Network Intelligence
7. Use keyboard-only navigation, test a mobile viewport, and verify readable text at increased zoom

Record screenshots/errors and Streamlit Cloud logs for failures. **Do not mark the live test complete based on CI alone.**
