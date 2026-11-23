# PayLens — Payment Network Intelligence

[**Launch the interactive demo**](https://paymentlens.streamlit.app) · [**View dashboard source**](dashboard/app.py)

**An analyst-oriented fintech portfolio project combining payment operations, merchant intelligence, and simulated fraud modelling.**

> **Try it in three minutes:** Start on **Overview** to inspect the transaction mix, open **Investigations** to trace a payment and make a session-only decision, then visit **Fraud Intelligence** to explore how changing the classification threshold affects precision and recall.

**What this demonstrates:** translating business questions into interactive KPI reporting, building an end-to-end Python/Scikit-learn workflow, designing investigation drill-downs, and communicating limitations clearly.

**Important:** The project uses approximately 25,000 reproducible **synthetic transactions**, including simulated fraud labels. It is not connected to real payment services. Risk scores on transaction and merchant screens include predictions on the training dataset, so they must not be taken as independent or production-ready fraud assessments. Model metrics in the Fraud Intelligence workspace are calculated on a held-out stratified test set.

---

## Business questions

- How are transaction volume and payment value distributed across regions and categories?
- What are the payment success and simulated fraud rates?
- Which transactions should be reviewed first based on model-estimated risk?
- Which merchants have elevated predicted risk or weak payment performance?
- Who are the highest-value and most frequent customers?

## Features

### 1. Overview
- Transaction count, payment value, success rate and simulated fraud rate
- Regional transaction value and category-level transaction volume
- Daily payment activity trend
- Horizontal workspace navigation, compact KPI panels and dynamic date, region, category, payment method and transaction amount filters

### 2. Investigations
- Prioritize payments by risk, amount, or recency
- Inspect customer and merchant activity associated with a selected payment
- Review rule-based risk cues and calculate on-demand **SHAP** contributions for an individual Random Forest model score
- Record investigation notes and statuses temporarily in the Streamlit session; export notes to CSV

### 3. Fraud intelligence
- Random Forest classifier trained on synthetic fraud labels
- Precision, recall, F1 score, accuracy and ROC-AUC
- Low / medium / high predicted-risk bands
- High-risk transaction review queue
- Model feature-importance chart

### 4. Merchant risk
- Merchant-level transaction volume and value
- Average ticket size and payment success rate
- Observed synthetic fraud rate and average predicted risk
- Merchant comparison, merchant-specific drill-down, transaction history, concise evidence-based performance brief and portfolio scorecard

### 5. Network intelligence
- Summarize distinct customers, merchants, and observed customer–merchant links
- Review repeated customer–merchant payment relationships with an adjustable minimum transaction count
- Explore merchants that share customers within the selected synthetic dataset
- Export observed repeat relationships to CSV
- Network observations are descriptive; overlapping customers and repeat payments are **not inherently fraudulent**

### 6. Customer insights
- Customer transaction value and frequency
- Average ticket size and payment success rate
- One-time, occasional and frequent customer segments

### 7. Transaction explorer
- Search by transaction ID under active filters
- Sort and paginate transaction records without arbitrary 500-row truncation
- Export all matching records to CSV, not only the displayed page

## Tech stack

- Python
- Pandas and NumPy
- Scikit-learn (Random Forest)
- Streamlit
- Plotly
- SHAP (on-demand local explanations)

## Project structure

```text
paylens-payment-analytics/
├── dashboard/
│   └── app.py
├── data/
│   └── payments.csv          # generated locally / at first app launch
├── src/
│   ├── analytics.py
│   ├── data_quality.py
│   ├── network_analytics.py
│   └── generate_data.py
├── requirements.txt
└── README.md
```

## Run locally

From the project directory:

```bash
python -m pip install -r requirements.txt
python src/generate_data.py
streamlit run dashboard/app.py
```

The dashboard also generates the dataset automatically if `data/payments.csv` is missing.

## Model methodology and evaluation

PayLens uses a **Random Forest classifier** trained on earlier synthetic transactions (first 75% chronologically), with the newest 25% held out for testing. Model features include amount, timestamp-derived hour/weekend indicators, merchant risk attribute, category, region, payment method and device type. Transaction/customer identifiers, the fraud label and payment outcome are excluded as predictor features.

The synthetic fraud label is sampled from an engineered process based on amount, merchant risk, night hours, unknown devices and card payments. The timestamp generator now produces hour and weekend fields from the timestamp itself; the dataset is versioned so older inconsistent samples are regenerated.

**Evaluation:** The Fraud Intelligence page shows precision, recall, F1, ROC-AUC, average precision (AP), the fraud-prevalence/random-ranking AP baseline, Brier score and a threshold-adjustable confusion matrix on the chronological holdout. It now also includes an **uncalibrated-score reliability diagram** and a held-out threshold-comparison table showing precision, recall, alert volume and false alerts. AP is especially useful for these imbalanced synthetic labels. Scores used for older records are marked **in-sample**; newer held-out predictions are marked **out-of-sample**.

For this educational demo, Random Forest class weights are configured as `{0: 1, 1: 5}` rather than fully balancing the classes. The dashboard's illustrative review bands are **Low: under 10%**, **Medium: 10–25%** and **High: over 25%**. The scores have **not been calibrated or externally validated**; the bands are convenience categories, *not* deployable risk policies. Rule-based investigation cues are separate from the optional per-transaction **SHAP** chart. The SHAP bars explain contributions to this model score, not causality.

**Limitations:** Because the dataset and fraud mechanism are synthetic, good test metrics only demonstrate recovery of engineered patterns. A chronological split reduces one form of evaluation optimism but cannot establish performance on real payment data. The CSV is a simulation, and no fraud-alerting or case-management integration is connected to real institutions.

## Run the dataset integrity tests

From the inner `paylens-payment-analytics/` directory:

```bash
python -m unittest discover -s tests -v
```

The tests regenerate the deterministic sample and check schema, identifiers, binary fields, timestamp/hour/weekend consistency, dataset version, chronological holdout boundaries, network summaries, and data-quality error cases. GitHub Actions runs these checks on pushes.

## Live demo verification

These are manual browser checks; passing Python unit tests alone does not prove the Streamlit Cloud interactions work.

1. In **Investigations**, select a payment, expand **Explain this model prediction · SHAP**, and select **Calculate SHAP explanation**. The baseline plus contributions should reconstruct the model score.
2. Change the case decision, enter a note and choose **Save case note**. Navigate away and back; the note should remain during the same Streamlit session.
3. Select **Export session case notes** and verify the CSV has the transaction ID, status and note.
4. In **Fraud Intelligence**, move the threshold slider and confirm precision, recall, false positives and missed cases change when classifications change.
5. Check the model-reliability chart and threshold trade-off table; these describe uncalibrated model predictions.
6. In **Merchant Risk**, choose a merchant and inspect the evidence-based business brief.

Case notes are temporary per session and intentionally not a production data store. If any control fails, capture the visible error and check the Streamlit deployment logs.

## Product quality and accessibility review

The [product-quality audit](../docs/product-quality-audit.md) documents code-level accessibility improvements, UI consistency, cross-workspace drill-downs, automated Streamlit interactions and remaining browser checks. This is **not** a claim of WCAG conformance. Full Streamlit Cloud keyboard, mobile, CSV-download and screen-reader verification remain manual.

## Data quality and usability safeguards

The Streamlit app checks its CSV at startup and regenerates the reproducible synthetic dataset if the schema, version, timestamps or required fields are invalid. Filters support region, category, payment method, amount, date and illustrative risk band. An empty result set produces an explanatory message instead of showing misleading charts.

**Known limits:** Case notes persist only for the current Streamlit session. SHAP computation is performed on demand and may be slower on first use. The app does not establish real-world fraud performance or provide actual connected payment-network information.

## Future improvements

- Walk-forward validation on multiple time windows
- Probability calibration and repeated temporal evaluation beyond the current holdout
- Expand on-demand SHAP to global model explainability and stability checks
- Merchant network graph and customer–merchant relationship analysis
- Automated model monitoring and drift checks
- Expand existing dataset smoke tests and publish a data dictionary


[Portfolio case study · PayLens](https://vedkhairnar.vercel.app/projects/paylens/)
