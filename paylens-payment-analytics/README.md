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
- Review transparent rule-based risk cues (not SHAP/causal explanations)
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
- Merchant comparison, merchant-specific drill-down, transaction history and portfolio scorecard

### 5. Customer insights
- Customer transaction value and frequency
- Average ticket size and payment success rate
- One-time, occasional and frequent customer segments

### 6. Transaction explorer
- Filtered transaction table
- CSV export for further analysis

## Tech stack

- Python
- Pandas and NumPy
- Scikit-learn (Random Forest)
- Streamlit
- Plotly

## Project structure

```text
paylens-payment-analytics/
├── dashboard/
│   └── app.py
├── data/
│   └── payments.csv          # generated locally / at first app launch
├── src/
│   ├── analytics.py
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

## Model methodology

A Random Forest classifier is trained on a stratified train/test split. The synthetic label is generated using a combination of transaction amount, merchant risk score, transaction hour, device type and payment method. The model reports classification metrics on the held-out test set and produces estimated risk scores for the transaction explorer and merchant analysis.

Because the labels and data are simulated, the metrics demonstrate a machine-learning workflow only. They must not be interpreted as evidence of real-world fraud-detection performance.

## Future improvements

- Time-based validation to reduce temporal leakage
- Improve model calibration and cross-validation beyond the existing held-out precision–recall curve and threshold tuning
- Explainability with SHAP
- Merchant network graph and customer–merchant relationship analysis
- Automated model monitoring and drift checks
- Expand existing dataset smoke tests and publish a data dictionary
