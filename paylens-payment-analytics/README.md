# PayLens — Payment Network Intelligence

PayLens is a student portfolio project that combines payment analytics, synthetic fraud detection, merchant risk monitoring and customer segmentation in an interactive Streamlit dashboard.

> **Data disclaimer:** The project uses reproducible synthetic transactions. It is not connected to a bank, payment processor or live fraud system. Fraud labels are simulated for learning and model evaluation.

## Business questions

- How are transaction volume and payment value distributed across regions and categories?
- What are the payment success and simulated fraud rates?
- Which transactions should be reviewed first based on model-estimated risk?
- Which merchants have elevated predicted risk or weak payment performance?
- Who are the highest-value and most frequent customers?

## Features

### 1. Executive overview
- Transaction count, payment value, success rate and simulated fraud rate
- Regional transaction value and category-level transaction volume
- Daily payment activity trend
- Sidebar filters for region, category and transaction date

### 2. Fraud intelligence
- Random Forest classifier trained on synthetic fraud labels
- Precision, recall, F1 score, accuracy and ROC-AUC
- Low / medium / high predicted-risk bands
- High-risk transaction review queue
- Model feature-importance chart

### 3. Merchant risk
- Merchant-level transaction volume and value
- Average ticket size and payment success rate
- Observed synthetic fraud rate and average predicted risk
- Merchant portfolio scatter plot and scorecard

### 4. Customer insights
- Customer transaction value and frequency
- Average ticket size and payment success rate
- One-time, occasional and frequent customer segments

### 5. Data explorer
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
- Precision-recall curve and threshold tuning for imbalanced classes
- Explainability with SHAP
- Merchant network graph and customer–merchant relationship analysis
- Automated model monitoring and drift checks
- Unit tests and a documented data dictionary
