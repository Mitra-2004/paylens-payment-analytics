from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="PayLens | Payment Intelligence", page_icon="💳", layout="wide")

st.markdown("""
<style>
.stApp {background: radial-gradient(ellipse at 80% 0%, #14294a 0%, #091322 43%, #07101c 100%); color: #eaf2ff;}
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1450px;}
[data-testid="stMetric"] {background: linear-gradient(135deg,#132741,#0e1d32); border:1px solid #284462; border-radius:16px; padding:18px; box-shadow:0 7px 22px #00000022;}
[data-testid="stMetricLabel"] {color:#b4c9de;}
[data-testid="stMetricValue"] {color:#f3f8ff;}
[data-testid="stSidebar"] {background:#0d1b2e; border-right:1px solid #263c56;}
.stTabs [data-baseweb="tab-list"] {gap:8px; flex-wrap:wrap;}
.stTabs [data-baseweb="tab"] {border-radius:12px; background:#14263d; padding:10px 16px;}
.stTabs [aria-selected="true"] {background:#146d85; color:white;}
h1,h2,h3 {letter-spacing:-0.025em;}
[data-testid="stAlert"] {border-radius:12px;}
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "payments.csv"
GENERATOR_FILE = PROJECT_ROOT / "src" / "generate_data.py"

st.title("💳 PayLens | Intelligence Console")
st.caption("PAYMENT OPERATIONS  /  FRAUD INTELLIGENCE  /  BUSINESS ANALYTICS")
st.info("🧪 Public demo • 25,000 synthetic transactions • No real customer or banking data")

REQUIRED_COLUMNS = {
    "transaction_id", "timestamp", "customer_id", "merchant_id",
    "category", "region", "amount", "success", "payment_method",
    "device_type", "transaction_hour", "is_weekend",
    "merchant_risk_score", "is_fraud",
}

def dataset_needs_refresh(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        existing_columns = set(pd.read_csv(path, nrows=0).columns)
        return not REQUIRED_COLUMNS.issubset(existing_columns)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return True

if dataset_needs_refresh(DATA_FILE):
    if not GENERATOR_FILE.exists():
        st.error(f"Dataset generator not found: {GENERATOR_FILE}")
        st.stop()
    with st.spinner("Preparing the latest synthetic payment dataset..."):
        runpy.run_path(str(GENERATOR_FILE))

if dataset_needs_refresh(DATA_FILE):
    st.error("The generated dataset is missing required columns. Check the generator and Streamlit logs.")
    st.stop()

@st.cache_data
def load_data(file_path: str, modified_time: float) -> pd.DataFrame:
    data = pd.read_csv(file_path, parse_dates=["timestamp"])
    return data


@st.cache_resource
def train_fraud_model(data: pd.DataFrame):
    numeric_features = [
        "amount",
        "transaction_hour",
        "is_weekend",
        "merchant_risk_score",
    ]
    categorical_features = ["category", "region", "payment_method", "device_type"]
    features = pd.get_dummies(
        data[numeric_features + categorical_features],
        columns=categorical_features,
        dtype=int,
    )
    target = data["is_fraud"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42, stratify=target
    )
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    test_probability = model.predict_proba(x_test)[:, 1]
    test_prediction = (test_probability >= 0.50).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, test_prediction),
        "precision": precision_score(y_test, test_prediction, zero_division=0),
        "recall": recall_score(y_test, test_prediction, zero_division=0),
        "f1": f1_score(y_test, test_prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_test, test_probability),
        "average_precision": average_precision_score(y_test, test_probability),
        "test_labels": y_test.to_numpy(),
        "test_probabilities": test_probability,
    }

    all_probability = model.predict_proba(features)[:, 1]
    scored = data.copy()
    scored["predicted_fraud_risk"] = all_probability
    scored["risk_band"] = pd.cut(
        scored["predicted_fraud_risk"],
        bins=[-0.001, 0.30, 0.60, 1.0],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    importance = (
        pd.DataFrame({"feature": features.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(12)
    )
    return scored, metrics, importance


file_modified = DATA_FILE.stat().st_mtime
df = load_data(str(DATA_FILE), file_modified)
scored_df, model_metrics, feature_importance = train_fraud_model(df)

with st.sidebar:
    st.header("🎛️ Control Center")
    st.caption("Explore the synthetic payment network")
    selected_regions = st.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
    selected_categories = st.multiselect("Category", sorted(df["category"].unique()), default=sorted(df["category"].unique()))
    selected_methods = st.multiselect(
        "Payment method", sorted(df["payment_method"].unique()),
        default=sorted(df["payment_method"].unique()),
    )
    min_amount, max_amount = float(df["amount"].min()), float(df["amount"].max())
    amount_range = st.slider(
        "Transaction amount (₹)", min_value=0.0,
        max_value=float(np.ceil(max_amount)), value=(0.0, float(np.ceil(max_amount))),
    )
    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()
    date_range = st.date_input("Transaction dates", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    st.caption("All transactions are synthetic and intended for portfolio demonstration.")

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered = scored_df[
    scored_df["region"].isin(selected_regions)
    & scored_df["category"].isin(selected_categories)
    & scored_df["payment_method"].isin(selected_methods)
    & scored_df["amount"].between(*amount_range)
    & scored_df["timestamp"].dt.date.between(start_date, end_date)
].copy()

if filtered.empty:
    st.warning("No transactions match these filters. Adjust the filters in the sidebar.")
    st.stop()

tab_overview, tab_fraud, tab_merchants, tab_customers, tab_explorer = st.tabs(
    ["Executive Overview", "Fraud Intelligence", "Merchant Risk", "Customer Insights", "Data Explorer"]
)

with tab_overview:
    st.subheader("📊 Executive Overview")
    st.caption("Explore regional performance, payment trends and operational KPIs. All figures reflect your active filters.")
    total_value = filtered["amount"].sum()
    success_rate = filtered["success"].mean() * 100
    fraud_count = int(filtered["is_fraud"].sum())
    fraud_rate = filtered["is_fraud"].mean() * 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions", f"{len(filtered):,}")
    m2.metric("Transaction value", f"₹{total_value:,.0f}")
    m3.metric("Payment success rate", f"{success_rate:.1f}%")
    m4.metric("Simulated fraud rate", f"{fraud_rate:.2f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("**Transaction value by region**")
        region_value = filtered.groupby("region", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
        st.plotly_chart(px.bar(region_value, x="region", y="amount", title="Transaction value (₹)", text_auto=".2s"), use_container_width=True)
    with right:
        st.markdown("**Transaction volume by category**")
        category_volume = filtered.groupby("category", as_index=False).size().sort_values("size", ascending=False)
        st.plotly_chart(px.bar(category_volume, x="category", y="size", title="Number of transactions"), use_container_width=True)

    daily = filtered.set_index("timestamp").resample("D").agg(transaction_count=("transaction_id", "count"), transaction_value=("amount", "sum")).reset_index()
    st.markdown("**Payment activity over time**")
    st.plotly_chart(px.line(daily, x="timestamp", y="transaction_count", title="Daily transaction volume", markers=True), use_container_width=True)
    st.markdown("#### 💡 Executive highlights")
    top_region = region_value.iloc[0]
    top_category = category_volume.iloc[0]
    a, b, c = st.columns(3)
    a.info(f"**Top region:** {top_region['region']} — ₹{top_region['amount']:,.0f} processed")
    b.info(f"**Most active category:** {top_category['category']} — {int(top_category['size']):,} transactions")
    c.info(f"**Simulated fraud alerts:** {fraud_count:,} transactions labelled fraudulent")
    report = (
        "PAYLENS — EXECUTIVE SUMMARY (SYNTHETIC DATA)\n"
        f"Transactions: {len(filtered):,}\n"
        f"Transaction value: INR {total_value:,.2f}\n"
        f"Success rate: {success_rate:.2f}%\n"
        f"Simulated fraud rate: {fraud_rate:.2f}%\n"
        f"Top region by value: {top_region['region']}\n"
        f"Top category by volume: {top_category['category']}\n"
        "This report uses simulated transactions and is not a real-world fraud assessment.\n"
    )
    st.download_button("📥 Export executive summary", report, file_name="paylens_executive_summary.txt", mime="text/plain")

with tab_fraud:
    st.subheader("Fraud Intelligence")
    st.caption("Random Forest on synthetic fraud labels. Test metrics use held-out transactions; transaction queue scores include in-sample predictions and are illustrative, not independently validated.")
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Accuracy", f"{model_metrics['accuracy'] * 100:.1f}%")
    f2.metric("Precision", f"{model_metrics['precision'] * 100:.1f}%")
    f3.metric("Recall", f"{model_metrics['recall'] * 100:.1f}%")
    f4.metric("F1 score", f"{model_metrics['f1']:.3f}")
    f5.metric("ROC-AUC", f"{model_metrics['roc_auc']:.3f}")

    st.markdown("**Classification threshold analysis (held-out test set)**")
    threshold = st.slider("Fraud alert threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
    test_labels = model_metrics["test_labels"]
    test_probabilities = model_metrics["test_probabilities"]
    threshold_predictions = (test_probabilities >= threshold).astype(int)
    cm = confusion_matrix(test_labels, threshold_predictions, labels=[0, 1])
    tp = int(cm[1, 1])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tn = int(cm[0, 0])
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Test precision", f"{precision_score(test_labels, threshold_predictions, zero_division=0):.1%}")
    p2.metric("Test recall", f"{recall_score(test_labels, threshold_predictions, zero_division=0):.1%}")
    p3.metric("False positives", f"{fp:,}")
    p4.metric("Missed fraud cases", f"{fn:,}")
    st.caption(f"Held-out test set: {len(test_labels):,} transactions; {tp:,} true positives and {tn:,} true negatives. Average precision (PR-AUC summary): {model_metrics['average_precision']:.3f}.")
    cm_frame = pd.DataFrame(
        cm,
        index=["Actual legitimate", "Actual fraud"],
        columns=["Predicted legitimate", "Predicted fraud"],
    )
    left_eval, right_eval = st.columns(2)
    with left_eval:
        st.markdown("**Confusion matrix**")
        st.dataframe(cm_frame, use_container_width=True)
    with right_eval:
        st.markdown("**Precision–recall curve (held-out test set)**")
        pr_precision, pr_recall, _ = precision_recall_curve(test_labels, test_probabilities)
        pr_frame = pd.DataFrame({"Recall": pr_recall, "Precision": pr_precision})
        selected_precision = precision_score(test_labels, threshold_predictions, zero_division=0)
        selected_recall = recall_score(test_labels, threshold_predictions, zero_division=0)
        pr_chart = px.line(
            pr_frame,
            x="Recall",
            y="Precision",
            title="Precision–recall curve and selected threshold",
            labels={"Recall": "Recall", "Precision": "Precision"},
        )
        pr_chart.add_scatter(
            x=[selected_recall],
            y=[selected_precision],
            mode="markers+text",
            marker={"size": 14, "color": "#ef4444", "symbol": "circle"},
            text=[f"Threshold {threshold:.2f}"],
            textposition="top center",
            name="Selected threshold",
            hovertemplate=(
                f"Threshold: {threshold:.2f}<br>"
                f"Precision: {selected_precision:.3f}<br>"
                f"Recall: {selected_recall:.3f}<extra></extra>"
            ),
        )
        pr_chart.update_layout(xaxis_range=[0, 1.02], yaxis_range=[0, 1.05])
        st.plotly_chart(pr_chart, use_container_width=True)
        st.caption(
            f"Selected threshold {threshold:.2f}: "
            f"precision {selected_precision:.1%}, recall {selected_recall:.1%}. "
            "The full curve stays fixed because it represents all possible thresholds."
        )

    high_risk = filtered[filtered["risk_band"] == "High"].sort_values("predicted_fraud_risk", ascending=False)
    left, right = st.columns(2)
    with left:
        st.markdown("**Transactions by predicted risk band**")
        risk_counts = filtered.groupby("risk_band", observed=False).size().reindex(["Low", "Medium", "High"], fill_value=0).rename("transactions").reset_index()
        st.plotly_chart(px.bar(risk_counts, x="risk_band", y="transactions", color="risk_band", title="Model risk classification"), use_container_width=True)
    with right:
        st.markdown("**Most influential model features**")
        st.plotly_chart(px.bar(feature_importance.sort_values("importance"), x="importance", y="feature", orientation="h", title="Random Forest feature importance"), use_container_width=True)

    st.markdown("**High-risk transaction queue**")
    display_cols = ["transaction_id", "timestamp", "customer_id", "merchant_id", "amount", "region", "category", "predicted_fraud_risk", "risk_band", "is_fraud"]
    st.dataframe(high_risk[display_cols].head(100), use_container_width=True, hide_index=True)
    st.caption("The is_fraud column is the synthetic ground-truth label included only to evaluate this demonstration dataset.")

with tab_merchants:
    st.subheader("Merchant Risk & Performance")
    merchant_summary = filtered.groupby("merchant_id").agg(
        transactions=("transaction_id", "count"),
        transaction_value=("amount", "sum"),
        avg_ticket=("amount", "mean"),
        payment_success_rate=("success", "mean"),
        observed_fraud_rate=("is_fraud", "mean"),
        predicted_risk=("predicted_fraud_risk", "mean"),
        merchant_risk_score=("merchant_risk_score", "mean"),
    ).reset_index()
    merchant_summary["payment_success_rate"] *= 100
    merchant_summary["observed_fraud_rate"] *= 100
    merchant_summary["predicted_risk"] *= 100
    merchant_summary["risk_tier"] = pd.cut(merchant_summary["predicted_risk"], [-0.01, 2, 5, 100], labels=["Low", "Medium", "High"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Merchants with highest predicted risk**")
        top_risk = merchant_summary.nlargest(12, "predicted_risk")
        st.plotly_chart(px.bar(top_risk.sort_values("predicted_risk"), x="predicted_risk", y="merchant_id", orientation="h", title="Average predicted fraud risk (%)"), use_container_width=True)
    with right:
        st.markdown("**Merchant value vs. predicted risk**")
        st.plotly_chart(px.scatter(merchant_summary, x="transaction_value", y="predicted_risk", size="transactions", hover_name="merchant_id", title="Merchant portfolio map", labels={"transaction_value": "Transaction value (₹)", "predicted_risk": "Predicted risk (%)"}), use_container_width=True)
    st.markdown("**Merchant scorecard**")
    st.dataframe(merchant_summary.sort_values("predicted_risk", ascending=False).head(100).round(2), use_container_width=True, hide_index=True)

with tab_customers:
    st.subheader("Customer Insights")
    customer_summary = filtered.groupby("customer_id").agg(
        transactions=("transaction_id", "count"),
        lifetime_value=("amount", "sum"),
        average_ticket=("amount", "mean"),
        success_rate=("success", "mean"),
        predicted_risk=("predicted_fraud_risk", "mean"),
    ).reset_index()
    customer_summary["success_rate"] *= 100
    customer_summary["predicted_risk"] *= 100
    left, right = st.columns(2)
    with left:
        st.markdown("**Top customers by transaction value**")
        top_customers = customer_summary.nlargest(12, "lifetime_value")
        st.plotly_chart(px.bar(top_customers.sort_values("lifetime_value"), x="lifetime_value", y="customer_id", orientation="h", title="Customer transaction value (₹)"), use_container_width=True)
    with right:
        st.markdown("**Customer activity segments**")
        customer_summary["activity_segment"] = pd.cut(customer_summary["transactions"], bins=[0, 1, 3, 10000], labels=["One-time", "Occasional", "Frequent"])
        segments = customer_summary.groupby("activity_segment", observed=False).size().rename("customers").reset_index()
        st.plotly_chart(px.pie(segments, names="activity_segment", values="customers", title="Customer segmentation"), use_container_width=True)
    st.dataframe(customer_summary.sort_values("lifetime_value", ascending=False).head(100).round(2), use_container_width=True, hide_index=True)

with tab_explorer:
    st.subheader("🔎 Transaction Data Explorer")
    search_id = st.text_input("Find a transaction by ID", placeholder="e.g. 1042")
    if search_id.strip():
        if search_id.strip().isdigit():
            filtered = filtered.loc[filtered["transaction_id"] == int(search_id.strip())].copy()
        else:
            st.warning("Enter a numeric transaction ID.")
    st.write(f"Showing {min(500, len(filtered)):,} of {len(filtered):,} filtered transactions.")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered transactions (CSV)", data=csv, file_name="paylens_filtered_transactions.csv", mime="text/csv")

st.divider()
st.caption("PayLens is a student portfolio project using reproducible synthetic data. It is not connected to a bank, payment processor or live fraud system.")
