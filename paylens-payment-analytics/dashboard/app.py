from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="PayLens | Payment Intelligence", page_icon="💳", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "payments.csv"
GENERATOR_FILE = PROJECT_ROOT / "src" / "generate_data.py"

st.title("💳 PayLens")
st.caption("Payment Network Intelligence • Analytics, fraud detection and merchant risk")

if not DATA_FILE.exists():
    if not GENERATOR_FILE.exists():
        st.error(f"Dataset generator not found: {GENERATOR_FILE}")
        st.stop()
    runpy.run_path(str(GENERATOR_FILE))

if not DATA_FILE.exists():
    st.error(f"Dataset generation did not create: {DATA_FILE}")
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
    st.header("Explore data")
    selected_regions = st.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
    selected_categories = st.multiselect("Category", sorted(df["category"].unique()), default=sorted(df["category"].unique()))
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
    & scored_df["timestamp"].dt.date.between(start_date, end_date)
].copy()

if filtered.empty:
    st.warning("No transactions match these filters. Adjust the filters in the sidebar.")
    st.stop()

tab_overview, tab_fraud, tab_merchants, tab_customers, tab_explorer = st.tabs(
    ["Executive Overview", "Fraud Intelligence", "Merchant Risk", "Customer Insights", "Data Explorer"]
)

with tab_overview:
    st.subheader("Executive Overview")
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
    st.plotly_chart(px.line(daily, x="timestamp", y="transaction_count", title="Daily transaction volume"), use_container_width=True)

with tab_fraud:
    st.subheader("Fraud Intelligence")
    st.caption("A Random Forest classifier is trained on simulated fraud labels. Scores are educational estimates, not production fraud decisions.")
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Accuracy", f"{model_metrics['accuracy'] * 100:.1f}%")
    f2.metric("Precision", f"{model_metrics['precision'] * 100:.1f}%")
    f3.metric("Recall", f"{model_metrics['recall'] * 100:.1f}%")
    f4.metric("F1 score", f"{model_metrics['f1']:.3f}")
    f5.metric("ROC-AUC", f"{model_metrics['roc_auc']:.3f}")

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
    st.subheader("Transaction Data Explorer")
    st.write(f"Showing {min(500, len(filtered)):,} of {len(filtered):,} filtered transactions.")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered transactions (CSV)", data=csv, file_name="paylens_filtered_transactions.csv", mime="text/csv")

st.divider()
st.caption("PayLens is a student portfolio project using reproducible synthetic data. It is not connected to a bank, payment processor or live fraud system.")
