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

st.set_page_config(page_title="PayLens | Payment Intelligence", page_icon="💳", layout="wide", initial_sidebar_state="expanded")
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = ["#FF813F", "#F17C9B", "#E6B05D", "#A38BDB", "#68B6A3", "#8D96A5"]

st.markdown("""
<style>
:root{--orange:#ff762f;--pink:#f26489;--panel:#202023;--line:#39383b}
.stApp{background:radial-gradient(ellipse 55% 28% at 80% -4%,rgba(255,110,40,.16),transparent 80%),radial-gradient(ellipse 35% 25% at 14% 98%,rgba(221,72,37,.075),transparent 80%),#111113;color:#f4f3f3}
.block-container{max-width:1580px;padding:1.25rem 2rem 3rem}
[data-testid="stSidebar"]{background:linear-gradient(160deg,#1b1b1e,#111113 82%);border-right:1px solid #363538}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.65rem}
[data-testid="stMetric"]{background:radial-gradient(ellipse at 15% 0%,#343032 0%,transparent 70%),linear-gradient(145deg,#262629,#1b1b1e);border:1px solid #414044;border-radius:17px;padding:20px;min-height:116px;box-shadow:inset 0 1px 0 #ffffff13,0 10px 28px #0005;transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease}
[data-testid="stMetric"]:hover{transform:translateY(-3px);border-color:#9d5537;box-shadow:0 12px 30px #0007,0 0 22px #ff762f14}
[data-testid="stMetricLabel"]{color:#bdbcc0;font-size:.85rem}
[data-testid="stMetricValue"]{color:#faf9f8;letter-spacing:-.04em;font-weight:680}
.stTabs [data-baseweb="tab-list"]{gap:5px;flex-wrap:wrap;border-bottom:1px solid #343438;padding-bottom:10px}
.stTabs [data-baseweb="tab"]{background:#242326;border:1px solid #3b393c;border-radius:10px;padding:10px 15px;color:#c6c2c3;transition:all .17s ease}
.stTabs [data-baseweb="tab"]:hover{background:#342a29;color:white}
.stTabs [aria-selected="true"]{background:linear-gradient(90deg,#523128,#332526);color:#ffad79;border-color:#94553a}
h1,h2,h3{letter-spacing:-.04em}h1{font-weight:740}
[data-testid="stPlotlyChart"]{background:linear-gradient(140deg,#27272a,#1c1c1f);border:1px solid #3c3b3f;border-radius:17px;overflow:hidden;box-shadow:inset 0 1px 0 #ffffff0d,0 10px 25px #0003}
[data-testid="stDataFrame"]{border:1px solid #373639;border-radius:12px;overflow:hidden}
[data-testid="stVerticalBlockBorderWrapper"]{background:linear-gradient(145deg,#28282b,#1c1c1f);border:1px solid #424044!important;border-radius:17px!important;box-shadow:inset 0 1px 0 #ffffff0b,0 12px 32px #0004}
[data-testid="stAlert"]{border-radius:12px}
div.stButton button,div.stDownloadButton button{border-radius:10px;transition:transform .18s ease,border-color .18s ease}
div.stButton button:hover,div.stDownloadButton button:hover{transform:translateY(-1px);border-color:#ff762f}
div[data-baseweb="select"]>div{border-radius:10px}
@media(max-width:850px){.block-container{padding:1rem .75rem 2rem}[data-testid="stMetric"]{padding:12px;min-height:95px}.stTabs [data-baseweb="tab"]{padding:8px 10px}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{transition:none!important;animation:none!important}}
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "payments.csv"
GENERATOR_FILE = PROJECT_ROOT / "src" / "generate_data.py"

st.title("PayLens")
st.caption("Payment intelligence  /  Fraud analytics  /  Merchant performance")



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
    st.markdown("### 🟠 PayLens")
    st.caption("INTELLIGENCE WORKSPACE")
    st.divider()
    st.header("Filters")
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

tab_overview, tab_investigate, tab_fraud, tab_merchants, tab_customers, tab_explorer = st.tabs(
    ["Overview", "Investigations", "Fraud model", "Merchants", "Customers", "Transactions"]
)

with tab_overview:
    st.subheader("Overview")
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

    daily = filtered.set_index("timestamp").resample("D").agg(
        transaction_count=("transaction_id", "count"),
        transaction_value=("amount", "sum"),
        simulated_fraud=("is_fraud", "sum"),
    ).reset_index()
    region_value = filtered.groupby("region", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    category_volume = filtered.groupby("category", as_index=False).size().sort_values("size", ascending=False)
    chart_left, chart_right = st.columns([1.55, 1], gap="medium")
    with chart_left:
        with st.container(border=True):
            st.markdown("#### Transaction trend")
            st.caption("Daily payment activity across your selected period")
            trend_metric = st.radio(
                "Metric", ["Transaction volume", "Transaction value"],
                horizontal=True, key="overview_trend",
            )
            trend_column = "transaction_count" if trend_metric == "Transaction volume" else "transaction_value"
            trend_chart = px.area(daily, x="timestamp", y=trend_column)
            trend_chart.update_traces(
                line={"width": 3, "color": "#ff762f"},
                fillcolor="rgba(255,118,47,.18)",
                hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra></extra>",
            )
            trend_chart.update_layout(
                height=310, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#d4d1d2"}, margin={"l": 5, "r": 8, "t": 10, "b": 4},
                xaxis={"showgrid": False, "title": None},
                yaxis={"gridcolor": "#383638", "title": None},
                showlegend=False, hovermode="x unified",
            )
            st.plotly_chart(trend_chart, use_container_width=True)
    with chart_right:
        with st.container(border=True):
            st.markdown("#### Intelligence brief")
            st.caption("Observations generated from the filtered synthetic data")
            high_risk_count = int((filtered["risk_band"] == "High").sum())
            largest_region = region_value.iloc[0]
            highest_category = category_volume.iloc[0]
            st.markdown(f"**High-risk model flags** · {high_risk_count:,}")
            st.caption("Transactions with model scores above 60%.")
            st.divider()
            st.markdown(f"**Largest market** · {largest_region['region']}")
            st.caption(f"₹{largest_region['amount']:,.0f} in transaction value.")
            st.divider()
            st.markdown(f"**Most active category** · {highest_category['category']}")
            st.caption(f"{int(highest_category['size']):,} payments in the selected period.")
            st.divider()
            st.caption("These are descriptive signals, not real-time alerts or AI-generated causal findings.")

    bottom_left, bottom_right = st.columns([1, 1.55], gap="medium")
    with bottom_left:
        with st.container(border=True):
            st.markdown("#### Risk distribution")
            st.caption("Share of transactions by model risk band")
            risk_counts = filtered["risk_band"].value_counts().reindex(
                ["Low", "Medium", "High"], fill_value=0
            )
            risk_pie = px.pie(
                names=risk_counts.index, values=risk_counts.values, hole=.68,
                color=risk_counts.index,
                color_discrete_map={"Low": "#57565d", "Medium": "#ff852f", "High": "#f05a82"},
            )
            risk_pie.update_traces(textinfo="percent", textposition="outside")
            risk_pie.update_layout(
                height=260, margin={"l": 5, "r": 5, "t": 0, "b": 0},
                paper_bgcolor="rgba(0,0,0,0)", font={"color": "#f1eded"},
                legend={"orientation": "h", "y": -.05},
            )
            st.plotly_chart(risk_pie, use_container_width=True)
    with bottom_right:
        with st.container(border=True):
            st.markdown("#### Recent high-risk transactions")
            st.caption("Investigate individual payments in the Investigations workspace")
            recent_queue = filtered.sort_values(
                ["predicted_fraud_risk", "timestamp"], ascending=False
            ).head(7)
            st.dataframe(
                recent_queue[["transaction_id", "merchant_id", "amount", "predicted_fraud_risk"]],
                hide_index=True, use_container_width=True,
                column_config={
                    "transaction_id": "Transaction",
                    "merchant_id": "Merchant",
                    "amount": st.column_config.NumberColumn("Amount", format="₹%.2f"),
                    "predicted_fraud_risk": st.column_config.ProgressColumn(
                        "Risk score", min_value=0, max_value=1, format="%.2f"
                    ),
                },
            )
            st.caption("Scores are illustrative; they include in-sample model predictions.")

    with st.expander("Explore regional and category performance"):
        breakdown_left, breakdown_right = st.columns(2)
        with breakdown_left:
            region_chart = px.bar(region_value, x="region", y="amount", title="Value by region (₹)")
            region_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(region_chart, use_container_width=True)
        with breakdown_right:
            category_chart = px.bar(category_volume, x="category", y="size", title="Transactions by category")
            category_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(category_chart, use_container_width=True)
    st.markdown("#### What stands out")
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
    st.download_button("Download executive summary", report, file_name="paylens_executive_summary.txt", mime="text/plain")


with tab_investigate:
    st.subheader("Transaction investigations")
    st.caption("Explore a simulated payment, its merchant context and model score. Investigation notes are saved only for this browser session.")
    queue = filtered.sort_values("predicted_fraud_risk", ascending=False)
    left_queue, right_queue = st.columns([1, 2], gap="large")
    with left_queue:
        st.markdown("**Review queue**")
        queue_mode = st.radio(
            "Queue", ["Highest risk", "Largest amount", "Recent"],
            horizontal=True, key="investigation_queue",
        )
        if queue_mode == "Largest amount":
            queue = filtered.sort_values("amount", ascending=False)
        elif queue_mode == "Recent":
            queue = filtered.sort_values("timestamp", ascending=False)
        candidate_ids = queue["transaction_id"].head(150).astype(int).tolist()
        selected_id = st.selectbox(
            "Select transaction",
            candidate_ids,
            format_func=lambda value: f"TX-{value:06d}",
            key="investigation_transaction",
        )
        st.dataframe(
            queue[["transaction_id", "amount", "merchant_id", "predicted_fraud_risk"]].head(15),
            use_container_width=True, hide_index=True,
            column_config={
                "predicted_fraud_risk": st.column_config.ProgressColumn(
                    "Risk score", min_value=0.0, max_value=1.0, format="%.1f"
                )
            },
        )
        st.caption("Queue shows up to 150 candidate transactions under your current filters.")
    selected = filtered.loc[filtered["transaction_id"] == selected_id].iloc[0]
    merchant_history = scored_df.loc[scored_df["merchant_id"] == selected["merchant_id"]]
    customer_history = scored_df.loc[scored_df["customer_id"] == selected["customer_id"]]
    with right_queue:
        st.markdown(f"### TX-{int(selected_id):06d}")
        st.caption(f"{selected['timestamp']:%d %b %Y · %H:%M}  ·  {selected['region']}  ·  {selected['category']}")
        risk_value = float(selected["predicted_fraud_risk"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Transaction amount", f"₹{selected['amount']:,.2f}")
        m2.metric("Model risk score", f"{risk_value:.1%}")
        m3.metric("Risk band", str(selected["risk_band"]))
        st.progress(risk_value, text="Illustrative model risk estimate")
        st.markdown("**Payment details**")
        detail = pd.DataFrame({
            "Field": ["Merchant", "Customer", "Payment method", "Device", "Payment result", "Simulated fraud label"],
            "Value": [
                f"M-{int(selected['merchant_id']):04d}",
                f"C-{int(selected['customer_id']):04d}",
                str(selected["payment_method"]),
                str(selected["device_type"]),
                "Successful" if int(selected["success"]) else "Failed",
                "Fraud" if int(selected["is_fraud"]) else "Legitimate",
            ],
        })
        st.dataframe(detail, hide_index=True, use_container_width=True)
        st.markdown("**What deserves a closer look?**")
        amount_cutoff = float(scored_df["amount"].quantile(0.94))
        indicators = []
        if selected["amount"] > amount_cutoff:
            indicators.append("This amount is above the 94th percentile of the synthetic dataset.")
        if int(selected["transaction_hour"]) <= 4 or int(selected["transaction_hour"]) >= 23:
            indicators.append("The recorded transaction hour is in the late-night window.")
        if selected["device_type"] == "Unknown":
            indicators.append("The device type is unknown.")
        if float(selected["merchant_risk_score"]) > 0.5:
            indicators.append("The merchant has an elevated simulated risk attribute.")
        if selected["payment_method"] in ["Credit Card", "Debit Card"]:
            indicators.append("Card payment is one of the signals used in the synthetic label-generation process.")
        if indicators:
            for indicator in indicators:
                st.markdown(f"- {indicator}")
        else:
            st.write("No predefined synthetic risk indicators stand out for this transaction.")
        st.caption(
            "These are transparent, rule-based investigation cues—not SHAP explanations or causal explanations of this model prediction."
        )
    st.divider()
    st.markdown("#### Connected activity")
    h1, h2, h3 = st.columns(3)
    h1.metric("Merchant transactions", f"{len(merchant_history):,}")
    h2.metric("Merchant average model risk", f"{merchant_history['predicted_fraud_risk'].mean():.1%}")
    h3.metric("Customer transactions", f"{len(customer_history):,}")
    history_view = st.radio(
        "Related activity", ["Merchant", "Customer"], horizontal=True, key="investigation_history"
    )
    related = merchant_history if history_view == "Merchant" else customer_history
    related = related.sort_values("timestamp", ascending=False)
    st.dataframe(
        related[["transaction_id", "timestamp", "amount", "region", "payment_method", "predicted_fraud_risk"]].head(30),
        hide_index=True, use_container_width=True,
    )
    st.markdown("#### Analyst decision")
    if "case_records" not in st.session_state:
        st.session_state["case_records"] = {}
    previous = st.session_state["case_records"].get(int(selected_id), {})
    with st.form(f"case_form_{int(selected_id)}"):
        decision_options = ["Unreviewed", "Monitor", "Escalate for review", "No action"]
        previous_status = previous.get("status", "Unreviewed")
        status = st.selectbox("Decision", decision_options, index=decision_options.index(previous_status))
        notes = st.text_area("Investigation notes", value=previous.get("notes", ""), max_chars=1000,
                             placeholder="Document the signals reviewed and why this decision was made.")
        save = st.form_submit_button("Save case note", type="primary")
    if save:
        st.session_state["case_records"][int(selected_id)] = {"status": status, "notes": notes}
        st.success("Case note saved for this session.")
    if st.session_state["case_records"]:
        records = pd.DataFrame([
            {"transaction_id": tx, **entry}
            for tx, entry in st.session_state["case_records"].items()
        ])
        st.download_button(
            "Export session case notes", records.to_csv(index=False).encode("utf-8"),
            file_name="paylens_investigation_notes.csv", mime="text/csv",
        )
    st.caption("Session-only notes are not shared or persisted. This is a demonstration workflow, not a production case-management system.")

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

    with st.expander("How to interpret these results"):
        st.write("Lower thresholds catch more potentially fraudulent transactions but may create more false alarms. Higher thresholds reduce reviews but can miss fraud cases. Compare held-out test metrics before deciding.")

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
    st.subheader("Transaction explorer")
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
