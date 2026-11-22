from html import escape
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import shap
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
from sklearn.metrics import brier_score_loss

st.set_page_config(page_title="PayLens | Payment Intelligence", page_icon="💳", layout="wide", initial_sidebar_state="expanded")
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = ["#FF7A3D", "#F0508A", "#E6B05D", "#A38BDB", "#68B6A3", "#8D96A5"]

ACCENT = "#FF7A3D"
PINK = "#F0508A"
PLOT_CONFIG = {"displayModeBar": False}

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;650;700&display=swap');
:root{--bg:#0b0b0d;--surface:#17171b;--line:#2a2a31;--line-2:#383840;--text:#f5f4f4;--muted:#8e8b91;--orange:#ff7a3d;--orange-soft:#ffb38a;--pink:#f0508a;--radius:18px}
html,body,.stApp{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.stApp{background:radial-gradient(ellipse 60% 30% at 85% -6%,rgba(255,122,61,.13),transparent 75%),radial-gradient(ellipse 40% 26% at 6% 100%,rgba(240,80,138,.06),transparent 80%),var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:transparent}
[data-testid="stDecoration"]{display:none}
.block-container{max-width:1520px;padding:1.6rem 2.2rem 3rem}
h1,h2,h3,h4{letter-spacing:-.03em;color:var(--text)}

/* Sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#131316,#0d0d10);border-right:1px solid #222228}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.6rem}
.pl-brand{display:flex;align-items:center;gap:12px;padding:6px 4px 14px}
.pl-logo{width:38px;height:38px;border-radius:12px;background:linear-gradient(140deg,#ffa066,#ff5a2e 60%,#e0356b);display:grid;place-items:center;color:#fff;font-weight:800;font-size:1.1rem;box-shadow:0 10px 22px #ff5a2e40}
.pl-brand-name{font-weight:700;font-size:1.05rem;color:var(--text);letter-spacing:-.02em}
.pl-brand-sub{font-size:.72rem;color:var(--muted)}
.pl-nav-label{color:#6f6c72;font-size:.68rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;margin:10px 6px 6px}
[data-testid="stSidebar"] [role="radiogroup"]{gap:4px}
[data-testid="stSidebar"] [role="radiogroup"] label{border:1px solid transparent;padding:9px 12px;border-radius:12px;transition:background .2s ease,border-color .2s ease;cursor:pointer}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#1c1c21}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:linear-gradient(90deg,rgba(255,122,61,.18),rgba(240,80,138,.06));border-color:#5a3423;box-shadow:inset 3px 0 var(--orange)}
[data-testid="stSidebar"] [role="radiogroup"] label p{font-size:.92rem;color:#c9c6cb}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{color:#ffc5a4;font-weight:600}
[data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"]>div:first-child{display:none}

/* Horizontal navigation: intentional single-line workstation tabs */
.st-key-paylens_page{margin:0 0 18px;padding:7px;border:1px solid #353137;border-radius:15px;background:linear-gradient(150deg,#1f1d21,#151519);box-shadow:inset 0 1px 0 #ffffff0c,0 10px 25px #0003;overflow-x:auto}
.st-key-paylens_page [role="radiogroup"]{display:flex;flex-direction:row!important;flex-wrap:nowrap!important;gap:7px;width:max-content;min-width:100%}
.st-key-paylens_page [role="radiogroup"] label{display:flex;align-items:center;justify-content:center;flex:1 0 auto;white-space:nowrap;padding:11px 15px;border:1px solid transparent;border-radius:10px;cursor:pointer;transition:background .18s ease,box-shadow .18s ease,border-color .18s ease}
.st-key-paylens_page [role="radiogroup"] label:hover{background:#30272a;color:#fff}
.st-key-paylens_page [role="radiogroup"] label:has(input:checked){background:linear-gradient(115deg,#623323,#342225);border-color:#82503a;box-shadow:inset 0 1px 0 #ffa56c29,0 0 16px #ff7a3d15}
.st-key-paylens_page [role="radiogroup"] label p{font-size:.87rem;color:#c9c5cb}
.st-key-paylens_page [role="radiogroup"] label:has(input:checked) p{color:#ffc9a7;font-weight:650}
.st-key-paylens_page [role="radiogroup"] [data-baseweb="radio"]>div:first-child{display:none}
@media(max-width:900px){.st-key-paylens_page [role="radiogroup"] label{padding:10px 11px}.st-key-paylens_page [role="radiogroup"] label p{font-size:.8rem}}

/* Top bar */
.pl-topbar{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;margin:0 0 20px;padding-bottom:16px;border-bottom:1px solid #222228}
.pl-page-title{font-size:1.65rem;font-weight:700;letter-spacing:-.035em;color:var(--text)}
.pl-page-sub{color:var(--muted);font-size:.9rem;margin-top:4px;max-width:760px;line-height:1.5}
.pl-status{white-space:nowrap;font-size:.78rem;color:var(--orange-soft);background:#2a1b14;border:1px solid #4a2c1d;border-radius:999px;padding:6px 12px;font-weight:600}

/* KPI cards */
.pl-kpi-grid{display:grid;grid-template-columns:repeat(var(--cols,4),minmax(0,1fr));gap:14px;margin:4px 0 16px}
.pl-kpi{position:relative;overflow:hidden;background:linear-gradient(150deg,#1f1f25,#15151a);border:1px solid var(--line-2);border-radius:var(--radius);padding:18px 20px;transition:transform .25s cubic-bezier(.2,.8,.2,1),border-color .25s ease,box-shadow .25s ease;box-shadow:inset 0 1px 0 #ffffff0d,0 8px 24px #0004}
.pl-kpi:hover{transform:translateY(-2px);border-color:#5a3a2c;box-shadow:inset 0 1px 0 #ffffff0d,0 14px 30px #0006,0 0 24px #ff7a3d12}
.pl-kpi-label{color:var(--muted);font-size:.8rem;font-weight:500}
.pl-kpi-value{color:var(--text);font-size:1.75rem;font-weight:700;letter-spacing:-.03em;margin-top:6px;font-variant-numeric:tabular-nums}
.pl-kpi-note{color:var(--muted);font-size:.78rem;margin-top:6px}
.pl-kpi-compact{padding:14px 16px;border-radius:14px}
.pl-kpi-compact .pl-kpi-value{font-size:1.4rem}

/* Panels */
.pl-panel-head{display:flex;gap:12px;align-items:center;margin:2px 0 14px}
.pl-chip{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;background:linear-gradient(145deg,#3a2419,#231a17);border:1px solid #5a3626;color:var(--orange-soft);font-size:1.05rem;flex:none}
.pl-panel-title{color:var(--text);font-weight:650;font-size:1rem;letter-spacing:-.01em}
.pl-panel-sub{color:var(--muted);font-size:.82rem;margin-top:2px}
[data-testid="stVerticalBlockBorderWrapper"]{background:linear-gradient(160deg,#1a1a1f,#131316)!important;border:1px solid var(--line)!important;border-radius:var(--radius)!important;box-shadow:inset 0 1px 0 #ffffff08,0 14px 34px #0005}
[data-testid="stPlotlyChart"]{background:none;border:none;box-shadow:none;padding:0}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden}
[data-testid="stExpander"]{background:var(--surface);border:1px solid var(--line)!important;border-radius:16px!important;overflow:hidden}
[data-testid="stExpander"] summary{padding:12px 16px;font-weight:600;color:#dcd9dc}

/* Rows, insights, cues */
.pl-row{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:12px 0;border-bottom:1px solid #26262c}
.pl-row:last-child{border-bottom:none}
.pl-row-k{color:var(--text);font-weight:600;font-size:.94rem}
.pl-row-s{color:var(--muted);font-size:.82rem;margin-top:3px}
.pl-row-v{color:var(--orange-soft);font-weight:700;font-size:1rem;white-space:nowrap;font-variant-numeric:tabular-nums}
.pl-insights{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.pl-insight{background:linear-gradient(150deg,#1f1f25,#15151a);border:1px solid var(--line-2);border-radius:16px;padding:16px 18px}
.pl-insight-k{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.09em;font-weight:600}
.pl-insight-v{color:var(--text);font-weight:650;margin-top:8px;font-size:1rem}
.pl-insight-s{color:var(--muted);font-size:.82rem;margin-top:4px;line-height:1.45}
.pl-cue{display:flex;gap:10px;align-items:flex-start;background:#1d1a1c;border:1px solid #352a2c;border-radius:12px;padding:10px 12px;color:#e6e2e4;font-size:.9rem;margin:6px 0;line-height:1.45}
.pl-cue::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--orange);margin-top:7px;flex:none}
.pl-txn-head{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:12px}
.pl-txn-id{font-size:1.4rem;font-weight:700;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.pl-meter{height:8px;border-radius:999px;background:#26262c;overflow:hidden;margin:8px 0 6px}
.pl-meter-fill{height:100%;border-radius:999px;transition:width .4s ease}
.pl-badge{display:inline-flex;align-items:center;padding:4px 11px;border-radius:999px;font-size:.76rem;font-weight:600;border:1px solid;white-space:nowrap}
.pl-badge-high{color:#ff8fb1;background:#3a1622;border-color:#6b2a3f}
.pl-badge-medium{color:#ffb38a;background:#33200f;border-color:#6a3d1d}
.pl-badge-low{color:#b9b6bc;background:#202026;border-color:#3a3a42}

/* Buttons and inputs */
div.stButton button,div.stDownloadButton button,div.stFormSubmitButton button{border-radius:12px;border:1px solid #3a3a42;background:#1d1d23;color:#e9e6e8;font-weight:550;padding:.45rem 1rem;transition:transform .18s ease,border-color .18s ease,filter .18s ease}
div.stButton button:hover,div.stDownloadButton button:hover{transform:translateY(-1px);border-color:var(--orange);color:#fff}
div.stButton button[kind="primary"],div.stFormSubmitButton button[kind="primary"]{background:linear-gradient(95deg,#ff7a3d,#f0508a);border:0;color:#fff;box-shadow:0 10px 24px #ff5a2e33}
div.stButton button[kind="primary"]:hover,div.stFormSubmitButton button[kind="primary"]:hover{filter:brightness(1.08);transform:translateY(-1px)}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div,div[data-baseweb="textarea"]{border-radius:12px!important;background:#16161a!important;border-color:#33333b!important}
[data-testid="stSlider"] [role="slider"]{background:var(--orange)!important}

/* Segmented navigation (replaces in-page radio tabs) */
[data-testid="stSegmentedControl"]{width:fit-content;margin:0 0 14px}
[data-testid="stSegmentedControl"] button{background:#1b1b20;border:1px solid #33333b;border-radius:10px;color:#a9a6ac;font-weight:550;padding:6px 14px;transition:all .18s ease}
[data-testid="stSegmentedControl"] button:hover{color:#fff;border-color:#4a3428}
[data-testid="stSegmentedControl"] button[aria-checked="true"],[data-testid="stSegmentedControl"] button[aria-pressed="true"]{background:linear-gradient(90deg,#4a2c1f,#2c1f1f);color:#ffc7a8;border-color:#8a4c2f}

@media(max-width:900px){.block-container{padding:1rem .8rem 2rem}.pl-kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.pl-insights{grid-template-columns:1fr}.pl-topbar{flex-direction:column;align-items:flex-start}}
/* Readability, keyboard access, and touch targets */
:root{--muted:#b9b6be}
.pl-nav-label{color:#b9b6be}
.pl-page-sub,.pl-panel-sub,.pl-kpi-note,.pl-row-s,.pl-insight-s,.pl-brand-sub{color:#bab6be}
[data-testid="stWidgetLabel"] p,[data-testid="stCaptionContainer"] p{color:#bcb9c1}
.st-key-paylens_page [role="radiogroup"] label{min-height:44px}
.st-key-paylens_page [role="radiogroup"] label:focus-within,
div.stButton button:focus-visible,div.stDownloadButton button:focus-visible,
div.stFormSubmitButton button:focus-visible,
[data-testid="stExpander"] summary:focus-visible{outline:3px solid #ffc296;outline-offset:3px}
[data-testid="stSegmentedControl"] button{min-height:40px}
[data-testid="stSegmentedControl"] button:focus-visible{outline:3px solid #ffc296;outline-offset:2px}
a{color:#ffbc94}
a:focus-visible{outline:3px solid #ffc296;outline-offset:3px;border-radius:4px}
.pl-guide{display:flex;gap:12px;align-items:flex-start;background:linear-gradient(110deg,#221c1c,#1a1a20);border:1px solid #49352c;border-radius:16px;padding:16px 18px;margin:4px 0 20px}
.pl-guide-mark{font-size:1.1rem;line-height:1.3;color:#ffab76}
.pl-guide-title{font-weight:650;font-size:.93rem;color:#f7f4f4}
.pl-guide-copy{font-size:.85rem;line-height:1.6;color:#c6c1c6;margin-top:3px}
.pl-guide strong{color:#ffe0cd}
@media(max-width:680px){
 .block-container{padding:1rem .8rem 2rem}
 .pl-kpi-grid{grid-template-columns:1fr 1fr;gap:9px}
 .pl-kpi{padding:13px 12px;border-radius:13px}
 .pl-kpi-value{font-size:1.35rem;overflow-wrap:anywhere}
 .pl-guide{padding:13px;gap:8px}
 .pl-page-title{font-size:1.4rem}
 .st-key-paylens_page{scrollbar-width:thin}
}
@media(max-width:390px){.pl-kpi-grid{grid-template-columns:1fr}}

@media(prefers-reduced-motion:reduce){*,*:before,*:after{transition:none!important;animation:none!important}}
"""
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

PAGES = ["Overview", "Investigations", "Fraud Intelligence", "Merchant Risk", "Customer Insights", "Transactions"]
PAGE_ICONS = {
    "Overview": "▦",
    "Investigations": "⌕",
    "Fraud Intelligence": "⚑",
    "Merchant Risk": "◫",
    "Customer Insights": "◔",
    "Transactions": "≡",
}
PAGE_META = {
    "Overview": ("Overview", "Explore regional performance, payment trends and operational KPIs. All figures reflect your active filters."),
    "Investigations": ("Investigations", "Review suspicious payments, trace related activity, and document decisions. Changes remain in this browser session."),
    "Fraud Intelligence": ("Fraud Intelligence", "Random Forest on synthetic fraud labels. Test metrics use held-out transactions; transaction queue scores include in-sample predictions and are illustrative, not independently validated."),
    "Merchant Risk": ("Merchant intelligence", "Identify concentrated exposure, compare merchant performance and investigate individual merchant profiles."),
    "Customer Insights": ("Customer insights", "Understand where customer value concentrates and how often customers transact under your active filters."),
    "Transactions": ("Transaction explorer", "Search, filter and export individual payment records."),
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "payments.csv"
GENERATOR_FILE = PROJECT_ROOT / "src" / "generate_data.py"


# ---------- Shared UI components ----------

def kpi_row(items, compact=False):
    """Render a row of KPI cards. items: list of (label, value, note)."""
    card_class = "pl-kpi pl-kpi-compact" if compact else "pl-kpi"
    cards = []
    for label, value, note in items:
        note_html = f'<div class="pl-kpi-note">{escape(note)}</div>' if note else ""
        cards.append(
            f'<div class="{card_class}"><div class="pl-kpi-label">{escape(label)}</div>'
            f'<div class="pl-kpi-value">{escape(value)}</div>{note_html}</div>'
        )
    st.markdown(
        f'<div class="pl-kpi-grid" style="--cols:{len(items)}">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def panel_header(title, subtitle, icon):
    st.markdown(
        f'<div class="pl-panel-head"><div class="pl-chip">{escape(icon)}</div>'
        f'<div><div class="pl-panel-title">{escape(title)}</div>'
        f'<div class="pl-panel-sub">{escape(subtitle)}</div></div></div>',
        unsafe_allow_html=True,
    )


def segmented(label, options, key):
    """Pill-style segmented navigation used inside pages."""
    if hasattr(st, "segmented_control"):
        choice = st.segmented_control(
            label, options, default=options[0], selection_mode="single",
            key=key, label_visibility="collapsed",
        )
        return choice or options[0]
    return st.radio(label, options, horizontal=True, key=key, label_visibility="collapsed")


def style_fig(fig, height=300, legend=False, axes=True):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#c9c6c8", "size": 12},
        margin={"l": 8, "r": 10, "t": 10, "b": 8},
        showlegend=legend,
        hoverlabel={"bgcolor": "#1c1c21", "bordercolor": "#3a3a41",
                    "font": {"color": "#f4f3f3", "family": "Inter, sans-serif"}},
    )
    if axes:
        fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#2c2c33", tickfont={"color": "#8d8a8f"})
        fig.update_yaxes(gridcolor="#26262c", zeroline=False, linecolor="#2c2c33", tickfont={"color": "#8d8a8f"})
    return fig


def risk_meter(value):
    pct = max(0.0, min(1.0, value)) * 100
    color = PINK if value >= 0.25 else ("#ff8a3d" if value >= 0.10 else "#7b7880")
    st.markdown(
        f'<div class="pl-meter"><div class="pl-meter-fill" style="width:{pct:.1f}%;background:{color}"></div></div>',
        unsafe_allow_html=True,
    )


def band_badge(band):
    return f'<span class="pl-badge pl-badge-{escape(band.lower())}">{escape(band)} risk</span>'


def insight_card(kicker, value, detail):
    return (
        f'<div class="pl-insight"><div class="pl-insight-k">{escape(kicker)}</div>'
        f'<div class="pl-insight-v">{escape(value)}</div>'
        f'<div class="pl-insight-s">{escape(detail)}</div></div>'
    )


# ---------- Data preparation ----------

REQUIRED_COLUMNS = {
    "transaction_id", "timestamp", "customer_id", "merchant_id",
    "category", "region", "amount", "success", "payment_method",
    "device_type", "transaction_hour", "is_weekend",
    "merchant_risk_score", "is_fraud", "dataset_version",
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

    # Chronological hold-out: train on earlier records, evaluate on later ones.
    # Transaction IDs, customer IDs and post-payment outcomes are excluded.
    ordered = data.sort_values(["timestamp", "transaction_id"]).index
    split_at = int(len(ordered) * 0.75)
    train_idx, test_idx = ordered[:split_at], ordered[split_at:]
    x_train, x_test = features.loc[train_idx], features.loc[test_idx]
    y_train, y_test = target.loc[train_idx], target.loc[test_idx]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Chronological training/test partitions must contain both classes.")
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        min_samples_leaf=3,
        class_weight={0: 1, 1: 5},
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    test_probability = model.predict_proba(x_test)[:, 1]
    test_prediction = (test_probability >= 0.15).astype(int)  # illustrative review threshold
    metrics = {
        "training_prevalence": float(y_train.mean()),        "accuracy": accuracy_score(y_test, test_prediction),
        "precision": precision_score(y_test, test_prediction, zero_division=0),
        "recall": recall_score(y_test, test_prediction, zero_division=0),
        "f1": f1_score(y_test, test_prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_test, test_probability),
        "average_precision": average_precision_score(y_test, test_probability),
        "baseline_precision": float(y_test.mean()),
        "brier_score": brier_score_loss(y_test, test_probability),
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "test_start": data.loc[test_idx, "timestamp"].min(),
        "train_end": data.loc[train_idx, "timestamp"].max(),
        "test_labels": y_test.to_numpy(),
        "test_probabilities": test_probability,
    }

    all_probability = model.predict_proba(features)[:, 1]
    scored = data.copy()
    scored["predicted_fraud_risk"] = all_probability
    scored["score_provenance"] = "Training-data score (in-sample)"
    scored.loc[test_idx, "score_provenance"] = "Held-out score (out-of-sample)"
    scored["risk_band"] = pd.cut(
        scored["predicted_fraud_risk"],
        bins=[-0.001, 0.10, 0.25, 1.0],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    importance = (
        pd.DataFrame({"feature": features.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(12)
    )
    return scored, metrics, importance, model, features.columns.tolist()


file_modified = DATA_FILE.stat().st_mtime
df = load_data(str(DATA_FILE), file_modified)
scored_df, model_metrics, feature_importance, fraud_model, model_feature_columns = train_fraud_model(df)

@st.cache_resource(show_spinner=False)
def fraud_shap_explainer(model):
    """Explain raw model class outputs; no live data leaves the app."""
    return shap.TreeExplainer(model)


def explain_transaction(row: pd.Series, model, model_columns: list[str]):
    """Return base prediction and class-1 SHAP contributions for one transaction."""
    numeric = ["amount", "transaction_hour", "is_weekend", "merchant_risk_score"]
    categorical = ["category", "region", "payment_method", "device_type"]
    one = pd.DataFrame([row[numeric + categorical].to_dict()])
    encoded = pd.get_dummies(one, columns=categorical, dtype=int)
    encoded = encoded.reindex(columns=model_columns, fill_value=0)
    explainer = fraud_shap_explainer(model)
    shap_result = explainer.shap_values(encoded)
    if isinstance(shap_result, list):
        contributions = np.asarray(shap_result[1])[0]
    else:
        array = np.asarray(shap_result)
        contributions = array[0, :, 1] if array.ndim == 3 else array[0]
    base_array = np.asarray(explainer.expected_value)
    base = float(base_array[1] if base_array.ndim else base_array)
    predicted = float(model.predict_proba(encoded)[0, 1])
    return base, predicted, pd.DataFrame({
        "feature": model_columns,
        "contribution": contributions.astype(float),
    })


def navigate_to(destination: str) -> None:
    """Switch workspace from a button callback before Streamlit redraws widgets."""
    st.session_state["paylens_page"] = destination


# ---------- Sidebar navigation and filters ----------

with st.sidebar:
    st.markdown(
        '<div class="pl-brand"><div class="pl-logo">P</div>'
        '<div><div class="pl-brand-name">PayLens</div>'
        '<div class="pl-brand-sub">Payment intelligence</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="pl-nav-label">Filters</div>', unsafe_allow_html=True)
    with st.expander("Refine your view", expanded=False):
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
    st.markdown('<div class="pl-nav-label">About this demo</div>', unsafe_allow_html=True)
    with st.expander("Quick guide and methodology", expanded=False):
        st.markdown(
            "1. **Overview:** Review KPIs, payment trends and risk distribution.\n"
            "2. **Investigations:** Open a payment, inspect signals and record a demo decision.\n"
            "3. **Fraud Intelligence:** Explore test-set precision and recall by threshold.\n"
            "4. **Merchant Risk:** Compare merchants and drill into transaction history."
        )
        st.caption("Synthetic labels and model scores are educational, not banking decisions.")
        st.link_button("Read source code", "https://github.com/Mitra-2004/paylens-payment-analytics", use_container_width=True)
    st.caption("Synthetic data · portfolio demonstration")

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

# ---------- Horizontal workspace navigation ----------

page = st.radio(
    "Workspace",
    PAGES,
    format_func=lambda name: f"{PAGE_ICONS[name]}  {name}",
    horizontal=True,
    label_visibility="collapsed",
    key="paylens_page",
)

# ---------- Page header ----------

page_title, page_subtitle = PAGE_META[page]
subtitle_html = f'<div class="pl-page-sub">{escape(page_subtitle)}</div>' if page_subtitle else ""
st.markdown(
    f'<div class="pl-topbar"><div><div class="pl-page-title">{escape(page_title)}</div>{subtitle_html}</div>'
    f'<div class="pl-status">{len(filtered):,} matching transactions</div></div>',
    unsafe_allow_html=True,
)


# ---------- Overview ----------

if page == "Overview":
    st.markdown(
        '<div class="pl-guide"><div class="pl-guide-mark">✦</div><div>'
        '<div class="pl-guide-title">A payment network, at a glance</div>'
        '<div class="pl-guide-copy">Start with the KPIs and trend chart, then open '
        '<strong>Investigations</strong> to review a transaction or '
        '<strong>Fraud Intelligence</strong> to examine model trade-offs. '
        'All transactions are synthetic and all filters affect the figures below.</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    total_value = filtered["amount"].sum()
    success_rate = filtered["success"].mean() * 100
    fraud_count = int(filtered["is_fraud"].sum())
    fraud_rate = filtered["is_fraud"].mean() * 100
    kpi_row([
        ("Transactions", f"{len(filtered):,}", "In selected period"),
        ("Transaction value", f"₹{total_value:,.0f}", "Gross processed"),
        ("Payment success rate", f"{success_rate:.1f}%", "Successful attempts"),
        ("Simulated fraud rate", f"{fraud_rate:.2f}%", "Labelled fraudulent"),
    ])

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
            panel_header("Transaction trend", "Daily payment activity across your selected period", "↗")
            trend_metric = segmented("Metric", ["Transaction volume", "Transaction value"], key="overview_trend")
            trend_column = "transaction_count" if trend_metric == "Transaction volume" else "transaction_value"
            trend_chart = px.area(daily, x="timestamp", y=trend_column)
            trend_chart.update_traces(
                line={"width": 2.5, "color": ACCENT},
                fillcolor="rgba(255,122,61,.16)",
                hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra></extra>",
            )
            style_fig(trend_chart, height=290)
            trend_chart.update_layout(hovermode="x unified")
            trend_chart.update_xaxes(title_text=None)
            trend_chart.update_yaxes(title_text=None)
            st.plotly_chart(trend_chart, use_container_width=True, config=PLOT_CONFIG)
    with chart_right:
        with st.container(border=True):
            panel_header("Intelligence brief", "Observations from the filtered synthetic data", "✦")
            high_risk_count = int((filtered["risk_band"] == "High").sum())
            largest_region = region_value.iloc[0]
            highest_category = category_volume.iloc[0]
            st.markdown(
                '<div class="pl-row"><div><div class="pl-row-k">High-risk model flags</div>'
                '<div class="pl-row-s">Transactions with model scores above 25%</div></div>'
                f'<div class="pl-row-v">{high_risk_count:,}</div></div>'
                '<div class="pl-row"><div><div class="pl-row-k">Largest market</div>'
                f'<div class="pl-row-s">₹{largest_region["amount"]:,.0f} in transaction value</div></div>'
                f'<div class="pl-row-v">{escape(str(largest_region["region"]))}</div></div>'
                '<div class="pl-row"><div><div class="pl-row-k">Most active category</div>'
                f'<div class="pl-row-s">{int(highest_category["size"]):,} payments in the selected period</div></div>'
                f'<div class="pl-row-v">{escape(str(highest_category["category"]))}</div></div>',
                unsafe_allow_html=True,
            )
            st.caption("Descriptive signals, not real-time alerts or AI-generated causal findings.")

    bottom_left, bottom_right = st.columns([1, 1.55], gap="medium")
    with bottom_left:
        with st.container(border=True):
            panel_header("Risk distribution", "Share of transactions by model risk band", "◔")
            risk_counts = filtered["risk_band"].value_counts().reindex(
                ["Low", "Medium", "High"], fill_value=0
            )
            risk_pie = px.pie(
                names=risk_counts.index, values=risk_counts.values, hole=.68,
                color=risk_counts.index,
                color_discrete_map={"Low": "#57565d", "Medium": "#ff852f", "High": PINK},
            )
            risk_pie.update_traces(textinfo="none", sort=False)
            style_fig(risk_pie, height=270, legend=True, axes=False)
            risk_pie.update_layout(legend={"orientation": "h", "y": -0.05, "x": 0.5, "xanchor": "center"})
            high_share = risk_counts.get("High", 0) / max(risk_counts.sum(), 1) * 100
            risk_pie.add_annotation(
                text=f"<b>{high_share:.1f}%</b><br><span style='font-size:11px;color:#8e8b91'>High risk</span>",
                showarrow=False, font={"size": 20, "color": "#f5f4f4"},
            )
            st.plotly_chart(risk_pie, use_container_width=True, config=PLOT_CONFIG)
    with bottom_right:
        with st.container(border=True):
            panel_header("Recent high-risk transactions", "Investigate individual payments in the Investigations workspace", "⚑")
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
            footer_left, footer_right = st.columns([2, 1])
            with footer_left:
                st.caption("Scores are illustrative and include in-sample model predictions.")
            with footer_right:
                st.button(
                    "Open investigations", type="primary",
                    on_click=navigate_to, args=("Investigations",),
                    use_container_width=True,
                )

    with st.expander("Explore regional and category performance"):
        breakdown_left, breakdown_right = st.columns(2)
        with breakdown_left:
            st.markdown("**Value by region (₹)**")
            region_chart = px.bar(region_value, x="region", y="amount")
            region_chart.update_traces(marker_color=ACCENT)
            style_fig(region_chart, height=300)
            st.plotly_chart(region_chart, use_container_width=True, config=PLOT_CONFIG)
        with breakdown_right:
            st.markdown("**Transactions by category**")
            category_chart = px.bar(category_volume, x="category", y="size")
            category_chart.update_traces(marker_color=PINK)
            style_fig(category_chart, height=300)
            st.plotly_chart(category_chart, use_container_width=True, config=PLOT_CONFIG)

    top_region = region_value.iloc[0]
    top_category = category_volume.iloc[0]
    st.markdown(
        '<div class="pl-insights">'
        + insight_card("Top region", str(top_region["region"]), f"₹{top_region['amount']:,.0f} processed")
        + insight_card("Most active category", str(top_category["category"]), f"{int(top_category['size']):,} transactions")
        + insight_card("Simulated fraud alerts", f"{fraud_count:,}", "Transactions labelled fraudulent")
        + "</div>",
        unsafe_allow_html=True,
    )
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


# ---------- Investigations ----------

if page == "Investigations":
    queue = filtered.sort_values("predicted_fraud_risk", ascending=False)
    left_queue, right_queue = st.columns([1, 1.65], gap="medium")
    with left_queue:
        with st.container(border=True):
            panel_header("Review queue", "Sort, select and inspect a payment", "⌕")
            queue_mode = segmented("Queue", ["Highest risk", "Largest amount", "Recent"], key="investigation_queue")
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
    risk_value = float(selected["predicted_fraud_risk"])
    band = str(selected["risk_band"])

    with right_queue:
        with st.container(border=True):
            st.markdown(
                '<div class="pl-txn-head"><div>'
                f'<div class="pl-txn-id">TX-{int(selected_id):06d}</div>'
                f'<div class="pl-panel-sub">{selected["timestamp"]:%d %b %Y · %H:%M} · '
                f'{escape(str(selected["region"]))} · {escape(str(selected["category"]))}</div>'
                f'</div>{band_badge(band)}</div>',
                unsafe_allow_html=True,
            )
            kpi_row([
                ("Transaction amount", f"₹{selected['amount']:,.2f}", None),
                ("Model risk score", f"{risk_value:.1%}", None),
                ("Risk band", band, None),
            ], compact=True)
            st.caption(f"Risk model · {selected['score_provenance']}. Scores are not calibrated fraud probabilities.")
            risk_meter(risk_value)

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
            cues = []
            if selected["amount"] > amount_cutoff:
                cues.append("This amount is above the 94th percentile of the synthetic dataset.")
            if int(selected["transaction_hour"]) <= 4 or int(selected["transaction_hour"]) >= 23:
                cues.append("The recorded transaction hour is in the late-night window.")
            if selected["device_type"] == "Unknown":
                cues.append("The device type is unknown.")
            if float(selected["merchant_risk_score"]) > 0.5:
                cues.append("The merchant has an elevated simulated risk attribute.")
            if selected["payment_method"] in ["Credit Card", "Debit Card"]:
                cues.append("Card payment is one of the signals used in the synthetic label-generation process.")
            if cues:
                st.markdown("".join(f'<div class="pl-cue">{escape(cue)}</div>' for cue in cues), unsafe_allow_html=True)
            else:
                st.caption("No predefined synthetic risk indicators stand out for this transaction.")
            st.caption(
                "The cues above are simple rules, not explanations of the trained model."
            )
            with st.expander("Explain this model prediction · SHAP", expanded=False):
                st.caption(
                    "SHAP values attribute how each encoded feature shifts this Random Forest's "
                    "class-1 score relative to its baseline. They do not prove causality or "
                    "establish real-world fraud probability."
                )
                if st.button("Calculate SHAP explanation", key=f"shap_tx_{int(selected_id)}"):
                    with st.spinner("Calculating feature contributions..."):
                        base, predicted, contributions = explain_transaction(
                            selected, fraud_model, model_feature_columns
                        )
                    contribution_sum = base + contributions["contribution"].sum()
                    st.caption(
                        f"Baseline model score: {base:.1%} · "
                        f"Feature contributions: {contributions['contribution'].sum():+.1%} · "
                        f"Prediction: {predicted:.1%}"
                    )
                    if not np.isclose(contribution_sum, predicted, atol=1e-4):
                        st.warning("SHAP contributions did not fully reconstruct the prediction.")
                    top_effects = contributions.assign(
                        magnitude=lambda frame: frame["contribution"].abs(),
                        direction=lambda frame: np.where(
                            frame["contribution"] >= 0, "Raises score", "Lowers score"
                        ),
                    ).nlargest(10, "magnitude").sort_values("contribution")
                    effect_chart = px.bar(
                        top_effects, x="contribution", y="feature", orientation="h",
                        color="direction",
                        color_discrete_map={
                            "Raises score": PINK, "Lowers score": "#7db4eb"
                        },
                        labels={
                            "contribution": "Contribution to model score",
                            "feature": "Input feature",
                        },
                    )
                    effect_chart.update_traces(
                        hovertemplate="%{y}<br>Effect: %{x:+.4f}<extra></extra>"
                    )
                    style_fig(effect_chart, height=340, legend=True)
                    effect_chart.update_xaxes(tickformat="+.1%")
                    st.plotly_chart(effect_chart, use_container_width=True, config=PLOT_CONFIG)
                    st.caption(
                        "Positive bars raise the simulated model's risk score; "
                        "negative bars lower it. Feature contributions reflect "
                        "encoded columns and can be correlated."
                    )

    with st.container(border=True):
        panel_header("Connected activity", "Other payments linked to this merchant or customer", "◎")
        kpi_row([
            ("Merchant transactions", f"{len(merchant_history):,}", None),
            ("Merchant average model risk", f"{merchant_history['predicted_fraud_risk'].mean():.1%}", None),
            ("Customer transactions", f"{len(customer_history):,}", None),
        ], compact=True)
        history_view = segmented("Related activity", ["Merchant", "Customer"], key="investigation_history")
        related = merchant_history if history_view == "Merchant" else customer_history
        related = related.sort_values("timestamp", ascending=False)
        st.dataframe(
            related[["transaction_id", "timestamp", "amount", "region", "payment_method", "predicted_fraud_risk"]].head(30),
            hide_index=True, use_container_width=True,
        )

    with st.container(border=True):
        panel_header("Analyst decision", "Record a decision and notes for this transaction", "✎")
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


# ---------- Fraud Intelligence ----------

if page == "Fraud Intelligence":
    kpi_row([
        ("Accuracy", f"{model_metrics['accuracy'] * 100:.1f}%", "Held-out test set"),
        ("Precision", f"{model_metrics['precision'] * 100:.1f}%", "Held-out · 15% threshold"),
        ("Recall", f"{model_metrics['recall'] * 100:.1f}%", "Held-out · 15% threshold"),
        ("F1 score", f"{model_metrics['f1']:.3f}", "Held-out · 15% threshold"),
        ("ROC-AUC", f"{model_metrics['roc_auc']:.3f}", "Held-out test set"),
    ])

    with st.expander("Model validation and limitations", expanded=True):
        st.markdown(
            f"**Validation:** Chronological holdout — {model_metrics['train_size']:,} older transactions "
            f"for training and {model_metrics['test_size']:,} newer transactions for evaluation. "
            f"Training data ends **{model_metrics['train_end']:%d %b %Y}**; "
            f"the test period starts **{model_metrics['test_start']:%d %b %Y}**."
        )
        st.markdown(
            f"**Fraud prevalence / random-ranking average-precision baseline:** "
            f"{model_metrics['baseline_precision']:.2%}  ·  "
            f"**Model average precision:** {model_metrics['average_precision']:.3f}  ·  "
            f"**Brier score (lower is better):** {model_metrics['brier_score']:.4f}"
        )
        st.caption(
            "The synthetic fraud labels are created from engineered signals, so strong results "
            "would demonstrate pattern recovery—not real-world fraud detection. "
            "Features exclude transaction IDs, customer IDs, fraud labels and payment success. "
            "Scores on older training records are in-sample; newer test scores are out-of-sample."
        )

    with st.container(border=True):
        panel_header("Threshold analysis", "Classification threshold on the held-out test set", "⚑")
        threshold = st.slider("Fraud alert threshold", min_value=0.05, max_value=0.95, value=0.15, step=0.05)
        test_labels = model_metrics["test_labels"]
        test_probabilities = model_metrics["test_probabilities"]
        threshold_predictions = (test_probabilities >= threshold).astype(int)
        cm = confusion_matrix(test_labels, threshold_predictions, labels=[0, 1])
        tp = int(cm[1, 1])
        fp = int(cm[0, 1])
        fn = int(cm[1, 0])
        tn = int(cm[0, 0])
        selected_precision = precision_score(test_labels, threshold_predictions, zero_division=0)
        selected_recall = recall_score(test_labels, threshold_predictions, zero_division=0)
        kpi_row([
            ("Test precision", f"{selected_precision:.1%}", None),
            ("Test recall", f"{selected_recall:.1%}", None),
            ("False positives", f"{fp:,}", None),
            ("Missed fraud cases", f"{fn:,}", None),
        ], compact=True)
        st.caption(
            f"Held-out test set: {len(test_labels):,} transactions; {tp:,} true positives and {tn:,} true negatives. "
            f"Average precision (PR-AUC summary): {model_metrics['average_precision']:.3f}."
        )
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
            st.markdown("**Precision–recall curve**")
            pr_precision, pr_recall, _ = precision_recall_curve(test_labels, test_probabilities)
            pr_frame = pd.DataFrame({"Recall": pr_recall, "Precision": pr_precision})
            pr_chart = px.line(pr_frame, x="Recall", y="Precision")
            pr_chart.update_traces(line={"color": ACCENT, "width": 2.5})
            pr_chart.add_scatter(
                x=[selected_recall],
                y=[selected_precision],
                mode="markers+text",
                marker={"size": 14, "color": PINK, "symbol": "circle", "line": {"width": 2, "color": "#f5f4f4"}},
                text=[f"Threshold {threshold:.2f}"],
                textposition="top center",
                name="Selected threshold",
                hovertemplate=(
                    f"Threshold: {threshold:.2f}<br>"
                    f"Precision: {selected_precision:.3f}<br>"
                    f"Recall: {selected_recall:.3f}<extra></extra>"
                ),
            )
            style_fig(pr_chart, height=300)
            pr_chart.update_layout(xaxis_range=[0, 1.02], yaxis_range=[0, 1.05])
            st.plotly_chart(pr_chart, use_container_width=True, config=PLOT_CONFIG)
        st.caption(
            f"Selected threshold {threshold:.2f}: precision {selected_precision:.1%}, recall {selected_recall:.1%}. "
            "The full curve stays fixed because it represents all possible thresholds."
        )

    high_risk = filtered[filtered["risk_band"] == "High"].sort_values("predicted_fraud_risk", ascending=False)
    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            panel_header("Transactions by risk band", "Model risk classification across the filtered set", "◔")
            risk_counts = filtered.groupby("risk_band", observed=False).size().reindex(["Low", "Medium", "High"], fill_value=0).rename("transactions").reset_index()
            band_chart = px.bar(risk_counts, x="risk_band", y="transactions", color="risk_band",
                                color_discrete_map={"Low": "#57565d", "Medium": "#ff852f", "High": PINK})
            style_fig(band_chart, height=280)
            st.plotly_chart(band_chart, use_container_width=True, config=PLOT_CONFIG)
    with right:
        with st.container(border=True):
            panel_header("Most influential features", "Random Forest feature importance", "◫")
            importance_chart = px.bar(feature_importance.sort_values("importance"), x="importance", y="feature", orientation="h")
            importance_chart.update_traces(marker_color=ACCENT)
            style_fig(importance_chart, height=280)
            st.plotly_chart(importance_chart, use_container_width=True, config=PLOT_CONFIG)

    with st.expander("How to interpret these results"):
        st.write("Lower thresholds catch more potentially fraudulent transactions but may create more false alarms. Higher thresholds reduce reviews but can miss fraud cases. Compare held-out test metrics before deciding.")

    with st.container(border=True):
        panel_header("High-risk transaction queue", "Top transactions in the High risk band", "⚑")
        display_cols = ["transaction_id", "timestamp", "customer_id", "merchant_id", "amount", "region", "category", "predicted_fraud_risk", "risk_band", "score_provenance", "is_fraud"]
        st.dataframe(high_risk[display_cols].head(100), use_container_width=True, hide_index=True)
        st.caption("Ground-truth labels are synthetic. Use score_provenance to distinguish in-sample from genuinely held-out predictions.")
        st.caption("Low: <10%, Medium: 10–25%, High: >25%. These are illustrative prioritization bands, not validated operational thresholds.")


# ---------- Merchant Risk ----------

if page == "Merchant Risk":
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

    kpi_row([
        ("Merchants in view", f"{len(merchant_summary):,}", "Under active filters"),
        ("Total processed", f"₹{merchant_summary['transaction_value'].sum():,.0f}", "Gross value"),
        ("Average success rate", f"{filtered['success'].mean():.1%}", "Successful attempts"),
        ("Portfolio model risk", f"{filtered['predicted_fraud_risk'].mean():.1%}", "Mean predicted score"),
    ])

    with st.container(border=True):
        panel_header("Merchant comparison", "Top twelve merchants by the selected measure", "◫")
        chart_mode = segmented("Compare merchants by", ["Model risk", "Transaction value", "Payment success"], key="merchant_chart_mode")
        chart_columns = {
            "Model risk": ("predicted_risk", "Average model risk (%)"),
            "Transaction value": ("transaction_value", "Processed value (₹)"),
            "Payment success": ("payment_success_rate", "Success rate (%)"),
        }
        chart_column, chart_label = chart_columns[chart_mode]
        chart_top = merchant_summary.nlargest(12, chart_column).sort_values(chart_column)
        chart = px.bar(
            chart_top, x=chart_column, y="merchant_id", orientation="h",
            labels={chart_column: chart_label, "merchant_id": "Merchant"},
        )
        chart.update_traces(marker_color=ACCENT)
        style_fig(chart, height=340)
        chart.update_yaxes(type="category")
        st.plotly_chart(chart, use_container_width=True, config=PLOT_CONFIG)

    with st.container(border=True):
        panel_header("Merchant deep dive", "Profile, value and transaction history for one merchant", "◎")
        merchant_ids = merchant_summary.sort_values("predicted_risk", ascending=False)["merchant_id"].tolist()
        selected_merchant = st.selectbox(
            "Select merchant", merchant_ids,
            format_func=lambda merchant: f"Merchant M-{int(merchant):04d}",
            key="merchant_profile_id",
        )
        merchant_profile = merchant_summary.loc[
            merchant_summary["merchant_id"] == selected_merchant
        ].iloc[0]
        kpi_row([
            ("Transactions", f"{int(merchant_profile['transactions']):,}", None),
            ("Processed value", f"₹{merchant_profile['transaction_value']:,.0f}", None),
            ("Average model risk", f"{merchant_profile['predicted_risk']:.1f}%", None),
            ("Payment success", f"{merchant_profile['payment_success_rate']:.1f}%", None),
        ], compact=True)
        risk_meter(float(merchant_profile["predicted_risk"]) / 100)
        profile_payments = filtered.loc[filtered["merchant_id"] == selected_merchant].sort_values(
            "timestamp", ascending=False
        )
        st.markdown("**Merchant transaction history**")
        st.dataframe(
            profile_payments[[
                "transaction_id", "timestamp", "amount",
                "category", "payment_method", "predicted_fraud_risk",
            ]].head(50),
            hide_index=True, use_container_width=True,
            column_config={
                "predicted_fraud_risk": st.column_config.ProgressColumn(
                    "Risk estimate", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )
        st.caption("Model scores are generated from synthetic transactions and are not independently validated risk assessments.")

    with st.expander("Portfolio exploration charts", expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown("**Merchants with highest predicted risk**")
            top_risk = merchant_summary.nlargest(12, "predicted_risk")
            risk_chart = px.bar(top_risk.sort_values("predicted_risk"), x="predicted_risk",
                                y="merchant_id", orientation="h")
            risk_chart.update_traces(marker_color=PINK)
            style_fig(risk_chart, height=340)
            risk_chart.update_yaxes(type="category")
            st.plotly_chart(risk_chart, use_container_width=True, config=PLOT_CONFIG)
        with right:
            st.markdown("**Merchant value vs. predicted risk**")
            scatter = px.scatter(merchant_summary, x="transaction_value", y="predicted_risk",
                                 size="transactions", hover_name="merchant_id")
            scatter.update_traces(marker={"color": ACCENT, "opacity": 0.75, "line": {"width": 0}})
            style_fig(scatter, height=340)
            st.plotly_chart(scatter, use_container_width=True, config=PLOT_CONFIG)
        st.markdown("**Merchant scorecard**")
        st.dataframe(
            merchant_summary.sort_values("predicted_risk", ascending=False).head(100).round(2),
            use_container_width=True, hide_index=True,
        )


# ---------- Customer Insights ----------

if page == "Customer Insights":
    customer_summary = filtered.groupby("customer_id").agg(
        transactions=("transaction_id", "count"),
        lifetime_value=("amount", "sum"),
        average_ticket=("amount", "mean"),
        success_rate=("success", "mean"),
        predicted_risk=("predicted_fraud_risk", "mean"),
    ).reset_index()
    customer_summary["success_rate"] *= 100
    customer_summary["predicted_risk"] *= 100

    kpi_row([
        ("Customers in view", f"{len(customer_summary):,}", "Under active filters"),
        ("Average lifetime value", f"₹{customer_summary['lifetime_value'].mean():,.0f}", "Per customer"),
        ("Average ticket", f"₹{filtered['amount'].mean():,.0f}", "Per transaction"),
        ("Repeat customers", f"{(customer_summary['transactions'] > 1).mean():.1%}", "Two or more payments"),
    ])

    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            panel_header("Top customers", "Twelve highest customers by transaction value", "◔")
            top_customers = customer_summary.nlargest(12, "lifetime_value")
            customer_chart = px.bar(top_customers.sort_values("lifetime_value"), x="lifetime_value", y="customer_id", orientation="h")
            customer_chart.update_traces(marker_color=ACCENT)
            style_fig(customer_chart, height=360)
            customer_chart.update_yaxes(type="category")
            st.plotly_chart(customer_chart, use_container_width=True, config=PLOT_CONFIG)
    with right:
        with st.container(border=True):
            panel_header("Activity segments", "Customers grouped by number of payments", "◫")
            customer_summary["activity_segment"] = pd.cut(customer_summary["transactions"], bins=[0, 1, 3, 10000], labels=["One-time", "Occasional", "Frequent"])
            segments = customer_summary.groupby("activity_segment", observed=False).size().rename("customers").reset_index()
            segment_chart = px.pie(segments, names="activity_segment", values="customers", hole=.68,
                                   color="activity_segment",
                                   color_discrete_map={"One-time": "#57565d", "Occasional": "#ff852f", "Frequent": PINK})
            segment_chart.update_traces(textinfo="percent", textposition="outside")
            style_fig(segment_chart, height=360, legend=True, axes=False)
            st.plotly_chart(segment_chart, use_container_width=True, config=PLOT_CONFIG)

    with st.container(border=True):
        panel_header("Customer table", "Ranked by lifetime transaction value", "≡")
        st.dataframe(customer_summary.sort_values("lifetime_value", ascending=False).head(100).round(2), use_container_width=True, hide_index=True)


# ---------- Transactions ----------

if page == "Transactions":
    with st.container(border=True):
        panel_header("Find a transaction", "Search by numeric transaction ID within the active filters", "⌕")
        search_id = st.text_input("Transaction ID", placeholder="e.g. 1042", label_visibility="collapsed")
        if search_id.strip():
            if search_id.strip().isdigit():
                filtered = filtered.loc[filtered["transaction_id"] == int(search_id.strip())].copy()
            else:
                st.warning("Enter a numeric transaction ID.")
        st.caption(f"Showing {min(500, len(filtered)):,} of {len(filtered):,} filtered transactions.")
        st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered transactions (CSV)", data=csv, file_name="paylens_filtered_transactions.csv", mime="text/csv")

st.divider()
st.caption(
    "PayLens · Student portfolio demonstration · Synthetic data · No live banking connection. "
    "Fraud scores are illustrative; held-out model metrics do not establish real-world performance."
)