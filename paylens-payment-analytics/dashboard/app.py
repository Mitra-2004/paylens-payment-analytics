from pathlib import Path
import runpy

import pandas as pd
import streamlit as st

st.set_page_config(page_title="PayLens", layout="wide")
st.title("PayLens — Payment Network Intelligence")

# Resolve paths from this file, not Streamlit's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "payments.csv"
GENERATOR_FILE = PROJECT_ROOT / "src" / "generate_data.py"

# The dataset is generated locally and is intentionally not required in Git.
# Create it automatically on first deployment if it is missing.
if not DATA_FILE.exists():
    if not GENERATOR_FILE.exists():
        st.error(f"Dataset and generator are both missing. Expected: {DATA_FILE}")
        st.stop()
    runpy.run_path(str(GENERATOR_FILE))

if not DATA_FILE.exists():
    st.error(f"Dataset generation did not create the expected file: {DATA_FILE}")
    st.stop()

df = pd.read_csv(DATA_FILE)

c1, c2, c3 = st.columns(3)
c1.metric("Transactions", f"{len(df):,}")
c2.metric("Transaction Value", f"₹{df['amount'].sum():,.0f}")
c3.metric("Success Rate", f"{df['success'].mean() * 100:.2f}%")

st.subheader("Transaction Value by Region")
st.bar_chart(df.groupby("region")["amount"].sum())

st.subheader("Transaction Volume by Category")
st.bar_chart(df.groupby("category").size())

st.subheader("Payment Success Rate")
st.bar_chart(df.groupby("category")["success"].mean())

st.subheader("Data Explorer")
st.dataframe(df.head(200), use_container_width=True)
