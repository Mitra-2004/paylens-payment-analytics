import streamlit as st
import pandas as pd

st.set_page_config(page_title="PayLens", layout="wide")
st.title("PayLens — Payment Network Intelligence")

df=pd.read_csv("data/payments.csv")
c1,c2,c3=st.columns(3)
c1.metric("Transactions",f"{len(df):,}")
c2.metric("Transaction Value",f"₹{df.amount.sum():,.0f}")
c3.metric("Success Rate",f"{df.success.mean()*100:.2f}%")

st.subheader("Transaction Value by Region")
st.bar_chart(df.groupby("region")["amount"].sum())

st.subheader("Transaction Volume by Category")
st.bar_chart(df.groupby("category").size())

st.subheader("Payment Success Rate")
st.bar_chart(df.groupby("category")["success"].mean())

st.subheader("Data Explorer")
st.dataframe(df.head(200),use_container_width=True)
