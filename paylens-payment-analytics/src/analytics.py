from pathlib import Path
import runpy

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "payments.csv"
GENERATOR_FILE = PROJECT_ROOT / "src" / "generate_data.py"

if not DATA_FILE.exists():
    runpy.run_path(str(GENERATOR_FILE))

df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])

print("\nPAYLENS — PAYMENT NETWORK SUMMARY")
print("=" * 42)
print(f"Transactions: {len(df):,}")
print(f"Total payment value: ₹{df['amount'].sum():,.2f}")
print(f"Payment success rate: {df['success'].mean() * 100:.2f}%")
print(f"Simulated fraud rate: {df['is_fraud'].mean() * 100:.2f}%")

print("\nTransaction volume by category:")
print(df.groupby("category").size().sort_values(ascending=False))

print("\nTransaction value by region:")
print(df.groupby("region")["amount"].sum().sort_values(ascending=False).round(2))

print("\nMerchant performance (top 10 by transaction value):")
merchant_summary = df.groupby("merchant_id").agg(
    transactions=("transaction_id", "count"),
    transaction_value=("amount", "sum"),
    success_rate=("success", "mean"),
    simulated_fraud_rate=("is_fraud", "mean"),
)
merchant_summary["success_rate"] *= 100
merchant_summary["simulated_fraud_rate"] *= 100
print(merchant_summary.sort_values("transaction_value", ascending=False).head(10).round(2))

print("\nCustomer value (top 10):")
customer_value = df.groupby("customer_id")["amount"].sum().sort_values(ascending=False)
print(customer_value.head(10).round(2))
