from pathlib import Path

import numpy as np
import pandas as pd

# Reproducible synthetic data for portfolio demonstrations.
rng = np.random.default_rng(42)
n = 25000

categories = np.array(["Retail", "Travel", "Dining", "Electronics", "Groceries", "Entertainment"])
regions = np.array(["Pune", "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai"])
payment_methods = np.array(["UPI", "Credit Card", "Debit Card", "Net Banking", "Wallet"])
devices = np.array(["Android", "iOS", "Windows", "Mac", "Unknown"])

customer_ids = rng.integers(1, 5001, n)
merchant_ids = rng.integers(1, 1001, n)
amounts = np.round(rng.lognormal(mean=6.2, sigma=0.95, size=n), 2)
# Hours and weekends must correspond to each transaction timestamp.
start = np.datetime64("2026-08-01T00:00:00")
end = np.datetime64("2026-11-22T23:59:59")
seconds = int((end - start) / np.timedelta64(1, "s"))
timestamps = pd.to_datetime(start + rng.integers(0, seconds, n).astype("timedelta64[s]"))
hours = timestamps.hour.to_numpy()
weekends = (timestamps.dayofweek >= 5).astype(int)
merchant_risk_map = rng.beta(2.0, 5.0, 1000)
merchant_risk = merchant_risk_map[merchant_ids - 1]
payment_method = rng.choice(payment_methods, n, p=[0.48, 0.20, 0.16, 0.10, 0.06])
device_type = rng.choice(devices, n, p=[0.42, 0.25, 0.15, 0.13, 0.05])

# Fraud labels are simulated from risk signals; they are not real-world findings.
high_value_signal = (amounts > np.quantile(amounts, 0.94)).astype(float)
night_signal = ((hours <= 4) | (hours >= 23)).astype(float)
unknown_device_signal = (device_type == "Unknown").astype(float)
card_signal = np.isin(payment_method, ["Credit Card", "Debit Card"]).astype(float)

logit = (
    -5.0
    + 1.45 * high_value_signal
    + 1.65 * merchant_risk
    + 0.95 * night_signal
    + 0.85 * unknown_device_signal
    + 0.35 * card_signal
)
fraud_probability = 1 / (1 + np.exp(-logit))
is_fraud = rng.binomial(1, fraud_probability)

# Failed payments are slightly more common among simulated fraudulent attempts.
success_probability = np.clip(0.95 - 0.34 * is_fraud - 0.10 * merchant_risk, 0.45, 0.99)
success = rng.binomial(1, success_probability)

df = pd.DataFrame({
    "transaction_id": np.arange(1, n + 1),
    "dataset_version": np.full(n, 2),
    "timestamp": timestamps,
    "customer_id": customer_ids,
    "merchant_id": merchant_ids,
    "category": rng.choice(categories, n),
    "region": rng.choice(regions, n),
    "amount": amounts,
    "success": success,
    "payment_method": payment_method,
    "device_type": device_type,
    "transaction_hour": hours,
    "is_weekend": weekends,
    "merchant_risk_score": np.round(merchant_risk, 4),
    "is_fraud": is_fraud,
})

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)
output_file = data_dir / "payments.csv"
df.to_csv(output_file, index=False)
print(f"Created {len(df):,} synthetic transactions at {output_file}")
