from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
n = 25000
categories = np.array(["Retail", "Travel", "Dining", "Electronics", "Groceries", "Entertainment"])
regions = np.array(["Pune", "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai"])

df = pd.DataFrame({
    "transaction_id": np.arange(1, n + 1),
    "customer_id": rng.integers(1, 5001, n),
    "merchant_id": rng.integers(1, 1001, n),
    "category": rng.choice(categories, n),
    "region": rng.choice(regions, n),
    "amount": np.round(rng.lognormal(6.5, 1, n), 2),
    "success": rng.choice([0, 1], n, p=[0.06, 0.94]),
})

# Always write the CSV beside the project, regardless of the process working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = DATA_DIR / "payments.csv"
df.to_csv(OUTPUT_FILE, index=False)
print(f"Created {OUTPUT_FILE}")
