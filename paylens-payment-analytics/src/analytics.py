from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "payments.csv"
if not DATA_FILE.exists():
    from runpy import run_path
    run_path(str(PROJECT_ROOT / "src" / "generate_data.py"))

df = pd.read_csv(DATA_FILE)
print("\nTop categories by transaction volume:")
print(df.groupby("category").size().sort_values(ascending=False))
print("\nRegional transaction value:")
print(df.groupby("region")["amount"].sum().sort_values(ascending=False))
print("\nPayment success rate:")
print(df.groupby("region")["success"].mean().sort_values())
