import pandas as pd

df=pd.read_csv("data/payments.csv")
print("\nTop categories by transaction volume:")
print(df.groupby("category").size().sort_values(ascending=False))
print("\nRegional transaction value:")
print(df.groupby("region")["amount"].sum().sort_values(ascending=False))
print("\nPayment success rate:")
print(df.groupby("region")["success"].mean().sort_values())
