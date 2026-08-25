from pathlib import Path
import numpy as np, pandas as pd

rng=np.random.default_rng(11)
n=25000
categories=np.array(["Retail","Travel","Dining","Electronics","Groceries","Entertainment"])
regions=np.array(["Pune","Mumbai","Delhi","Bengaluru","Hyderabad","Chennai"])
df=pd.DataFrame({
    "transaction_id":np.arange(1,n+1),
    "customer_id":rng.integers(1,5001,n),
    "merchant_id":rng.integers(1,1001,n),
    "category":rng.choice(categories,n),
    "region":rng.choice(regions,n),
    "amount":np.round(rng.lognormal(6.5,1,n),2),
    "success":rng.choice([0,1],n,p=[.06,.94])
})
Path("data").mkdir(exist_ok=True)
df.to_csv("data/payments.csv",index=False)
print("Created data/payments.csv")
