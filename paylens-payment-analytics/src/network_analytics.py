"""Pure, testable network summaries for synthetic customer–merchant activity."""
import pandas as pd


def merchant_connections(transactions: pd.DataFrame, min_transactions: int = 2) -> pd.DataFrame:
    """Customer/merchant pairs with repeat activity, based only on observed records."""
    required = {"customer_id", "merchant_id", "transaction_id", "amount", "predicted_fraud_risk"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing network columns: {sorted(missing)}")
    if transactions.empty:
        return pd.DataFrame(columns=[
            "customer_id", "merchant_id", "transactions", "total_value", "avg_model_score"
        ])
    summary = transactions.groupby(["customer_id", "merchant_id"], as_index=False).agg(
        transactions=("transaction_id", "count"),
        total_value=("amount", "sum"),
        avg_model_score=("predicted_fraud_risk", "mean"),
    )
    return summary.loc[summary["transactions"] >= min_transactions].sort_values(
        ["transactions", "total_value"], ascending=False
    ).reset_index(drop=True)


def network_summary(transactions: pd.DataFrame) -> dict:
    """Descriptive bipartite graph statistics; no inferred shared identity."""
    if transactions.empty:
        return {
            "customers": 0, "merchants": 0, "unique_links": 0,
            "repeat_links": 0, "repeat_transactions": 0,
        }
    link_counts = transactions.groupby(["customer_id", "merchant_id"]).size()
    repeats = link_counts[link_counts >= 2]
    return {
        "customers": int(transactions["customer_id"].nunique()),
        "merchants": int(transactions["merchant_id"].nunique()),
        "unique_links": int(len(link_counts)),
        "repeat_links": int(len(repeats)),
        "repeat_transactions": int(repeats.sum()),
    }


def merchant_customer_overlap(transactions: pd.DataFrame, merchant_id: int) -> pd.DataFrame:
    """Other merchants sharing distinct customers with one selected merchant."""
    pairs = transactions[["merchant_id", "customer_id"]].drop_duplicates()
    customers = pairs.loc[pairs["merchant_id"] == merchant_id, "customer_id"]
    shared = pairs[pairs["customer_id"].isin(customers)]
    shared = shared.loc[shared["merchant_id"] != merchant_id]
    if shared.empty:
        return pd.DataFrame(columns=["merchant_id", "shared_customers"])
    return (
        shared.groupby("merchant_id", as_index=False)["customer_id"]
        .nunique().rename(columns={"customer_id": "shared_customers"})
        .sort_values(["shared_customers", "merchant_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
