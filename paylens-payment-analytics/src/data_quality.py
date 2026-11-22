"""Dataset quality rules for the reproducible PayLens demo."""
import pandas as pd


REQUIRED_COLUMNS = {
    "transaction_id", "dataset_version", "timestamp", "customer_id", "merchant_id",
    "category", "region", "amount", "success", "payment_method",
    "device_type", "transaction_hour", "is_weekend",
    "merchant_risk_score", "is_fraud",
}


def validate_transactions(data: pd.DataFrame) -> list[str]:
    """Return explicit quality issues, without mutating the input dataframe."""
    issues: list[str] = []
    absent = REQUIRED_COLUMNS - set(data.columns)
    if absent:
        return [f"Missing required columns: {', '.join(sorted(absent))}"]
    if data.empty:
        return ["The transaction dataset is empty."]
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        issues.append("Required dataset fields contain missing values.")
    if data["transaction_id"].duplicated().any():
        issues.append("Transaction IDs must be unique.")
    if not data["amount"].gt(0).all():
        issues.append("Transaction amounts must be positive.")
    if not data["merchant_risk_score"].between(0, 1).all():
        issues.append("Merchant risk attributes must be between 0 and 1.")
    for field in ("success", "is_fraud", "is_weekend"):
        if not data[field].isin([0, 1]).all():
            issues.append(f"{field} must contain only 0 or 1.")
    if not data["dataset_version"].eq(2).all():
        issues.append("Unexpected dataset version (expected version 2).")
    timestamps = pd.to_datetime(data["timestamp"], errors="coerce")
    if timestamps.isna().any():
        issues.append("Invalid transaction timestamps.")
    else:
        if not timestamps.dt.hour.eq(data["transaction_hour"]).all():
            issues.append("Transaction hours do not match timestamps.")
        if not (timestamps.dt.dayofweek >= 5).astype(int).eq(data["is_weekend"]).all():
            issues.append("Weekend flags do not match timestamps.")
    return issues
