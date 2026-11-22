"""Basic synthetic data smoke tests. Run: python -m unittest discover -s tests -v"""
from pathlib import Path
import runpy
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "src" / "generate_data.py"
DATA_FILE = ROOT / "data" / "payments.csv"

REQUIRED_COLUMNS = {
    "transaction_id", "timestamp", "customer_id", "merchant_id",
    "category", "region", "amount", "success", "payment_method",
    "device_type", "transaction_hour", "is_weekend",
    "merchant_risk_score", "is_fraud", "dataset_version",
}


class PaymentDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate a fresh, deterministic dataset to avoid testing stale CSVs.
        runpy.run_path(str(GENERATOR))
        cls.df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])

    def test_dataset_shape(self):
        self.assertEqual(len(self.df), 25000)
        self.assertTrue(REQUIRED_COLUMNS.issubset(self.df.columns))

    def test_transaction_ids_unique(self):
        self.assertTrue(self.df["transaction_id"].is_unique)

    def test_amounts_positive(self):
        self.assertTrue((self.df["amount"] > 0).all())

    def test_binary_outcomes(self):
        for column in ("success", "is_fraud", "is_weekend"):
            self.assertTrue(self.df[column].isin([0, 1]).all())

    def test_risk_scores_bounded(self):
        self.assertTrue(self.df["merchant_risk_score"].between(0, 1).all())

    def test_timestamps_present(self):
        self.assertFalse(self.df["timestamp"].isna().any())

    def test_timestamps_match_engineered_features(self):
        self.assertTrue(
            (self.df["timestamp"].dt.hour == self.df["transaction_hour"]).all()
        )
        self.assertTrue(
            ((self.df["timestamp"].dt.dayofweek >= 5).astype(int) == self.df["is_weekend"]).all()
        )

    def test_dataset_version(self):
        self.assertTrue((self.df["dataset_version"] == 2).all())

    def test_chronological_holdout(self):
        ordered = self.df.sort_values(["timestamp", "transaction_id"])
        split_at = int(len(ordered) * 0.75)
        train, test = ordered.iloc[:split_at], ordered.iloc[split_at:]
        self.assertLessEqual(train["timestamp"].max(), test["timestamp"].min())
        self.assertEqual(len(train), 18750)
        self.assertEqual(len(test), 6250)
        self.assertEqual(train["is_fraud"].nunique(), 2)
        self.assertEqual(test["is_fraud"].nunique(), 2)



if __name__ == "__main__":
    unittest.main()
