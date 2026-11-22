"""Quality guard tests: valid sample, damaged dates, duplicates and mismatched fields."""
import unittest
import pandas as pd

from src.data_quality import validate_transactions


class DataQualityTests(unittest.TestCase):
    def setUp(self):
        self.valid = pd.DataFrame([{
            "transaction_id": 1, "dataset_version": 2,
            "timestamp": pd.Timestamp("2026-11-21 23:10:00"),
            "customer_id": 10, "merchant_id": 20, "category": "Retail",
            "region": "Pune", "amount": 125.0, "success": 1,
            "payment_method": "UPI", "device_type": "Android",
            "transaction_hour": 23, "is_weekend": 1,
            "merchant_risk_score": 0.25, "is_fraud": 0,
        }])

    def test_valid_record(self):
        self.assertEqual(validate_transactions(self.valid), [])

    def test_duplicate_id(self):
        duplicate = pd.concat([self.valid, self.valid], ignore_index=True)
        self.assertTrue(any("unique" in msg for msg in validate_transactions(duplicate)))

    def test_incorrect_hour(self):
        invalid = self.valid.copy()
        invalid.loc[0, "transaction_hour"] = 12
        self.assertTrue(any("hours" in msg for msg in validate_transactions(invalid)))

    def test_unexpected_version(self):
        invalid = self.valid.copy()
        invalid.loc[0, "dataset_version"] = 1
        self.assertTrue(any("version" in msg for msg in validate_transactions(invalid)))

    def test_missing_column(self):
        invalid = self.valid.drop(columns=["is_fraud"])
        self.assertTrue(any("Missing" in msg for msg in validate_transactions(invalid)))


if __name__ == "__main__":
    unittest.main()
