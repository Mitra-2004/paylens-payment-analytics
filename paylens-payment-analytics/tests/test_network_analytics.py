"""Tests for network summaries using a tiny, inspectable example dataset."""
import unittest
import pandas as pd
from src.network_analytics import (
    merchant_connections, network_summary, merchant_customer_overlap,
)


class NetworkAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame([
            {"transaction_id": 1, "customer_id": 10, "merchant_id": 100, "amount": 50, "predicted_fraud_risk": .1},
            {"transaction_id": 2, "customer_id": 10, "merchant_id": 100, "amount": 80, "predicted_fraud_risk": .2},
            {"transaction_id": 3, "customer_id": 10, "merchant_id": 200, "amount": 20, "predicted_fraud_risk": .1},
            {"transaction_id": 4, "customer_id": 20, "merchant_id": 200, "amount": 30, "predicted_fraud_risk": .3},
            {"transaction_id": 5, "customer_id": 20, "merchant_id": 100, "amount": 40, "predicted_fraud_risk": .1},
        ])

    def test_summary(self):
        summary = network_summary(self.data)
        self.assertEqual(summary["customers"], 2)
        self.assertEqual(summary["merchants"], 2)
        self.assertEqual(summary["unique_links"], 4)
        self.assertEqual(summary["repeat_links"], 1)
        self.assertEqual(summary["repeat_transactions"], 2)

    def test_connections(self):
        links = merchant_connections(self.data, min_transactions=2)
        self.assertEqual(len(links), 1)
        self.assertEqual(int(links.iloc[0]["transactions"]), 2)
        self.assertAlmostEqual(float(links.iloc[0]["total_value"]), 130.0)

    def test_overlap(self):
        result = merchant_customer_overlap(self.data, 100)
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.iloc[0]["merchant_id"]), 200)
        self.assertEqual(int(result.iloc[0]["shared_customers"]), 2)

    def test_empty(self):
        empty = self.data.iloc[0:0]
        self.assertEqual(network_summary(empty)["unique_links"], 0)
        self.assertTrue(merchant_connections(empty).empty)
        self.assertTrue(merchant_customer_overlap(empty, 100).empty)

    def test_missing_column_rejected(self):
        with self.assertRaises(ValueError):
            merchant_connections(self.data.drop(columns=["amount"]))


if __name__ == "__main__":
    unittest.main()
