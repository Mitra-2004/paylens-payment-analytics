"""Check SHAP's binary RandomForest output and prediction additivity."""
import unittest

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier


class ExplainabilityTests(unittest.TestCase):
    def test_shap_reconstructs_positive_class_score(self):
        rng = np.random.default_rng(42)
        features = pd.DataFrame({
            "amount": rng.lognormal(size=120),
            "late_night": rng.integers(0, 2, size=120),
            "merchant_risk": rng.random(size=120),
        })
        labels = (
            (features["amount"] > 1.2) &
            (features["merchant_risk"] > 0.35)
        ).astype(int)
        model = RandomForestClassifier(n_estimators=8, max_depth=4, random_state=42)
        model.fit(features, labels)
        row = features.iloc[[0]]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(row)
        if isinstance(shap_values, list):
            positive_contributions = np.asarray(shap_values[1])[0]
        else:
            arr = np.asarray(shap_values)
            positive_contributions = arr[0, :, 1] if arr.ndim == 3 else arr[0]
        expected = np.asarray(explainer.expected_value)
        baseline = float(expected[1] if expected.ndim else expected)
        prediction = model.predict_proba(row)[0, 1]
        self.assertAlmostEqual(
            baseline + float(positive_contributions.sum()),
            float(prediction),
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
