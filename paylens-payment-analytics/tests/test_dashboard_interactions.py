"""Streamlit widget-level smoke tests; complements live browser testing.

Run from the inner project folder:
python -m unittest tests.test_dashboard_interactions -v
"""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


class DashboardWorkflowTests(unittest.TestCase):
    def test_recruiter_workflow(self):
        app = AppTest.from_file(str(APP), default_timeout=180).run()
        self.assertFalse(app.exception, f"Overview errors: {app.exception}")
        self.assertTrue(any("Overview" in str(r.value) for r in app.radio))
        self.assertTrue(any("Investigate top-risk payment" in button.label for button in app.button))

        overview_drilldown = next(
            button for button in app.button if button.label == "Investigate top-risk payment"
        )
        app = overview_drilldown.click().run()
        self.assertFalse(app.exception, f"Investigations errors: {app.exception}")
        self.assertEqual(app.session_state["paylens_page"], "Investigations")
        self.assertTrue(any("Select transaction" == select.label for select in app.selectbox))

        status_control = next(select for select in app.selectbox if select.label == "Decision")
        app = status_control.set_value("Monitor").run()
        note = next(widget for widget in app.text_area if widget.label == "Investigation notes")
        app = note.set_value("Reviewed payment context in automated smoke test").run()
        save = next(button for button in app.button if button.label == "Save case note")
        app = save.click().run()
        self.assertFalse(app.exception, f"Save case errors: {app.exception}")
        self.assertTrue(app.session_state["case_records"])

        nav = next(radio for radio in app.radio if radio.key == "paylens_page")
        app = nav.set_value("Fraud Intelligence").run()
        self.assertFalse(app.exception, f"Fraud Intelligence errors: {app.exception}")
        self.assertTrue(any(slider.label == "Fraud alert threshold" for slider in app.slider))
        threshold = next(slider for slider in app.slider if slider.label == "Fraud alert threshold")
        app = threshold.set_value(0.25).run()
        self.assertFalse(app.exception, f"Threshold errors: {app.exception}")

        nav = next(radio for radio in app.radio if radio.key == "paylens_page")
        app = nav.set_value("Merchant Risk").run()
        self.assertFalse(app.exception, f"Merchant Risk errors: {app.exception}")
        self.assertTrue(any(select.label == "Select merchant" for select in app.selectbox))


if __name__ == "__main__":
    unittest.main()
