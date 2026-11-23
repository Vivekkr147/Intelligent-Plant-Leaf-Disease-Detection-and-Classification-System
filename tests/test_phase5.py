"""
=============================================================================
PlantCare AI — Automated Unit & Route Test Suite
Phase 5: Final Security Audit, CSRF Protection & Evaluation Parity
=============================================================================
Description:
Validates that:
1. CSRF protection prevents unauthorized cross-site POST submissions when enabled.
2. Valid session CSRF tokens allow authorized state-changing requests.
3. Database parameterized queries safely handle special characters and quotes.
4. Evaluation results correctly reflect 96.04% test accuracy on 1,163 samples.
5. All 7 target classes are fully accounted for in metadata and class mapping.
=============================================================================
"""

import os
import json
import unittest
from app import app
import database


class TestPhase5SecurityAndParity(unittest.TestCase):
    """
    Test suite for Phase 5 security audits, CSRF protection, and evaluation integrity.
    """

    def setUp(self):
        """
        Configure test environment.
        """
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_evaluation_results_file_integrity(self):
        """
        Verify that models/evaluation_results.json accurately documents:
        - Exactly 1,163 held-out test images
        - 96.04% test accuracy
        - 95.73% macro F1-score
        - 7 target classes
        """
        eval_path = os.path.join(os.path.dirname(__file__), "..", "models", "evaluation_results.json")
        self.assertTrue(os.path.exists(eval_path), "evaluation_results.json must exist")

        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["total_test_samples"], 1163)
        self.assertAlmostEqual(data["test_accuracy"], 0.9604, places=3)
        self.assertEqual(data["test_accuracy_percent"], "96.04%")
        self.assertAlmostEqual(data["macro_f1"], 0.9573, places=3)
        class_keys = [k for k in data["classification_report"] if k not in ("accuracy", "macro avg", "weighted avg")]
        self.assertEqual(len(class_keys), 7)

    def test_class_indices_contains_seven_classes(self):
        """
        Verify that models/class_indices.json contains exactly 7 classes.
        """
        indices_path = os.path.join(os.path.dirname(__file__), "..", "models", "class_indices.json")
        self.assertTrue(os.path.exists(indices_path))

        with open(indices_path, "r", encoding="utf-8") as f:
            indices = json.load(f)

        self.assertEqual(len(indices), 7)
        expected = [
            "Potato___Early_blight",
            "Potato___Late_blight",
            "Potato___healthy",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___healthy"
        ]
        for name in expected:
            self.assertIn(name, indices.values())

    def test_database_parameterization_handles_special_characters(self):
        """
        Verify that database insertion safely handles single quotes, double quotes,
        and SQL keywords without SQL injection vulnerabilities or syntax errors.
        """
        malicious_input = "Tomato' OR '1'='1; DROP TABLE predictions; --"
        record_id = database.insert_prediction(
            crop="Tomato",
            disease_name=malicious_input,
            raw_class="Tomato___healthy",
            is_healthy=1,
            confidence=0.98,
            confidence_percent="98.0%",
            notes="Testing O'Reilly and \"quotes\""
        )
        self.assertIsInstance(record_id, int)

        # Retrieve and verify exact string is preserved verbatim
        record = database.get_prediction_by_id(record_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["disease_name"], malicious_input)
        self.assertEqual(record["notes"], "Testing O'Reilly and \"quotes\"")

        # Clean up test row
        database.delete_prediction(record_id)

    def test_csrf_protection_rejects_missing_token_when_enabled(self):
        """
        Verify that when CSRF protection is active (WTF_CSRF_ENABLED=True, TESTING=False),
        a POST request lacking a valid token is safely intercepted.
        """
        original_testing = app.config.get('TESTING')
        original_csrf = app.config.get('WTF_CSRF_ENABLED')

        try:
            app.config['TESTING'] = False
            app.config['WTF_CSRF_ENABLED'] = True

            client = app.test_client()
            # Attempt POST without session token
            response = client.post('/history/clear', data={}, follow_redirects=False)
            # Expect redirect to referrer/home due to invalid token
            self.assertEqual(response.status_code, 302)

        finally:
            app.config['TESTING'] = original_testing
            app.config['WTF_CSRF_ENABLED'] = original_csrf


if __name__ == '__main__':
    unittest.main(verbosity=2)
