"""
=============================================================================
PlantCare AI — Automated Unit & Route Test Suite
Phase 4: Prediction History and Analytics Dashboard
=============================================================================
Description:
Validates that:
1. SQLite database tables and indexes are created safely.
2. Prediction records are created ONLY upon successful model inference.
3. Privacy consent controls whether leaf images are stored permanently on disk.
4. History page displays real records, applies crop/status filters, and handles empty state.
5. Deletion routes are protected (POST only) and safely purge rows and disk files.
6. Dashboard calculates accurate real-time statistics (counts, percentages, averages).
7. Chart.js dashboard views and API endpoints return real database metrics.
=============================================================================
"""

import io
import os
import shutil
import tempfile
import unittest
from PIL import Image

from app import app
import database


class TestPhase4HistoryAndDashboard(unittest.TestCase):
    """
    Test suite for Phase 4 SQLite history storage, analytics dashboard,
    and privacy-preserving image storage.
    """

    def setUp(self):
        """
        Configure an isolated temporary SQLite database and test client for each test.
        """
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False

        # Create temporary database file for isolation
        self.test_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.test_dir, 'test_plantcare.db')
        app.config['DATABASE'] = self.test_db_path

        # Initialize the fresh test database
        database.init_db(self.test_db_path)

        self.client = app.test_client()

    def tearDown(self):
        """
        Clean up temporary test database and files.
        """
        # Ensure any open connection in flask.g is closed
        with app.app_context():
            database.close_db()

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    @staticmethod
    def create_in_memory_image(img_format='JPEG', size=(224, 224), color='forestgreen'):
        """
        Generate in-memory image byte stream using Pillow.
        """
        stream = io.BytesIO()
        img = Image.new('RGB', size, color=color)
        img.save(stream, format=img_format)
        stream.seek(0)
        return stream

    # -----------------------------------------------------------------------
    # 1. Database Operations & Schema Tests
    # -----------------------------------------------------------------------
    def test_database_initialization_and_schema(self):
        """
        Verify that database.init_db creates the predictions table with all required columns.
        """
        conn = database.get_db(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(predictions)")
        columns = {row["name"]: row["type"] for row in cursor.fetchall()}

        expected_columns = [
            "id", "timestamp", "crop", "disease_name", "raw_class",
            "is_healthy", "confidence", "confidence_percent", "image_path", "notes"
        ]
        for col in expected_columns:
            self.assertIn(col, columns, f"Missing required column: {col}")

    def test_insert_and_retrieve_prediction_record(self):
        """
        Verify parameterized insertion and retrieval of prediction records.
        """
        record_id = database.insert_prediction(
            crop="Tomato",
            disease_name="Early Blight",
            raw_class="Tomato___Early_blight",
            is_healthy=0,
            confidence=0.9654,
            confidence_percent="96.5%",
            image_path=None,
            db_path=self.test_db_path
        )
        self.assertIsInstance(record_id, int)
        self.assertGreater(record_id, 0)

        record = database.get_prediction_by_id(record_id, db_path=self.test_db_path)
        self.assertIsNotNone(record)
        self.assertEqual(record["crop"], "Tomato")
        self.assertEqual(record["disease_name"], "Early Blight")
        self.assertEqual(record["raw_class"], "Tomato___Early_blight")
        self.assertEqual(record["is_healthy"], 0)
        self.assertAlmostEqual(record["confidence"], 0.9654, places=4)
        self.assertEqual(record["confidence_percent"], "96.5%")
        self.assertIsNone(record["image_path"])

    def test_delete_prediction_record(self):
        """
        Verify that database.delete_prediction deletes the record and returns image_path.
        """
        record_id = database.insert_prediction(
            crop="Potato",
            disease_name="Healthy Foliage",
            raw_class="Potato___healthy",
            is_healthy=1,
            confidence=0.9912,
            confidence_percent="99.1%",
            image_path="uploads/test_sample.jpg",
            db_path=self.test_db_path
        )
        success, img_path = database.delete_prediction(record_id, db_path=self.test_db_path)
        self.assertTrue(success)
        self.assertEqual(img_path, "uploads/test_sample.jpg")

        # Verify it no longer exists
        record = database.get_prediction_by_id(record_id, db_path=self.test_db_path)
        self.assertIsNone(record)

    def test_clear_all_predictions(self):
        """
        Verify that database.clear_all_predictions empties the table.
        """
        database.insert_prediction(
            crop="Tomato", disease_name="Leaf Mold", raw_class="Tomato___Leaf_Mold",
            is_healthy=0, confidence=0.88, confidence_percent="88.0%", db_path=self.test_db_path
        )
        database.insert_prediction(
            crop="Potato", disease_name="Late Blight", raw_class="Potato___Late_blight",
            is_healthy=0, confidence=0.92, confidence_percent="92.0%", db_path=self.test_db_path
        )

        all_records = database.get_all_predictions(db_path=self.test_db_path)
        self.assertEqual(len(all_records), 2)

        database.clear_all_predictions(db_path=self.test_db_path)
        remaining = database.get_all_predictions(db_path=self.test_db_path)
        self.assertEqual(len(remaining), 0)

    # -----------------------------------------------------------------------
    # 2. History Filtering & Web Routes Tests
    # -----------------------------------------------------------------------
    def test_history_page_empty_state(self):
        """
        Verify that /history displays a friendly empty state when no scans exist.
        """
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("No Prediction Records Found", html)
        self.assertIn("Scan a Leaf Now", html)

    def test_history_page_with_records_and_filtering(self):
        """
        Verify that /history renders records and correctly filters by crop and status.
        """
        database.insert_prediction(
            crop="Tomato", disease_name="Healthy Foliage", raw_class="Tomato___healthy",
            is_healthy=1, confidence=0.98, confidence_percent="98.0%", db_path=self.test_db_path
        )
        database.insert_prediction(
            crop="Potato", disease_name="Early Blight", raw_class="Potato___Early_blight",
            is_healthy=0, confidence=0.91, confidence_percent="91.0%", db_path=self.test_db_path
        )

        # 1. Unfiltered GET
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("Tomato___healthy", html)
        self.assertIn("Potato___Early_blight", html)
        self.assertIn("Total Scans", html)

        # 2. Filter by crop = Tomato
        response_tomato = self.client.get('/history?crop=Tomato')
        html_tomato = response_tomato.data.decode('utf-8')
        self.assertIn("Tomato___healthy", html_tomato)
        self.assertNotIn("Potato___Early_blight", html_tomato)

        # 3. Filter by status = diseased
        response_diseased = self.client.get('/history?status=diseased')
        html_diseased = response_diseased.data.decode('utf-8')
        self.assertIn("Potato___Early_blight", html_diseased)
        self.assertNotIn("Tomato___healthy", html_diseased)

    def test_history_deletion_requires_post_method(self):
        """
        Verify security protection: GET request to state-changing deletion route
        returns HTTP 405 Method Not Allowed.
        """
        response = self.client.get('/history/delete/1')
        self.assertEqual(response.status_code, 405)

        response_clear = self.client.get('/history/clear')
        self.assertEqual(response_clear.status_code, 405)

    def test_history_delete_single_record_post(self):
        """
        Verify POST /history/delete/<id> deletes the record and redirects with flash message.
        """
        record_id = database.insert_prediction(
            crop="Tomato", disease_name="Early Blight", raw_class="Tomato___Early_blight",
            is_healthy=0, confidence=0.95, confidence_percent="95.0%", db_path=self.test_db_path
        )
        response = self.client.post(f'/history/delete/{record_id}', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn(f"Prediction record #{record_id} successfully deleted", html)

        # Verify record no longer exists in database
        self.assertIsNone(database.get_prediction_by_id(record_id, db_path=self.test_db_path))

    def test_history_clear_all_post(self):
        """
        Verify POST /history/clear deletes all records and redirects with flash message.
        """
        database.insert_prediction(
            crop="Tomato", disease_name="Early Blight", raw_class="Tomato___Early_blight",
            is_healthy=0, confidence=0.95, confidence_percent="95.0%", db_path=self.test_db_path
        )
        response = self.client.post('/history/clear', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("All prediction history and stored images have been cleared", html)
        self.assertEqual(len(database.get_all_predictions(db_path=self.test_db_path)), 0)

    # -----------------------------------------------------------------------
    # 3. Analytics Dashboard & Real-Time Stats Tests
    # -----------------------------------------------------------------------
    def test_dashboard_empty_state(self):
        """
        Verify that /dashboard renders gracefully with 0 records and no fake statistics.
        """
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("No Diagnostic Data Available Yet", html)

    def test_dashboard_accurate_statistics_calculation(self):
        """
        Verify that get_dashboard_analytics accurately calculates genuine metrics:
        total scans, healthy vs diseased counts and percentages, and average confidence.
        """
        # Insert 3 records: 2 diseased, 1 healthy
        database.insert_prediction(
            crop="Tomato", disease_name="Early Blight", raw_class="Tomato___Early_blight",
            is_healthy=0, confidence=0.90, confidence_percent="90.0%", db_path=self.test_db_path
        )
        database.insert_prediction(
            crop="Potato", disease_name="Late Blight", raw_class="Potato___Late_blight",
            is_healthy=0, confidence=0.80, confidence_percent="80.0%", db_path=self.test_db_path
        )
        database.insert_prediction(
            crop="Tomato", disease_name="Healthy Foliage", raw_class="Tomato___healthy",
            is_healthy=1, confidence=1.00, confidence_percent="100.0%", db_path=self.test_db_path
        )

        stats = database.get_dashboard_analytics(db_path=self.test_db_path)
        self.assertEqual(stats["total_scans"], 3)
        self.assertEqual(stats["healthy_count"], 1)
        self.assertEqual(stats["diseased_count"], 2)
        self.assertAlmostEqual(stats["healthy_percent"], 33.3, places=1)
        self.assertAlmostEqual(stats["diseased_percent"], 66.7, places=1)
        # Average confidence: (0.90 + 0.80 + 1.00) / 3 = 0.90 -> 90.0%
        self.assertAlmostEqual(stats["avg_confidence"], 90.0, places=1)
        self.assertEqual(stats["crop_distribution"]["Tomato"], 2)
        self.assertEqual(stats["crop_distribution"]["Potato"], 1)

    def test_dashboard_web_route_and_json_api(self):
        """
        Verify that /dashboard renders KPI cards and /api/dashboard-stats returns JSON data.
        """
        database.insert_prediction(
            crop="Tomato", disease_name="Healthy Foliage", raw_class="Tomato___healthy",
            is_healthy=1, confidence=0.97, confidence_percent="97.0%", db_path=self.test_db_path
        )
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("Foliage Health Ratio", html)
        self.assertIn("Total Scans", html)
        self.assertIn("chart.umd.min.js", html)

        # Test JSON API endpoint
        api_res = self.client.get('/api/dashboard-stats')
        self.assertEqual(api_res.status_code, 200)
        data = api_res.get_json()
        self.assertEqual(data["total_scans"], 1)
        self.assertEqual(data["healthy_count"], 1)

    # -----------------------------------------------------------------------
    # 4. Integration: /predict Logs Record & Privacy Consent Tests
    # -----------------------------------------------------------------------
    def test_predict_post_without_consent_does_not_save_image_file(self):
        """
        Verify that uploading an image without checking save_image logs the record in SQLite
        but leaves image_path as None (0 permanent image files saved).
        """
        jpeg_stream = self.create_in_memory_image('JPEG')
        payload = {'file': (jpeg_stream, 'privacy_test_leaf.jpg')}

        response = self.client.post('/predict', data=payload, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)

        # Query database to inspect logged record
        records = database.get_all_predictions(db_path=self.test_db_path)
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0]["image_path"])

    def test_predict_post_with_consent_saves_image_file_and_cleans_up_on_delete(self):
        """
        Verify that uploading with save_image='on' saves an image reference, and deleting
        the record purges the disk file.
        """
        jpeg_stream = self.create_in_memory_image('JPEG')
        payload = {
            'file': (jpeg_stream, 'consent_leaf.jpg'),
            'save_image': 'on'
        }

        response = self.client.post('/predict', data=payload, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)

        records = database.get_all_predictions(db_path=self.test_db_path)
        self.assertEqual(len(records), 1)
        saved_rel_path = records[0]["image_path"]
        self.assertIsNotNone(saved_rel_path)

        # Check file exists on disk
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_disk_path = os.path.join(base_dir, 'static', saved_rel_path.replace('/', os.sep))
        self.assertTrue(os.path.exists(full_disk_path))

        # Delete the record via POST
        del_res = self.client.post(f'/history/delete/{records[0]["id"]}')
        self.assertEqual(del_res.status_code, 302)

        # Verify disk file was deleted
        self.assertFalse(os.path.exists(full_disk_path))

    def test_predict_post_invalid_file_does_not_create_database_record(self):
        """
        Verify that invalid or corrupted uploads rejected by security validation
        DO NOT create prediction records in SQLite.
        """
        corrupted = {'file': (io.BytesIO(b"not a valid image"), 'fake.jpg')}
        response = self.client.post('/predict', data=corrupted, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 302)

        records = database.get_all_predictions(db_path=self.test_db_path)
        self.assertEqual(len(records), 0)

    # -----------------------------------------------------------------------
    # 5. Navigation Links Verification
    # -----------------------------------------------------------------------
    def test_navigation_bar_contains_phase4_links(self):
        """
        Verify that base.html includes navigation links to History and Dashboard.
        """
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("/history", html)
        self.assertIn("/dashboard", html)
        self.assertIn("Phase 4", html)


if __name__ == '__main__':
    unittest.main(verbosity=2)
