"""
=============================================================================
PlantCare AI — Automated Unit & Route Test Suite
Phase 1: Web Routes, Input Security, and Upload Validation
Author: vivek kumar (MCA Minor Project)
=============================================================================
Description:
Validates that:
1. All website navigation routes (Home, Predict, About, Contact) return HTTP 200.
2. File extension whitelisting behaves correctly.
3. In-memory image streams are verified for structural integrity using Pillow.
4. Non-image or corrupted files are strictly rejected with friendly warnings.
5. Valid uploads display the Phase 1 message:
   "AI model not trained yet. Prediction will be available in Phase 2."
6. NO fabricated or fake AI prediction classes/percentages exist in Phase 1.
=============================================================================
"""

import io
import unittest
from PIL import Image

# Import the core Flask application and validation helpers
from app import app, allowed_file, validate_image_stream


class PlantCareAITestCase(unittest.TestCase):
    """
    Test suite for Flask routes, security boundaries, and upload pipeline.
    """

    def setUp(self):
        """
        Configure the Flask test client and enable test mode.
        """
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    # -----------------------------------------------------------------------
    # Helper Generator Methods
    # -----------------------------------------------------------------------
    @staticmethod
    def create_in_memory_image(img_format='JPEG', size=(120, 120), color='green'):
        """
        Generate a genuine in-memory image byte stream using Pillow.
        """
        stream = io.BytesIO()
        mode = 'RGBA' if img_format == 'PNG' else 'RGB'
        img = Image.new(mode, size, color=color)
        img.save(stream, format=img_format)
        stream.seek(0)
        return stream

    # -----------------------------------------------------------------------
    # 1. Navigation Route Tests
    # -----------------------------------------------------------------------
    def test_homepage_route(self):
        """
        Test that GET '/' loads successfully with proper hero heading and CTA.
        """
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("Protect Your Plants with the Power of AI", html)
        self.assertIn("Detect Plant Disease", html)
        self.assertIn("PlantCare", html)

    def test_predict_page_get_route(self):
        """
        Test that GET '/predict' renders the upload area and drop zone.
        """
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("Plant Leaf Disease Detection", html)
        self.assertIn("Drag and drop your leaf photo here", html)
        self.assertIn("Browse Image", html)

    def test_about_page_route(self):
        """
        Test that GET '/about' explains CNN, MobileNetV2, and project roadmap.
        """
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("Convolutional Neural Networks", html)
        self.assertIn("MobileNetV2", html)
        self.assertIn("PlantVillage", html)
        self.assertIn("Phase 1", html)

    def test_contact_page_get_and_post_route(self):
        """
        Test that GET and POST '/contact' handle inquiries in Phase 1 preview mode.
        """
        get_res = self.client.get('/contact')
        self.assertEqual(get_res.status_code, 200)
        html = get_res.data.decode('utf-8')
        self.assertIn("vivek kumar", html)
        self.assertIn("MCA University Student", html)

        post_res = self.client.post('/contact', data={
            'name': 'Test Researcher',
            'email': 'researcher@example.org',
            'subject': 'academic_review',
            'message': 'Testing Phase 1 contact submission.'
        }, follow_redirects=True)
        self.assertEqual(post_res.status_code, 200)
        post_html = post_res.data.decode('utf-8')
        self.assertIn("Thank you, Test Researcher!", post_html)

    # -----------------------------------------------------------------------
    # 2. File Whitelist & Pillow Integrity Unit Tests
    # -----------------------------------------------------------------------
    def test_allowed_file_extensions(self):
        """
        Verify that allowed_file accepts only JPG, JPEG, and PNG.
        """
        self.assertTrue(allowed_file('leaf.jpg'))
        self.assertTrue(allowed_file('leaf.jpeg'))
        self.assertTrue(allowed_file('LEAF.PNG'))
        self.assertTrue(allowed_file('tomato_late_blight.JPG'))

        # Reject unsupported or dangerous extensions
        self.assertFalse(allowed_file('plant_disease.pdf'))
        self.assertFalse(allowed_file('malicious_script.py'))
        self.assertFalse(allowed_file('exploit.exe'))
        self.assertFalse(allowed_file('no_extension_file'))
        self.assertFalse(allowed_file(''))

    def test_validate_image_stream_with_valid_images(self):
        """
        Verify validate_image_stream passes authentic JPEG and PNG files.
        """
        jpeg_stream = self.create_in_memory_image('JPEG', (200, 150))
        is_valid, info, err = validate_image_stream(jpeg_stream)
        self.assertTrue(is_valid)
        self.assertEqual(info['format'], 'JPEG')
        self.assertEqual(info['dimensions'], '200 × 150 px')
        self.assertEqual(err, "")

        png_stream = self.create_in_memory_image('PNG', (100, 100))
        is_valid, info, err = validate_image_stream(png_stream)
        self.assertTrue(is_valid)
        self.assertEqual(info['format'], 'PNG')
        self.assertEqual(err, "")

    def test_validate_image_stream_with_corrupted_data(self):
        """
        Verify validate_image_stream safely rejects corrupted data.
        """
        fake_stream = io.BytesIO(b"NOT_A_REAL_IMAGE_DATA_BYTES_123456789")
        is_valid, info, err = validate_image_stream(fake_stream)
        self.assertFalse(is_valid)
        self.assertIn("Corrupted or unrecognized", err)

    # -----------------------------------------------------------------------
    # 3. Upload Security & Phase 1 Assertion Tests
    # -----------------------------------------------------------------------
    def test_predict_post_without_file(self):
        """
        POST to /predict without a file part triggers a friendly warning.
        """
        response = self.client.post('/predict', data={}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("No file part detected in the upload request", html)

    def test_predict_post_empty_filename(self):
        """
        POST to /predict with empty file upload triggers a prompt.
        """
        empty_data = {'file': (io.BytesIO(b''), '')}
        response = self.client.post('/predict', data=empty_data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("No file selected", html)

    def test_predict_post_unsupported_file_extension(self):
        """
        POST to /predict with .txt file is rejected with a clear message.
        """
        text_data = {'file': (io.BytesIO(b"plain text content"), 'notes.txt')}
        response = self.client.post('/predict', data=text_data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("Invalid file extension", html)

    def test_predict_post_corrupted_image_file(self):
        """
        POST to /predict with corrupted JPEG header is safely trapped.
        """
        corrupted_data = {'file': (io.BytesIO(b"garbage binary header 999"), 'fake_leaf.jpg')}
        response = self.client.post('/predict', data=corrupted_data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("Corrupted or unrecognized image file", html)

    def test_predict_post_valid_leaf_image_returns_live_prediction(self):
        """
        Verify that uploading a valid leaf image returns a live MobileNetV2 prediction:
        1. Must return HTTP 200.
        2. Must display diagnosis details (crop, disease name, display name).
        3. Must display real model confidence percentage.
        4. Must display validated image metadata (dimensions, format).
        5. Must not contain Phase 1 untrained notice when model is loaded.
        """
        valid_jpeg = self.create_in_memory_image('JPEG', (256, 256), color='forestgreen')
        upload_payload = {'file': (valid_jpeg, 'sample_tomato_leaf.jpg')}

        response = self.client.post(
            '/predict',
            data=upload_payload,
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')

        # Check for image metadata
        self.assertIn("sample_tomato_leaf.jpg", html)
        self.assertIn("256 × 256 px", html)

        # Check for live prediction content
        self.assertIn("Deep Learning Diagnosis", html)
        self.assertIn("Model Confidence Probability", html)
        self.assertIn("Softmax probability output", html)
        # Should not show the model-offline notice
        self.assertNotIn("AI model not trained yet. Prediction will be available in Phase 2.", html)

    def test_predict_post_valid_png_leaf_image(self):
        """
        Test that authentic PNG uploads pass through the validation pipeline
        and return live prediction results.
        """
        valid_png = self.create_in_memory_image('PNG', (180, 180), color='green')
        upload_payload = {'file': (valid_png, 'healthy_apple_leaf.png')}

        response = self.client.post(
            '/predict',
            data=upload_payload,
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("PNG", html)
        self.assertIn("Deep Learning Diagnosis", html)

    def test_predict_post_when_model_is_not_ready(self):
        """
        Verify that if the AI model is offline or uninitialized, the system
        safely falls back to displaying the offline notice without crashing.
        """
        from app import predictor
        original_state = predictor.is_ready
        try:
            predictor.is_ready = False
            valid_jpeg = self.create_in_memory_image('JPEG', (200, 200), color='forestgreen')
            upload_payload = {'file': (valid_jpeg, 'fallback_test_leaf.jpg')}

            response = self.client.post(
                '/predict',
                data=upload_payload,
                content_type='multipart/form-data'
            )
            self.assertEqual(response.status_code, 200)
            html = response.data.decode('utf-8')
            self.assertIn("AI model not trained yet. Prediction will be available in Phase 2.", html)
            self.assertIn("fallback_test_leaf.jpg", html)
        finally:
            predictor.is_ready = original_state

    def test_custom_404_error_page(self):
        """
        Test that accessing an undefined route returns HTTP 404 with friendly error UI.
        """
        response = self.client.get('/invalid-route-that-does-not-exist')
        self.assertEqual(response.status_code, 404)
        html = response.data.decode('utf-8')
        self.assertIn("404", html)
        self.assertIn("Page Not Found", html)

    def test_predict_post_exceeding_file_size_limit_returns_413(self):
        """
        Test that uploading a file exceeding the 5 MB MAX_CONTENT_LENGTH limit
        triggers the custom HTTP 413 error handler and returns a friendly warning.
        """
        # Create a payload of 6 MB (exceeds 5 MB limit)
        six_mb_data = b'0' * (6 * 1024 * 1024)
        payload = {'file': (io.BytesIO(six_mb_data), 'oversized_leaf.jpg')}

        response = self.client.post(
            '/predict',
            data=payload,
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 413)
        html = response.data.decode('utf-8')
        self.assertIn("Uploaded file is too large", html)
        self.assertIn("5 MB", html)

    def test_predict_post_actual_plantvillage_test_image(self):
        """
        Test end-to-end inference on a genuine PlantVillage test image.
        """
        import os
        test_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'test', 'Tomato___healthy')
        alt_test_dir = r"C:\Users\Abhishek Kumar\PlantCareData\prepared\test\Tomato___healthy"
        target_file = None
        for d in [test_dir, alt_test_dir]:
            if os.path.exists(d):
                files = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    target_file = os.path.join(d, files[0])
                    break

        if not target_file:
            self.skipTest("No PlantVillage test image found on disk")

        with open(target_file, 'rb') as f:
            img_bytes = f.read()

        payload = {'file': (io.BytesIO(img_bytes), 'plantvillage_test_leaf.jpg')}
        response = self.client.post('/predict', data=payload, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("Tomato", html)
        self.assertIn("Model Confidence Probability", html)

    def test_predict_post_low_confidence_displays_uncertainty_warning(self):
        """
        Test that predictions with confidence below 70% render the uncertainty alert.
        """
        from unittest.mock import patch
        mock_result = {
            'status': 'success',
            'raw_class': 'Tomato___Early_blight',
            'display_name': 'Tomato — Early Blight (Alternaria solani)',
            'crop': 'Tomato',
            'disease_name': 'Early Blight',
            'is_healthy': False,
            'confidence': 0.584,
            'confidence_percent': '58.4%',
            'is_uncertain': True,
            'uncertainty_warning': 'The model confidence is 58.4%, which is below the 70.0% certainty threshold.',
            'advisory_notice': 'General agricultural advisory.',
            'pathogen': 'Fungus',
            'symptoms': 'Leaf lesions',
            'organic_remedies': ['Prune lower leaves'],
            'chemical_treatments': ['Copper fungicides'],
            'prevention': ['Drip irrigation'],
            'distribution': {'Tomato___Early_blight': 0.584, 'Tomato___healthy': 0.416}
        }
        with patch('app.predictor.predict', return_value=mock_result):
            valid_jpeg = self.create_in_memory_image('JPEG', (224, 224), color='green')
            payload = {'file': (valid_jpeg, 'uncertain_leaf.jpg')}
            response = self.client.post('/predict', data=payload, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 200)
            html = response.data.decode('utf-8')
            self.assertIn("Low Confidence Prediction", html)
            self.assertIn("below the 70.0% certainty threshold", html)
            self.assertIn("Confidence thresholds cannot reliably detect", html)


if __name__ == '__main__':
    # Run all automated tests with verbose output
    unittest.main(verbosity=2)
