"""
=============================================================================
PlantCare AI — Automated Unit Tests for Phase 2 ML Modules
Author: Abhishek Kumar (MCA Minor Project)
=============================================================================
Description:
Tests that:
1. 'ml/preprocessing.py' normalizes input images to (1, 224, 224, 3) and [-1, 1] range.
2. 'ml/predictor.py' initializes in a safe un-trained state when model binary is not present.
3. 'predict()' returns the truthful Phase 1/Phase 2 message:
   "AI model not trained yet. Prediction will be available in Phase 2."
4. NO fake predictions or random confidence scores are ever generated.
5. All 7 target classes have complete agronomic metadata (pathogen, organic cures, etc.).
=============================================================================
"""

import io
import unittest
from PIL import Image

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from ml.preprocessing import load_and_preprocess_image, inspect_image
from ml.predictor import PlantDiseasePredictor, TARGET_CLASSES_METADATA


class TestPlantCareML(unittest.TestCase):
    """
    Test suite for Phase 2 Machine Learning Preprocessing and Predictor modules.
    """

    def setUp(self):
        """
        Create in-memory test images for testing.
        """
        # 100x150 RGB image
        self.img_rgb = Image.new("RGB", (100, 150), color=(34, 139, 34))
        # 80x80 RGBA image (with alpha channel)
        self.img_rgba = Image.new("RGBA", (80, 80), color=(46, 139, 87, 200))

    @unittest.skipUnless(HAS_NUMPY, "Requires NumPy (Phase 2 ML environment)")
    def test_preprocessing_output_shape_and_range(self):
        """
        Verify that load_and_preprocess_image generates a (1, 224, 224, 3) tensor
        with values strictly bounded within [-1.0, 1.0].
        """
        tensor = load_and_preprocess_image(self.img_rgb, target_size=(224, 224))
        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        self.assertEqual(tensor.dtype, np.float32)

        # Value bounds test for MobileNetV2 scaling
        self.assertTrue(np.all(tensor >= -1.0))
        self.assertTrue(np.all(tensor <= 1.0))

    @unittest.skipUnless(HAS_NUMPY, "Requires NumPy (Phase 2 ML environment)")
    def test_preprocessing_handles_rgba_conversion(self):
        """
        Verify that RGBA images with transparency are cleanly converted to 3-channel RGB.
        """
        tensor = load_and_preprocess_image(self.img_rgba, target_size=(224, 224))
        self.assertEqual(tensor.shape, (1, 224, 224, 3))

    @unittest.skipUnless(HAS_NUMPY, "Requires NumPy (Phase 2 ML environment)")
    def test_preprocessing_handles_bytesio_stream(self):
        """
        Verify that in-memory BytesIO streams are processed seamlessly.
        """
        stream = io.BytesIO()
        self.img_rgb.save(stream, format="JPEG")
        stream.seek(0)

        tensor = load_and_preprocess_image(stream)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))

    def test_inspect_image_metadata(self):
        """
        Verify inspect_image extracts accurate dimension and mode metadata.
        """
        info = inspect_image(self.img_rgb)
        self.assertEqual(info["width"], 100)
        self.assertEqual(info["height"], 150)
        self.assertEqual(info["mode"], "RGB")

    def test_predictor_untrained_state_returns_phase1_notice(self):
        """
        CRITICAL TEST FOR ACADEMIC INTEGRITY:
        When models/plantcare.keras does not exist, the predictor MUST:
        1. Have is_ready == False
        2. Return the exact Phase 1/Phase 2 message:
           'AI model not trained yet. Prediction will be available in Phase 2.'
        3. Never generate fake class predictions or fake confidence numbers.
        """
        # Point to a non-existent model path to test un-trained state
        predictor = PlantDiseasePredictor(model_path="models/non_existent_model.keras")
        self.assertFalse(predictor.is_ready)

        result = predictor.predict(self.img_rgb)
        self.assertEqual(result["status"], "model_not_ready")
        self.assertFalse(result["is_trained"])
        self.assertIn("AI model not trained yet. Prediction will be available in Phase 2.", result["message"])

        # Strictly assert absence of fake predictions
        self.assertIsNone(result["prediction"])
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["crop"])

    def test_all_seven_target_classes_present_in_knowledgebase(self):
        """
        Verify that all 7 target classes are fully documented with agronomic remedies.
        """
        expected_classes = [
            "Potato___Early_blight",
            "Potato___Late_blight",
            "Potato___healthy",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___healthy"
        ]

        for cls_name in expected_classes:
            self.assertIn(cls_name, TARGET_CLASSES_METADATA)
            meta = TARGET_CLASSES_METADATA[cls_name]
            self.assertIn("crop", meta)
            self.assertIn("disease_name", meta)
            self.assertIn("pathogen", meta)
            self.assertIn("symptoms", meta)
            self.assertIn("organic_remedies", meta)
            self.assertIn("chemical_treatments", meta)
            self.assertIn("prevention", meta)
            self.assertGreater(len(meta["organic_remedies"]), 0)

    @unittest.skipUnless(HAS_NUMPY, "Requires NumPy (Phase 2/3 ML environment)")
    def test_predictor_with_trained_model_loads_successfully(self):
        """
        Verify that PlantDiseasePredictor loads the trained MobileNetV2 model
        and initializes class names accurately.
        """
        predictor = PlantDiseasePredictor()
        if not predictor.is_ready:
            self.skipTest("models/plantcare.keras not found on disk")

        self.assertTrue(predictor.is_ready)
        self.assertEqual(len(predictor.class_names), 7)
        self.assertIsNotNone(predictor.model)

    @unittest.skipUnless(HAS_NUMPY, "Requires NumPy (Phase 2/3 ML environment)")
    def test_predictor_live_inference_on_synthetic_image(self):
        """
        Verify live inference produces real, normalized probability distribution,
        correct status, and structured metadata.
        """
        predictor = PlantDiseasePredictor()
        if not predictor.is_ready:
            self.skipTest("models/plantcare.keras not found on disk")

        result = predictor.predict(self.img_rgb)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["is_trained"])
        self.assertIn(result["raw_class"], predictor.class_names)
        self.assertIn(result["crop"], ["Potato", "Tomato"])
        self.assertIsInstance(result["confidence"], float)
        self.assertGreater(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertTrue(result["confidence_percent"].endswith("%"))
        self.assertEqual(len(result["distribution"]), 7)
        self.assertAlmostEqual(sum(result["distribution"].values()), 100.0, places=1)

    @unittest.skipUnless(HAS_NUMPY, "Requires NumPy (Phase 2/3 ML environment)")
    def test_predictor_uncertainty_threshold_flag(self):
        """
        Verify that when confidence is below 70%, is_uncertain is True and
        uncertainty_warning is populated.
        """
        from unittest.mock import MagicMock
        predictor = PlantDiseasePredictor()
        if not predictor.is_ready:
            self.skipTest("models/plantcare.keras not found on disk")

        # Mock model predict to return uniform probabilities (1/7 = ~0.1429 each)
        mock_probs = np.full((1, 7), 1.0 / 7.0, dtype=np.float32)
        original_model = predictor.model
        try:
            predictor.model = MagicMock()
            predictor.model.predict.return_value = mock_probs

            result = predictor.predict(self.img_rgb)
            self.assertEqual(result["status"], "success")
            self.assertTrue(result["is_uncertain"])
            self.assertIn("Confidence thresholds cannot reliably", result["uncertainty_warning"])
        finally:
            predictor.model = original_model

    @unittest.skipUnless(HAS_NUMPY, "Requires NumPy (Phase 2/3 ML environment)")
    def test_predictor_live_inference_on_actual_test_image(self):
        """
        Verify inference on an authentic test image from the dataset.
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
            self.skipTest("No test image found on disk")

        predictor = PlantDiseasePredictor()
        if not predictor.is_ready:
            self.skipTest("models/plantcare.keras not found on disk")

        result = predictor.predict(target_file)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["crop"], "Tomato")
        self.assertTrue(result["is_healthy"])
        self.assertGreater(result["confidence"], 0.80)


if __name__ == "__main__":
    unittest.main(verbosity=2)
