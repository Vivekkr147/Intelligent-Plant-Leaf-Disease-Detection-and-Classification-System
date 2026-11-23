"""
=============================================================================
PlantCare AI — Plant Disease Inference & Agronomic Remedy Engine
Phase 2 Machine Learning Architecture
Author: Abhishek Kumar (MCA Minor Project)
=============================================================================
Description:
Provides the prediction interface connecting incoming plant leaf images
to the MobileNetV2 Convolutional Neural Network.
Manages:
1. Seven target disease classes across Tomato and Potato crops from the
   official PlantVillage dataset.
2. Scientific pathogen taxonomy, visible symptom profiles, certified organic
   cures, chemical fungicide advice, and agronomic prevention practices.
3. Strict adherence to academic integrity:
   If the real model binary ('models/plantcare.keras') has not been trained yet,
   it explicitly returns:
   "AI model not trained yet. Prediction will be available in Phase 2."
   and NEVER fabricates random or simulated prediction percentages.
=============================================================================
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from ml.preprocessing import load_and_preprocess_image

logger = logging.getLogger(__name__)

# =============================================================================
# Agronomic Knowledgebase for the 7 Target Classes
# =============================================================================
# Note: Recommendations are provided as general agricultural guidance.
# They do not constitute guaranteed cures or formal chemical/organic certifications.
# =============================================================================
TARGET_CLASSES_METADATA: Dict[str, Dict[str, Any]] = {
    "Potato___Early_blight": {
        "crop": "Potato",
        "disease_name": "Early Blight",
        "display_name": "Potato — Early Blight",
        "is_healthy": False,
        "pathogen": "Alternaria solani (Fungal Pathogen)",
        "symptoms": (
            "Small, circular dark brown to black spots with characteristic concentric rings "
            "forming a 'target-board' pattern on older, lower foliage."
        ),
        "organic_remedies": [
            "Apply copper-based protectant sprays (e.g., Bordeaux mixture or copper octanoate) at early onset.",
            "Use bio-control foliar sprays containing Bacillus subtilis or Trichoderma viride.",
            "Apply diluted neem oil (0.5% concentration) to help suppress fungal spore germination."
        ],
        "chemical_treatments": [
            "Mancozeb 75% WP or Chlorothalonil 75% WP protectant fungicides where permitted.",
            "Under severe infection pressure, consult local agricultural extension for systemic options such as Azoxystrobin or Difenoconazole."
        ],
        "prevention": [
            "Practice minimum 2-to-3 year crop rotation with non-solanaceous crops.",
            "Water plants at the base using drip irrigation; avoid overhead sprinkler wetting.",
            "Prune and safely dispose of lower infected leaves after harvest."
        ]
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "disease_name": "Late Blight",
        "display_name": "Potato — Late Blight",
        "is_healthy": False,
        "pathogen": "Phytophthora infestans (Oomycete Water Mold)",
        "symptoms": (
            "Water-soaked, pale-green to dark brown lesions that expand rapidly across leaf tips, "
            "often accompanied by white fuzzy mildew growth on leaf undersides in cool, humid weather."
        ),
        "organic_remedies": [
            "Apply preventive fixed copper hydroxide or liquid copper octanoate before extended rainy periods.",
            "Spray bio-protective formulations based on Streptomyces or beneficial mycorrhizae.",
            "Ensure wide plant spacing to maximize canopy airflow and rapid leaf drying."
        ],
        "chemical_treatments": [
            "Metalaxyl + Mancozeb formulations at first sign of regional blight warnings.",
            "Cymoxanil + Mancozeb or Dimethomorph during active outbreaks, following label safety intervals."
        ],
        "prevention": [
            "Plant only certified disease-free seed tubers.",
            "Destroy volunteer potato plants and cull piles before the planting season.",
            "Eliminate standing water and ensure well-draining soil mounds."
        ]
    },
    "Potato___healthy": {
        "crop": "Potato",
        "disease_name": "Healthy Potato Leaf",
        "display_name": "Potato — Healthy Foliage",
        "is_healthy": True,
        "pathogen": "None (Plant is healthy and vigorous)",
        "symptoms": "Uniform deep green color, intact leaf margins, no fungal spotting, lesion rings, or chlorosis.",
        "organic_remedies": [
            "Continue balanced cultural practices with compost tea or well-rotted organic matter.",
            "Apply beneficial soil inoculants (Pseudomonas fluorescens) to support root vigor."
        ],
        "chemical_treatments": [
            "No chemical intervention needed. Avoid unnecessary prophylactic fungicide applications."
        ],
        "prevention": [
            "Maintain a balanced N-P-K nutrient schedule, avoiding excessive nitrogen fertilization.",
            "Inspect underside of foliage bi-weekly for early pest vectors like aphids."
        ]
    },
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "disease_name": "Early Blight",
        "display_name": "Tomato — Early Blight",
        "is_healthy": False,
        "pathogen": "Alternaria solani (Fungal Pathogen)",
        "symptoms": (
            "Concentric dark brown rings surrounded by a chlorotic yellow halo, starting on bottom leaves "
            "and progressing upwards, leading to premature leaf drop."
        ),
        "organic_remedies": [
            "Apply copper sulfate or copper soap spray every 7–10 days during warm, damp conditions.",
            "Foliar spray with Bacillus amyloliquefaciens or compost extract formulations.",
            "Mulch heavily around tomato base to prevent soil-splash onto lower foliage."
        ],
        "chemical_treatments": [
            "Mancozeb 75% WP or Chlorothalonil protectant sprays.",
            "Alternate with systemic fungicides like Pyraclostrobin or Tebuconazole as advised by local extension agents."
        ],
        "prevention": [
            "Stake and prune indeterminate tomatoes to maintain an upright, well-ventilated canopy.",
            "Always sanitize pruning shears with 70% alcohol between plants.",
            "Do not plant tomatoes in soil where potatoes, eggplants, or peppers were grown in the previous season."
        ]
    },
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "disease_name": "Late Blight",
        "display_name": "Tomato — Late Blight",
        "is_healthy": False,
        "pathogen": "Phytophthora infestans (Oomycete Water Mold)",
        "symptoms": (
            "Irregular water-soaked greasy patches turning purplish-black, collapsing leaf stems, "
            "and white spore mildew visible underneath leaves in humid morning hours."
        ),
        "organic_remedies": [
            "Preventive sprays of fixed copper formulations before extended rainy periods.",
            "Immediately remove and bag severely infected leaves; do not add infected material to home compost."
        ],
        "chemical_treatments": [
            "Metalaxyl-M + Mancozeb at first warning of regional blight alerts.",
            "Mandipropamid or Fluopicolide formulations during active sporulation, adhering strictly to pre-harvest intervals."
        ],
        "prevention": [
            "Select late-blight resistant tomato cultivars when available.",
            "Water exclusively via ground drip lines or soaker hoses.",
            "Maintain wide row spacing for maximum sunlight penetration and air movement."
        ]
    },
    "Tomato___Leaf_Mold": {
        "crop": "Tomato",
        "disease_name": "Leaf Mold",
        "display_name": "Tomato — Leaf Mold",
        "is_healthy": False,
        "pathogen": "Passalora fulva (syn. Cladosporium fulvum, Fungus)",
        "symptoms": (
            "Pale greenish-yellow chlorotic spots on upper leaf surfaces that correspond with "
            "velvety olive-green to grayish-brown mold mats on leaf undersides."
        ),
        "organic_remedies": [
            "Apply bio-fungicides containing Trichoderma harzianum or Bacillus subtilis.",
            "Apply sulfur-based wettable powder or copper soap formulations.",
            "Prune dense foliage to lower relative humidity below 85%."
        ],
        "chemical_treatments": [
            "Chlorothalonil 75% WP or Difenoconazole 25% EC.",
            "Thiophanate-methyl or Azoxystrobin applied according to manufacturer label safety directions."
        ],
        "prevention": [
            "In greenhouse/polyhouse settings, maximize ventilation and utilize circulation fans.",
            "Avoid late evening overhead irrigation to prevent leaves from remaining wet overnight.",
            "Sterilize greenhouse benches and frames between planting cycles."
        ]
    },
    "Tomato___healthy": {
        "crop": "Tomato",
        "disease_name": "Healthy Tomato Leaf",
        "display_name": "Tomato — Healthy Foliage",
        "is_healthy": True,
        "pathogen": "None (Plant is healthy and vigorous)",
        "symptoms": "Vibrant green leaves, robust pubescence (leaf hairs), upright stems with zero lesion spots.",
        "organic_remedies": [
            "Feed regularly with balanced organic compost or seaweed kelp extract.",
            "Maintain consistent soil moisture to prevent calcium-related blossom-end rot."
        ],
        "chemical_treatments": [
            "No chemical intervention required."
        ],
        "prevention": [
            "Inspect foliage regularly for pests like whiteflies or hornworms.",
            "Keep mulch fresh and clean around the root zone."
        ]
    }
}


class PlantDiseasePredictor:
    """
    Manages MobileNetV2 inference and agronomic remedy mapping.
    """

    def __init__(
        self,
        model_path: str = "models/plantcare.keras",
        class_indices_path: str = "models/class_indices.json"
    ):
        """
        Initialize the predictor instance.

        Args:
            model_path (str): Relative or absolute path to the saved Keras model.
            class_indices_path (str): Path to class index JSON mapping.
        """
        # Resolve application-root-relative paths if relative strings are provided
        if not os.path.isabs(model_path):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            candidate = os.path.join(base_dir, model_path)
            if os.path.exists(candidate):
                model_path = candidate
        if not os.path.isabs(class_indices_path):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            candidate = os.path.join(base_dir, class_indices_path)
            if os.path.exists(candidate):
                class_indices_path = candidate

        self.model_path = model_path
        self.class_indices_path = class_indices_path
        self.model = None
        self.class_indices: Dict[int, str] = {}
        self.is_ready: bool = False

        self._load_class_indices()
        self._load_model_if_available()

    def _load_class_indices(self) -> None:
        """
        Load or fallback class index mapping.
        """
        if os.path.exists(self.class_indices_path):
            try:
                with open(self.class_indices_path, "r", encoding="utf-8") as f:
                    raw_indices = json.load(f)
                    # Convert string keys to integer class indices
                    self.class_indices = {int(k): v for k, v in raw_indices.items()}
                return
            except Exception as e:
                logger.warning(f"Could not parse class_indices.json: {e}")

        # Standard alphabetical class indices (PlantVillage standard)
        self.class_indices = {
            0: "Potato___Early_blight",
            1: "Potato___Late_blight",
            2: "Potato___healthy",
            3: "Tomato___Early_blight",
            4: "Tomato___Late_blight",
            5: "Tomato___Leaf_Mold",
            6: "Tomato___healthy"
        }

    @property
    def class_names(self) -> List[str]:
        """
        List of class names sorted by class index.
        """
        return [self.class_indices[i] for i in sorted(self.class_indices.keys())]

    def _load_model_if_available(self) -> None:
        """
        Attempt to load the trained Keras model if the binary file exists.
        If not found, gracefully sets is_ready = False.
        """
        if not os.path.exists(self.model_path):
            self.model = None
            self.is_ready = False
            return

        try:
            # Lazy import TensorFlow so the web framework remains lightweight
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
            self.is_ready = True
            logger.info(f"Loaded trained PlantCare AI model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load trained model from {self.model_path}: {e}")
            self.model = None
            self.is_ready = False

    def predict(self, image_input: Union[str, bytes, Any]) -> Dict[str, Any]:
        """
        Perform real plant leaf disease classification.

        CRITICAL REQUIREMENT:
        If the real model is not trained yet (Phase 1 / early Phase 2),
        returns a truthful notice:
        "AI model not trained yet. Prediction will be available in Phase 2."
        and NEVER generates mock/fake predictions or fabricated confidences.

        Args:
            image_input: Image file path, bytes, or stream.

        Returns:
            dict: Structured prediction response.
        """
        # If the real trained model is not present, report truthfully
        if not self.is_ready or self.model is None:
            return {
                "status": "model_not_ready",
                "is_trained": False,
                "message": "AI model not trained yet. Prediction will be available in Phase 2.",
                "prediction": None,
                "confidence": None,
                "confidence_percent": None,
                "crop": None,
                "disease_name": None,
                "details": None
            }

        # Model is present: perform real MobileNetV2 inference
        try:
            # 1. Preprocess input into (1, 224, 224, 3) normalized tensor
            tensor = load_and_preprocess_image(image_input, target_size=(224, 224))

            # 2. Verify single-pass normalization:
            # The PlantCare MobileNetV2 architecture incorporates in-model preprocess_input
            # that operates on raw [0, 255] RGB float inputs. Unscaling tensor back to [0, 255]
            # ensures that the in-model preprocess_input layer executes normalization exactly once.
            model_input = (tensor + 1.0) * 127.5

            # 3. Run forward pass
            probabilities = self.model.predict(model_input, verbose=0)[0]

            # 3. Extract top prediction
            predicted_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_idx])
            class_name = self.class_indices.get(predicted_idx, "Unknown")

            # 4. Lookup agricultural metadata
            meta = TARGET_CLASSES_METADATA.get(class_name, {
                "crop": "Unknown",
                "disease_name": class_name,
                "display_name": class_name,
                "is_healthy": False,
                "pathogen": "Under investigation",
                "symptoms": "Lesion detection completed.",
                "organic_remedies": ["Consult agricultural extension officer."],
                "chemical_treatments": ["Apply broad-spectrum registered fungicide."],
                "prevention": ["Rotate crops and sanitize farm tools."]
            })

            # 5. Build all-probabilities distribution dictionary
            all_scores = {
                self.class_indices.get(i, f"Class_{i}"): round(float(prob) * 100, 2)
                for i, prob in enumerate(probabilities)
            }

            # 6. Uncertainty Detection (Confidence Thresholding & Out-of-Domain Notice)
            # Predictions below 70% confidence are flagged as uncertain.
            # Academic note: Confidence thresholds alone cannot reliably detect all unsupported
            # plant species or pathogens; unrepresented plant leaves may still receive skewed predictions.
            is_uncertain = confidence < 0.70
            uncertainty_warning = (
                f"Model confidence is moderate ({round(confidence * 100, 1)}%). "
                "This leaf image may display early/subtle infection symptoms, non-standard lighting, "
                "or may belong to a plant species or disease outside the current 7-class scope. "
                "Note: Confidence thresholds cannot reliably distinguish all unsupported plants or diseases. "
                "In-person agricultural extension verification is recommended."
                if is_uncertain else ""
            )

            advisory_notice = (
                "Advisory Notice: Predictions and agronomic guidance provided by PlantCare AI are automated "
                "decision-support estimates. They do not constitute guaranteed laboratory diagnoses or registered "
                "prescriptions. Always consult local agricultural extension officers and strictly follow manufacturer "
                "safety directions on registered product labels before application."
            )

            return {
                "status": "success",
                "is_trained": True,
                "message": "Disease prediction generated successfully by MobileNetV2.",
                "raw_class": class_name,
                "display_name": meta["display_name"],
                "crop": meta["crop"],
                "disease_name": meta["disease_name"],
                "is_healthy": meta["is_healthy"],
                "confidence": round(confidence, 4),
                "confidence_percent": f"{round(confidence * 100, 2)}%",
                "is_uncertain": is_uncertain,
                "uncertainty_warning": uncertainty_warning,
                "advisory_notice": advisory_notice,
                "pathogen": meta["pathogen"],
                "symptoms": meta["symptoms"],
                "organic_remedies": meta["organic_remedies"],
                "chemical_treatments": meta["chemical_treatments"],
                "prevention": meta["prevention"],
                "distribution": all_scores
            }

        except Exception as e:
            logger.error(f"Inference error during model prediction: {e}")
            return {
                "status": "error",
                "is_trained": True,
                "message": f"Inference pipeline error: {str(e)}",
                "prediction": None,
                "confidence": None
            }
