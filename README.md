# PlantCare AI — Intelligent Plant Leaf Disease Detection and Classification System

> **MCA University Minor Project**  
> **Author:** Abhishek Kumar (Master of Computer Applications)  
> **Academic Focus:** Applied Computer Vision, Web Engineering & Agricultural AI

---

## 🌿 Project Overview

**PlantCare AI** is an intelligent web application designed to detect, classify, and diagnose plant leaf diseases from digital photographs. Agricultural pathogens (fungi, bacteria, and oomycetes) destroy between **20% to 40%** of global crop yields annually. By empowering farmers, gardeners, and agronomists with early AI-assisted disease diagnosis, PlantCare AI helps prevent widespread infection and minimizes excessive chemical pesticide usage.

---

## 📌 Phase 1 & Phase 2 Architecture

### Phase 1: Robust Web Framework & Multi-Tier Validation
1. **Modular Flask Backend:** Clean MVC routing (`/`, `/predict`, `/about`, `/contact`), secure filenames with `werkzeug.utils.secure_filename`, and friendly error handling for `404` and `500`.
2. **5 MB Upload Enforcement & HTTP 413 Handling:** Strictly enforced 5 MB payload limit with a custom `413 Request Entity Too Large` error handler that renders the predict page with friendly guidance.
3. **Pillow Deep Image Inspection:** Multi-tier byte verification (`img.verify()`), color channel checks, and corrupt file rejection.
4. **Modern UI/UX:** Responsive green-and-white agricultural theme, drag-and-drop file portal, live HTML5 `FileReader` preview, and custom SVG illustrations.
5. **Academic Integrity:** Zero fake predictions or randomized percentages.

### Phase 2: Machine Learning & Transfer Learning Pipeline
1. **Target 7-Class Disease Scope (Tomato & Potato):**
   - `Potato___Early_blight` (*Alternaria solani*)
   - `Potato___Late_blight` (*Phytophthora infestans*)
   - `Potato___healthy` (Healthy foliage)
   - `Tomato___Early_blight` (*Alternaria solani*)
   - `Tomato___Late_blight` (*Phytophthora infestans*)
   - `Tomato___Leaf_Mold` (*Passalora fulva*)
   - `Tomato___healthy` (Healthy foliage)
2. **Dedicated Python 3.12 ML Environment:**
   - Isolated from Flask's web environment to guarantee compatibility with official TensorFlow 2.16+ Windows builds.
   - Managed via `requirements-ml.txt` and `setup_ml_env.ps1`.
3. **Dataset Ingestion & Leaf-Grouped Splitting (`training/prepare_dataset.py`):**
   - Ingests the official PlantVillage color dataset.
   - Preserves leaf-grouped train/test separation (UUID grouping) to prevent data leakage across train (70%), validation (15%), and test (15%) splits.
4. **MobileNetV2 Transfer Learning Architecture (`training/train_model.py`):**
   - Two-stage training: Feature Extraction (frozen base, lr=1e-3) followed by Fine-Tuning top 30 layers (lr=1e-5).
   - In-model data augmentation (RandomFlip, RandomRotation, RandomZoom, RandomContrast).
   - Normalization using MobileNetV2 formula: `(pixel / 127.5) - 1.0`.
   - Callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`, and `TensorBoard`.
   - Saves model to `models/plantcare.keras` and index mapping to `models/class_indices.json`.
5. **Authentic Performance Evaluation (`training/evaluate_model.py`):**
   - Evaluates on unseen `dataset/test`.
   - Generates confusion matrix heatmap (`models/confusion_matrix.png`) and classification reports (`models/evaluation_results.json`).
6. **Inference Modules (`ml/preprocessing.py` & `ml/predictor.py`):**
   - `PlantDiseasePredictor` with rich agronomic knowledgebase (pathogen taxonomy, symptoms, organic cures, chemical sprays, and prevention tips).

---

## 📂 Project Directory Structure

```text
plant-disease-detection-app/
├── app.py                      # Flask web application & route handlers
├── requirements.txt            # Flask web dependencies (Flask, Pillow, pytest)
├── requirements-ml.txt         # Phase 2 ML dependencies (TensorFlow, Keras, NumPy, scikit-learn)
├── setup_ml_env.ps1            # Automated setup script for Python 3.12 ML environment
├── README.md                   # Complete documentation & execution guide
├── .gitignore                  # Git exclusions (datasets, heavy models, venvs)
├── ml/                         # Machine learning inference package
│   ├── __init__.py             # Package exports
│   ├── preprocessing.py        # Image resizing (224x224) and MobileNetV2 normalization
│   └── predictor.py            # PlantDiseasePredictor & 7-class agronomic remedies
├── training/                   # Model training and dataset pipeline
│   ├── __init__.py             # Package exports
│   ├── prepare_dataset.py      # PlantVillage extraction & leaf-grouped splitting
│   ├── train_model.py          # MobileNetV2 transfer learning (2-stage training)
│   └── evaluate_model.py       # Test split evaluation, confusion matrix & reports
├── models/                     # Trained weights & mappings
│   ├── .gitkeep                # Directory anchor
│   └── class_indices.json      # 7-class index mapping
├── templates/                  # Jinja2 HTML templates
│   ├── base.html               # Master layout with navbar, alerts & footer
│   ├── index.html              # Homepage (Hero, How It Works, Features)
│   ├── predict.html            # Drag-and-drop leaf upload & validation results
│   ├── about.html              # Architecture, CNN explanation & roadmap
│   └── contact.html            # Student profile, inquiry form & FAQ
├── static/                     # Static web assets
│   ├── css/style.css           # Custom green & white agricultural stylesheet
│   ├── js/script.js            # Drag-and-drop & client validation logic
│   └── images/                 # Custom SVG illustrations (logo, hero, CNN diagram)
└── tests/
    ├── test_app.py             # 15 automated web route & 5 MB upload tests
    └── test_ml.py              # ML preprocessing & predictor unit tests
```

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Web Backend** | Python 3, Flask 3.x, Werkzeug |
| **Image Verification** | Pillow (PIL) |
| **Deep Learning (Phase 2)** | TensorFlow 2.16+, Keras 3.x, MobileNetV2 Architecture |
| **Data Processing & Metrics** | NumPy, scikit-learn, Matplotlib, Seaborn |
| **Frontend UI/UX** | HTML5, CSS3, JavaScript, Bootstrap 5.3, Bootstrap Icons |
| **Automated Testing** | pytest, unittest |

---

## 🚀 Execution Guide (Windows PowerShell)

### 1. Running the Flask Web Application & Tests (Existing Environment)

```powershell
# Navigate to the workspace
cd "c:\Users\Abhishek Kumar\OneDrive\Desktop\Minor Project\plant disease detection app"

# Activate the web virtual environment
.\venv\Scripts\Activate.ps1

# Run the full automated test suite (all 15 web tests)
python -m pytest tests/test_app.py -v

# Start the Flask web application
python app.py
```
Open your browser at: **`http://127.0.0.1:5000`**

---

### 2. Setting Up the Python 3.12 ML Environment (`venv-ml`)

TensorFlow on Windows officially supports Python 3.9–3.12. To keep your lightweight Flask environment (`venv`) clean, Phase 2 uses a separate virtual environment (`venv-ml`):

```powershell
# Run the automated ML environment setup script
.\setup_ml_env.ps1
```

*Or manually in PowerShell:*
```powershell
# 1. Create Python 3.12 virtual environment
py -3.12 -m venv venv-ml

# 2. Activate venv-ml
.\venv-ml\Scripts\Activate.ps1

# 3. Install Phase 2 machine learning dependencies
pip install -r requirements-ml.txt
```

---

### 3. Phase 2 Model Training & Evaluation Workflow (When Ready)

Once you download the official PlantVillage color dataset:

```powershell
# Step 1: Prepare and split dataset with leaf-grouping
python training/prepare_dataset.py --source-dir <path_to_plantvillage_color>

# Step 2: Train MobileNetV2 transfer learning model
python training/train_model.py --epochs1 10 --epochs2 15

# Step 3: Evaluate on unseen test dataset and generate confusion matrix
python training/evaluate_model.py
```

---

## 📜 Academic Integrity Note
This project strictly enforces academic integrity: all tests, metrics, and training routines use authentic data pipelines. No mock weights or fabricated confidence values are generated.
