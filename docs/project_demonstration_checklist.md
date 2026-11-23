# PlantCare AI: University Project Demonstration Checklist

**Candidate:** Vivek Kumar (MCA University Candidate)  
**System:** PlantCare AI 5.0 (Production Verified)  
**Target Audience:** MCA Minor Project External / Internal Examination Committee  
**Total Demonstration Time:** ~10 to 15 minutes  

---

## 1. Pre-Flight Preparation Checklist (15 Minutes Before Viva)

- [ ] **Open PowerShell as Administrator / User:**
  ```powershell
  cd "C:\Users\Abhishek Kumar\OneDrive\Desktop\Minor Project\plant disease detection app"
  ```
- [ ] **Verify Python Environment:**
  ```powershell
  & "C:\Users\Abhishek Kumar\pcml\Scripts\python.exe" --version
  # Must output: Python 3.12.10
  ```
- [ ] **Verify TensorFlow & Keras:**
  ```powershell
  & "C:\Users\Abhishek Kumar\pcml\Scripts\python.exe" -c "import tensorflow as tf; print(tf.__version__)"
  # Must output: 2.17.1
  ```
- [ ] **Verify Model File Existence:**
  - `models/plantcare.keras` (~23.7 MB)
  - `models/class_indices.json` (7 classes)
  - `models/evaluation_results.json` (96.04% test metrics)
- [ ] **Verify Sample Test Images are Ready:**
  Keep sample leaf photos on Desktop or Downloads for quick drag-and-drop:
  - `sample_tomato_healthy.jpg`
  - `sample_tomato_early_blight.jpg`
  - `sample_potato_late_blight.jpg`
  - `fake_text_file.txt` (for security demonstration)
  - `corrupted_image.jpg` (for header validation demonstration)
- [ ] **Run Full Test Suite Pre-Check:**
  ```powershell
  & "C:\Users\Abhishek Kumar\pcml\Scripts\python.exe" -m pytest tests/ -v
  # Verify: 48 passed
  ```

---

## 2. Launching the Live Application

Run the Flask server using the dedicated `pcml` environment:
```powershell
& "C:\Users\Abhishek Kumar\pcml\Scripts\python.exe" app.py
```
**Expected Terminal Output:**
```text
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```
Open Chrome or Edge and navigate to:
```text
http://127.0.0.1:5000
```

---

## 3. Step-by-Step Live Demonstration Script

### Step 1: Homepage & System Overview (2 Minutes)
1. **Show the Homepage (`http://127.0.0.1:5000/`):**
   - Point out the agricultural green-and-white theme (`#1b4332`, `#2d6a4f`, `#e8f5e9`).
   - Highlight the **"Phase 4/5"** active status badge.
   - Explain the 3-step pipeline: *Upload Leaf Photo* \(\to\) *MobileNetV2 Deep Learning* \(\to\) *Actionable Agronomic Guidance*.
   - Mention the 3 summary metric highlights (7 Target Classes, < 100 ms Inference, 100% Secure Validation).

---

### Step 2: Upload Security & Defensive Validation (3 Minutes)
*(Examiners love to see robust error handling and defensive programming!)*
1. **Navigate to the Diagnostic Portal (`/predict`).**
2. **Test 1 — File Type Rejection:**
   - Attempt to upload `fake_text_file.txt`.
   - **Show Result:** Flash alert: *"Invalid file extension... Only JPG, JPEG, and PNG images are allowed."*
3. **Test 2 — Corrupted Header Trapping:**
   - Attempt to upload `corrupted_image.jpg` (a text file renamed to `.jpg`).
   - **Show Result:** Flash alert: *"Corrupted or unrecognized image file. Please upload an authentic photo."*
   - **Explain to Panel:** Explain that Pillow's `img.verify()` inspects the internal binary magic numbers and header structures, preventing arbitrary script execution.
4. **Test 3 — File Size Limit (5 MB):**
   - Explain that files exceeding 5 MB trigger a custom HTTP 413 error handler without crashing Flask.

---

### Step 3: Live Model Inference & Prediction Output (3 Minutes)
1. **Upload a Real Healthy Leaf (e.g., Tomato Healthy):**
   - Drag and drop the leaf image onto the dashed container.
   - Show the instant **Client-Side Preview** card displaying filename, dimensions, and payload size.
   - Leave the opt-in toggle switch **unchecked** (to show privacy-preserving non-storage).
   - Click **"Detect Plant Disease"**.
2. **Review the Diagnostic Result Card:**
   - **Header:** Green gradient with *"Foliage Healthy — Crop: Tomato"*.
   - **Diagnosis Title:** *"Tomato — Healthy Foliage"*.
   - **Confidence Meter:** Show the high probability percentage (typically 99%+ on authentic leaves).
   - **Pathogen:** *"None (Plant is healthy and vigorous)"*.
   - **Preventative Guidance:** Balanced fertilizer recommendations and inspection advice.
   - **7-Class Distribution:** Show the transparent Softmax probability distribution bars across all 7 classes.
3. **Demonstrate Infected Leaf (e.g., Tomato Early Blight):**
   - Upload an infected leaf and **check** the opt-in toggle: *"Save leaf photo with prediction history"*.
   - Click **"Detect Plant Disease"**.
   - **Show Result:**
     - Status: *"Infection Detected"*.
     - Pathogen: *"Alternaria solani (Fungal Pathogen)"*.
     - Symptoms: Concentric target-board rings.
     - Cultural remedies (diluted neem oil, pruning lower foliage).
     - Conventional protectants (Mancozeb, Copper fungicides).
     - Badge: *"Logged to History #X"*.

---

### Step 4: Low-Confidence / Uncertainty Warning (1 Minute)
1. **Explain the Uncertainty Guardrail:**
   - Point out that any leaf with confidence below 70% automatically displays a prominent amber warning.
   - **Explain to Panel:** Explain the Softmax closed-world limitation and why PlantCare AI explicitly warns farmers when inputs sit near decision boundaries.

---

### Step 5: Prediction History & Privacy Demonstration (2 Minutes)
1. **Click "History" in the Navbar (`/history`):**
   - Show the responsive table of previous scans with timestamps, crop badges, condition names, and confidence bars.
2. **Demonstrate Privacy-Preserving Image Storage:**
   - Show the record saved **without consent**: Displays a shield placeholder (*"No image saved — Privacy protected"*). Zero files on disk!
   - Show the record saved **with consent**: Displays a clickable thumbnail. Click it to open the **enlarged modal preview**!
3. **Demonstrate Dynamic Filters:**
   - Select **Crop: Tomato** \(\to\) table filters dynamically.
   - Select **Condition: Healthy Only** \(\to\) table filters instantly.
4. **Demonstrate Protected Deletion:**
   - Click the red delete icon on a record \(\to\) confirm prompt \(\to\) record and disk file are purged with flash alert: *"Prediction record #X successfully deleted."*
   - Show the **"Clear All"** button and its safety modal.

---

### Step 6: Live Analytics Dashboard & Chart.js (2 Minutes)
1. **Click "Dashboard" in the Navbar (`/dashboard`):**
   - Point out the 4 KPI metric cards:
     - **Total Scans**
     - **Healthy Foliage** (count and %)
     - **Infected Leaves** (count and %)
     - **Average Softmax Confidence** (%)
2. **Demonstrate the 4 Interactive Chart.js Visualizations:**
   - **Foliage Health Ratio (Donut Chart):** Hover over segments to show interactive tooltips.
   - **Scans by Crop (Pie Chart):** Tomato vs. Potato volume.
   - **Daily Scan Activity (Line Chart):** Temporal volume trend.
   - **Predicted Class Distribution (Horizontal Bar Chart):** Color-coded bars across all 7 classes.
3. **Explain to Panel:** Emphasize that **zero fake statistics** are used; all charts and cards query live rows in `plantcare.db` via SQL aggregations.

---

### Step 7: Automated Test Suite Execution (1 Minute)
1. **Open PowerShell in front of the examiners:**
   ```powershell
   & "C:\Users\Abhishek Kumar\pcml\Scripts\python.exe" -m pytest tests/ -v
   ```
2. **Show the Clean 48 Passed Results:**
   - `test_app.py`: 18 tests passed (Routes, 404/413 handlers, security).
   - `test_ml.py`: 10 tests passed (Preprocessing range [-1, 1], predictor parity).
   - `test_phase4.py`: 16 tests passed (Database CRUD, privacy consent, Chart.js metrics).
   - `test_phase5.py`: 4 tests passed (CSRF protection, evaluation parity, SQL parameterization).
   - **Total:** **48 passed, 0 failed (100% success rate)**.

---

## 4. Code Walkthrough Quick Reference (When Examiners Ask to See Code)

| If Examiners Ask About... | Open File & Point To: | Key Technical Explanation |
| :--- | :--- | :--- |
| **Model Loading Optimization** | `app.py` (Line 51 & 73) | Point out `predictor` is initialized once at startup, not per request. |
| **Input Security & Pillow** | `app.py` (`validate_image_stream`) | Point to `img.verify()`, Pillow header check, and format whitelisting. |
| **CSRF Protection** | `app.py` (`csrf_protect`) | Show session token generation and `before_request` verification. |
| **Preprocessing Parity** | `ml/predictor.py` (`predict`) | Show `(tensor + 1.0) * 127.5` fed to in-model `preprocess_input`. |
| **SQL Injection Prevention** | `database.py` (`insert_prediction`) | Show parameterized `?` query bindings; zero string formatting. |
| **Privacy Consent Handling** | `app.py` (`predict`) | Show check for `save_image` form switch and cleanup in `delete_prediction`. |
| **Model Evaluation Metrics** | `models/evaluation_results.json` | Show 1,163 test samples, 96.04% accuracy, and confusion matrix. |
| **Interactive Charts** | `templates/dashboard.html` | Show Chart.js instantiation consuming Jinja2-serialized JSON data. |

---

## 5. Summary Closing Statement for Viva

> *"In conclusion, PlantCare AI is not merely an isolated deep learning experiment, but an end-to-end, production-grade agricultural decision-support application. By choosing MobileNetV2 with depthwise separable convolutions, we achieved an optimal balance of 96.04% test split accuracy and rapid sub-100ms inference on standard CPUs with a compact 23.7 MB model. Supported by rigorous input validation, CSRF defense, parameterized SQLite persistence, and an interactive Chart.js dashboard, the system is fully verified with 48 automated test cases and stands ready for mobile edge expansion."*
