# PlantCare AI: University Viva Presentation Outline (15 Slides)

**Candidate:** Vivek Kumar  
**Degree:** Master of Computer Applications (MCA) — Minor Project  
**Title:** PlantCare AI — Intelligent Plant Leaf Disease Detection System  
**Framework & Tech:** TensorFlow 2.17.1, Keras 3.0, MobileNetV2, Flask, SQLite, Chart.js  

---

### Slide 1: Title & Candidate Credentials
- **Slide Title:** PlantCare AI — Intelligent Plant Leaf Disease Detection System
- **Subtitle:** An End-to-End Deep Learning & Web Architecture for Foliar Pathogen Classification
- **Presenter:** Vivek Kumar (MCA University Candidate)
- **Department:** Department of Computer Applications
- **Key Visual Elements:** PlantCare AI logo, university crest placeholder, technology badges (TensorFlow, Flask, MobileNetV2, SQLite, Chart.js).
- **Speaking Points (30s):** Introduce self, department, and the core purpose of the project: bridging digital computer vision with real-world smallholder agriculture.

---

### Slide 2: Motivation & Problem Statement
- **Slide Title:** Agricultural Background & Problem Statement
- **Bullet Points:**
  - Crop diseases destroy 20% to 40% of global agricultural yields annually.
  - Solanaceous staples (Potato and Tomato) suffer devastating fungal and oomycete infections (Early Blight, Late Blight, Leaf Mold).
  - Severe rural shortage of certified agronomists leads to late interventions.
  - Farmers frequently misidentify blights, leading to inappropriate chemical pesticide overuse and environmental contamination.
- **Key Metric:** \$220B estimated annual global crop yield loss from foliar pathogens.
- **Speaking Points (45s):** Emphasize the economic and nutritional impact of foliar diseases, and explain why automated visual diagnosis is urgent for developing agricultural regions.

---

### Slide 3: Project Objectives & Scope
- **Slide Title:** Project Objectives & Deliverables
- **Bullet Points:**
  1. Develop an ImageNet transfer-learned MobileNetV2 model for seven target foliar classes.
  2. Implement strict software security: 5 MB limit, Pillow header integrity checks, and CSRF protection.
  3. Guarantee single-pass preprocessing parity without double normalization artifacts.
  4. Build a responsive, agricultural green-and-white Flask web portal with live softmax confidence meters.
  5. Engineer persistent SQLite prediction logging with privacy-compliant opt-in photo storage.
  6. Deliver interactive Chart.js analytics for crop distributions, health ratios, and scan trends.
- **Speaking Points (45s):** Summarize the end-to-end scope: not merely an isolated ML model, but a complete, secure, database-backed web application.

---

### Slide 4: Literature Review & Architectural Choice
- **Slide Title:** Literature Survey: Why MobileNetV2?
- **Comparison Table:**
  - *VGG-16:* 138M parameters, ~528 MB model size (High accuracy, but impractical for web CPU servers).
  - *ResNet-50:* 25.6M parameters, ~100 MB model size (Deep residual learning, high computational overhead).
  - *MobileNetV2 (Selected):* **3.5M parameters, 23.7 MB model size** (Depthwise separable convolutions + inverted residuals; ~9x fewer MAC operations with 96%+ accuracy).
- **Speaking Points (60s):** Justify why MobileNetV2 is the optimal architectural choice for responsive web inference on standard CPU servers without GPU dependency.

---

### Slide 5: System Architecture & Multi-Tier Pipeline
- **Slide Title:** Multi-Tiered System Architecture
- **Diagram Summary:**
  - *Client Layer:* Browser (HTML5, Bootstrap 5, Chart.js, Drag-and-drop JS).
  - *Security Layer:* Content-Length (5 MB limit) \(\to\) CSRF Token \(\to\) Extension Whitelist \(\to\) Pillow `img.verify()`.
  - *Inference Layer:* Preprocessing (224×224 RGB, [-1.0, 1.0]) \(\to\) MobileNetV2 \(\to\) Softmax Probabilities.
  - *Persistence Layer:* Parameterized SQLite (`predictions` table) + Opt-in `static/uploads/`.
- **Speaking Points (60s):** Walk the panel through the flow of an image from the user's browser, through the security boundary, into the CNN inference graph, and into persistent storage.

---

### Slide 6: Dataset Composition & Partitioning
- **Slide Title:** PlantVillage Dataset: 7 Solanaceous Classes
- **Data Table:**
  - Total Images: **7,684** authentic foliar photographs.
  - Training Split (70%): **5,369 images**
  - Validation Split (15%): **1,152 images**
  - Testing Split (15%): **1,163 images** (strictly held-out)
  - Classes: Potato Early Blight, Potato Late Blight, Potato Healthy, Tomato Early Blight, Tomato Late Blight, Tomato Leaf Mold, Tomato Healthy.
- **Integrity Guarantee:** Verified zero UUID / leaf overlap across training, validation, and test splits.
- **Speaking Points (45s):** Emphasize rigorous partitioning: no data leakage between splits, zero corrupted images, and balanced coverage across crops.

---

### Slide 7: MobileNetV2 Mathematical Foundations
- **Slide Title:** Deep Learning Foundations: Depthwise Separable Convolutions
- **Mathematical Formulations:**
  - Standard Conv MACs: \(D_K^2 \cdot M \cdot N \cdot D_F^2\)
  - Depthwise + Pointwise MACs: \(D_K^2 \cdot M \cdot D_F^2 + M \cdot N \cdot D_F^2\)
  - Computational Reduction: \(\frac{1}{N} + \frac{1}{D_K^2} \approx \frac{1}{9}\) reduction factor.
  - *Inverted Residuals & Linear Bottlenecks:* Expands channel dimension by \(t=6\) for depthwise spatial filtering, then projects linearly back to low-dimensional manifold without destroying manifold non-linearities.
- **Speaking Points (60s):** Explain how MobileNetV2 achieves high representational capacity with an order-of-magnitude reduction in arithmetic operations.

---

### Slide 8: Preprocessing & Mathematical Parity
- **Slide Title:** Preprocessing Pipeline & Single-Pass Parity
- **Key Concepts:**
  - In-memory decoding via Pillow (`Image.open(stream)`).
  - RGB conversion (discarding RGBA alpha channels cleanly).
  - Bicubic resizing to \(224 \times 224\) pixels.
  - Mathematical Parity: Avoided double normalization artifacts by scaling raw RGB \([0, 255]\) directly to the in-model `preprocess_input` layer, bounded strictly to \([-1.0, 1.0]\).
  - Preprocessing parity tested with zero absolute numerical difference between direct Keras and predictor inference.
- **Speaking Points (45s):** Highlight this engineering detail: double normalization is a common defect in junior ML projects, and we proved mathematical parity with automated tests.

---

### Slide 9: Two-Stage Model Training Protocol
- **Slide Title:** Two-Stage Transfer Learning Protocol
- **Stage 1 — Feature Extraction:**
  - Frozen MobileNetV2 base (`layer.trainable = False`).
  - Trained custom head: GlobalAveragePooling2D \(\to\) BatchNormalization \(\to\) Dropout (0.3) \(\to\) Dense(7, Softmax).
  - Adam Optimizer (\(\text{lr} = 1 \times 10^{-3}\)) for 10 initial epochs.
- **Stage 2 — Fine-Tuning:**
  - Unfroze top 30 layers; kept BatchNormalization in inference mode to preserve ImageNet statistics.
  - Adam Optimizer with reduced learning rate (\(\text{lr} = 1 \times 10^{-5}\)).
  - EarlyStopping & ModelCheckpoint monitoring validation accuracy \(\to\) converged at Epoch 22.
- **Speaking Points (45s):** Explain the rationale for freezing first (preventing large initial gradients from destroying ImageNet weights) followed by gentle fine-tuning.

---

### Slide 10: Empirical Results & Evaluation Metrics
- **Slide Title:** Empirical Model Evaluation (Held-Out Test Set)
- **Headline Results on 1,163 Test Images:**
  - **Overall Test Accuracy:** **96.04%** (1,117 / 1,163 correct)
  - **Overall Test Loss:** **0.1164**
  - **Macro Average F1-Score:** **95.73%**
  - **Weighted Average F1-Score:** **96.03%**
  - Tomato Healthy Recall: **100.0%** (240 / 240 correctly classified)
  - Potato Early Blight Precision: **99.32%** (147 / 148 predictions correct)
- **Speaking Points (60s):** Present the empirical findings clearly. Clarify that 96.04% is the measured performance on the held-out test split, not a field guarantee.

---

### Slide 11: Confusion Matrix & Diagnostic Insights
- **Slide Title:** Confusion Matrix Analysis & Misclassifications
- **Visual:** 7×7 Confusion Matrix Table / Heatmap.
- **Diagnostic Observations:**
  - Total Misclassifications: 46 out of 1,163 images (3.96% error rate).
  - *Cross-Crop Confusion:* 4 potato late blight leaves misclassified as tomato late blight, and 2 tomato late blight misclassified as potato late blight.
  - *Pathological Reason:* Both diseases are caused by the identical oomycete pathogen (*Phytophthora infestans*), which manifests identical necrotic water-soaked lesion morphologies across both Solanaceous hosts.
- **Speaking Points (60s):** Discuss the confusion matrix analytically: explain *why* errors occurred from a biological perspective, demonstrating deep understanding of the problem domain.

---

### Slide 12: Web Application Security Pipeline
- **Slide Title:** Web Application Architecture & Security Pipeline
- **Security Pillars:**
  1. *Payload Throttling:* 5 MB MAX_CONTENT_LENGTH enforced by web server.
  2. *Deep Header Verification:* Pillow `img.verify()` guarantees valid JPEG/PNG headers; prevents arbitrary binary/script execution.
  3. *Path Traversal Defense:* Werkzeug `secure_filename()` sanitization.
  4. *CSRF Token Validation:* Cryptographic session tokens on all state-changing POST endpoints.
  5. *SQL Injection Defense:* 100% parameterized SQL bindings (`?`).
  6. *Error Shielding:* Custom 404, 413, and 500 handlers preventing raw traceback leakage.
- **Speaking Points (45s):** Show that software security was engineered as a primary concern rather than an afterthought.

---

### Slide 13: Prediction History & Analytics Dashboard
- **Slide Title:** Persistent History & Real-Time Analytics Dashboard
- **Key Features:**
  - *SQLite Database:* Thread-safe connection pooling with auto-initialized schema.
  - *Privacy-Preserving Image Storage:* Opt-in toggle; leaf photos are NOT saved to disk without explicit user consent.
  - *History Page:* Dynamic filtering by Crop and Health condition, thumbnail modal previews, protected single deletion and clear-all actions.
  - *Analytics Dashboard:* Real-time KPI counters and 4 interactive Chart.js visualizations (Donut, Pie, Line, and Horizontal Bar charts).
  - *Zero Fake Statistics:* All numbers dynamically computed from live database rows.
- **Speaking Points (60s):** Highlight Phase 4 contributions: persistence, privacy compliance, and real-time visualization of diagnostic trends.

---

### Slide 14: System Limitations & Responsible Agronomic AI
- **Slide Title:** System Limitations & Responsible AI Engineering
- **Ethical & Operational Bounds:**
  - *Laboratory Background Bias:* PlantVillage images use detached leaves on uniform paper backgrounds; performance may degrade on outdoor foliage with complex soil clutter.
  - *Single Leaf Constraint:* Requires close-up shots; cannot diagnose whole canopy or distant tree photos.
  - *Uncertainty Warning Mechanism:* Flags any prediction with confidence \(< 70\%\) as potentially unmodeled or subtle.
  - *Out-of-Distribution Warning:* Softmax closed-world assumption means non-plant objects still sum to 1.0; system explicitly warns users that confidence thresholds cannot reliably detect all unsupported plant species.
  - *Agronomic Disclaimers:* Recommendations are framed as decision-support guidance, not legal or laboratory guarantees.
- **Speaking Points (45s):** Demonstrate academic maturity by openly discussing model limitations and the responsible guardrails engineered into the application.

---

### Slide 15: Conclusion, Future Scope & Acknowledgements
- **Slide Title:** Conclusion & Future Scope
- **Conclusion:**
  - Successfully built and validated an end-to-end plant disease diagnosis system.
  - Achieved 96.04% test accuracy with an ultra-lightweight 23.7 MB MobileNetV2 model.
  - 48 out of 48 automated test cases passed across all project phases.
- **Future Enhancements:**
  - TFLite quantization for offline on-device mobile Android app.
  - Domain expansion to all 38 PlantVillage classes.
  - Integration with multi-spectral agricultural drone cameras.
- **Acknowledgements:** Guide, department faculty, and open-source agricultural researchers.
- **Speaking Points (30s):** Conclude the presentation, thank the examination committee, and invite questions.
