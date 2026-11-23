"""
=============================================================================
PlantCare AI — Model Evaluation & Performance Metrics Pipeline
Phase 2 Deep Learning Architecture
Author: Abhishek Kumar (MCA Minor Project)
=============================================================================
Description:
Evaluates the trained MobileNetV2 model ('models/plantcare.keras') against
the unseen, held-out test dataset split ('dataset/test').

Metrics Computed:
1. Overall Test Loss & Test Categorical Accuracy.
2. Per-class Precision, Recall, and F1-Score via scikit-learn.
3. Confusion Matrix: Normalized and raw counts, saved as a high-resolution
   heatmap in 'models/confusion_matrix.png'.
4. Full numerical results serialized to 'models/evaluation_results.json'.

Academic Integrity Rule:
If the model file or test dataset is missing, the script halts with informative
guidance and NEVER generates simulated accuracy reports or fake confusion matrices.
=============================================================================
"""

import os
import sys
import json
import argparse
import numpy as np

# Verify dependencies
try:
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless server execution
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"[ERROR] Required evaluation dependency missing: {e}")
    print("Please activate your Python 3.12 ML environment:")
    print("  .\\venv-ml\\Scripts\\Activate.ps1")
    print("  pip install -r requirements-ml.txt")
    sys.exit(1)


def evaluate_model(
    model_path: str = "models/plantcare.keras",
    test_dir: str = "dataset/test",
    class_indices_path: str = "models/class_indices.json",
    output_dir: str = "models"
) -> None:
    """
    Run authentic evaluation on the test dataset split.
    """
    print("\n" + "=" * 70)
    print("  PlantCare AI — Model Evaluation Pipeline")
    print(f"  Target Model: {model_path}")
    print(f"  Test Dataset: {test_dir}")
    print("=" * 70 + "\n")

    # 1. Verification of assets
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model file '{model_path}' not found.")
        print("Please train the model first by running:")
        print("  python training/train_model.py")
        sys.exit(1)

    if not os.path.exists(test_dir) or not os.listdir(test_dir):
        print(f"[ERROR] Test dataset folder '{test_dir}' is empty or does not exist.")
        print("Please prepare the dataset first by running:")
        print("  python training/prepare_dataset.py")
        sys.exit(1)

    # 2. Load class names mapping
    if os.path.exists(class_indices_path):
        with open(class_indices_path, "r", encoding="utf-8") as f:
            class_indices = json.load(f)
            class_names = [class_indices[str(i)] for i in range(len(class_indices))]
    else:
        class_names = sorted(os.listdir(test_dir))

    print(f"[INFO] Evaluating across {len(class_names)} classes: {class_names}")

    # 3. Load Trained Model
    print(f"[MODEL] Loading {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # 4. Load Test Dataset (without shuffling to align predictions with ground truth)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=32,
        image_size=(224, 224),
        shuffle=False
    )

    # 5. Evaluate Overall Loss and Accuracy
    print("\n[EVAL] Computing overall test loss and categorical accuracy...")
    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\n>> Real Test Loss:     {loss:.4f}")
    print(f">> Real Test Accuracy: {accuracy * 100:.2f}%")

    # 6. Gather Predictions and True Labels
    print("\n[PREDICT] Generating inference predictions on all test batches...")
    file_paths = test_ds.file_paths
    y_true_indices = []
    y_pred_indices = []
    confidences = []
    misclassified_samples = []

    batch_idx = 0
    batch_size = 32
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        b_true = np.argmax(labels.numpy(), axis=1)
        b_pred = np.argmax(preds, axis=1)
        b_conf = np.max(preds, axis=1)

        start_offset = batch_idx * batch_size
        for j in range(len(b_true)):
            idx = start_offset + j
            t = int(b_true[j])
            p = int(b_pred[j])
            c = float(b_conf[j])
            y_true_indices.append(t)
            y_pred_indices.append(p)
            confidences.append(c)

            if t != p:
                fp = file_paths[idx] if idx < len(file_paths) else f"sample_{idx}.jpg"
                misclassified_samples.append({
                    "filename": os.path.basename(fp),
                    "true_class": class_names[t],
                    "predicted_class": class_names[p],
                    "confidence": round(c, 4),
                    "distribution": {class_names[k]: round(float(preds[j][k]), 4) for k in range(len(class_names))}
                })

        batch_idx += 1

    y_true = np.array(y_true_indices)
    y_pred = np.array(y_pred_indices)

    # 7. Classification Report (Precision, Recall, F1-Score)
    print("\n" + "-" * 60)
    print("  Detailed Classification Performance Report")
    print("-" * 60)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True
    )
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print(report_str)

    macro_f1 = float(report_dict["macro avg"]["f1-score"])
    weighted_f1 = float(report_dict["weighted avg"]["f1-score"])
    print(f">> Macro F1-Score:    {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
    print(f">> Weighted F1-Score: {weighted_f1:.4f} ({weighted_f1 * 100:.2f}%)")

    # 8. Confusion Patterns & Model Limitation Analysis
    confusion_patterns = {}
    for item in misclassified_samples:
        key = f"{item['true_class']} -> {item['predicted_class']}"
        confusion_patterns[key] = confusion_patterns.get(key, 0) + 1

    sorted_patterns = sorted(confusion_patterns.items(), key=lambda x: x[1], reverse=True)
    print("\n" + "-" * 60)
    print(f"  Error Analysis: {len(misclassified_samples)} Misclassified Samples ({len(misclassified_samples)/len(y_true)*100:.2f}% Error Rate)")
    print("-" * 60)
    for pair, count in sorted_patterns[:10]:
        print(f"  {pair}: {count} instances")

    # 9. Confusion Matrix & Heatmap Visualization
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(f"PlantCare AI MobileNetV2 — Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[OUTPUT] Confusion matrix heatmap saved to: {cm_path}")

    # 10. Serialize Complete Numerical Results
    eval_summary = {
        "model": model_path,
        "test_dataset": test_dir,
        "total_test_samples": int(len(y_true)),
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "test_accuracy_percent": f"{accuracy * 100:.2f}%",
        "macro_f1": macro_f1,
        "macro_f1_percent": f"{macro_f1 * 100:.2f}%",
        "weighted_f1": weighted_f1,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "misclassified_count": len(misclassified_samples),
        "confusion_patterns": dict(sorted_patterns),
        "misclassified_samples": misclassified_samples
    }

    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"[OUTPUT] Full evaluation metrics saved to: {results_path}")

    print("\n" + "=" * 70)
    print("  Model Evaluation Successfully Finished!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained MobileNetV2 on test split.")
    parser.add_argument("--model", type=str, default="models/plantcare.keras", help="Path to model.")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Test split folder.")
    parser.add_argument("--class-indices", type=str, default="models/class_indices.json", help="Class mapping.")
    parser.add_argument("--out-dir", type=str, default="models", help="Output directory for reports.")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        test_dir=args.test_dir,
        class_indices_path=args.class_indices,
        output_dir=args.out_dir
    )
