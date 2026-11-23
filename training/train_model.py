"""
=============================================================================
PlantCare AI — MobileNetV2 Transfer Learning Pipeline
Phase 2 Deep Learning Architecture
Author: Abhishek Kumar (MCA Minor Project)
=============================================================================
Description:
Trains a Convolutional Neural Network for plant disease classification using
Transfer Learning with MobileNetV2 pre-trained on ImageNet.
Target: 7 disease classes across Tomato and Potato crops.

Two-Stage Training Architecture:
Stage 1: Feature Extraction
  - Base MobileNetV2 weights frozen.
  - Custom classification head trained with Adam(lr=1e-3) for 10 epochs.
Stage 2: Fine-Tuning
  - Top 30 layers of MobileNetV2 unfrozen.
  - End-to-end network fine-tuned with Adam(lr=1e-5) for 15 epochs.

Callbacks:
  - ModelCheckpoint: Saves best model weights to 'models/plantcare.keras'.
  - EarlyStopping: Halts training if val_loss plateaus for 5 epochs.
  - ReduceLROnPlateau: Scales down learning rate by 0.2× on plateau.
  - TensorBoard: Logs training scalar metrics to 'training/logs'.

Outputs:
  - models/plantcare.keras (Saved Keras Model)
  - models/class_indices.json (Class mapping dictionary)
  - models/model_metadata.json (Architecture & training hyperparameter log)
=============================================================================
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Any

# Ensure TensorFlow is imported inside script execution
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        ReduceLROnPlateau,
        TensorBoard
    )
except ImportError:
    print("[ERROR] TensorFlow is required for model training.")
    print("Please activate your Python 3.12 ML environment:")
    print("  .\\venv-ml\\Scripts\\Activate.ps1")
    print("  pip install -r requirements-ml.txt")
    sys.exit(1)


# Training Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 7
SEED = 42


def build_plantcare_mobilenet_model(
    input_shape=(224, 224, 3),
    num_classes=7,
    dropout_rate=0.3
) -> tf.keras.Model:
    """
    Construct the MobileNetV2 transfer learning model with data augmentation
    and custom classification head.
    """
    # 1. Input layer
    inputs = layers.Input(shape=input_shape, name="input_leaf_image")

    # 2. In-model Data Augmentation Pipeline
    # (Executed on GPU/CPU during training only; deactivated during inference)
    x = layers.RandomFlip("horizontal_and_vertical", seed=SEED)(inputs)
    x = layers.RandomRotation(0.15, seed=SEED)(x)
    x = layers.RandomZoom(0.15, seed=SEED)(x)
    x = layers.RandomContrast(0.1, seed=SEED)(x)

    # 3. MobileNetV2 Normalization: Scales [0, 255] -> [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # 4. Pre-trained MobileNetV2 Base (ImageNet weights, excluding top dense layers)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    # Initially freeze all base layers for Stage 1 feature extraction
    base_model.trainable = False

    x = base_model(x, training=False)

    # 5. Custom Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)

    # 6. Softmax probability output layer
    outputs = layers.Dense(num_classes, activation="softmax", name="prediction_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PlantCare_MobileNetV2")
    return model, base_model


def train_model(
    data_dir: str = "dataset",
    output_model_path: str = "models/plantcare.keras",
    epochs_stage1: int = 10,
    epochs_stage2: int = 15,
    batch_size: int = BATCH_SIZE,
    smoke_test: bool = False
) -> None:
    """
    Execute the two-stage transfer learning pipeline.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # Fallback to user profile directory if relative default is missing
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        fallback_path = os.path.expanduser(r"~\PlantCareData\prepared")
        if os.path.exists(os.path.join(fallback_path, "train")):
            data_dir = fallback_path
            train_dir = os.path.join(data_dir, "train")
            val_dir = os.path.join(data_dir, "val")

    print("\n" + "=" * 70)
    print("  PlantCare AI — MobileNetV2 Transfer Learning Pipeline")
    print(f"  Training Data Directory: {train_dir}")
    print(f"  Validation Data Directory: {val_dir}")
    print(f"  Target Model Output: {output_model_path}")
    if smoke_test:
        print("  MODE: Lightweight Smoke Test (2 Batches, Non-Destructive)")
    print("=" * 70 + "\n")

    # Safety check: Verify dataset presence
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        print(f"[ERROR] Training dataset not found at '{train_dir}'.")
        print("Please prepare the PlantVillage dataset first by running:")
        print("  python training/prepare_dataset.py --source-dir <path_to_plantvillage>")
        sys.exit(1)

    # In smoke test mode, protect production files
    if smoke_test:
        if output_model_path == "models/plantcare.keras":
            output_model_path = "models/smoke_test_plantcare.keras"
        metadata_filename = "models/smoke_test_metadata.json"
        epochs_stage1 = 1
        epochs_stage2 = 1
        steps_per_epoch = 2
        validation_steps = 2
    else:
        metadata_filename = "models/model_metadata.json"
        steps_per_epoch = None
        validation_steps = None

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    os.makedirs("training/logs", exist_ok=True)

    # 1. Load Datasets using image_dataset_from_directory
    print("[DATA] Loading training dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=SEED
    )

    print("[DATA] Loading validation dataset...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMG_SIZE,
        shuffle=False,
        seed=SEED
    )

    class_names = train_ds.class_names
    print(f"[INFO] Discovered {len(class_names)} classes: {class_names}")

    # Export class index mapping (only during full training or if missing)
    class_indices = {i: name for i, name in enumerate(class_names)}
    if not smoke_test or not os.path.exists("models/class_indices.json"):
        with open("models/class_indices.json", "w", encoding="utf-8") as f:
            json.dump(class_indices, f, indent=2)
        print("[INFO] Class indices saved to 'models/class_indices.json'")

    # Optimize data pipeline performance with prefetching and caching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 2. Build MobileNetV2 Model
    print("\n[MODEL] Constructing MobileNetV2 Transfer Learning Architecture...")
    model, base_model = build_plantcare_mobilenet_model(
        input_shape=(224, 224, 3),
        num_classes=len(class_names)
    )

    # Callbacks Configuration
    callbacks = [
        ModelCheckpoint(
            filepath=output_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    if not smoke_test:
        callbacks.append(
            TensorBoard(
                log_dir=os.path.join("training/logs", datetime.now().strftime("%Y%m%d-%H%M%S")),
                histogram_freq=1
            )
        )

    # -------------------------------------------------------------------------
    # STAGE 1: Feature Extraction (Frozen Base)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"  STAGE 1: Feature Extraction ({epochs_stage1} Epochs, lr=1e-3)")
    print("  Base MobileNetV2 layers are frozen.")
    print("-" * 60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    # -------------------------------------------------------------------------
    # STAGE 2: Fine-Tuning (Unfreeze Top 30 MobileNetV2 Layers)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"  STAGE 2: Fine-Tuning ({epochs_stage2} Epochs, lr=1e-5)")
    print("  Unfreezing the top 30 layers of MobileNetV2 for domain adaptation.")
    print("  Keeping BatchNormalization layers frozen in inference mode.")
    print("-" * 60)

    base_model.trainable = True
    # Freeze all layers except the top 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    # Explicitly keep all BatchNormalization layers in inference mode
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    total_epochs = epochs_stage1 + epochs_stage2
    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=len(history_stage1.epoch),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    # 3. Save Final Model & Training Metadata
    metadata = {
        "model_name": "PlantCare AI MobileNetV2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_resolution": [224, 224, 3],
        "num_classes": len(class_names),
        "classes": class_names,
        "epochs_completed": len(history_stage1.epoch) + len(history_stage2.epoch),
        "final_train_accuracy": float(history_stage2.history["accuracy"][-1]),
        "final_val_accuracy": float(history_stage2.history["val_accuracy"][-1]),
        "final_train_loss": float(history_stage2.history["loss"][-1]),
        "final_val_loss": float(history_stage2.history["val_loss"][-1]),
        "smoke_test": smoke_test
    }

    with open(metadata_filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    status_title = "Smoke Test Successfully Completed!" if smoke_test else "MobileNetV2 Training Successfully Completed!"
    print("\n" + "=" * 70)
    print(f"  {status_title}")
    print(f"  Model Saved to: {output_model_path}")
    print(f"  Class Index Mapping: models/class_indices.json")
    print(f"  Metadata Logged to: {metadata_filename}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on PlantVillage dataset.")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Prepared dataset root.")
    parser.add_argument("--model-out", type=str, default="models/plantcare.keras", help="Model path.")
    parser.add_argument("--epochs1", type=int, default=10, help="Stage 1 feature extraction epochs.")
    parser.add_argument("--epochs2", type=int, default=15, help="Stage 2 fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run lightweight 2-batch smoke test without touching production models."
    )
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        output_model_path=args.model_out,
        epochs_stage1=args.epochs1,
        epochs_stage2=args.epochs2,
        batch_size=args.batch_size,
        smoke_test=args.smoke_test
    )
