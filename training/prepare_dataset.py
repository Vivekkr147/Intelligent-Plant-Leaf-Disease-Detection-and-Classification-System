"""
=============================================================================
PlantCare AI — PlantVillage Dataset Preparation & Leaf-Grouped Splitting
Phase 2 Machine Learning Pipeline
Author: Abhishek Kumar (MCA Minor Project)
=============================================================================
Description:
Prepares the official PlantVillage color dataset for training MobileNetV2.
Target: 7 disease classes across Tomato and Potato crops:
  1. Potato___Early_blight
  2. Potato___Late_blight
  3. Potato___healthy
  4. Tomato___Early_blight
  5. Tomato___Late_blight
  6. Tomato___Leaf_Mold
  7. Tomato___healthy

Key Principles:
1. Strict Data Integrity: Validates every image file with Pillow (PIL)
   to eliminate corrupt or truncated files.
2. Leakage Prevention: Groups images originating from the same leaf (UUID grouping)
   before splitting, ensuring that photos of the same leaf never leak across
   training, validation, and testing sets.
3. Split Ratios: 70% Train, 15% Validation, 15% Test.
4. Generates an authentic dataset summary in 'dataset/dataset_summary.json'.
5. Zero Fake Data: Does not generate fake or placeholder images.
=============================================================================
"""

import os
import re
import sys
import json
import shutil
import random
import argparse
from typing import Dict, List, Tuple
from PIL import Image, UnidentifiedImageError

# Standard 7 Target Classes from PlantVillage Color Dataset
TARGET_CLASSES: List[str] = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy"
]

# Supported image extensions in PlantVillage
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def is_valid_image(file_path: str) -> bool:
    """
    Verify that an image file is readable, uncorrupted, and has valid dimensions.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            w, h = img.size
            if w < 10 or h < 10:
                return False
        return True
    except (UnidentifiedImageError, OSError, SyntaxError):
        return False


def extract_leaf_id(filename: str) -> str:
    """
    Extract leaf identifier from PlantVillage naming conventions.
    PlantVillage filenames often follow patterns like:
    '<UUID>___RS_Early.B 7192.JPG' or '<leaf_id>_<view>.jpg'
    If a UUID or stem pattern is detected, it is returned to prevent
    the same physical leaf from appearing in both train and test splits.
    """
    base_name = os.path.splitext(filename)[0]
    # Check for UUID prefix (e.g. c03dd08f-2ee1-4ad9-bf9f-2f76cb3ad969___...)
    uuid_match = re.match(r"^([0-9a-fA-F\-]{12,36})___", base_name)
    if uuid_match:
        return uuid_match.group(1).lower()

    # Check for underscore or dash prefix groupings
    prefix_match = re.match(r"^([a-zA-Z0-9]+)[_\-]", base_name)
    if prefix_match:
        return prefix_match.group(1).lower()

    # Default to filename itself if no grouping pattern matches
    return base_name.lower()


def split_grouped_dataset(
    class_images: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split images into train, val, and test sets while preserving leaf grouping.
    All images from the same leaf ID are assigned to the same split.
    """
    random.seed(seed)

    # 1. Group images by leaf ID
    leaf_groups: Dict[str, List[str]] = {}
    for img_path in class_images:
        leaf_id = extract_leaf_id(os.path.basename(img_path))
        leaf_groups.setdefault(leaf_id, []).append(img_path)

    # 2. Shuffle unique leaf groups
    unique_leaf_ids = list(leaf_groups.keys())
    random.shuffle(unique_leaf_ids)

    # 3. Partition leaf IDs according to target ratios
    total_leaves = len(unique_leaf_ids)
    train_end = int(total_leaves * train_ratio)
    val_end = train_end + int(total_leaves * val_ratio)

    train_leaves = set(unique_leaf_ids[:train_end])
    val_leaves = set(unique_leaf_ids[train_end:val_end])
    test_leaves = set(unique_leaf_ids[val_end:])

    # 4. Flatten back to image path lists
    train_imgs = [img for lid in train_leaves for img in leaf_groups[lid]]
    val_imgs = [img for lid in val_leaves for img in leaf_groups[lid]]
    test_imgs = [img for lid in test_leaves for img in leaf_groups[lid]]

    return train_imgs, val_imgs, test_imgs


def prepare_dataset(
    source_dir: str,
    target_dir: str = "dataset",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> None:
    """
    Process, validate, and split the raw PlantVillage dataset into:
      target_dir/train/<class_name>/
      target_dir/val/<class_name>/
      target_dir/test/<class_name>/
    """
    print("\n" + "=" * 70)
    print("  PlantCare AI — PlantVillage Dataset Preparation Pipeline")
    print(f"  Source Directory: {source_dir}")
    print(f"  Target Directory: {target_dir}")
    print(f"  Target Classes ({len(TARGET_CLASSES)}): {', '.join(TARGET_CLASSES)}")
    print("=" * 70 + "\n")

    if not os.path.exists(source_dir):
        print(f"[ERROR] Source dataset directory '{source_dir}' does not exist.")
        print("Please download the official PlantVillage dataset or provide a valid path:")
        print("  python training/prepare_dataset.py --source-dir <path_to_plantvillage_color>")
        sys.exit(1)

    # Create destination directories
    splits = ["train", "val", "test"]
    for split in splits:
        for cls_name in TARGET_CLASSES:
            os.makedirs(os.path.join(target_dir, split, cls_name), exist_ok=True)

    summary: Dict[str, Any] = {
        "dataset_name": "PlantVillage (Tomato & Potato 7-Class Subset)",
        "classes": TARGET_CLASSES,
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "counts": {},
        "total_images": 0
    }

    # Process each target class
    for cls_name in TARGET_CLASSES:
        cls_source_dir = os.path.join(source_dir, cls_name)

        if not os.path.exists(cls_source_dir):
            print(f"[WARNING] Class directory not found: '{cls_source_dir}'. Skipping.")
            continue

        # Collect and validate image files
        valid_images: List[str] = []
        corrupted_count = 0

        for root, _, files in os.walk(cls_source_dir):
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in VALID_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    if is_valid_image(full_path):
                        valid_images.append(full_path)
                    else:
                        corrupted_count += 1

        print(f"[CLASS] {cls_name}: Found {len(valid_images)} valid images (Corrupted/skipped: {corrupted_count})")

        if not valid_images:
            print(f"[WARNING] No valid images for {cls_name}.")
            continue

        # Leaf-grouped train/val/test splitting
        train_imgs, val_imgs, test_imgs = split_grouped_dataset(
            valid_images, train_ratio, val_ratio, test_ratio, seed=seed
        )

        # Copy images into structured folders
        for img_path in train_imgs:
            shutil.copy2(img_path, os.path.join(target_dir, "train", cls_name, os.path.basename(img_path)))
        for img_path in val_imgs:
            shutil.copy2(img_path, os.path.join(target_dir, "val", cls_name, os.path.basename(img_path)))
        for img_path in test_imgs:
            shutil.copy2(img_path, os.path.join(target_dir, "test", cls_name, os.path.basename(img_path)))

        summary["counts"][cls_name] = {
            "total": len(valid_images),
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs)
        }
        summary["total_images"] += len(valid_images)

    # Save summary report
    summary_path = os.path.join(target_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  Dataset Preparation Complete!")
    print(f"  Total Validated Images Processed: {summary['total_images']}")
    print(f"  Summary saved to: {summary_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare and split PlantVillage Tomato/Potato dataset for MobileNetV2."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="PlantVillage/color",
        help="Path to the raw PlantVillage color dataset directory."
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="dataset",
        help="Destination directory for prepared train/val/test splits."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic, reproducible leaf-grouped splitting."
    )
    args = parser.parse_args()

    prepare_dataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        seed=args.seed
    )
