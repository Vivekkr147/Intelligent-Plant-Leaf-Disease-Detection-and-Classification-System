"""
=============================================================================
PlantCare AI — Leaf Image Preprocessing Pipeline for MobileNetV2
Phase 2 Machine Learning Pipeline
Author: Abhishek Kumar (MCA Minor Project)
=============================================================================
Description:
Prepares plant leaf images for ingestion by the MobileNetV2 CNN model.
Implements:
1. High-quality image loading from files, byte arrays, or PIL objects.
2. RGB color mode enforcement (strips alpha channels or grayscale).
3. Resizing to MobileNetV2 input dimensions (224 × 224 pixels) with LANCZOS resampling.
4. MobileNetV2 normalization: scales raw pixel range [0, 255] to [-1.0, 1.0]
   using the formula: (pixel / 127.5) - 1.0.
5. Batch dimension expansion to shape (1, 224, 224, 3).
=============================================================================
"""

import io
from typing import Union, Tuple, Dict, Any
from PIL import Image, UnidentifiedImageError

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Standard input resolution required by MobileNetV2
MOBILENET_INPUT_SHAPE: Tuple[int, int] = (224, 224)


def inspect_image(image_input: Union[str, bytes, io.BytesIO, Image.Image]) -> Dict[str, Any]:
    """
    Extract metadata from an image input without modifying its contents.

    Args:
        image_input: File path (str), raw bytes, BytesIO stream, or PIL Image.

    Returns:
        dict: Metadata containing width, height, format, mode, and size_kb.
    """
    if isinstance(image_input, Image.Image):
        img = image_input
        size_bytes = len(img.tobytes())
        return {
            "width": img.width,
            "height": img.height,
            "dimensions": f"{img.width} × {img.height} px",
            "format": img.format or "PIL_IMAGE",
            "mode": img.mode,
            "size_kb": round(size_bytes / 1024, 2),
        }

    # If stream or bytes, read with PIL
    if isinstance(image_input, (bytes, bytearray)):
        stream = io.BytesIO(image_input)
    elif isinstance(image_input, io.BytesIO):
        stream = image_input
        stream.seek(0)
    elif isinstance(image_input, str):
        with open(image_input, "rb") as f:
            stream = io.BytesIO(f.read())
    else:
        raise ValueError(f"Unsupported image_input type: {type(image_input)}")

    stream.seek(0, io.SEEK_END)
    size_bytes = stream.tell()
    stream.seek(0)

    with Image.open(stream) as img:
        return {
            "width": img.width,
            "height": img.height,
            "dimensions": f"{img.width} × {img.height} px",
            "format": img.format or "UNKNOWN",
            "mode": img.mode,
            "size_kb": round(size_bytes / 1024, 2),
        }


def load_and_preprocess_image(
    image_input: Union[str, bytes, io.BytesIO, Image.Image],
    target_size: Tuple[int, int] = MOBILENET_INPUT_SHAPE
) -> np.ndarray:
    """
    Load and preprocess an authentic plant leaf photo for MobileNetV2 inference.

    Args:
        image_input: File path, byte array, BytesIO stream, or PIL Image.
        target_size: Tuple of (height, width), defaults to (224, 224).

    Returns:
        np.ndarray: Preprocessed float32 array with shape (1, target_size[0], target_size[1], 3)
                    and pixel values normalized to the range [-1.0, 1.0].

    Raises:
        ValueError: If the input cannot be read or has invalid dimensions.
    """
    # 1. Convert input into an active PIL Image instance
    if isinstance(image_input, Image.Image):
        img = image_input.copy()
    elif isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, io.BytesIO):
        image_input.seek(0)
        img = Image.open(image_input)
    elif isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    try:
        # 2. Convert to standard RGB mode (handling RGBA, Grayscale, CMYK, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 3. Resize to 224 × 224 using high-fidelity LANCZOS interpolation
        # (Preserves leaf edge features and lesion textures)
        resampling_filter = getattr(Image.Resampling, 'LANCZOS', Image.LANCZOS)
        img = img.resize(target_size, resample=resampling_filter)

        # 4. Convert PIL Image to float32 NumPy array
        if not HAS_NUMPY:
            raise ImportError("NumPy is required for tensor preprocessing. Activate the Phase 2 ML environment.")
        arr = np.asarray(img, dtype=np.float32)

        # 5. MobileNetV2 Normalization:
        # Scales [0.0, 255.0] -> [-1.0, 1.0]
        # (matches tf.keras.applications.mobilenet_v2.preprocess_input)
        normalized = (arr / 127.5) - 1.0

        # 6. Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        batched_tensor = np.expand_dims(normalized, axis=0)

        return batched_tensor

    except Exception as exc:
        raise ValueError(f"Failed to preprocess image for MobileNetV2: {str(exc)}") from exc
