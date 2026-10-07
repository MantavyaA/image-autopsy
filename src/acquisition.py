"""Image acquisition, metadata, sampling, and quantization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


SUPPORTED_FORMATS = {"jpg", "jpeg", "png", "bmp"}


@dataclass(frozen=True)
class ImageMetadata:
    """Basic acquisition metadata shown in the case file."""

    filename: str
    width: int
    height: int
    channels: int
    image_type: str
    file_size_kb: float
    is_grayscale: bool
    mean_intensity: float
    std_intensity: float
    min_intensity: int
    max_intensity: int


def _to_rgb_array(pil_image: Image.Image) -> np.ndarray:
    if pil_image.mode in {"RGBA", "LA"}:
        background = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
        background.alpha_composite(pil_image.convert("RGBA"))
        pil_image = background.convert("RGB")
    elif pil_image.mode not in {"RGB", "L"}:
        pil_image = pil_image.convert("RGB")

    if pil_image.mode == "L":
        gray = np.asarray(pil_image)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return np.asarray(pil_image.convert("RGB"))


def load_image(source: str | Path | BinaryIO) -> np.ndarray:
    """Load an image from a path or file-like object and return RGB uint8 data."""

    try:
        with Image.open(source) as pil_image:
            return _to_rgb_array(pil_image)
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc
    except OSError as exc:
        raise ValueError("The image could not be opened. It may be corrupted.") from exc


def get_image_metadata(
    image: np.ndarray,
    filename: str = "uploaded image",
    file_size_bytes: int | None = None,
) -> ImageMetadata:
    """Compute acquisition metadata and basic intensity statistics."""

    if image.ndim not in {2, 3}:
        raise ValueError("Expected a 2-D grayscale or 3-D color image.")

    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    gray = to_grayscale(image)
    file_size_kb = round((file_size_bytes or image.nbytes) / 1024, 2)
    suffix = Path(filename).suffix.lower().lstrip(".")
    image_type = suffix.upper() if suffix in SUPPORTED_FORMATS else "ARRAY"

    is_grayscale = channels == 1 or bool(
        np.allclose(image[..., 0], image[..., 1]) and np.allclose(image[..., 1], image[..., 2])
    )

    return ImageMetadata(
        filename=filename,
        width=int(width),
        height=int(height),
        channels=int(channels),
        image_type=image_type,
        file_size_kb=float(file_size_kb),
        is_grayscale=is_grayscale,
        mean_intensity=float(np.mean(gray)),
        std_intensity=float(np.std(gray)),
        min_intensity=int(np.min(gray)),
        max_intensity=int(np.max(gray)),
    )


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB/BGR-like or grayscale image data to uint8 grayscale."""

    if image.ndim == 2:
        return image.astype(np.uint8)
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)


def convert_color_spaces(image: np.ndarray) -> dict[str, np.ndarray]:
    """Return common color-space views used in forensic inspection."""

    rgb = image.astype(np.uint8)
    return {
        "RGB": rgb,
        "Grayscale": to_grayscale(rgb),
        "HSV": cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV),
        "LAB": cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB),
    }


def sample_image(image: np.ndarray, scale_percent: int = 100) -> np.ndarray:
    """Reduce resolution by percentage to demonstrate sampling."""

    scale_percent = int(np.clip(scale_percent, 10, 100))
    if scale_percent == 100:
        return image.copy()

    height, width = image.shape[:2]
    new_size = (
        max(1, int(width * scale_percent / 100)),
        max(1, int(height * scale_percent / 100)),
    )
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def quantize_image(image: np.ndarray, levels: int = 256) -> np.ndarray:
    """Reduce the number of intensity levels in an image."""

    levels = int(np.clip(levels, 2, 256))
    if levels == 256:
        return image.copy()

    image_float = image.astype(np.float32)
    step = 256 / levels
    quantized = np.floor(image_float / step) * step + step / 2
    return np.clip(quantized, 0, 255).astype(np.uint8)


def downscale_for_analysis(image: np.ndarray, max_side: int = 720) -> tuple[np.ndarray, float]:
    """Downscale expensive analysis input while returning the scale factor."""

    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return image.copy(), 1.0

    scale = max_side / largest
    resized = cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale
