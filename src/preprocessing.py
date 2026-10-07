"""Classical preprocessing and histogram operations."""

from __future__ import annotations

import cv2
import numpy as np

from src.acquisition import to_grayscale


def calculate_histogram(gray_image: np.ndarray) -> np.ndarray:
    """Return a 256-bin grayscale histogram."""

    gray = to_grayscale(gray_image)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    return hist.astype(np.float32)


def histogram_stretch(gray_image: np.ndarray) -> np.ndarray:
    """Linearly stretch intensities to the full 0-255 range."""

    gray = to_grayscale(gray_image)
    min_val, max_val = int(gray.min()), int(gray.max())
    if max_val == min_val:
        return gray.copy()
    stretched = (gray.astype(np.float32) - min_val) * (255.0 / (max_val - min_val))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def histogram_equalization(gray_image: np.ndarray) -> np.ndarray:
    """Apply global histogram equalization."""

    return cv2.equalizeHist(to_grayscale(gray_image))


def apply_clahe(gray_image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Apply contrast-limited adaptive histogram equalization."""

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    return clahe.apply(to_grayscale(gray_image))


def log_transform(gray_image: np.ndarray) -> np.ndarray:
    """Apply logarithmic intensity transformation."""

    gray = to_grayscale(gray_image).astype(np.float32)
    transformed = np.log1p(gray)
    transformed *= 255.0 / max(float(transformed.max()), 1.0)
    return np.clip(transformed, 0, 255).astype(np.uint8)


def image_negative(image: np.ndarray) -> np.ndarray:
    """Return image negative."""

    return 255 - image.astype(np.uint8)


def gaussian_smoothing(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian smoothing."""

    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def median_filtering(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median filtering."""

    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
    return cv2.medianBlur(image.astype(np.uint8), kernel_size)


def bilateral_filtering(image: np.ndarray, diameter: int = 7) -> np.ndarray:
    """Apply bilateral filtering while preserving edges."""

    return cv2.bilateralFilter(image.astype(np.uint8), diameter, 75, 75)


def preprocessing_gallery(image: np.ndarray) -> dict[str, np.ndarray]:
    """Build the standard CO2 comparison view."""

    gray = to_grayscale(image)
    return {
        "Original": image,
        "Grayscale": gray,
        "Histogram Equalized": histogram_equalization(gray),
        "CLAHE": apply_clahe(gray),
        "Log Transform": log_transform(gray),
        "Negative": image_negative(gray),
    }
