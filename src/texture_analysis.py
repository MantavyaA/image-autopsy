"""Local statistical and texture anomaly analysis."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk

from src.acquisition import to_grayscale


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize arbitrary numeric image data to uint8."""

    image = image.astype(np.float32)
    if float(image.max()) == float(image.min()):
        return np.zeros(image.shape, dtype=np.uint8)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def local_mean(gray_image: np.ndarray, window_size: int = 31) -> np.ndarray:
    gray = to_grayscale(gray_image).astype(np.float32)
    return cv2.blur(gray, (window_size, window_size))


def local_variance(gray_image: np.ndarray, window_size: int = 31) -> np.ndarray:
    gray = to_grayscale(gray_image).astype(np.float32)
    mean = cv2.blur(gray, (window_size, window_size))
    sq_mean = cv2.blur(gray * gray, (window_size, window_size))
    return np.maximum(sq_mean - mean * mean, 0)


def local_std(gray_image: np.ndarray, window_size: int = 31) -> np.ndarray:
    return np.sqrt(local_variance(gray_image, window_size))


def local_entropy(gray_image: np.ndarray, radius: int = 5) -> np.ndarray:
    gray = to_grayscale(gray_image)
    return entropy(gray, disk(radius)).astype(np.float32)


def local_gradient_magnitude(gray_image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(gray_image)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def texture_anomaly_map(image: np.ndarray, window_size: int = 31) -> np.ndarray:
    """Find regions whose local texture statistics differ from their neighborhood."""

    gray = to_grayscale(image)
    variance = local_variance(gray, window_size)
    std = local_std(gray, window_size)
    gradient = local_gradient_magnitude(gray)

    variance_context = cv2.blur(variance, (window_size * 2 + 1, window_size * 2 + 1))
    std_context = cv2.blur(std, (window_size * 2 + 1, window_size * 2 + 1))
    gradient_context = cv2.blur(gradient, (window_size * 2 + 1, window_size * 2 + 1))

    anomaly = (
        0.45 * np.abs(variance - variance_context)
        + 0.30 * np.abs(std - std_context)
        + 0.25 * np.abs(gradient - gradient_context)
    )
    return normalize_uint8(anomaly)


def texture_feature_panel(image: np.ndarray) -> dict[str, np.ndarray]:
    """Return visual texture/statistical maps for the dashboard."""

    gray = to_grayscale(image)
    return {
        "Local Mean": normalize_uint8(local_mean(gray)),
        "Local Variance": normalize_uint8(local_variance(gray)),
        "Local Std Dev": normalize_uint8(local_std(gray)),
        "Local Entropy": normalize_uint8(local_entropy(gray)),
        "Local Gradient": normalize_uint8(local_gradient_magnitude(gray)),
        "Texture Anomaly": texture_anomaly_map(gray),
    }
