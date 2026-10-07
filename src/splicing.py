"""Classical local inconsistency analysis for potential splice-like regions."""

from __future__ import annotations

import cv2
import numpy as np

from src.acquisition import to_grayscale
from src.texture_analysis import local_variance, normalize_uint8


def color_inconsistency(image: np.ndarray, window_size: int = 35) -> np.ndarray:
    """Estimate local color deviations from surrounding illumination/color context."""

    if image.ndim == 2:
        return np.zeros(image.shape, dtype=np.uint8)
    lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    local = cv2.blur(lab, (window_size, window_size))
    broader = cv2.blur(lab, (window_size * 2 + 1, window_size * 2 + 1))
    delta = np.linalg.norm(local - broader, axis=2)
    return normalize_uint8(delta)


def splicing_anomaly_map(image: np.ndarray) -> np.ndarray:
    """Fuse variance, gradient discontinuity, and color inconsistency cues."""

    gray = to_grayscale(image)
    variance = normalize_uint8(local_variance(gray, 31)).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    gradient_context = cv2.blur(gradient, (41, 41))
    gradient_discontinuity = normalize_uint8(np.abs(gradient - gradient_context)).astype(np.float32)

    color = color_inconsistency(image).astype(np.float32)
    fused = 0.35 * variance + 0.35 * gradient_discontinuity + 0.30 * color
    return normalize_uint8(fused)
