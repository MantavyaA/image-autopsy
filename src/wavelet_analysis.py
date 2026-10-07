"""Wavelet-domain analysis with Haar decomposition."""

from __future__ import annotations

import cv2
import numpy as np
import pywt

from src.acquisition import to_grayscale
from src.texture_analysis import normalize_uint8


def haar_decomposition(image: np.ndarray) -> dict[str, np.ndarray]:
    """Compute one-level 2-D Haar wavelet subbands."""

    gray = to_grayscale(image).astype(np.float32)
    coeffs = pywt.dwt2(gray, "haar")
    ll, (lh, hl, hh) = coeffs
    return {
        "LL": normalize_uint8(ll),
        "LH": normalize_uint8(np.abs(lh)),
        "HL": normalize_uint8(np.abs(hl)),
        "HH": normalize_uint8(np.abs(hh)),
    }


def wavelet_anomaly_map(image: np.ndarray) -> np.ndarray:
    """Use detail subband energy to highlight localized high-frequency anomalies."""

    gray = to_grayscale(image)
    coeffs = pywt.dwt2(gray.astype(np.float32), "haar")
    _, (lh, hl, hh) = coeffs
    detail_energy = np.sqrt(lh * lh + hl * hl + hh * hh)
    detail_energy = normalize_uint8(detail_energy)
    resized = cv2.resize(detail_energy, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return normalize_uint8(cv2.blur(resized, (15, 15)))
