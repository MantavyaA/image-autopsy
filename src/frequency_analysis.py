"""2-D Fourier analysis and frequency-domain anomaly evidence."""

from __future__ import annotations

import cv2
import numpy as np

from src.acquisition import to_grayscale
from src.texture_analysis import normalize_uint8


def fft_magnitude_spectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return shifted FFT data and log-scaled magnitude spectrum."""

    gray = to_grayscale(image).astype(np.float32)
    fft = np.fft.fft2(gray)
    shifted = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(shifted))
    return shifted, normalize_uint8(magnitude)


def frequency_anomaly_map(image: np.ndarray, low_frequency_radius_ratio: float = 0.12) -> np.ndarray:
    """Construct a high-frequency residual map from the Fourier domain."""

    gray = to_grayscale(image).astype(np.float32)
    shifted, _ = fft_magnitude_spectrum(gray)
    rows, cols = gray.shape
    cy, cx = rows // 2, cols // 2
    yy, xx = np.ogrid[:rows, :cols]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    cutoff = low_frequency_radius_ratio * min(rows, cols)
    high_pass = shifted.copy()
    high_pass[radius < cutoff] = 0
    residual = np.abs(np.fft.ifft2(np.fft.ifftshift(high_pass)))
    local_energy = cv2.blur(residual.astype(np.float32), (17, 17))
    return normalize_uint8(local_energy)


def frequency_statistics(image: np.ndarray) -> dict[str, float]:
    """Compute compact summary statistics for the FFT magnitude spectrum."""

    _, spectrum = fft_magnitude_spectrum(image)
    return {
        "fft_mean": float(np.mean(spectrum)),
        "fft_std": float(np.std(spectrum)),
        "fft_max": float(np.max(spectrum)),
    }
