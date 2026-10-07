"""Point, line, and edge detection for structural examination."""

from __future__ import annotations

import cv2
import numpy as np
from skimage import filters

from src.acquisition import to_grayscale


def _normalize_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if float(image.max()) == float(image.min()):
        return np.zeros(image.shape, dtype=np.uint8)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def sobel_edges(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return _normalize_uint8(cv2.magnitude(gx, gy))


def scharr_edges(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    return _normalize_uint8(cv2.magnitude(gx, gy))


def prewitt_edges(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    return _normalize_uint8(filters.prewitt(gray) * 255)


def laplacian_edges(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    return _normalize_uint8(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))


def canny_edges(image: np.ndarray, low_threshold: int = 80, high_threshold: int = 180) -> np.ndarray:
    gray = to_grayscale(image)
    return cv2.Canny(gray, int(low_threshold), int(high_threshold))


def point_response(image: np.ndarray) -> np.ndarray:
    """Detect point/corner-like responses using Harris corner energy."""

    gray = np.float32(to_grayscale(image))
    response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    return _normalize_uint8(response)


def hough_lines_overlay(image: np.ndarray, edges: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Draw probabilistic Hough line detections over the image."""

    rgb = image.copy()
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    edge_map = edges if edges is not None else canny_edges(image)
    min_len = max(20, min(image.shape[:2]) // 8)
    lines = cv2.HoughLinesP(
        edge_map,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_len,
        maxLineGap=8,
    )
    count = 0
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4)[:120]:
            cv2.line(rgb, (x1, y1), (x2, y2), (255, 55, 55), 2)
            count += 1
    return rgb, count


def edge_density(edge_map: np.ndarray) -> float:
    """Return the fraction of non-zero edge pixels."""

    return float(np.count_nonzero(edge_map) / edge_map.size)


def edge_inconsistency_map(image: np.ndarray) -> np.ndarray:
    """Highlight locally unusual edge energy."""

    sobel = sobel_edges(image).astype(np.float32)
    local_mean = cv2.blur(sobel, (31, 31))
    residual = np.abs(sobel - local_mean)
    return _normalize_uint8(residual)


def structural_analysis(image: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """Compute the standard structural evidence panel."""

    canny = canny_edges(image)
    line_overlay, line_count = hough_lines_overlay(image, canny)
    return {
        "Point Response": point_response(image),
        "Sobel": sobel_edges(image),
        "Scharr": scharr_edges(image),
        "Prewitt": prewitt_edges(image),
        "Laplacian": laplacian_edges(image),
        "Canny": canny,
        "Line Overlay": line_overlay,
        "Line Count": line_count,
        "Edge Density": edge_density(canny),
        "Edge Inconsistency": edge_inconsistency_map(image),
    }
