"""Morphological processing for suspicious-region mask refinement."""

from __future__ import annotations

import cv2
import numpy as np

from src.acquisition import to_grayscale


def structuring_element(size: int = 5, shape: str = "ellipse") -> np.ndarray:
    shape_map = {
        "rectangle": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
    }
    cv_shape = shape_map.get(shape.lower(), cv2.MORPH_ELLIPSE)
    size = max(3, int(size))
    return cv2.getStructuringElement(cv_shape, (size, size))


def erosion(mask: np.ndarray, size: int = 3) -> np.ndarray:
    return cv2.erode(to_grayscale(mask), structuring_element(size), iterations=1)


def dilation(mask: np.ndarray, size: int = 3) -> np.ndarray:
    return cv2.dilate(to_grayscale(mask), structuring_element(size), iterations=1)


def opening(mask: np.ndarray, size: int = 5) -> np.ndarray:
    return cv2.morphologyEx(to_grayscale(mask), cv2.MORPH_OPEN, structuring_element(size))


def closing(mask: np.ndarray, size: int = 7) -> np.ndarray:
    return cv2.morphologyEx(to_grayscale(mask), cv2.MORPH_CLOSE, structuring_element(size))


def refine_mask(mask: np.ndarray, min_area: int = 60) -> np.ndarray:
    """Clean tiny regions, fill small holes, and smooth suspicious mask boundaries."""

    binary = (to_grayscale(mask) > 0).astype(np.uint8) * 255
    cleaned = opening(binary, 5)
    cleaned = closing(cleaned, 9)

    count, labels, stats, _ = cv2.connectedComponentsWithStats((cleaned > 0).astype(np.uint8), 8)
    refined = np.zeros_like(cleaned)
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            refined[labels == label] = 255
    return refined


def morphology_panel(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Return before/after morphology views for the UI."""

    return {
        "Before Morphology": (to_grayscale(mask) > 0).astype(np.uint8) * 255,
        "Erosion": erosion(mask),
        "Dilation": dilation(mask),
        "Opening": opening(mask),
        "Closing": closing(mask),
        "After Morphology": refine_mask(mask),
    }
