"""Classical segmentation and connected-component analysis."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.acquisition import to_grayscale


@dataclass(frozen=True)
class Region:
    """Connected suspicious region summary."""

    label: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area_percent: float
    severity: str
    score: float


def global_threshold(gray_image: np.ndarray, threshold: int = 128) -> np.ndarray:
    gray = to_grayscale(gray_image)
    _, mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
    return mask


def otsu_threshold(gray_image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(gray_image)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def adaptive_threshold(gray_image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(gray_image)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        -5,
    )


def percentile_anomaly_threshold(anomaly_map: np.ndarray, percentile: float = 88.0) -> np.ndarray:
    """Threshold anomaly maps by high percentile, robust to different score scales."""

    gray = to_grayscale(anomaly_map)
    cutoff = np.percentile(gray, percentile)
    _, mask = cv2.threshold(gray, float(cutoff), 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


def connected_components(
    mask: np.ndarray,
    score_map: np.ndarray | None = None,
    min_area_percent: float = 0.15,
) -> list[Region]:
    """Extract and rank connected suspicious components."""

    binary = (to_grayscale(mask) > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    total_area = binary.shape[0] * binary.shape[1]
    regions: list[Region] = []
    score_gray = to_grayscale(score_map) if score_map is not None else None

    for label in range(1, count):
        x, y, w, h, area = stats[label]
        area_percent = 100.0 * float(area) / float(total_area)
        if area_percent < min_area_percent:
            continue

        if score_gray is not None:
            component_scores = score_gray[labels == label]
            score = float(np.mean(component_scores) / 255.0)
        else:
            score = min(1.0, area_percent / 20.0)

        severity = "High" if score >= 0.66 else "Medium" if score >= 0.38 else "Low"
        regions.append(
            Region(
                label=label,
                area=int(area),
                bbox=(int(x), int(y), int(w), int(h)),
                centroid=(float(centroids[label][0]), float(centroids[label][1])),
                area_percent=float(area_percent),
                severity=severity,
                score=score,
            )
        )

    return sorted(regions, key=lambda region: (region.score, region.area), reverse=True)


def draw_regions(image: np.ndarray, regions: list[Region]) -> np.ndarray:
    """Draw bounding boxes and labels for suspicious connected components."""

    overlay = image.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    for index, region in enumerate(regions, start=1):
        x, y, w, h = region.bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 35, 35), 2)
        cv2.putText(
            overlay,
            f"R{index}",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay
