"""Classical copy-move forgery detection using patch descriptors and KD-tree search."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from src.acquisition import downscale_for_analysis, to_grayscale


@dataclass(frozen=True)
class CopyMoveMatch:
    """A pair of spatially separated similar patches."""

    box_a: tuple[int, int, int, int]
    box_b: tuple[int, int, int, int]
    score: float
    distance: float


@dataclass(frozen=True)
class CopyMoveResult:
    """Copy-move detector output."""

    mask: np.ndarray
    overlay: np.ndarray
    matches: list[CopyMoveMatch]
    message: str
    evidence_score: float


def _patch_descriptors(
    gray: np.ndarray,
    patch_size: int,
    stride: int,
    descriptor_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    descriptors: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []
    height, width = gray.shape

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = gray[y : y + patch_size, x : x + patch_size]
            small = cv2.resize(patch, (descriptor_size, descriptor_size), interpolation=cv2.INTER_AREA)
            raw_vector = small.astype(np.float32).flatten() / 255.0
            centered = small.astype(np.float32).flatten()
            centered -= float(centered.mean())
            std = float(centered.std())
            if std < 12.0:
                continue
            centered /= std
            vector = np.concatenate([raw_vector * 2.0, centered * 0.25])
            descriptors.append(vector)
            positions.append((x, y))

    if not descriptors:
        return np.empty((0, descriptor_size * descriptor_size * 2), dtype=np.float32), np.empty((0, 2), dtype=np.int32)
    return np.vstack(descriptors).astype(np.float32), np.asarray(positions, dtype=np.int32)


def detect_copy_move(
    image: np.ndarray,
    patch_size: int = 24,
    stride: int = 8,
    max_side: int = 760,
    distance_threshold: float = 0.22,
    min_separation: int = 90,
    min_consistent_matches: int = 5,
    max_matches: int = 40,
) -> CopyMoveResult:
    """Detect likely duplicated regions within the same image without O(N^2) search."""

    analysis_image, scale = downscale_for_analysis(image, max_side=max_side)
    gray = to_grayscale(analysis_image)
    if min(gray.shape[:2]) < patch_size * 2:
        empty = np.zeros(gray.shape, dtype=np.uint8)
        return CopyMoveResult(empty, image.copy(), [], "Image is too small for copy-move analysis.", 0.0)

    descriptors, positions = _patch_descriptors(gray, patch_size, stride)
    if len(descriptors) < 12:
        empty = np.zeros(gray.shape, dtype=np.uint8)
        return CopyMoveResult(empty, image.copy(), [], "No strong internal duplication pattern detected.", 0.0)

    tree = cKDTree(descriptors)
    neighbor_count = min(30, len(descriptors))
    distances, indices = tree.query(descriptors, k=neighbor_count)

    candidates: list[tuple[float, int, int]] = []
    for idx, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        for distance, candidate_idx in zip(row_distances, row_indices):
            if candidate_idx == idx or candidate_idx < idx or distance > distance_threshold:
                continue
            separation = float(np.linalg.norm(positions[idx] - positions[candidate_idx]))
            if separation < max(min_separation, patch_size * 1.5):
                continue
            candidates.append((float(distance), idx, int(candidate_idx)))

    clusters: dict[tuple[int, int], list[tuple[float, int, int]]] = {}
    for distance, idx_a, idx_b in candidates:
        dx, dy = positions[idx_b] - positions[idx_a]
        key = (int(round(dx / stride)), int(round(dy / stride)))
        clusters.setdefault(key, []).append((distance, idx_a, idx_b))

    max_cluster_median_distance = min(0.08, distance_threshold * 0.40)
    consistent_clusters = []
    for cluster in clusters.values():
        if len(cluster) < min_consistent_matches:
            continue
        median_distance = float(np.median([item[0] for item in cluster]))
        if median_distance <= max_cluster_median_distance:
            consistent_clusters.append(sorted(cluster, key=lambda item: item[0]))
    consistent_clusters.sort(key=lambda cluster: (-len(cluster), float(np.mean([item[0] for item in cluster]))))
    candidates = []
    for cluster in consistent_clusters[:4]:
        candidates.extend(cluster[: max(3, max_matches // 4)])
    candidates.sort(key=lambda item: item[0])

    if not candidates:
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        return CopyMoveResult(
            full_mask,
            image.copy(),
            [],
            "No strong internal duplication pattern detected.",
            0.0,
        )

    mask_small = np.zeros(gray.shape, dtype=np.uint8)
    overlay_small = analysis_image.copy()
    matches: list[CopyMoveMatch] = []

    for distance, idx_a, idx_b in candidates:
        x1, y1 = positions[idx_a]
        x2, y2 = positions[idx_b]

        cv2.rectangle(mask_small, (x1, y1), (x1 + patch_size, y1 + patch_size), 255, -1)
        cv2.rectangle(mask_small, (x2, y2), (x2 + patch_size, y2 + patch_size), 255, -1)
        cv2.rectangle(overlay_small, (x1, y1), (x1 + patch_size, y1 + patch_size), (0, 255, 255), 2)
        cv2.rectangle(overlay_small, (x2, y2), (x2 + patch_size, y2 + patch_size), (255, 70, 70), 2)
        cv2.line(
            overlay_small,
            (x1 + patch_size // 2, y1 + patch_size // 2),
            (x2 + patch_size // 2, y2 + patch_size // 2),
            (255, 255, 255),
            1,
        )
        score = max(0.0, 1.0 - distance / max(distance_threshold, 1e-6))
        matches.append(
            CopyMoveMatch(
                box_a=(int(x1), int(y1), patch_size, patch_size),
                box_b=(int(x2), int(y2), patch_size, patch_size),
                score=float(score),
                distance=float(distance),
            )
        )
        if len(matches) >= max_matches:
            break

    if not matches:
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        return CopyMoveResult(
            full_mask,
            image.copy(),
            [],
            "No strong internal duplication pattern detected.",
            0.0,
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel)

    if scale != 1.0:
        full_size = (image.shape[1], image.shape[0])
        mask = cv2.resize(mask_small, full_size, interpolation=cv2.INTER_NEAREST)
        overlay = cv2.resize(overlay_small, full_size, interpolation=cv2.INTER_LINEAR)
    else:
        mask = mask_small
        overlay = overlay_small

    evidence_score = min(1.0, 0.18 + 0.035 * len(matches) + float(np.mean([m.score for m in matches])) * 0.45)
    return CopyMoveResult(
        mask=mask,
        overlay=overlay,
        matches=matches,
        message=f"Detected {len(matches)} spatially separated similar patch pair(s).",
        evidence_score=float(evidence_score),
    )
