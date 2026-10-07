"""Evidence fusion and heuristic forensic interpretation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.acquisition import to_grayscale


DEFAULT_WEIGHTS: dict[str, float] = {
    "texture": 0.20,
    "edge": 0.15,
    "frequency": 0.15,
    "wavelet": 0.15,
    "copy_move": 0.25,
    "local_statistical": 0.10,
}


@dataclass(frozen=True)
class FusionResult:
    """Final fused evidence output."""

    fusion_map: np.ndarray
    heatmap: np.ndarray
    overlay: np.ndarray
    score: int
    verdict: str
    likely_manipulation: str
    channel_scores: dict[str, float]
    evidence_flags: list[str]
    disclaimer: str


def normalize_map(map_image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Normalize an evidence map to [0, 1] and resize to target shape."""

    gray = to_grayscale(map_image).astype(np.float32)
    if gray.shape != shape:
        gray = cv2.resize(gray, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    min_val, max_val = float(gray.min()), float(gray.max())
    if max_val == min_val:
        return np.zeros(shape, dtype=np.float32)
    return (gray - min_val) / (max_val - min_val)


def suspicion_band(score: int) -> str:
    if score <= 20:
        return "Low evidence of manipulation"
    if score <= 40:
        return "Weak evidence"
    if score <= 60:
        return "Moderate suspicion"
    if score <= 80:
        return "High suspicion"
    return "Very high suspicion"


def classify_manipulation(channel_scores: dict[str, float]) -> str:
    copy_move = channel_scores.get("copy_move", 0.0)
    texture = channel_scores.get("texture", 0.0)
    edge = channel_scores.get("edge", 0.0)
    frequency = channel_scores.get("frequency", 0.0)
    wavelet = channel_scores.get("wavelet", 0.0)
    local = channel_scores.get("local_statistical", 0.0)

    strong_channels = sum(value > 0.45 for value in channel_scores.values())
    if copy_move > 0.42:
        return "Possible copy-move manipulation"
    if texture > 0.45 and edge > 0.38 and local > 0.38:
        return "Possible splicing/insertion"
    if texture > 0.42 and (frequency > 0.38 or wavelet > 0.38):
        return "Possible local enhancement/edit"
    if strong_channels >= 3:
        return "Multiple suspicious inconsistencies"
    if max(channel_scores.values() or [0.0]) < 0.25:
        return "No strong evidence of manipulation"
    return "Inconclusive"


def _channel_score(norm_map: np.ndarray) -> float:
    high = norm_map[norm_map >= np.percentile(norm_map, 88)]
    if high.size == 0:
        return 0.0
    return float(np.clip(np.mean(high), 0.0, 1.0))


def fuse_evidence(
    original_rgb: np.ndarray,
    evidence_maps: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
    copy_move_score: float = 0.0,
) -> FusionResult:
    """Fuse normalized evidence maps into a heuristic forensic suspicion score."""

    weights = weights or DEFAULT_WEIGHTS
    shape = original_rgb.shape[:2]
    normalized: dict[str, np.ndarray] = {}
    channel_scores: dict[str, float] = {}

    for name, map_image in evidence_maps.items():
        norm = normalize_map(map_image, shape)
        normalized[name] = norm
        channel_scores[name] = _channel_score(norm)

    if "copy_move" in normalized:
        channel_scores["copy_move"] = max(float(copy_move_score), channel_scores.get("copy_move", 0.0))

    fusion = np.zeros(shape, dtype=np.float32)
    total_weight = 0.0
    for name, weight in weights.items():
        if name not in normalized:
            continue
        fusion += normalized[name] * float(weight)
        total_weight += float(weight)

    if total_weight > 0:
        fusion /= total_weight

    map_component = float(np.mean(fusion) * 0.40 + np.percentile(fusion, 94) * 0.60)
    weighted_channel_sum = 0.0
    weighted_channel_total = 0.0
    for name, weight in weights.items():
        if name in channel_scores:
            weighted_channel_sum += channel_scores[name] * float(weight)
            weighted_channel_total += float(weight)
    channel_component = weighted_channel_sum / weighted_channel_total if weighted_channel_total else 0.0
    score_float = 100.0 * float(np.clip(0.58 * map_component + 0.42 * channel_component, 0, 1))
    copy_move_channel = channel_scores.get("copy_move", 0.0)
    if copy_move_channel > 0.55:
        region_significance = 55.0 + 25.0 * ((copy_move_channel - 0.55) / 0.45)
        score_float = max(score_float, region_significance)
    score = int(round(score_float))
    fusion_uint8 = np.clip(fusion * 255, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(fusion_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_rgb.astype(np.uint8), 0.62, heatmap, 0.38, 0)

    flags: list[str] = []
    label_map = {
        "edge": "Edge inconsistency detected",
        "texture": "Texture anomaly detected",
        "frequency": "Frequency-domain anomaly detected",
        "wavelet": "Wavelet-domain anomaly detected",
        "copy_move": "Repeated internal region detected",
        "local_statistical": "Local statistical inconsistency detected",
    }
    for name, value in channel_scores.items():
        if value > 0.38 and name in label_map:
            flags.append(label_map[name])
    if not flags:
        flags.append("No individual channel produced strong evidence")

    return FusionResult(
        fusion_map=fusion_uint8,
        heatmap=heatmap,
        overlay=overlay,
        score=score,
        verdict=suspicion_band(score),
        likely_manipulation=classify_manipulation(channel_scores),
        channel_scores=channel_scores,
        evidence_flags=flags,
        disclaimer=(
            "The score is a heuristic evidence-aggregation score for academic demonstration "
            "and is not a certified forensic probability."
        ),
    )
