"""Deterministic demo-case generator for IMAGE AUTOPSY presentations."""

from __future__ import annotations

import cv2
import numpy as np


def create_base_image(width: int = 640, height: int = 420) -> np.ndarray:
    """Create a synthetic but image-like RGB scene with texture and objects."""

    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[..., 0] = np.clip(35 + 90 * x + 20 * y, 0, 255)
    base[..., 1] = np.clip(48 + 95 * y, 0, 255)
    base[..., 2] = np.clip(70 + 70 * (1 - x), 0, 255)

    rng = np.random.default_rng(12)
    noise = rng.normal(0, 5, base.shape)
    base = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    cv2.rectangle(base, (62, 70), (230, 260), (70, 130, 185), -1)
    cv2.circle(base, (440, 145), 72, (205, 175, 80), -1)
    cv2.ellipse(base, (415, 310), (120, 46), 8, 0, 360, (92, 170, 112), -1)
    cv2.line(base, (40, 330), (590, 365), (220, 220, 220), 4)
    cv2.putText(base, "CASE", (92, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (235, 235, 235), 4)
    return base


def copy_move_demo(image: np.ndarray) -> np.ndarray:
    manipulated = image.copy()
    rng = np.random.default_rng(42)
    patch = rng.integers(35, 230, size=(96, 96, 3), dtype=np.uint8)
    cv2.circle(patch, (48, 48), 34, (245, 238, 82), -1)
    cv2.line(patch, (12, 16), (84, 76), (24, 32, 42), 5)
    cv2.putText(patch, "CM", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 15, 15), 3)
    manipulated[88:184, 96:192] = patch
    manipulated[248:344, 288:384] = patch
    return manipulated


def brightness_demo(image: np.ndarray) -> np.ndarray:
    manipulated = image.copy()
    region = manipulated[90:245, 360:525].astype(np.float32)
    manipulated[90:245, 360:525] = np.clip(region * 1.18 + 32, 0, 255).astype(np.uint8)
    return manipulated


def contrast_demo(image: np.ndarray) -> np.ndarray:
    manipulated = image.copy()
    region = manipulated[210:350, 80:245].astype(np.float32)
    manipulated[210:350, 80:245] = np.clip((region - 128) * 1.65 + 128, 0, 255).astype(np.uint8)
    return manipulated


def splicing_demo(image: np.ndarray) -> np.ndarray:
    manipulated = image.copy()
    donor = np.zeros((105, 145, 3), dtype=np.uint8)
    donor[:] = (178, 65, 85)
    cv2.circle(donor, (72, 52), 42, (230, 205, 95), -1)
    cv2.putText(donor, "X", (48, 73), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (40, 40, 40), 4)
    mask = np.zeros(donor.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (72, 52), (70, 48), 0, 0, 360, 255, -1)
    roi = manipulated[250:355, 455:600]
    blended = np.where(mask[..., None] > 0, donor, roi)
    manipulated[250:355, 455:600] = blended
    return manipulated


def demo_cases() -> dict[str, np.ndarray]:
    """Return in-memory demo cases that do not depend on the internet."""

    original = create_base_image()
    return {
        "CASE 001 - Original": original,
        "CASE 002 - Copy-Move": copy_move_demo(original),
        "CASE 003 - Local Contrast Manipulation": contrast_demo(original),
        "CASE 004 - Spliced Image": splicing_demo(original),
        "CASE 005 - Local Brightness Manipulation": brightness_demo(original),
    }
