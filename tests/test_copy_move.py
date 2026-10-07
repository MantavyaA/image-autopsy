import numpy as np

from src.copy_move import detect_copy_move


def test_copy_move_detects_synthetic_duplicate():
    image = np.zeros((180, 220, 3), dtype=np.uint8)
    patch = np.zeros((48, 48, 3), dtype=np.uint8)
    for row in range(48):
        patch[row, :, 0] = row * 4
        patch[:, row, 1] = row * 3
    patch[..., 2] = 120

    image[32:80, 32:80] = patch
    image[112:160, 136:184] = patch

    result = detect_copy_move(
        image,
        patch_size=24,
        stride=8,
        distance_threshold=5.0,
        min_separation=50,
        max_matches=10,
    )

    assert result.evidence_score > 0
    assert len(result.matches) > 0
    assert result.mask.sum() > 0
