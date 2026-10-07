import numpy as np

from src.evidence_fusion import fuse_evidence


def test_evidence_fusion_outputs_score_and_maps():
    original = np.zeros((80, 80, 3), dtype=np.uint8)
    evidence = np.zeros((80, 80), dtype=np.uint8)
    evidence[20:45, 20:45] = 255
    maps = {
        "texture": evidence,
        "edge": evidence,
        "frequency": evidence,
        "wavelet": evidence,
        "copy_move": np.zeros_like(evidence),
        "local_statistical": evidence,
    }

    result = fuse_evidence(original, maps)

    assert result.fusion_map.shape == (80, 80)
    assert result.heatmap.shape == original.shape
    assert result.overlay.shape == original.shape
    assert 0 <= result.score <= 100
