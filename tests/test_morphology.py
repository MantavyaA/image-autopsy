import numpy as np

from src.morphology import closing, dilation, erosion, opening, refine_mask


def test_morphology_operations_preserve_shape():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    mask[5, 5] = 255

    assert erosion(mask).shape == mask.shape
    assert dilation(mask).shape == mask.shape
    assert opening(mask).shape == mask.shape
    assert closing(mask).shape == mask.shape

    refined = refine_mask(mask, min_area=30)
    assert refined.shape == mask.shape
    assert refined[5, 5] == 0
