import numpy as np

from src.segmentation import connected_components, otsu_threshold, percentile_anomaly_threshold


def test_thresholding_and_connected_components():
    image = np.zeros((80, 80), dtype=np.uint8)
    image[20:45, 25:55] = 220

    otsu = otsu_threshold(image)
    percentile = percentile_anomaly_threshold(image, 85)
    regions = connected_components(percentile, image, min_area_percent=1.0)

    assert otsu.shape == image.shape
    assert percentile.shape == image.shape
    assert len(regions) == 1
    assert regions[0].area_percent > 5.0
