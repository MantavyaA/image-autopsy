import numpy as np

from src.acquisition import get_image_metadata, quantize_image, sample_image
from src.preprocessing import histogram_equalization, log_transform


def test_metadata_and_sampling_quantization():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, 30:] = 200

    metadata = get_image_metadata(image, filename="case.png", file_size_bytes=1200)
    assert metadata.width == 60
    assert metadata.height == 40
    assert metadata.channels == 3

    sampled = sample_image(image, 50)
    assert sampled.shape[:2] == (20, 30)

    quantized = quantize_image(image, 16)
    assert len(np.unique(quantized)) <= 16


def test_histogram_equalization_and_log_transform_shapes():
    gradient = np.tile(np.arange(64, dtype=np.uint8), (64, 1)) * 4
    equalized = histogram_equalization(gradient)
    logged = log_transform(gradient)
    assert equalized.shape == gradient.shape
    assert logged.shape == gradient.shape
    assert equalized.dtype == np.uint8
    assert logged.dtype == np.uint8
