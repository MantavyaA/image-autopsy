import numpy as np

from src.frequency_analysis import fft_magnitude_spectrum, frequency_anomaly_map
from src.wavelet_analysis import haar_decomposition, wavelet_anomaly_map


def test_fft_and_frequency_map_shapes():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = 255

    _, spectrum = fft_magnitude_spectrum(image)
    anomaly = frequency_anomaly_map(image)

    assert spectrum.shape == (64, 64)
    assert anomaly.shape == (64, 64)
    assert spectrum.dtype == np.uint8


def test_wavelet_decomposition_and_anomaly():
    image = np.zeros((64, 64), dtype=np.uint8)
    image[:, 32:] = 180

    subbands = haar_decomposition(image)
    anomaly = wavelet_anomaly_map(image)

    assert set(subbands) == {"LL", "LH", "HL", "HH"}
    assert anomaly.shape == image.shape
    assert anomaly.dtype == np.uint8
