"""Unit tests for GPU histogram matching."""
import numpy as np
import sys
sys.path.insert(0, "src")

from tilefusion.utils import match_histograms, CUDA_AVAILABLE
from skimage.exposure import match_histograms as skimage_match


def test_histogram_range():
    img = np.random.rand(256, 256).astype(np.float32)
    ref = np.random.rand(256, 256).astype(np.float32) * 2 + 1
    result = match_histograms(img, ref)
    # Output should be in reference range
    assert result.min() >= ref.min() - 0.1
    assert result.max() <= ref.max() + 0.1


def test_histogram_correlation():
    img = np.random.rand(256, 256).astype(np.float32)
    ref = np.random.rand(256, 256).astype(np.float32)

    cpu = skimage_match(img, ref)
    gpu = match_histograms(img, ref)

    cpu_hist, _ = np.histogram(cpu.flatten(), bins=100)
    gpu_hist, _ = np.histogram(gpu.flatten(), bins=100)
    corr = np.corrcoef(cpu_hist, gpu_hist)[0, 1]
    assert corr > 0.99, f"Histogram correlation {corr} too low"


if __name__ == "__main__":
    test_histogram_range()
    test_histogram_correlation()
    print("All tests passed")
