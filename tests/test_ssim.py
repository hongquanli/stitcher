"""Unit tests for GPU SSIM."""

import numpy as np
import pytest
import sys

sys.path.insert(0, "src")

from tilefusion.utils import compute_ssim
from skimage.metrics import structural_similarity as skimage_ssim


class TestComputeSSIM:
    """Tests for compute_ssim function."""

    def test_ssim_similar_images(self, rng):
        """Test SSIM of similar images matches skimage."""
        arr1 = rng.random((256, 256)).astype(np.float32)
        arr2 = arr1 + rng.random((256, 256)).astype(np.float32) * 0.1

        data_range = arr1.max() - arr1.min()
        cpu = skimage_ssim(arr1, arr2, win_size=15, data_range=data_range)
        gpu = compute_ssim(arr1, arr2, win_size=15)

        assert abs(cpu - gpu) < 0.01, f"SSIM diff {abs(cpu - gpu)} too high"

    def test_ssim_identical_images(self, rng):
        """Test SSIM of identical images is ~1.0."""
        arr = rng.random((256, 256)).astype(np.float32)
        ssim = compute_ssim(arr, arr, win_size=15)
        assert ssim > 0.99, f"SSIM of identical images should be ~1.0, got {ssim}"

    def test_ssim_different_images(self, rng):
        """Test SSIM of random images is low."""
        arr1 = rng.random((256, 256)).astype(np.float32)
        arr2 = rng.random((256, 256)).astype(np.float32)
        ssim = compute_ssim(arr1, arr2, win_size=15)
        assert ssim < 0.5, f"SSIM of random images should be low, got {ssim}"

    def test_ssim_range(self, rng):
        """Test SSIM is in valid range [0, 1]."""
        arr1 = rng.random((128, 128)).astype(np.float32)
        arr2 = rng.random((128, 128)).astype(np.float32)
        ssim = compute_ssim(arr1, arr2, win_size=7)
        assert 0.0 <= ssim <= 1.0, f"SSIM {ssim} outside valid range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
