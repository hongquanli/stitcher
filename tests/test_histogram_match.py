"""Unit tests for GPU histogram matching."""

import numpy as np
import pytest
import sys

sys.path.insert(0, "src")

from tilefusion.utils import match_histograms
from skimage.exposure import match_histograms as skimage_match


class TestMatchHistograms:
    """Tests for match_histograms function."""

    def test_histogram_range(self, rng):
        """Test output is in reference range."""
        img = rng.random((256, 256)).astype(np.float32)
        ref = rng.random((256, 256)).astype(np.float32) * 2 + 1
        result = match_histograms(img, ref)
        # Output should be in reference range
        assert result.min() >= ref.min() - 0.1
        assert result.max() <= ref.max() + 0.1

    def test_histogram_correlation(self, rng):
        """Test histogram correlation with skimage."""
        img = rng.random((256, 256)).astype(np.float32)
        ref = rng.random((256, 256)).astype(np.float32)

        cpu = skimage_match(img, ref)
        gpu = match_histograms(img, ref)

        cpu_hist, _ = np.histogram(cpu.flatten(), bins=100)
        gpu_hist, _ = np.histogram(gpu.flatten(), bins=100)
        corr = np.corrcoef(cpu_hist, gpu_hist)[0, 1]
        assert corr > 0.99, f"Histogram correlation {corr} too low"

    def test_same_image(self, rng):
        """Test matching image to itself."""
        img = rng.random((128, 128)).astype(np.float32)
        result = match_histograms(img, img)
        np.testing.assert_allclose(result, img, rtol=1e-5)

    def test_different_sizes(self, rng):
        """Test matching images of different sizes."""
        img = rng.random((64, 64)).astype(np.float32)
        ref = rng.random((128, 128)).astype(np.float32)
        result = match_histograms(img, ref)
        assert result.shape == img.shape

    def test_pixel_values_match_skimage(self, rng):
        """Test pixel-by-pixel matching against skimage."""
        img = rng.random((64, 64)).astype(np.float32)
        ref = rng.random((64, 64)).astype(np.float32) * 2 + 1

        cpu = skimage_match(img, ref)
        gpu = match_histograms(img, ref)

        # Pixel values should be close (not just histogram shape)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
