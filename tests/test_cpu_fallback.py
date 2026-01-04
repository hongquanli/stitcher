"""Tests for CPU fallback paths and dtype preservation."""

import numpy as np
import pytest
import sys

sys.path.insert(0, "src")

from tilefusion.utils import (
    phase_cross_correlation,
    shift_array,
    match_histograms,
    block_reduce,
    compute_ssim,
)


class TestCPUFallback:
    """Test that CPU fallback paths work correctly."""

    def test_phase_cross_correlation_cpu(self, force_cpu, rng):
        """Test phase_cross_correlation with CPU fallback."""
        ref = rng.random((128, 128)).astype(np.float32)
        mov = np.roll(ref, 5, axis=0)

        shift, error, phasediff = phase_cross_correlation(ref, mov)

        assert abs(shift[0] - (-5)) < 1, f"Y shift {shift[0]} not close to -5"

    def test_shift_array_cpu(self, force_cpu, rng):
        """Test shift_array with CPU fallback."""
        arr = rng.random((128, 128)).astype(np.float32)
        result = shift_array(arr, (3.0, -2.0))

        assert result.shape == arr.shape
        assert result.dtype == arr.dtype

    def test_match_histograms_cpu(self, force_cpu, rng):
        """Test match_histograms with CPU fallback."""
        img = rng.random((128, 128)).astype(np.float32)
        ref = rng.random((128, 128)).astype(np.float32) * 2

        result = match_histograms(img, ref)

        assert result.shape == img.shape

    def test_block_reduce_cpu(self, force_cpu, rng):
        """Test block_reduce with CPU fallback."""
        arr = rng.random((128, 128)).astype(np.float32)
        result = block_reduce(arr, (4, 4), np.mean)

        assert result.shape == (32, 32)

    def test_compute_ssim_cpu(self, force_cpu, rng):
        """Test compute_ssim with CPU fallback."""
        arr1 = rng.random((128, 128)).astype(np.float32)
        arr2 = arr1 + rng.random((128, 128)).astype(np.float32) * 0.1

        ssim = compute_ssim(arr1, arr2, win_size=7)

        assert 0.0 <= ssim <= 1.0


class TestDtypePreservation:
    """Test that dtype is preserved when preserve_dtype=True."""

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
    def test_shift_array_dtype(self, dtype, force_cpu, rng):
        """Test shift_array preserves dtype."""
        arr = (rng.random((64, 64)) * 255).astype(dtype)
        result = shift_array(arr, (1.5, -1.5), preserve_dtype=True)

        assert result.dtype == dtype, f"Expected {dtype}, got {result.dtype}"

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
    def test_match_histograms_dtype(self, dtype, force_cpu, rng):
        """Test match_histograms preserves dtype."""
        img = (rng.random((64, 64)) * 255).astype(dtype)
        ref = (rng.random((64, 64)) * 255).astype(dtype)
        result = match_histograms(img, ref, preserve_dtype=True)

        assert result.dtype == dtype, f"Expected {dtype}, got {result.dtype}"

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
    def test_block_reduce_dtype(self, dtype, force_cpu, rng):
        """Test block_reduce preserves dtype."""
        arr = (rng.random((64, 64)) * 255).astype(dtype)
        result = block_reduce(arr, (4, 4), np.mean, preserve_dtype=True)

        assert result.dtype == dtype, f"Expected {dtype}, got {result.dtype}"

    def test_shift_array_no_preserve(self, force_cpu, rng):
        """Test shift_array returns float when preserve_dtype=False."""
        arr = (rng.random((64, 64)) * 255).astype(np.uint16)
        result = shift_array(arr, (1.5, -1.5), preserve_dtype=False)

        # Should return float64 (scipy default)
        assert result.dtype in [np.float32, np.float64]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_shift_zero(self, force_cpu, rng):
        """Test that zero shift returns nearly identical array."""
        arr = rng.random((64, 64)).astype(np.float32)
        result = shift_array(arr, (0.0, 0.0))

        np.testing.assert_allclose(result, arr, rtol=1e-5, atol=1e-5)

    def test_identical_images_ssim(self, force_cpu, rng):
        """Test SSIM of identical images is ~1.0."""
        arr = rng.random((64, 64)).astype(np.float32)
        ssim = compute_ssim(arr, arr, win_size=7)

        assert ssim > 0.99, f"SSIM of identical images should be ~1.0, got {ssim}"

    def test_block_reduce_3d(self, force_cpu, rng):
        """Test block_reduce with 3D array."""
        arr = rng.random((3, 64, 64)).astype(np.float32)
        result = block_reduce(arr, (1, 4, 4), np.mean)

        assert result.shape == (3, 16, 16)

    def test_different_size_histogram_match(self, force_cpu, rng):
        """Test histogram matching with different sized images."""
        img = rng.random((64, 64)).astype(np.float32)
        ref = rng.random((128, 128)).astype(np.float32)

        result = match_histograms(img, ref)

        assert result.shape == img.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
