"""Unit tests for GPU shift_array."""

import numpy as np
import pytest
import sys

sys.path.insert(0, "src")

from tilefusion.utils import shift_array
from scipy.ndimage import shift as scipy_shift


class TestShiftArray:
    """Tests for shift_array function."""

    def test_integer_shift(self, rng):
        """Test integer shift matches scipy."""
        arr = rng.random((256, 256)).astype(np.float32)
        cpu = scipy_shift(arr, (3.0, -5.0), order=1, prefilter=False)
        gpu = shift_array(arr, (3.0, -5.0))
        np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-4)

    def test_subpixel_mean_error(self, rng):
        """Test subpixel shift has low mean error vs scipy."""
        arr = rng.random((256, 256)).astype(np.float32)
        cpu = scipy_shift(arr, (5.5, -3.2), order=1, prefilter=False)
        gpu = shift_array(arr, (5.5, -3.2))
        mean_diff = np.abs(cpu - gpu).mean()
        assert mean_diff < 0.01, f"Mean diff {mean_diff} too high"

    def test_zero_shift(self, rng):
        """Test zero shift returns nearly identical array."""
        arr = rng.random((256, 256)).astype(np.float32)
        result = shift_array(arr, (0.0, 0.0))
        # Allow small tolerance due to grid_sample interpolation
        np.testing.assert_allclose(result, arr, rtol=1e-4, atol=1e-4)

    def test_small_array(self, rng):
        """Test shift works on small arrays (edge case)."""
        arr = rng.random((4, 4)).astype(np.float32)
        result = shift_array(arr, (1.0, 1.0))
        assert result.shape == arr.shape

    def test_1pixel_fallback(self):
        """Test 1-pixel array falls back to CPU."""
        arr = np.array([[1.0]], dtype=np.float32)
        result = shift_array(arr, (0.5, 0.5))
        assert result.shape == (1, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
