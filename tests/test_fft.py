"""Unit tests for GPU phase_cross_correlation (FFT)."""

import numpy as np
import pytest
import sys

sys.path.insert(0, "src")

from tilefusion.utils import phase_cross_correlation
from skimage.registration import phase_cross_correlation as skimage_pcc


class TestPhaseCorrelation:
    """Tests for phase_cross_correlation function."""

    def test_known_shift(self, rng):
        """Test detection of known integer shift."""
        ref = rng.random((256, 256)).astype(np.float32)

        # Create shifted version: mov is ref shifted by (+5, -3)
        # phase_cross_correlation returns shift to apply to mov to align with ref
        # So it should return (-5, +3)
        mov = np.zeros_like(ref)
        mov[5:, :253] = ref[:-5, 3:]

        shift, _, _ = phase_cross_correlation(ref, mov)

        assert abs(shift[0] - (-5)) < 1, f"Y shift {shift[0]} not close to -5"
        assert abs(shift[1] - 3) < 1, f"X shift {shift[1]} not close to 3"

    def test_zero_shift(self, rng):
        """Test that identical images give zero shift."""
        ref = rng.random((256, 256)).astype(np.float32)

        shift, _, _ = phase_cross_correlation(ref, ref)

        assert abs(shift[0]) < 0.5, f"Y shift {shift[0]} should be ~0"
        assert abs(shift[1]) < 0.5, f"X shift {shift[1]} should be ~0"

    def test_matches_skimage_direction(self, rng):
        """Test that shift direction matches skimage convention."""
        ref = rng.random((128, 128)).astype(np.float32)

        # Shift by rolling
        mov = np.roll(np.roll(ref, 10, axis=0), -7, axis=1)

        gpu_shift, _, _ = phase_cross_correlation(ref, mov)
        cpu_shift, _, _ = skimage_pcc(ref, mov)

        # Directions should match
        assert np.sign(gpu_shift[0]) == np.sign(cpu_shift[0]), "Y direction mismatch"
        assert np.sign(gpu_shift[1]) == np.sign(cpu_shift[1]), "X direction mismatch"


class TestSubpixelRefinement:
    """Tests for subpixel phase correlation refinement."""

    def test_subpixel_refinement(self, rng):
        """Test subpixel accuracy with upsample_factor > 1."""
        ref = rng.random((128, 128)).astype(np.float32)

        # Use integer shift for ground truth (subpixel refinement should still work)
        mov = np.roll(np.roll(ref, 7, axis=0), -4, axis=1)

        # Test with upsample_factor=10 for subpixel refinement
        shift_subpixel, _, _ = phase_cross_correlation(ref, mov, upsample_factor=10)

        # Should detect the shift direction correctly
        assert (
            abs(shift_subpixel[0] - (-7)) < 1
        ), f"Subpixel Y shift {shift_subpixel[0]} not close to -7"
        assert (
            abs(shift_subpixel[1] - 4) < 1
        ), f"Subpixel X shift {shift_subpixel[1]} not close to 4"

        # Verify reasonable range
        assert -10 < shift_subpixel[0] < 0, f"Subpixel Y shift {shift_subpixel[0]} out of range"
        assert 0 < shift_subpixel[1] < 10, f"Subpixel X shift {shift_subpixel[1]} out of range"

    def test_subpixel_vs_integer_consistency(self, rng):
        """Test that subpixel and integer modes give consistent direction."""
        ref = rng.random((64, 64)).astype(np.float32)
        mov = np.roll(np.roll(ref, 3, axis=0), -2, axis=1)

        shift_int, _, _ = phase_cross_correlation(ref, mov, upsample_factor=1)
        shift_sub, _, _ = phase_cross_correlation(ref, mov, upsample_factor=10)

        # Signs should match
        assert np.sign(shift_int[0]) == np.sign(
            shift_sub[0]
        ), "Y direction mismatch between int/subpixel"
        assert np.sign(shift_int[1]) == np.sign(
            shift_sub[1]
        ), "X direction mismatch between int/subpixel"

        # Magnitudes should be close
        assert abs(shift_int[0] - shift_sub[0]) < 1, "Y magnitude differs too much"
        assert abs(shift_int[1] - shift_sub[1]) < 1, "X magnitude differs too much"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
