"""Unit tests for GPU phase_cross_correlation (FFT)."""

import numpy as np
import sys

sys.path.insert(0, "src")

from tilefusion.utils import phase_cross_correlation, CUDA_AVAILABLE
from skimage.registration import phase_cross_correlation as skimage_pcc


def test_known_shift():
    """Test detection of known integer shift."""
    np.random.seed(42)
    ref = np.random.rand(256, 256).astype(np.float32)

    # Create shifted version: mov is ref shifted by (+5, -3)
    # phase_cross_correlation returns shift to apply to mov to align with ref
    # So it should return (-5, +3)
    mov = np.zeros_like(ref)
    mov[5:, :253] = ref[:-5, 3:]

    shift, _, _ = phase_cross_correlation(ref, mov)

    # Should detect shift close to (-5, +3)
    assert abs(shift[0] - (-5)) < 1, f"Y shift {shift[0]} not close to -5"
    assert abs(shift[1] - 3) < 1, f"X shift {shift[1]} not close to 3"


def test_zero_shift():
    """Test that identical images give zero shift."""
    np.random.seed(42)
    ref = np.random.rand(256, 256).astype(np.float32)

    shift, _, _ = phase_cross_correlation(ref, ref)

    assert abs(shift[0]) < 0.5, f"Y shift {shift[0]} should be ~0"
    assert abs(shift[1]) < 0.5, f"X shift {shift[1]} should be ~0"


def test_matches_skimage_direction():
    """Test that shift direction matches skimage convention."""
    np.random.seed(42)
    ref = np.random.rand(128, 128).astype(np.float32)

    # Shift by rolling
    mov = np.roll(np.roll(ref, 10, axis=0), -7, axis=1)

    gpu_shift, _, _ = phase_cross_correlation(ref, mov)
    cpu_shift, _, _ = skimage_pcc(ref, mov)

    # Directions should match
    assert np.sign(gpu_shift[0]) == np.sign(cpu_shift[0]), "Y direction mismatch"
    assert np.sign(gpu_shift[1]) == np.sign(cpu_shift[1]), "X direction mismatch"


def test_subpixel_refinement():
    """Test subpixel accuracy with upsample_factor > 1."""
    np.random.seed(42)
    ref = np.random.rand(128, 128).astype(np.float32)

    # Use integer shift for ground truth (subpixel refinement should still work)
    mov = np.roll(np.roll(ref, 7, axis=0), -4, axis=1)

    # Test with upsample_factor=10 for subpixel refinement
    shift_subpixel, _, _ = phase_cross_correlation(ref, mov, upsample_factor=10)
    shift_integer, _, _ = phase_cross_correlation(ref, mov, upsample_factor=1)

    # Both should detect the shift direction correctly
    assert (
        abs(shift_subpixel[0] - (-7)) < 1
    ), f"Subpixel Y shift {shift_subpixel[0]} not close to -7"
    assert abs(shift_subpixel[1] - 4) < 1, f"Subpixel X shift {shift_subpixel[1]} not close to 4"

    # Subpixel should give fractional values (may have decimal component)
    # Just verify it returns reasonable values
    assert -10 < shift_subpixel[0] < 0, f"Subpixel Y shift {shift_subpixel[0]} out of range"
    assert 0 < shift_subpixel[1] < 10, f"Subpixel X shift {shift_subpixel[1]} out of range"


def test_subpixel_vs_integer_consistency():
    """Test that subpixel and integer modes give consistent direction."""
    np.random.seed(123)
    ref = np.random.rand(64, 64).astype(np.float32)
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
    print(f"CUDA available: {CUDA_AVAILABLE}")
    test_known_shift()
    print("test_known_shift: PASSED")
    test_zero_shift()
    print("test_zero_shift: PASSED")
    test_matches_skimage_direction()
    print("test_matches_skimage_direction: PASSED")
    test_subpixel_refinement()
    print("test_subpixel_refinement: PASSED")
    test_subpixel_vs_integer_consistency()
    print("test_subpixel_vs_integer_consistency: PASSED")
    print("All tests passed!")
