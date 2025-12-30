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


if __name__ == "__main__":
    print(f"CUDA available: {CUDA_AVAILABLE}")
    test_known_shift()
    print("test_known_shift: PASSED")
    test_zero_shift()
    print("test_zero_shift: PASSED")
    test_matches_skimage_direction()
    print("test_matches_skimage_direction: PASSED")
    print("All tests passed!")
