"""Unit tests for GPU block_reduce."""

import numpy as np
import pytest
import sys
from skimage.measure import block_reduce as skimage_block_reduce

sys.path.insert(0, "src")

from tilefusion.utils import block_reduce


class TestBlockReduce:
    """Test block_reduce GPU vs CPU equivalence."""

    def test_2d_basic(self, rng):
        """Test 2D block reduce matches skimage."""
        arr = rng.random((256, 256)).astype(np.float32)
        block_size = (4, 4)

        result = block_reduce(arr, block_size, np.mean)
        expected = skimage_block_reduce(arr, block_size, np.mean)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_2d_large(self, rng):
        """Test larger 2D array."""
        arr = rng.random((1024, 1024)).astype(np.float32)
        block_size = (8, 8)

        result = block_reduce(arr, block_size, np.mean)
        expected = skimage_block_reduce(arr, block_size, np.mean)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_3d_multichannel(self, rng):
        """Test 3D array with channel dimension."""
        arr = rng.random((3, 256, 256)).astype(np.float32)
        block_size = (1, 4, 4)

        result = block_reduce(arr, block_size, np.mean)
        expected = skimage_block_reduce(arr, block_size, np.mean)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_output_shape(self, rng):
        """Test output shape is correct."""
        arr = rng.random((512, 512)).astype(np.float32)
        block_size = (4, 4)

        result = block_reduce(arr, block_size, np.mean)

        assert result.shape == (128, 128)

    def test_non_divisible_shape(self, rng):
        """Test block reduce with non-divisible dimensions."""
        arr = rng.random((100, 100)).astype(np.float32)
        block_size = (8, 8)

        result = block_reduce(arr, block_size, np.mean)
        expected = skimage_block_reduce(arr, block_size, np.mean)

        np.testing.assert_allclose(result, expected, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
