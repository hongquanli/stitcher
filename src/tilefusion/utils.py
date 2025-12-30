"""
Shared utilities for tilefusion.

GPU/CPU detection, array operations, and helper functions.
"""

import numpy as np

# GPU detection - PyTorch based
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    CUDA_AVAILABLE = False

# CPU fallbacks
from scipy.ndimage import shift as _shift_cpu
from skimage.exposure import match_histograms
from skimage.measure import block_reduce as _block_reduce_cpu
from skimage.metrics import structural_similarity as _ssim_cpu
from skimage.registration import phase_cross_correlation

# Legacy compatibility
USING_GPU = CUDA_AVAILABLE
xp = np
cp = None


def block_reduce(arr, block_size, func=np.mean):
    """
    Block reduce array using GPU (torch) or CPU (skimage).

    Parameters
    ----------
    arr : ndarray
        Input array (2D or 3D with channel dim first).
    block_size : tuple
        Reduction factors per dimension.
    func : callable
        Reduction function (only np.mean supported on GPU).

    Returns
    -------
    reduced : ndarray
    """
    arr_np = np.asarray(arr)

    if CUDA_AVAILABLE and func == np.mean and arr_np.ndim >= 2:
        return _block_reduce_torch(arr_np, block_size)

    return _block_reduce_cpu(arr_np, block_size, func)


def _block_reduce_torch(arr: np.ndarray, block_size: tuple) -> np.ndarray:
    """GPU block reduce using torch.nn.functional.avg_pool2d."""
    ndim = arr.ndim

    if ndim == 2:
        kernel = (block_size[0], block_size[1])
        t = torch.from_numpy(arr).float().cuda().unsqueeze(0).unsqueeze(0)
        out = torch.nn.functional.avg_pool2d(t, kernel, stride=kernel)
        return out.squeeze().cpu().numpy()

    elif ndim == 3:
        kernel = (block_size[1], block_size[2]) if len(block_size) == 3 else block_size
        t = torch.from_numpy(arr).float().cuda().unsqueeze(0)
        out = torch.nn.functional.avg_pool2d(t, kernel, stride=kernel)
        return out.squeeze(0).cpu().numpy()

    return _block_reduce_cpu(arr, block_size, np.mean)


def shift_array(arr, shift_vec):
    """Shift array using scipy (CPU)."""
    return _shift_cpu(np.asarray(arr), shift=shift_vec, order=1, prefilter=False)


def compute_ssim(arr1, arr2, win_size: int) -> float:
    """SSIM using skimage (CPU)."""
    arr1_np = np.asarray(arr1)
    arr2_np = np.asarray(arr2)
    data_range = float(arr1_np.max() - arr1_np.min())
    if data_range == 0:
        data_range = 1.0
    return float(_ssim_cpu(arr1_np, arr2_np, win_size=win_size, data_range=data_range))


def make_1d_profile(length: int, blend: int) -> np.ndarray:
    """Create a linear ramp profile over `blend` pixels at each end."""
    blend = min(blend, length // 2)
    prof = np.ones(length, dtype=np.float32)
    if blend > 0:
        ramp = np.linspace(0, 1, blend, endpoint=False, dtype=np.float32)
        prof[:blend] = ramp
        prof[-blend:] = ramp[::-1]
    return prof


def to_numpy(arr):
    """Convert array to numpy, handling both CPU and GPU arrays."""
    return np.asarray(arr)


def to_device(arr):
    """Move array to current device (GPU if available, else CPU)."""
    return np.asarray(arr)
