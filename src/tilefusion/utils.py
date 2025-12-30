"""
Shared utilities for tilefusion.

GPU/CPU detection, array operations, and helper functions.
"""

import numpy as np

# GPU detection - PyTorch based
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# CPU fallbacks
from scipy.ndimage import shift as _shift_cpu
from skimage.exposure import match_histograms as _match_histograms_cpu
from skimage.measure import block_reduce
from skimage.metrics import structural_similarity as _ssim_cpu
from skimage.registration import phase_cross_correlation

# Legacy compatibility
USING_GPU = CUDA_AVAILABLE
xp = np
cp = None


def match_histograms(image, reference):
    """
    Match histogram of image to reference using GPU (torch) or CPU (skimage).

    Parameters
    ----------
    image : ndarray
        Image to transform.
    reference : ndarray
        Reference image for histogram matching.

    Returns
    -------
    matched : ndarray
        Image with matched histogram.
    """
    image_np = np.asarray(image)
    reference_np = np.asarray(reference)

    if CUDA_AVAILABLE and image_np.ndim == 2:
        return _match_histograms_torch(image_np, reference_np)

    return _match_histograms_cpu(image_np, reference_np)


def _match_histograms_torch(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """GPU histogram matching using torch operations."""
    # Move to GPU
    img = torch.from_numpy(image.astype(np.float32)).cuda().flatten()
    ref = torch.from_numpy(reference.astype(np.float32)).cuda().flatten()

    # Get sorted indices
    img_sorted, img_indices = torch.sort(img)
    ref_sorted, _ = torch.sort(ref)

    # Create inverse mapping
    inv_indices = torch.empty_like(img_indices)
    inv_indices[img_indices] = torch.arange(len(img), device="cuda")

    # Interpolate reference values at image quantiles
    img_quantiles = torch.linspace(0, 1, len(img), device="cuda")
    ref_quantiles = torch.linspace(0, 1, len(ref), device="cuda")

    # Map image values to reference values via quantile matching
    interp_values = torch.zeros_like(img)
    interp_values[img_indices] = ref_sorted[
        (inv_indices.float() / len(img) * len(ref)).long().clamp(0, len(ref) - 1)
    ]

    return interp_values.reshape(image.shape).cpu().numpy()


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
    """Convert array to numpy."""
    if TORCH_AVAILABLE and torch is not None and isinstance(arr, torch.Tensor):
        return arr.cpu().numpy()
    return np.asarray(arr)


def to_device(arr):
    """Move array to GPU if available."""
    if CUDA_AVAILABLE:
        return torch.from_numpy(np.asarray(arr)).cuda()
    return np.asarray(arr)
