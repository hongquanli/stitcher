"""
Shared utilities for tilefusion.

GPU/CPU detection, array operations, and helper functions.
All functions support GPU acceleration via PyTorch with automatic CPU fallback.
"""

from typing import Any, Callable

import numpy as np

__all__ = [
    # GPU detection flags
    "TORCH_AVAILABLE",
    "CUDA_AVAILABLE",
    "USING_GPU",
    # Array module (legacy compatibility)
    "xp",
    "cp",
    # Core functions
    "phase_cross_correlation",
    "shift_array",
    "match_histograms",
    "block_reduce",
    "compute_ssim",
    # Utility functions
    "make_1d_profile",
    "to_numpy",
    "to_device",
]

# GPU detection - PyTorch based
try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# CPU fallbacks
from scipy.ndimage import shift as _shift_cpu
from skimage.exposure import match_histograms as _match_histograms_cpu
from skimage.measure import block_reduce as _block_reduce_cpu
from skimage.metrics import structural_similarity as _ssim_cpu
from skimage.registration import phase_cross_correlation as _phase_cross_correlation_cpu

# Legacy compatibility - used by core.py and registration.py
# xp: array module (numpy, since cupy was removed)
# cp: cupy module (always None now, kept for API compatibility)
# USING_GPU: exported in __init__.py for user code
USING_GPU = CUDA_AVAILABLE
xp = np
cp = None  # cupy removed; GPU ops now use PyTorch internally

# Constants
_FFT_EPS = 1e-10  # Epsilon for FFT normalization to avoid division by zero
_PARABOLIC_EPS = 1e-10  # Epsilon for parabolic fit denominator check
_SSIM_K1 = 0.01  # SSIM constant K1 (luminance)
_SSIM_K2 = 0.03  # SSIM constant K2 (contrast)


# =============================================================================
# Phase Cross-Correlation (GPU FFT)
# =============================================================================


def phase_cross_correlation(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    upsample_factor: int = 1,
    **kwargs,
) -> tuple[np.ndarray, float, float]:
    """
    Phase cross-correlation using GPU (torch FFT) or CPU (skimage).

    Parameters
    ----------
    reference_image : ndarray
        Reference image.
    moving_image : ndarray
        Image to register.
    upsample_factor : int
        Upsampling factor for subpixel precision.

    Returns
    -------
    shift : ndarray
        Shift vector (y, x).
    error : float
        Translation invariant normalized RMS error.
        Note: GPU path returns 0.0 (not computed).
    phasediff : float
        Global phase difference.
        Note: GPU path returns 0.0 (not computed).
    """
    ref_np = np.asarray(reference_image)
    mov_np = np.asarray(moving_image)

    if CUDA_AVAILABLE and ref_np.ndim == 2:
        return _phase_cross_correlation_torch(ref_np, mov_np, upsample_factor)
    return _phase_cross_correlation_cpu(ref_np, mov_np, upsample_factor=upsample_factor, **kwargs)


def _phase_cross_correlation_torch(
    reference_image: np.ndarray, moving_image: np.ndarray, upsample_factor: int = 1
) -> tuple:
    """GPU phase cross-correlation using torch FFT."""
    ref = torch.from_numpy(reference_image.astype(np.float32)).cuda()
    mov = torch.from_numpy(moving_image.astype(np.float32)).cuda()

    # Compute cross-power spectrum
    ref_fft = torch.fft.fft2(ref)
    mov_fft = torch.fft.fft2(mov)
    cross_power = ref_fft * torch.conj(mov_fft)
    cross_power = cross_power / (torch.abs(cross_power) + _FFT_EPS)

    # Inverse FFT to get correlation
    correlation = torch.fft.ifft2(cross_power).real

    # Find peak
    max_idx = torch.argmax(correlation)
    h, w = correlation.shape
    peak_y = (max_idx // w).item()
    peak_x = (max_idx % w).item()

    # Handle wraparound for negative shifts
    if peak_y > h // 2:
        peak_y -= h
    if peak_x > w // 2:
        peak_x -= w

    shift = np.array([float(peak_y), float(peak_x)])

    # Subpixel refinement if requested
    if upsample_factor > 1:
        shift = _subpixel_refine_torch(correlation, peak_y, peak_x, h, w)

    return shift, 0.0, 0.0


def _subpixel_refine_torch(correlation, peak_y, peak_x, h, w):
    """Subpixel refinement using parabolic fit around peak."""
    py = peak_y % h
    px = peak_x % w

    y_indices = [(py - 1) % h, py, (py + 1) % h]
    x_indices = [(px - 1) % w, px, (px + 1) % w]

    neighborhood = torch.zeros(3, 3, device="cuda")
    for i, yi in enumerate(y_indices):
        for j, xj in enumerate(x_indices):
            neighborhood[i, j] = correlation[yi, xj]

    center_val = neighborhood[1, 1].item()

    # Y direction parabolic fit
    if neighborhood[0, 1].item() != center_val or neighborhood[2, 1].item() != center_val:
        denom = 2 * (2 * center_val - neighborhood[0, 1].item() - neighborhood[2, 1].item())
        dy = (
            (neighborhood[0, 1].item() - neighborhood[2, 1].item()) / denom
            if abs(denom) > _PARABOLIC_EPS
            else 0.0
        )
    else:
        dy = 0.0

    # X direction parabolic fit
    if neighborhood[1, 0].item() != center_val or neighborhood[1, 2].item() != center_val:
        denom = 2 * (2 * center_val - neighborhood[1, 0].item() - neighborhood[1, 2].item())
        dx = (
            (neighborhood[1, 0].item() - neighborhood[1, 2].item()) / denom
            if abs(denom) > _PARABOLIC_EPS
            else 0.0
        )
    else:
        dx = 0.0

    dy = max(-0.5, min(0.5, dy))
    dx = max(-0.5, min(0.5, dx))

    return np.array([float(peak_y) + dy, float(peak_x) + dx])


# =============================================================================
# Shift Array (GPU grid_sample)
# =============================================================================


def shift_array(
    arr: np.ndarray,
    shift_vec: tuple[float, float],
    preserve_dtype: bool = True,
) -> np.ndarray:
    """
    Shift array by subpixel amounts using GPU (torch) or CPU (scipy).

    Parameters
    ----------
    arr : ndarray
        2D input array.
    shift_vec : tuple[float, float]
        (dy, dx) shift amounts.
    preserve_dtype : bool
        If True, output dtype matches input dtype. Default True.

    Returns
    -------
    shifted : ndarray
        Shifted array, same shape as input.
    """
    arr_np = np.asarray(arr)
    original_dtype = arr_np.dtype

    if CUDA_AVAILABLE and arr_np.ndim == 2:
        result = _shift_array_torch(arr_np, shift_vec)
    else:
        # Compute in float for consistency with GPU path
        arr_float = arr_np.astype(np.float64)
        result = _shift_cpu(arr_float, shift=shift_vec, order=1, prefilter=False)

    if preserve_dtype and result.dtype != original_dtype:
        return result.astype(original_dtype)
    return result


def _shift_array_torch(arr: np.ndarray, shift_vec: tuple[float, float]) -> np.ndarray:
    """GPU shift using torch.nn.functional.grid_sample."""
    h, w = arr.shape

    # Guard against degenerate arrays (need at least 2 pixels for interpolation)
    if h < 2 or w < 2:
        return _shift_cpu(arr, shift=shift_vec, order=1, prefilter=False)

    dy, dx = float(shift_vec[0]), float(shift_vec[1])

    # Create pixel coordinate grids
    y_coords = torch.arange(h, device="cuda", dtype=torch.float32)
    x_coords = torch.arange(w, device="cuda", dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Apply shift: to shift output by (dy, dx), sample from (y-dy, x-dx)
    sample_y = grid_y - dy
    sample_x = grid_x - dx

    # Normalize to [-1, 1] for grid_sample (align_corners=True)
    sample_x = 2 * sample_x / (w - 1) - 1
    sample_y = 2 * sample_y / (h - 1) - 1

    # Stack to (H, W, 2) with (x, y) order, add batch dim -> (1, H, W, 2)
    grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)

    # Input: (1, 1, H, W)
    t = torch.from_numpy(arr).float().cuda().unsqueeze(0).unsqueeze(0)

    # grid_sample with bilinear interpolation
    out = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    return out.squeeze().cpu().numpy()


# =============================================================================
# Match Histograms (GPU sort/quantile)
# =============================================================================


def match_histograms(
    image: np.ndarray,
    reference: np.ndarray,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """
    Match histogram of image to reference using GPU (torch) or CPU (skimage).

    Parameters
    ----------
    image : ndarray
        Image to transform.
    reference : ndarray
        Reference image for histogram matching.
    preserve_dtype : bool
        If True, output dtype matches input dtype. Default True.

    Returns
    -------
    matched : ndarray
        Image with matched histogram.
    """
    image_np = np.asarray(image)
    reference_np = np.asarray(reference)
    original_dtype = image_np.dtype

    if CUDA_AVAILABLE and image_np.ndim == 2:
        result = _match_histograms_torch(image_np, reference_np)
    else:
        result = _match_histograms_cpu(image_np, reference_np)

    if preserve_dtype and result.dtype != original_dtype:
        return result.astype(original_dtype)
    return result


def _match_histograms_torch(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """GPU histogram matching using torch operations."""
    # Move to GPU
    img = torch.from_numpy(image.astype(np.float32)).cuda().flatten()
    ref = torch.from_numpy(reference.astype(np.float32)).cuda().flatten()

    # Get sorted indices
    _, img_indices = torch.sort(img)
    ref_sorted, _ = torch.sort(ref)

    # Create inverse mapping (rank of each pixel)
    inv_indices = torch.empty_like(img_indices)
    inv_indices[img_indices] = torch.arange(len(img), device="cuda")

    # Map image values to reference values via quantile matching
    # inv_indices[i] = rank of pixel i, so look up ref value at that scaled rank
    interp_values = ref_sorted[
        (inv_indices.float() / len(img) * len(ref)).long().clamp(0, len(ref) - 1)
    ]

    return interp_values.reshape(image.shape).cpu().numpy()


# =============================================================================
# Block Reduce (GPU avg_pool2d)
# =============================================================================


def block_reduce(
    arr: np.ndarray,
    block_size: tuple[int, ...],
    func: Callable = np.mean,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """
    Block reduce array using GPU (torch) or CPU (skimage).

    Parameters
    ----------
    arr : ndarray
        Input array (2D or 3D with channel dim first).
    block_size : tuple[int, ...]
        Reduction factors per dimension.
    func : Callable
        Reduction function (only np.mean supported on GPU).
    preserve_dtype : bool
        If True, output dtype matches input dtype. Default True.

    Returns
    -------
    reduced : ndarray
    """
    arr_np = np.asarray(arr)
    original_dtype = arr_np.dtype

    if CUDA_AVAILABLE and func == np.mean and arr_np.ndim >= 2:
        result = _block_reduce_torch(arr_np, block_size)
    else:
        result = _block_reduce_cpu(arr_np, block_size, func)

    if preserve_dtype and result.dtype != original_dtype:
        return result.astype(original_dtype)
    return result


def _block_reduce_torch(arr: np.ndarray, block_size: tuple) -> np.ndarray:
    """GPU block reduce using torch.nn.functional.avg_pool2d."""
    ndim = arr.ndim

    if ndim == 2:
        kernel = (block_size[0], block_size[1])
        t = torch.from_numpy(arr).float().cuda().unsqueeze(0).unsqueeze(0)
        out = torch.nn.functional.avg_pool2d(t, kernel, stride=kernel)
        return out.squeeze().cpu().numpy()

    elif ndim == 3:
        # For 3D arrays (C, H, W), extract spatial kernel from block_size
        if len(block_size) == 3:
            # block_size is (c_factor, h_factor, w_factor)
            # Only use spatial dimensions for avg_pool2d
            kernel = (block_size[1], block_size[2])
        else:
            # block_size is (h_factor, w_factor) - apply to spatial dims
            kernel = (block_size[0], block_size[1])
        t = torch.from_numpy(arr).float().cuda().unsqueeze(0)
        out = torch.nn.functional.avg_pool2d(t, kernel, stride=kernel)
        return out.squeeze(0).cpu().numpy()

    return _block_reduce_cpu(arr, block_size, np.mean)


# =============================================================================
# Compute SSIM (GPU conv2d)
# =============================================================================


def compute_ssim(arr1: np.ndarray, arr2: np.ndarray, win_size: int) -> float:
    """
    Compute SSIM using GPU (torch) or CPU (skimage).

    Parameters
    ----------
    arr1, arr2 : ndarray
        Input images (2D).
    win_size : int
        Window size for local statistics.

    Returns
    -------
    ssim : float
        Mean SSIM value.
    """
    arr1_np = np.asarray(arr1, dtype=np.float32)
    arr2_np = np.asarray(arr2, dtype=np.float32)

    # Compute data range once
    data_range = float(arr1_np.max() - arr1_np.min())
    if data_range == 0:
        data_range = 1.0

    if CUDA_AVAILABLE and arr1_np.ndim == 2:
        return _compute_ssim_torch(arr1_np, arr2_np, win_size, data_range)

    return float(_ssim_cpu(arr1_np, arr2_np, win_size=win_size, data_range=data_range))


def _compute_ssim_torch(
    arr1: np.ndarray, arr2: np.ndarray, win_size: int, data_range: float
) -> float:
    """GPU SSIM using torch conv2d for local statistics."""
    C1 = (_SSIM_K1 * data_range) ** 2
    C2 = (_SSIM_K2 * data_range) ** 2

    # Create uniform window
    window = torch.ones(1, 1, win_size, win_size, device="cuda") / (win_size * win_size)

    # Convert to tensors (1, 1, H, W)
    img1 = torch.from_numpy(arr1).float().cuda().unsqueeze(0).unsqueeze(0)
    img2 = torch.from_numpy(arr2).float().cuda().unsqueeze(0).unsqueeze(0)

    # Compute local means
    mu1 = F.conv2d(img1, window, padding=win_size // 2)
    mu2 = F.conv2d(img2, window, padding=win_size // 2)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = F.conv2d(img1**2, window, padding=win_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, window, padding=win_size // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=win_size // 2) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return float(ssim_map.mean().cpu())


# =============================================================================
# Utility Functions
# =============================================================================


def make_1d_profile(length: int, blend: int) -> np.ndarray:
    """
    Create a linear ramp profile over `blend` pixels at each end.

    Parameters
    ----------
    length : int
        Number of pixels.
    blend : int
        Ramp width.

    Returns
    -------
    prof : (length,) float32
        Linear profile.
    """
    blend = min(blend, length // 2)
    prof = np.ones(length, dtype=np.float32)
    if blend > 0:
        ramp = np.linspace(0, 1, blend, endpoint=False, dtype=np.float32)
        prof[:blend] = ramp
        prof[-blend:] = ramp[::-1]
    return prof


def to_numpy(arr) -> np.ndarray:
    """Convert array to numpy, handling both CPU and GPU arrays."""
    if TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
        return arr.cpu().numpy()
    return np.asarray(arr)


def to_device(arr) -> Any:
    """Move array to GPU if available, else return numpy array.

    Returns torch.Tensor on GPU if CUDA available, else np.ndarray.
    """
    if CUDA_AVAILABLE:
        return torch.from_numpy(np.asarray(arr)).cuda()
    return np.asarray(arr)
