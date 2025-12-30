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
from skimage.measure import block_reduce
from skimage.metrics import structural_similarity as _ssim_cpu
from skimage.registration import phase_cross_correlation as _phase_cross_correlation_cpu

# Legacy compatibility
USING_GPU = CUDA_AVAILABLE
xp = np
cp = None


def phase_cross_correlation(reference_image, moving_image, upsample_factor=1, **kwargs):
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
        Translation invariant normalized RMS error (placeholder).
    phasediff : float
        Global phase difference (placeholder).
    """
    ref_np = np.asarray(reference_image)
    mov_np = np.asarray(moving_image)

    if CUDA_AVAILABLE and ref_np.ndim == 2:
        return _phase_cross_correlation_torch(ref_np, mov_np, upsample_factor)
    return _phase_cross_correlation_cpu(ref_np, mov_np, upsample_factor=upsample_factor, **kwargs)


def _phase_cross_correlation_torch(reference_image: np.ndarray, moving_image: np.ndarray,
                                    upsample_factor: int = 1) -> tuple:
    """GPU phase cross-correlation using torch FFT."""
    ref = torch.from_numpy(reference_image.astype(np.float32)).cuda()
    mov = torch.from_numpy(moving_image.astype(np.float32)).cuda()

    # Compute cross-power spectrum
    ref_fft = torch.fft.fft2(ref)
    mov_fft = torch.fft.fft2(mov)
    cross_power = ref_fft * torch.conj(mov_fft)
    eps = 1e-10
    cross_power = cross_power / (torch.abs(cross_power) + eps)

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
        dy = (neighborhood[0, 1].item() - neighborhood[2, 1].item()) / denom if abs(denom) > 1e-10 else 0.0
    else:
        dy = 0.0

    # X direction parabolic fit
    if neighborhood[1, 0].item() != center_val or neighborhood[1, 2].item() != center_val:
        denom = 2 * (2 * center_val - neighborhood[1, 0].item() - neighborhood[1, 2].item())
        dx = (neighborhood[1, 0].item() - neighborhood[1, 2].item()) / denom if abs(denom) > 1e-10 else 0.0
    else:
        dx = 0.0

    dy = max(-0.5, min(0.5, dy))
    dx = max(-0.5, min(0.5, dx))

    return np.array([float(peak_y) + dy, float(peak_x) + dx])


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
    return np.asarray(arr)


def to_device(arr):
    """Move array to current device."""
    return np.asarray(arr)
