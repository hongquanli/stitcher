"""
Flatfield correction module using BaSiCPy.

Provides functions to calculate and apply flatfield (and optionally darkfield)
correction for microscopy images.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from basicpy import BaSiC

    HAS_BASICPY = True
except ImportError:
    HAS_BASICPY = False


def calculate_flatfield(
    tiles: List[np.ndarray],
    use_darkfield: bool = False,
    constant_darkfield: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Calculate flatfield (and optionally darkfield) using BaSiCPy.

    Parameters
    ----------
    tiles : list of ndarray
        List of tile images, each with shape (C, Y, X).
    use_darkfield : bool
        Whether to also compute darkfield correction.
    constant_darkfield : bool
        If True, darkfield is reduced to a single constant value (median) per
        channel. This is physically appropriate since dark current is typically
        uniform across the sensor. Default is True.

    Returns
    -------
    flatfield : ndarray
        Flatfield correction array with shape (C, Y, X), float32.
    darkfield : ndarray or None
        Darkfield correction array with shape (C, Y, X), or None if not computed.
        If constant_darkfield=True, each channel slice will be a constant value.

    Raises
    ------
    ImportError
        If basicpy is not installed.
    ValueError
        If tiles list is empty or tiles have inconsistent shapes.
    """
    if not HAS_BASICPY:
        raise ImportError(
            "basicpy is required for flatfield calculation. " "Install with: pip install basicpy"
        )

    if not tiles:
        raise ValueError("tiles list is empty")

    # Get shape from first tile
    n_channels = tiles[0].shape[0]
    tile_shape = tiles[0].shape[1:]  # (Y, X)

    # Validate all tiles have same shape
    for i, tile in enumerate(tiles):
        if tile.shape[0] != n_channels:
            raise ValueError(f"Tile {i} has {tile.shape[0]} channels, expected {n_channels}")
        if tile.shape[1:] != tile_shape:
            raise ValueError(f"Tile {i} has shape {tile.shape[1:]}, expected {tile_shape}")

    # Calculate flatfield per channel
    flatfield = np.zeros((n_channels,) + tile_shape, dtype=np.float32)
    darkfield = np.zeros((n_channels,) + tile_shape, dtype=np.float32) if use_darkfield else None

    for ch in range(n_channels):
        # Stack channel data from all tiles: shape (n_tiles, Y, X)
        channel_stack = np.stack([tile[ch] for tile in tiles], axis=0)

        # Create BaSiC instance and fit
        basic = BaSiC(get_darkfield=use_darkfield, smoothness_flatfield=1.0)
        basic.fit(channel_stack)

        flatfield[ch] = basic.flatfield.astype(np.float32)

        if use_darkfield:
            if constant_darkfield:
                # Use median value for constant darkfield (more robust than mean)
                df_value = np.median(basic.darkfield)
                darkfield[ch] = np.full(tile_shape, df_value, dtype=np.float32)
            else:
                darkfield[ch] = basic.darkfield.astype(np.float32)

    return flatfield, darkfield


def apply_flatfield(
    tile: np.ndarray,
    flatfield: np.ndarray,
    darkfield: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply flatfield correction to a tile.

    Formula:
        If darkfield is provided: corrected = (raw - darkfield) / flatfield
        Otherwise: corrected = raw / flatfield

    Parameters
    ----------
    tile : ndarray
        Input tile with shape (C, Y, X).
    flatfield : ndarray
        Flatfield correction array with shape (C, Y, X).
    darkfield : ndarray, optional
        Darkfield correction array with shape (C, Y, X).

    Returns
    -------
    corrected : ndarray
        Corrected tile with shape (C, Y, X), same dtype as input.
    """
    # Avoid division by zero
    flatfield_safe = np.where(flatfield > 1e-6, flatfield, 1.0)

    if darkfield is not None:
        corrected = (tile - darkfield) / flatfield_safe
    else:
        corrected = tile / flatfield_safe

    return corrected.astype(tile.dtype)


def apply_flatfield_region(
    region: np.ndarray,
    flatfield: np.ndarray,
    darkfield: Optional[np.ndarray],
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray:
    """
    Apply flatfield correction to a tile region.

    Parameters
    ----------
    region : ndarray
        Input region with shape (C, h, w) or (h, w).
    flatfield : ndarray
        Full flatfield correction array with shape (C, Y, X).
    darkfield : ndarray, optional
        Full darkfield correction array with shape (C, Y, X).
    y_slice, x_slice : slice
        Slices defining the region within the full tile.

    Returns
    -------
    corrected : ndarray
        Corrected region with same shape as input.
    """
    # Extract corresponding flatfield/darkfield regions
    if region.ndim == 2:
        ff_region = flatfield[0, y_slice, x_slice]
        df_region = darkfield[0, y_slice, x_slice] if darkfield is not None else None
    else:
        ff_region = flatfield[:, y_slice, x_slice]
        df_region = darkfield[:, y_slice, x_slice] if darkfield is not None else None

    # Avoid division by zero
    ff_safe = np.where(ff_region > 1e-6, ff_region, 1.0)

    if df_region is not None:
        corrected = (region - df_region) / ff_safe
    else:
        corrected = region / ff_safe

    return corrected.astype(region.dtype)


def save_flatfield(
    path: Path,
    flatfield: np.ndarray,
    darkfield: Optional[np.ndarray] = None,
) -> None:
    """
    Save flatfield (and optionally darkfield) to a .npy file.

    Parameters
    ----------
    path : Path
        Output path (should end with .npy).
    flatfield : ndarray
        Flatfield array with shape (C, Y, X).
    darkfield : ndarray, optional
        Darkfield array with shape (C, Y, X).
    """
    data = {
        "flatfield": flatfield.astype(np.float32),
        "darkfield": darkfield.astype(np.float32) if darkfield is not None else None,
        "channels": flatfield.shape[0],
        "shape": flatfield.shape[1:],
    }
    np.save(path, data, allow_pickle=True)


def load_flatfield(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load flatfield (and optionally darkfield) from a .npy file.

    Parameters
    ----------
    path : Path
        Path to .npy file.

    Returns
    -------
    flatfield : ndarray
        Flatfield array with shape (C, Y, X).
    darkfield : ndarray or None
        Darkfield array with shape (C, Y, X), or None if not present.
    """
    data = np.load(path, allow_pickle=True).item()
    flatfield = data["flatfield"]
    darkfield = data.get("darkfield", None)
    return flatfield, darkfield
