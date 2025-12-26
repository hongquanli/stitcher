"""
Individual TIFFs format reader.

Reads folder format with individual TIFF files per tile/channel and coordinates.csv.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tifffile


def load_individual_tiffs_metadata(folder_path: Path) -> Dict[str, Any]:
    """
    Load metadata from individual TIFFs folder format.

    Expected structure:
    - Individual TIFF files: manual_{fov}_{z}_{channel}.tiff
    - coordinates.csv with fov, x (mm), y (mm)
    - Optional: acquisition parameters.json

    Parameters
    ----------
    folder_path : Path
        Path to the data folder.

    Returns
    -------
    metadata : dict
        Dictionary containing:
        - n_tiles: int
        - shape: (Y, X)
        - channels: int
        - channel_names: list of str
        - pixel_size: (py, px)
        - tile_positions: list of (y, x) tuples
        - fov_indices: list of int
        - image_folder: Path
    """
    # Find the subfolder containing images (usually "0" for single z-level)
    subfolders = [d for d in folder_path.iterdir() if d.is_dir()]
    if subfolders:
        image_folder = subfolders[0]
    else:
        image_folder = folder_path

    # Load coordinates
    coords_path = image_folder / "coordinates.csv"
    if not coords_path.exists():
        coords_path = folder_path / "coordinates.csv"
    if not coords_path.exists():
        raise FileNotFoundError(f"coordinates.csv not found in {folder_path}")

    coords = pd.read_csv(coords_path)
    n_tiles = len(coords)

    # Get channel names from TIFF filenames
    tiff_files = list(image_folder.glob("*.tiff"))
    if not tiff_files:
        tiff_files = list(image_folder.glob("*.tif"))

    channel_names = set()
    for f in tiff_files:
        parts = f.stem.split("_")
        if len(parts) >= 4:
            channel_name = "_".join(parts[3:])
            channel_names.add(channel_name)

    channel_names = sorted(channel_names)
    channels = len(channel_names)

    # Read first image to get dimensions
    first_fov = coords["fov"].iloc[0]
    first_channel = channel_names[0]
    first_img_path = image_folder / f"manual_{first_fov}_0_{first_channel}.tiff"
    if not first_img_path.exists():
        first_img_path = image_folder / f"manual_{first_fov}_0_{first_channel}.tif"

    first_img = tifffile.imread(first_img_path)
    Y, X = first_img.shape[-2:]

    # Load pixel size from acquisition parameters
    params_path = folder_path / "acquisition parameters.json"
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        magnification = params.get("objective", {}).get("magnification", 10.0)
        sensor_pixel_um = params.get("sensor_pixel_size_um", 7.52)
        pixel_size_um = sensor_pixel_um / magnification
    else:
        pixel_size_um = 0.752  # Default for 10x

    pixel_size = (pixel_size_um, pixel_size_um)

    # Convert mm coordinates to µm and store as (y, x)
    tile_positions = []
    for _, row in coords.iterrows():
        x_um = row["x (mm)"] * 1000
        y_um = row["y (mm)"] * 1000
        tile_positions.append((y_um, x_um))

    fov_indices = coords["fov"].tolist()

    return {
        "n_tiles": n_tiles,
        "n_series": n_tiles,
        "shape": (Y, X),
        "channels": channels,
        "channel_names": channel_names,
        "time_dim": 1,
        "position_dim": n_tiles,
        "pixel_size": pixel_size,
        "tile_positions": tile_positions,
        "fov_indices": fov_indices,
        "image_folder": image_folder,
    }


def read_individual_tiffs_tile(
    image_folder: Path,
    channel_names: List[str],
    fov_indices: List[int],
    tile_idx: int,
) -> np.ndarray:
    """
    Read all channels of a tile from individual TIFFs folder format.

    Parameters
    ----------
    image_folder : Path
        Path to folder containing TIFF files.
    channel_names : list of str
        Channel names.
    fov_indices : list of int
        FOV indices for each tile.
    tile_idx : int
        Index of the tile.

    Returns
    -------
    arr : ndarray of shape (C, Y, X)
        Tile data as float32.
    """
    fov = fov_indices[tile_idx]

    channels = []
    for channel_name in channel_names:
        img_path = image_folder / f"manual_{fov}_0_{channel_name}.tiff"
        if not img_path.exists():
            img_path = image_folder / f"manual_{fov}_0_{channel_name}.tif"
        arr = tifffile.imread(img_path)
        channels.append(arr)

    stacked = np.stack(channels, axis=0)
    return stacked.astype(np.float32)


def read_individual_tiffs_region(
    image_folder: Path,
    channel_names: List[str],
    fov_indices: List[int],
    tile_idx: int,
    y_slice: slice,
    x_slice: slice,
    channel_idx: int = 0,
) -> np.ndarray:
    """
    Read a region of a single channel from individual TIFFs format.

    Parameters
    ----------
    image_folder : Path
        Path to folder containing TIFF files.
    channel_names : list of str
        Channel names.
    fov_indices : list of int
        FOV indices for each tile.
    tile_idx : int
        Index of the tile.
    y_slice, x_slice : slice
        Region to read.
    channel_idx : int
        Channel index.

    Returns
    -------
    arr : ndarray of shape (1, h, w)
        Tile region as float32.
    """
    fov = fov_indices[tile_idx]
    channel_name = channel_names[channel_idx]

    img_path = image_folder / f"manual_{fov}_0_{channel_name}.tiff"
    if not img_path.exists():
        img_path = image_folder / f"manual_{fov}_0_{channel_name}.tif"

    arr = tifffile.imread(img_path)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    return arr[:, y_slice, x_slice].astype(np.float32)
