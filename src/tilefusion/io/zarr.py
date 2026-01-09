"""
Zarr format reader/writer.

Handles Zarr v3 with OME-NGFF metadata and per_index_metadata for stage positions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import tensorstore as ts


def load_zarr_metadata(zarr_path: Path) -> Dict[str, Any]:
    """
    Load metadata from Zarr format with per_index_metadata.

    Parameters
    ----------
    zarr_path : Path
        Path to the Zarr store.

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
        - is_3d: bool
        - tensorstore: ts.TensorStore
    """
    zarr_json = zarr_path / "zarr.json"
    with open(zarr_json) as f:
        meta = json.load(f)

    attrs = meta.get("attributes", {})
    per_index_meta = attrs.get("per_index_metadata", {})
    voxel_size = attrs.get("deskewed_voxel_size_um", [1.0, 1.0, 1.0])

    # Open tensorstore to get shape
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(zarr_path)},
    }
    zarr_ts = ts.open(spec, create=False, open=True).result()
    shape = zarr_ts.shape

    # Shape is (T, P, C, Z, Y, X) for 3D or (T, P, C, Y, X) for 2D
    if len(shape) == 6:
        time_dim, position_dim, channels, z_dim, Y, X = shape
        is_3d = True
        pixel_size = (voxel_size[1], voxel_size[2])
    elif len(shape) == 5:
        time_dim, position_dim, channels, Y, X = shape
        is_3d = False
        if len(voxel_size) == 3:
            pixel_size = (voxel_size[1], voxel_size[2])
        else:
            pixel_size = (voxel_size[0], voxel_size[1])
    else:
        raise ValueError(f"Unsupported Zarr data rank {len(shape)}; expected 5 or 6.")

    n_tiles = position_dim

    # Extract tile positions from per_index_metadata
    tile_positions = []
    t_meta = per_index_meta.get("0", {})
    for p in range(position_dim):
        p_meta = t_meta.get(str(p), {})
        z_meta = p_meta.get("0", {})
        stage_pos = z_meta.get("stage_position", [0.0, 0.0, 0.0])
        if len(stage_pos) == 3:
            tile_positions.append((stage_pos[1], stage_pos[2]))
        else:
            tile_positions.append((stage_pos[0], stage_pos[1]))

    channel_names = attrs.get("channels", [f"ch{i}" for i in range(channels)])

    return {
        "n_tiles": n_tiles,
        "n_series": n_tiles,
        "shape": (Y, X),
        "channels": channels,
        "channel_names": channel_names,
        "time_dim": time_dim,
        "position_dim": position_dim,
        "pixel_size": pixel_size,
        "tile_positions": tile_positions,
        "is_3d": is_3d,
        "tensorstore": zarr_ts,
    }


def read_zarr_tile(
    zarr_ts: ts.TensorStore,
    tile_idx: int,
    is_3d: bool = False,
) -> np.ndarray:
    """
    Read all channels of a tile from Zarr format.

    Parameters
    ----------
    zarr_ts : ts.TensorStore
        Open TensorStore handle.
    tile_idx : int
        Index of the tile.
    is_3d : bool
        If True, data is 3D and max projection is applied.

    Returns
    -------
    arr : ndarray of shape (C, Y, X)
        Tile data as float32.
    """
    if is_3d:
        arr = zarr_ts[0, tile_idx, :, :, :, :].read().result()
        arr = np.max(arr, axis=1)  # Max projection along Z
    else:
        arr = zarr_ts[0, tile_idx, :, :, :].read().result()
    return arr.astype(np.float32)


def read_zarr_region(
    zarr_ts: ts.TensorStore,
    tile_idx: int,
    y_slice: slice,
    x_slice: slice,
    channel_idx: int = 0,
    is_3d: bool = False,
) -> np.ndarray:
    """
    Read a region of a single channel from Zarr format.

    Parameters
    ----------
    zarr_ts : ts.TensorStore
        Open TensorStore handle.
    tile_idx : int
        Index of the tile.
    y_slice, x_slice : slice
        Region to read.
    channel_idx : int
        Channel index.
    is_3d : bool
        If True, data is 3D.

    Returns
    -------
    arr : ndarray of shape (1, h, w)
        Tile region as float32.
    """
    if is_3d:
        arr = zarr_ts[0, tile_idx, channel_idx, :, y_slice, x_slice].read().result()
        arr = np.max(arr, axis=0)
        arr = arr[np.newaxis, :, :]
    else:
        arr = zarr_ts[0, tile_idx, channel_idx, y_slice, x_slice].read().result()
        arr = arr[np.newaxis, :, :]
    return arr.astype(np.float32)


def create_zarr_store(
    output_path: Path,
    shape: Tuple[int, ...],
    chunk_shape: Tuple[int, ...],
    shard_chunk: Tuple[int, ...],
    max_workers: int = 8,
    zarr_version: int = 2,
) -> ts.TensorStore:
    """
    Create a Zarr store with optional sharding codec.

    Parameters
    ----------
    output_path : Path
        Path for the Zarr store.
    shape : tuple
        Full array shape (T, C, Z, Y, X).
    chunk_shape : tuple
        Codec chunk shape (for v3 sharding) or main chunk shape (for v2).
    shard_chunk : tuple
        Shard chunk shape (v3 only).
    max_workers : int
        I/O concurrency limit.
    zarr_version : int
        Zarr format version: 2 (default, napari-compatible) or 3 (with sharding).

    Returns
    -------
    store : ts.TensorStore
        Open TensorStore for writing.
    """
    if zarr_version == 3:
        # Zarr v3 with sharding (better performance, but not compatible with standard napari)
        config = {
            "context": {
                "file_io_concurrency": {"limit": max_workers},
                "data_copy_concurrency": {"limit": max_workers},
            },
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(output_path)},
            "metadata": {
                "shape": list(shape),
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": list(shard_chunk)},
                },
                "chunk_key_encoding": {"name": "default"},
                "codecs": [
                    {
                        "name": "sharding_indexed",
                        "configuration": {
                            "chunk_shape": list(chunk_shape),
                            "codecs": [
                                {"name": "bytes", "configuration": {"endian": "little"}},
                                {
                                    "name": "blosc",
                                    "configuration": {
                                        "cname": "zstd",
                                        "clevel": 5,
                                        "shuffle": "bitshuffle",
                                    },
                                },
                            ],
                            "index_codecs": [
                                {"name": "bytes", "configuration": {"endian": "little"}},
                                {"name": "crc32c"},
                            ],
                            "index_location": "end",
                        },
                    }
                ],
                "data_type": "uint16",
                "dimension_names": ["t", "c", "z", "y", "x"],
            },
        }
    else:
        # Zarr v2 (compatible with standard napari and most tools)
        config = {
            "context": {
                "file_io_concurrency": {"limit": max_workers},
                "data_copy_concurrency": {"limit": max_workers},
            },
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(output_path)},
            "metadata": {
                "shape": list(shape),
                "chunks": list(chunk_shape),
                "dtype": "<u2",  # uint16 little-endian
                "order": "C",
                "fill_value": 0,  # Explicitly set for napari compatibility
                "dimension_separator": "/",  # Nested format for OME-NGFF compatibility
                "compressor": {
                    "id": "blosc",
                    "cname": "zstd",
                    "clevel": 5,
                    "shuffle": 2,  # bitshuffle
                },
            },
        }

    return ts.open(config, create=True, open=True).result()


def write_ngff_metadata(
    omezarr_path: Path,
    pixel_size: Tuple[float, float],
    center: Tuple[float, float],
    resolution_multiples: Sequence[Union[int, Sequence[int]]],
    dataset_name: str = "image",
    version: str | None = None,
    zarr_version: int = 2,
) -> None:
    """
    Write OME-NGFF multiscales metadata for Zarr.

    Parameters
    ----------
    omezarr_path : Path
        Root path of the NGFF group.
    pixel_size : tuple of (py, px)
        Base pixel size.
    center : tuple of (cx, cy)
        Center coordinates for translation.
    resolution_multiples : sequence
        Resolution factors per scale.
    dataset_name : str
        Name of the dataset node.
    version : str
        NGFF version. Defaults to "0.4" for Zarr v2 and "0.5" for Zarr v3.
    """
    # Default version based on zarr_version
    if version is None:
        version = "0.4" if zarr_version == 2 else "0.5"

    axes = [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    norm_res = [tuple(r) if hasattr(r, "__len__") else (r, r) for r in resolution_multiples]
    # 5D: (t, c, z, y, x) - z scale is 1.0 (no z downsampling)
    base_scale = [1.0, 1.0, 1.0] + [float(s) for s in pixel_size]
    trans = [0.0, 0.0, 0.0] + list(center)

    datasets = []
    prev_sp = base_scale[3:]  # Only y, x for previous spatial
    for lvl, factors in enumerate(norm_res):
        spatial = [base_scale[i + 3] * factors[i] for i in range(2)]
        scale = [1.0, 1.0, 1.0] + spatial  # t, c, z stay at 1.0
        if lvl == 0:
            translation = trans
        else:
            translation = [
                0.0,
                0.0,
                0.0,  # z translation stays 0
                datasets[-1]["coordinateTransformations"][1]["translation"][3] + 0.5 * prev_sp[0],
                datasets[-1]["coordinateTransformations"][1]["translation"][4] + 0.5 * prev_sp[1],
            ]
        datasets.append(
            {
                "path": str(lvl),  # Standard OME-NGFF path: "0", "1", "2", etc.
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale},
                    {"type": "translation", "translation": translation},
                ],
            }
        )
        prev_sp = spatial

    mult = {
        "version": version,
        "axes": axes,
        "datasets": datasets,
        "name": dataset_name,
        "@type": "ngff:Image",
    }

    if zarr_version == 3:
        # Zarr v3 format
        metadata = {
            "attributes": {"ome": {"version": version, "multiscales": [mult]}},
            "zarr_format": 3,
            "node_type": "group",
        }
        with open(omezarr_path / "zarr.json", "w") as f:
            json.dump(metadata, f, indent=2)
    else:
        # Zarr v2 format
        metadata = {"multiscales": [mult]}
        # Write .zgroup file
        with open(omezarr_path / ".zgroup", "w") as f:
            json.dump({"zarr_format": 2}, f)
        # Write .zattrs file with OME metadata
        with open(omezarr_path / ".zattrs", "w") as f:
            json.dump(metadata, f, indent=2)


def write_scale_group_metadata(scale_path: Path, zarr_version: int = 2) -> None:
    """Write metadata for a scale group.

    Parameters
    ----------
    scale_path : Path
        Path to the scale group directory.
    zarr_version : int
        Zarr format version: 2 (default) or 3.
    """
    scale_path.mkdir(parents=True, exist_ok=True)

    if zarr_version == 3:
        # Zarr v3 format
        ngff = {
            "attributes": {"_ARRAY_DIMENSIONS": ["t", "c", "z", "y", "x"]},
            "zarr_format": 3,
            "node_type": "group",
        }
        with open(scale_path / "zarr.json", "w") as f:
            json.dump(ngff, f, indent=2)
    else:
        # Zarr v2 format
        ngff = {"_ARRAY_DIMENSIONS": ["t", "c", "z", "y", "x"]}
        with open(scale_path / ".zgroup", "w") as f:
            json.dump({"zarr_format": 2}, f)
        with open(scale_path / ".zattrs", "w") as f:
            json.dump(ngff, f, indent=2)
